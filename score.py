"""Language models asked whether each SCORE replication succeeded, scored against the outcomes and compared with the human forecasters.

Chat models (chat_servers) receive the prompt as one user message and answer in the format "Answer: yes or no / Probability: 0 to 100", enforced by vLLM's structured output. Two scores are read from each response: the token score, p(yes) / (p(yes) + p(no)) from the top 20 log-probabilities at the answer word, and the stated score, the written probability divided by 100. With thinking on, the model reasons before answering and both are read after the reasoning.

Base models (base_servers) receive the same prompt followed by a blank line and "Answer:", as plain text, and are scored by L = log p(" yes") - log p(" no") for the next token, from vLLM's prompt log-probabilities.

Responses and log-probabilities are cached in JSON lines files (results/qwen_responses.jsonl for chat models, results/olmo_perplexities.jsonl for base models), keyed by model, prompt and settings. Only prompts missing from a cache are sent; with everything cached, no server is contacted. A server that is needed must be running at the URL given for it in models.toml.

Templates: with_original_finding (abstract, claim, original finding; the full prompt), claim_only (abstract and claim; no finding), no_abstract (claim and original finding) and claim_alone (claim only).

For each run, a CSV with one row per claim is written to out_dir, and a line of summary.csv: weighted AUC and weighted accuracy. Options can be edited in CONFIG below or given on the command line with the same names, for example:

    uv run python score.py --chat_servers qwen27 --qwen_thinking_on false
    uv run python score.py --predictor_chat false --base_servers olmo32 olmo7b
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import openai

import serving

# ---------------------------------------------------------------- configuration

CONFIG = {
    # predictors
    "predictor_base": True,       # every server in base_servers
    "predictor_replicats": True,
    "predictor_markets": True,
    "predictor_chat": True,       # every server in chat_servers
    "template_claim_only": True,
    "template_with_original_finding": True,
    # prompt ablation: without the abstract, and the claim alone (no abstract, no original finding)
    "template_no_abstract": False,
    "template_claim_alone": False,
    # base models: the offset that turns L into a prediction, fitted on all rows ("fitted") or leaving each paper out ("loo")
    "mode_fitted": True,
    "mode_loo": True,
    # chat models: thinking on, off, or both; the stated score, the token score, or both
    "qwen_thinking_on": True,
    "qwen_thinking_off": True,
    "qwen_score_stated": True,
    "qwen_score_token": True,
    # accuracy of humans and chat models: "threshold" (p >= 0.5 counts as "succeeded") or "hedged" (credit = probability given to the actual outcome)
    "human_accuracy": "threshold",
    # row filters: lists of values to run; "all" collapses over that filter
    "subsets": ["all", "human", "highpower"],
    "data_types": ["all"],
    # files (relative paths are relative to this script's folder)
    "data_file": "data/SCORE_replicated_claims.csv",
    "out_dir": "results",
    "cache_file": "results/olmo_perplexities.jsonl",
    # base models (names in models.toml, kind "base") and number of requests in flight at once
    "base_servers": ["olmo32"],
    "concurrency": 32,
    # chat models (names in models.toml, kind "chat"), their response cache, requests in flight at once, and the cap on generated tokens (reasoning plus answer)
    "chat_servers": ["qwen27"],
    "qwen_cache_file": "results/qwen_responses.jsonl",
    "qwen_concurrency": 64,
    "qwen_max_tokens": 16384,
    # with thinking on, the reasoning is cut off after this many tokens and the model must then answer (vLLM's thinking_token_budget); 0 means no budget
    "qwen_thinking_budget": 4096,
    # with thinking on, a response that ends with no answer (usually cut off by qwen_max_tokens) is given one: a second request sends the prompt, the reasoning so far (at most qwen_max_tokens tokens of it) and the end-of-reasoning marker, with the answer format enforced (force_answer)
    "qwen_force_answer": False,
}

ROOT = Path(__file__).resolve().parent

BOOL_OPTIONS = (
    "predictor_base", "predictor_replicats", "predictor_markets", "predictor_chat",
    "template_claim_only", "template_with_original_finding", "template_no_abstract", "template_claim_alone", "mode_fitted", "mode_loo",
    "qwen_thinking_on", "qwen_thinking_off", "qwen_score_stated", "qwen_score_token", "qwen_force_answer",
)
SUBSETS = ("all", "human", "highpower")
DATA_TYPES = ("all", "New", "Secondary")
HUMAN_ACCURACY = ("threshold", "hedged")
TEMPLATES = ("claim_only", "with_original_finding")
ABLATION_TEMPLATES = ("no_abstract", "claim_alone")
ALL_TEMPLATES = TEMPLATES + ABLATION_TEMPLATES
MODES = ("fitted", "loo")
HUMANS = ("replicats", "markets")
ENDINGS = ("yes_no",)
ENDING_CONTINUATIONS = {"yes_no": (" yes", " no")}
THINKING = ("on", "off")
QWEN_SCORES = ("stated", "token")
QWEN_TEMPERATURE = 0
QWEN_TOP_LOGPROBS = 20
QWEN_INSTRUCTION = (
    "Answer yes or no, and give the probability from 0 to 100 that the replication succeeded, in exactly this format:\n"
    "Answer: yes or no\n"
    "Probability: a whole number from 0 to 100"
)
QWEN_FORMAT = r"Answer: (yes|no)\nProbability: (100|[1-9]?[0-9])"
# model scores are rounded to this many decimal places: vLLM computes logits in bfloat16, so equal scores reach us differing only by floating-point noise (about 1e-7), which would otherwise break their ties arbitrarily
SCORE_DECIMALS = 6

COLUMNS = (
    "Paper", "Journal", "Year", "Field", "DOI", "Data", "Replicated", "Replication p", "Original N", "Replication N",
    "Claim ID", "Claim tested", "Abstract", "Original finding", "Replication finding", "Paper weight",
    "Human set", "Human high-power set", "repliCATS estimate", "Replication Markets estimate",
    "Synthetic Markets estimate", "MACROSCORE estimate", "A+ estimate",
)
REPLICATED_LABELS = ("Yes", "No (n.s.)", "No (opposite)", "No (p n/a)")
NOT_RATED = "not rated"

OLMO_YES_NO_FILE_COLUMNS = (
    "row", "claim_id", "paper", "data_type", "weight", "ppl_no", "ppl_yes",
    "score", "offset", "final_score", "prediction", "correct_answer", "correct",
)
HUMAN_FILE_COLUMNS = ("row", "claim_id", "paper", "data_type", "weight", "estimate", "prediction", "correct_answer", "credit")
QWEN_FILE_COLUMNS = (
    "row", "claim_id", "paper", "data_type", "weight", "answer", "stated_probability", "token_probability", "token_note",
    "score", "prediction", "correct_answer", "credit", "answer_disagrees",
)
SUMMARY_COLUMNS = (
    "predictor", "server", "subset", "data_type", "template", "mode", "olmo_ending", "thinking", "qwen_score", "human_accuracy",
    "n_rows", "n_papers", "sum_weights", "weighted_auc", "fitted_offset", "weighted_accuracy",
)


class ScoreError(Exception):
    """A failure that stops the run: bad configuration, a failed integrity check, missing log-probabilities or responses, or a response without an answer."""


class DataError(ScoreError):
    """The data file failed an integrity check."""


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Record:
    row: int                # position in the data file, 1 to 274
    claim_id: str
    paper_id: str           # Claim ID before the first underscore
    paper: str              # short citation
    data_type: str          # "New" or "Secondary"
    replicated: str
    claim: str
    abstract: str
    finding: str            # Original finding
    paper_weight: float     # column 16 (checked only; weights are recomputed per selection)
    human_set: bool
    highpower: bool
    replicats: float | None
    markets: float | None

    @property
    def succeeded(self) -> bool:
        return self.replicated == "Yes"

    @property
    def correct_answer(self) -> str:
        return "succeeded" if self.succeeded else "failed"


def _estimate(text: str, where: str) -> float | None:
    if text == NOT_RATED:
        return None
    try:
        v = float(text)
    except ValueError:
        raise DataError(f"{where}: estimate {text!r} is neither a number nor {NOT_RATED!r}") from None
    if not 0 <= v <= 1:
        raise DataError(f"{where}: estimate {v} is outside [0, 1]")
    return v


def _yes_no(text: str, where: str) -> bool:
    if text not in ("yes", "no"):
        raise DataError(f"{where}: expected 'yes' or 'no', found {text!r}")
    return text == "yes"


def load_data(path: str | Path) -> list[Record]:
    """Read the data file and run the integrity checks, raising DataError on any failure."""
    with open(path, newline="", encoding="utf-8") as f:
        table = list(csv.reader(f))
    if not table or tuple(table[0]) != COLUMNS:
        raise DataError(f"{path}: header does not match the 23 columns of data/README.md")
    records = []
    for i, cells in enumerate(table[1:], start=1):
        where = f"data row {i}"
        if len(cells) != len(COLUMNS):
            raise DataError(f"{where}: {len(cells)} fields, expected {len(COLUMNS)}")
        c = dict(zip(COLUMNS, cells))
        empty = [k for k, v in c.items() if v == ""]
        if empty:
            raise DataError(f"{where}: empty field(s) {', '.join(empty)}")
        if c["Data"] not in ("New", "Secondary"):
            raise DataError(f"{where}: Data is {c['Data']!r}, expected 'New' or 'Secondary'")
        if c["Replicated"] not in REPLICATED_LABELS:
            raise DataError(f"{where}: Replicated is {c['Replicated']!r}")
        try:
            pw = float(c["Paper weight"])
        except ValueError:
            raise DataError(f"{where}: Paper weight {c['Paper weight']!r} is not a number") from None
        records.append(Record(
            row=i,
            claim_id=c["Claim ID"],
            paper_id=c["Claim ID"].split("_", 1)[0],
            paper=c["Paper"],
            data_type=c["Data"],
            replicated=c["Replicated"],
            claim=c["Claim tested"],
            abstract=c["Abstract"],
            finding=c["Original finding"],
            paper_weight=pw,
            human_set=_yes_no(c["Human set"], where),
            highpower=_yes_no(c["Human high-power set"], where),
            replicats=_estimate(c["repliCATS estimate"], where),
            markets=_estimate(c["Replication Markets estimate"], where),
        ))
    check_integrity(records)
    return records


def _expect(what: str, found, expected) -> None:
    if found != expected:
        raise DataError(f"integrity check failed: {what} is {found}, expected {expected}")


def _set_counts(rows: list[Record]) -> tuple[int, int, int, int]:
    return (len(rows), sum(r.data_type == "New" for r in rows), sum(r.data_type == "Secondary" for r in rows), sum(r.succeeded for r in rows))


def check_integrity(records: list[Record]) -> None:
    """Counts of rows, papers, sets, estimates and outcomes, and unique Claim IDs. Field-level checks (23 columns, no empty field, labels) are made while loading."""
    _expect("number of data records", len(records), 274)
    _expect("number of distinct paper IDs", len({r.paper_id for r in records}), 164)
    _expect("number of distinct Claim IDs", len({r.claim_id for r in records}), 274)
    _expect("Human set (rows, New, Secondary, replicated)", _set_counts([r for r in records if r.human_set]), (135, 84, 51, 66))
    _expect("Human high-power set (rows, New, Secondary, replicated)", _set_counts([r for r in records if r.highpower]), (57, 37, 20, 25))
    _expect("rows with a numeric repliCATS estimate", sum(r.replicats is not None for r in records), 138)
    _expect("rows with a numeric Replication Markets estimate", sum(r.markets is not None for r in records), 138)
    missing = [r.claim_id for r in records if (r.human_set or r.highpower) and (r.replicats is None or r.markets is None)]
    _expect("human-set or high-power rows lacking an estimate", missing, [])
    _expect("rows with Replicated = Yes", sum(r.succeeded for r in records), 151)
    _expect("sum of Paper weight (rounded to 9 places)", round(sum(r.paper_weight for r in records), 9), 164)


def select(records: list[Record], subset: str, data_type: str) -> list[Record]:
    """Rows of a subset ("all", "human", "highpower") and data type ("all", "New", "Secondary"), in data-file order."""
    keep = {"all": lambda r: True, "human": lambda r: r.human_set, "highpower": lambda r: r.highpower}[subset]
    return [r for r in records if keep(r) and (data_type == "all" or r.data_type == data_type)]


def weights(rows: list[Record]) -> list[float]:
    """w_i = 1 / number of selected rows from the same paper."""
    n: dict[str, int] = {}
    for r in rows:
        n[r.paper_id] = n.get(r.paper_id, 0) + 1
    return [1 / n[r.paper_id] for r in rows]


# ---------------------------------------------------------------- metrics


def weighted_accuracy(w, credit) -> float:
    w = np.asarray(w, dtype=float)
    return float(np.dot(w, np.asarray(credit, dtype=float)) / w.sum())


def weighted_auc(x, succeeded, w) -> float:
    """Pairwise weighted AUC: ties count 0.5. NaN if either outcome is absent."""
    x, s, w = np.asarray(x, dtype=float), np.asarray(succeeded, dtype=bool), np.asarray(w, dtype=float)
    xp, wp, xn, wn = x[s], w[s], x[~s], w[~s]
    if len(xp) == 0 or len(xn) == 0:
        return math.nan
    win = (xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])
    return float(wp @ win @ wn / (wp.sum() * wn.sum()))


def human_prediction(p: float) -> str:
    return "succeeded" if p >= 0.5 else "failed"


def human_credit(p: float, succeeded: bool, mode: str) -> float:
    if mode == "threshold":
        return float((p >= 0.5) == succeeded)
    if mode == "hedged":
        return p if succeeded else 1 - p
    raise ScoreError(f"human_accuracy must be one of {HUMAN_ACCURACY}, not {mode!r}")


def fit_offset(d, succeeded, w) -> float:
    """The offset b with the highest weighted accuracy of (score + b > 0), ties to the smallest |b|, then the smaller b."""
    d, s, w = np.asarray(d, dtype=float), np.asarray(succeeded, dtype=bool), np.asarray(w, dtype=float)
    if len(d) == 0:
        raise ScoreError("cannot fit an offset on no rows")
    u = np.unique(-d)
    cands = np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])
    correct = (d[None, :] + cands[:, None] > 0) == s[None, :]
    acc = np.round(correct @ w / w.sum(), 12)
    best = min(range(len(cands)), key=lambda k: (-acc[k], abs(cands[k]), cands[k]))
    return float(cands[best])


def loo_offsets(d, succeeded, w, papers) -> list[float]:
    """Leave one paper out: each row's offset is fitted on the other papers' rows, with the full selection's weights."""
    d, s, w, papers = np.asarray(d, dtype=float), np.asarray(succeeded, dtype=bool), np.asarray(w, dtype=float), np.asarray(papers)
    offsets = np.empty(len(d))
    for p in dict.fromkeys(papers.tolist()):
        held = papers == p
        offsets[held] = fit_offset(d[~held], s[~held], w[~held])
    return offsets.tolist()


# ---------------------------------------------------------------- prompts and base-model log-probabilities


def _template_parts(rec: Record, template: str) -> list[str]:
    """The paragraphs of the prompt up to and including the question, shared by base and chat models. The ablation templates drop the abstract, and its opening line becomes "Here is the claim the replicating team tested:"."""
    if template in ("claim_only", "with_original_finding"):
        parts = [
            "The following paper was subjected to a replication attempt. Here is the original paper's abstract:",
            rec.abstract,
            "And, here is the claim the replicating team tested:",
            rec.claim,
        ]
    elif template in ABLATION_TEMPLATES:
        parts = ["The following paper was subjected to a replication attempt. Here is the claim the replicating team tested:", rec.claim]
    else:
        raise ScoreError(f"unknown template {template!r}")
    if template in ("with_original_finding", "no_abstract"):
        parts += ["In particular the original paper found:", rec.finding]
    parts.append("What do you think happened in the replication attempt? Did it find a statistically significant result in the same direction as the original?")
    return parts


def context(rec: Record, template: str, ending: str = "yes_no") -> str:
    """A base model's prompt: the chat prompt, with its answer-format instruction, followed by a blank line and "Answer:"."""
    if ending != "yes_no":
        raise ScoreError(f"unknown ending {ending!r}")
    return qwen_prompt(rec, template) + "\n\nAnswer:"


class OlmoScorer:
    """A base model's tokenization and prompt log-probabilities from its vLLM server, over HTTP. Safe to share between threads."""

    def __init__(self, server: str):
        url, self.model = serving.endpoint(server)
        self.base = url.removesuffix("/v1")
        self.client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=64))

    def _post(self, path: str, body: dict) -> dict:
        r = self.client.post(f"{self.base}{path}", json=body)
        if r.status_code != 200:
            raise ScoreError(f"POST {path} returned {r.status_code}: {r.text[:300]}")
        return r.json()

    def tokenize(self, text: str) -> list[int]:
        """Token IDs with the tokenizer's default special-token handling. Raises if the text plus one generated token would not fit the server's context."""
        out = self._post("/tokenize", {"model": self.model, "prompt": text})
        if out["count"] + 1 > out["max_model_len"]:
            raise ScoreError(f"prompt of {out['count']} tokens exceeds the server's context length {out['max_model_len']}")
        return out["tokens"]

    def prompt_logprobs(self, ids: list[int]) -> list:
        """vLLM's prompt_logprobs for a token-ID prompt: None for the first token, then a dict keyed by token ID string for each later token."""
        out = self._post("/v1/completions", {"model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
        return out["choices"][0]["prompt_logprobs"]


def continuation_logprobs(ctx_ids: list[int], full_ids: list[int], plp: list) -> list[float]:
    """Log-probabilities of the continuation tokens: those of context+continuation after the context's own tokens."""
    n = len(ctx_ids)
    if full_ids[:n] != ctx_ids or len(full_ids) <= n:
        raise ScoreError("tokens of context + continuation do not start with the context's tokens, so the continuation tokens are ambiguous")
    return [plp[i][str(full_ids[i])]["logprob"] for i in range(n, len(full_ids))]


def perplexity(logprobs: list[float]) -> float:
    """exp(-(1/T) * sum of log p) over the T continuation tokens. For the single-token continuations " yes" and " no", log PPL = -log p."""
    return math.exp(-sum(logprobs) / len(logprobs))


def cache_key(model: str, ctx: str, continuation: str) -> str:
    return hashlib.sha256(json.dumps([model, ctx, continuation]).encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                cache[e["key"]] = e
            except (ValueError, KeyError):
                continue  # a line cut short by an interrupted run; its pair is recomputed
    return cache


def _score_row(scorer, rec: Record, template: str, ending: str = "yes_no") -> list[dict]:
    ctx = context(rec, template, ending)
    ctx_ids = scorer.tokenize(ctx)
    entries = []
    for cont in ENDING_CONTINUATIONS[ending]:
        full = scorer.tokenize(ctx + cont)
        lps = continuation_logprobs(ctx_ids, full, scorer.prompt_logprobs(full))
        entries.append({
            "key": cache_key(scorer.model, ctx, cont),
            "model": scorer.model,
            "template": template,
            "ending": ending,
            "row": rec.row,
            "claim_id": rec.claim_id,
            "continuation": cont,
            "n_context_tokens": len(ctx_ids),
            "first_token_id": ctx_ids[0],
            "continuation_token_ids": full[len(ctx_ids):],
            "continuation_logprobs": lps,
            "ppl": perplexity(lps),
        })
    return entries


def _require(server: str) -> None:
    """Stop with a ScoreError unless the server answers at its URL in models.toml."""
    try:
        serving.require(server)
    except serving.ServingError as e:
        raise ScoreError(str(e)) from None


def get_perplexities(records: list[Record], templates: list[str], cache_path: Path, server: str, concurrency: int, scorer=None, endings: list[str] = ("yes_no",)) -> dict[tuple[str, int, str], float]:
    """PPL for every (template, row, continuation), taken from the cache or computed on the server and added to it.

    The server is contacted only if something is missing. A given scorer (a fake, in the tests) is used instead.
    """
    model = scorer.model if scorer is not None else serving.endpoint(server)[1]
    cache = load_cache(cache_path)
    jobs = [(rec, t, e) for e in endings for t in templates for rec in records]
    todo = [(rec, t, e) for rec, t, e in jobs if any(cache_key(model, context(rec, t, e), c) not in cache for c in ENDING_CONTINUATIONS[e])]
    if todo:
        if scorer is None:
            print(f"{len(todo)} of {len(jobs)} prompts have no cached log-probabilities from {server}.", flush=True)
            _require(server)
        _compute_perplexities(todo, scorer or OlmoScorer(server), cache, cache_path, server, concurrency)
    return {(t, rec.row, c): cache[cache_key(model, context(rec, t, e), c)]["ppl"] for rec, t, e in jobs for c in ENDING_CONTINUATIONS[e]}


def _compute_perplexities(todo: list, scorer, cache: dict, cache_path: Path, server: str, concurrency: int) -> None:
    print(f"Computing log-probabilities for {len(todo)} prompts on {server} ({concurrency} at a time)...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(_score_row, scorer, rec, t, e) for rec, t, e in todo]
        for done, fut in enumerate(as_completed(futures), start=1):
            for e in fut.result():
                cache[e["key"]] = e
                f.write(json.dumps(e) + "\n")
            f.flush()
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- chat models: prompts and responses


def qwen_prompt(rec: Record, template: str) -> str:
    """The prompt up to the question, followed by a blank line and the answer-format instruction."""
    return "\n\n".join(_template_parts(rec, template) + [QWEN_INSTRUCTION])


def parse_answer(content: str | None) -> tuple[str, int] | None:
    """("yes" or "no", probability 0 to 100) from a response in the required format, or None."""
    m = re.fullmatch(QWEN_FORMAT, (content or "").strip())
    return (m.group(1), int(m.group(2))) if m else None


def token_probability(top: list[list]) -> tuple[float, str]:
    """p(yes) / (p(yes) + p(no)) from the top log-probabilities at the answer word, and a note if either word was missing.

    Tokens are matched ignoring surrounding spaces, and spellings of the same word are summed. A word missing from the list is given the lowest log-probability returned (masked tokens, which vLLM reports as about -9999, are ignored). The log odds log p(yes) - log p(no) are rounded to SCORE_DECIMALS places before conversion, so that equal scores are exactly equal.
    """
    lps = {"yes": [], "no": []}
    for tok, lp in top:
        w = tok.strip().lower()
        if w in lps:
            lps[w].append(lp)
    floor = min((lp for _, lp in top if lp > -1000), default=-1000.0)
    missing = [w for w in lps if not lps[w]]
    ly, ln = (float(np.logaddexp.reduce(lps[w])) if lps[w] else floor for w in ("yes", "no"))
    return 1 / (1 + math.exp(-round(ly - ln, SCORE_DECIMALS))), " ".join(f"{w} missing" for w in missing)


def qwen_key(model: str, prompt: str, thinking: str, budget: int = 0) -> str:
    """The budget is part of the key only with thinking on, so it never invalidates thinking-off responses."""
    parts = [model, prompt, thinking, QWEN_TEMPERATURE, QWEN_FORMAT] + ([budget] if thinking == "on" and budget else [])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class QwenClient:
    """Chat requests to a chat model's vLLM server, with the answer constrained to QWEN_FORMAT. Safe to share between threads."""

    def __init__(self, server: str):
        url, self.model = serving.endpoint(server)
        self.base = url.removesuffix("/v1")
        # long timeout: with thinking on, one request can generate tens of thousands of tokens while sharing the server with many others
        # (olmo32t at 12 requests at once makes about 2.7 tokens/s each, so a 60,000-token response takes over 6 hours)
        self.client = openai.OpenAI(base_url=url, api_key="EMPTY", timeout=24 * 3600, max_retries=0)

    def fit(self, prompt: str, max_tokens: int, budget: int) -> tuple[int, int]:
        """max_tokens and the reasoning budget, lowered if the prompt plus max_tokens would exceed the server's context length (vLLM rejects such a request). The budget is lowered by the same amount, so the room left for the answer after the reasoning stays as configured."""
        r = httpx.post(f"{self.base}/tokenize", json={"model": self.model, "messages": [{"role": "user", "content": prompt}]}, timeout=600)
        if r.status_code != 200:
            raise ScoreError(f"POST /tokenize returned {r.status_code}: {r.text[:300]}")
        out = r.json()
        cut = max(0, out["count"] + max_tokens - out["max_model_len"])
        if cut and (max_tokens - cut < 1 or (budget and budget - cut < 1)):
            raise ScoreError(f"prompt of {out['count']} tokens leaves no room in the server's context length {out['max_model_len']}")
        return max_tokens - cut, (budget - cut if budget else 0)

    def ask(self, prompt: str, thinking: str, max_tokens: int, budget: int = 0) -> dict:
        """One response: answer text, reasoning, finish reason, token counts, and the top log-probabilities at the answer word."""
        max_tokens, budget = self.fit(prompt, max_tokens, budget if thinking == "on" else 0)
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking == "on"}, "structured_outputs": {"regex": QWEN_FORMAT}}
        if thinking == "on" and budget:
            extra_body["thinking_token_budget"] = budget
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=QWEN_TEMPERATURE,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=QWEN_TOP_LOGPROBS,
                extra_body=extra_body,
            )
        except openai.OpenAIError as e:
            raise ScoreError(f"request failed: {e}") from None
        ch = r.choices[0]
        extra = ch.message.model_extra or {}
        # with thinking on, the log-probabilities also cover the reasoning, which ends with the </think> token; the answer word is
        # the last "yes" or "no" after it (without the token, only the last few tokens are searched, so no word in the reasoning is taken)
        toks = ch.logprobs.content if ch.logprobs and ch.logprobs.content else []
        end = max((i for i, t in enumerate(toks) if t.token == "</think>"), default=None)
        reasoning_tokens = end if end is not None else (len(toks) if thinking == "on" else 0)
        answer_top = None
        for t in reversed(toks[end + 1:] if end is not None else toks[-16:]):
            if t.token.strip().lower() in ("yes", "no"):
                answer_top = [[a.token, a.logprob] for a in t.top_logprobs]
                break
        return {
            "content": ch.message.content,
            "reasoning": extra.get("reasoning_content") or extra.get("reasoning") or "",
            "finish_reason": ch.finish_reason,
            "completion_tokens": r.usage.completion_tokens if r.usage else None,
            "reasoning_tokens": reasoning_tokens,
            "answer_top_logprobs": answer_top,
            # what was sent, which is below the configured values when the prompt is long (see fit); the cache key keeps the configured ones
            "max_tokens_used": max_tokens,
            "thinking_budget_used": budget,
        }


    def force(self, prompt: str, trace: str, cap: int) -> dict:
        """An answer after the end-of-reasoning marker, for a response that ended with none: the templated prompt (whose generation prompt ends with the start-of-reasoning marker, for Qwen and Olmo Think alike), the first `cap` tokens of the reasoning trace, then the end-of-reasoning marker, with QWEN_FORMAT enforced and greedy decoding."""
        def tok(body: dict) -> list[int]:
            r = httpx.post(f"{self.base}/tokenize", json={"model": self.model, **body}, timeout=600)
            if r.status_code != 200:
                raise ScoreError(f"POST /tokenize returned {r.status_code}: {r.text[:300]}")
            return r.json()["tokens"]

        head = tok({"messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": True}, "add_generation_prompt": True})
        trace_ids = tok({"prompt": trace.rstrip("\n"), "add_special_tokens": False})[:cap]
        end = tok({"prompt": "\n</think>\n\n", "add_special_tokens": False})
        r = httpx.post(f"{self.base}/v1/completions", timeout=3600, json={
            "model": self.model, "prompt": head + trace_ids + end, "max_tokens": 32, "temperature": QWEN_TEMPERATURE,
            "logprobs": QWEN_TOP_LOGPROBS, "structured_outputs": {"regex": QWEN_FORMAT},
        })
        if r.status_code != 200:
            raise ScoreError(f"forced answer: POST /v1/completions returned {r.status_code}: {r.text[:300]}")
        ch = r.json()["choices"][0]
        answer_top = None
        for t, top in zip(ch["logprobs"]["tokens"], ch["logprobs"]["top_logprobs"]):
            if t.strip().lower() in ("yes", "no"):
                answer_top = [[a, v] for a, v in top.items()]
                break
        return {"content": ch["text"], "finish_reason": ch["finish_reason"], "answer_top_logprobs": answer_top, "trace_tokens": len(trace_ids)}


def force_entry(client, e: dict, prompt: str, cap: int) -> dict:
    """A cached response with no answer, given a forced one (QwenClient.force). The unfinished trace is the reasoning or, when the server's reasoning parser found no end-of-reasoning marker (Olmo Think), the content. The original ending is kept in first_content and first_finish_reason."""
    trace = e["reasoning"] or e["content"] or ""
    f = client.force(prompt, trace, cap)
    parsed = parse_answer(f["content"])
    return {
        **e, "reasoning": trace, "forced": True, "forced_trace_tokens": f["trace_tokens"],
        "first_content": e["content"], "first_finish_reason": e["finish_reason"], "first_answer_top_logprobs": e.get("answer_top_logprobs"),
        "content": f["content"], "finish_reason": f["finish_reason"], "answer_top_logprobs": f["answer_top_logprobs"],
        "answer": parsed[0] if parsed else None, "probability": parsed[1] if parsed else None,
    }


def unforced(e: dict) -> dict:
    """A cached response as it was before force_entry gave it an answer (the same entry if it was never forced)."""
    if not e.get("forced"):
        return e
    keep = {k: v for k, v in e.items() if k not in ("forced", "forced_trace_tokens", "first_content", "first_finish_reason", "first_answer_top_logprobs")}
    return {**keep, "content": e["first_content"], "finish_reason": e["first_finish_reason"], "answer": None, "probability": None, "answer_top_logprobs": e.get("first_answer_top_logprobs"),
            "reasoning": e["reasoning"] if e["first_content"] != e["reasoning"] else ""}


def _ask_row(client, rec: Record, template: str, thinking: str, max_tokens: int, budget: int = 0) -> dict:
    prompt = qwen_prompt(rec, template)
    budget = budget if thinking == "on" else 0
    resp = client.ask(prompt, thinking, max_tokens, budget)
    parsed = parse_answer(resp["content"])
    return {
        "key": qwen_key(client.model, prompt, thinking, budget),
        "model": client.model,
        "template": template,
        "thinking": thinking,
        "row": rec.row,
        "claim_id": rec.claim_id,
        "max_tokens": max_tokens,
        "thinking_budget": budget,
        "answer": parsed[0] if parsed else None,
        "probability": parsed[1] if parsed else None,
        **resp,
    }


def reasoning_capped(e: dict, budget: int) -> bool:
    """True if a response's reasoning used its whole budget, so vLLM ended it. vLLM counts the </think> token it inserts against the budget, so a capped reasoning has budget - 1 tokens; completion_tokens also counts the answer, so it cannot tell a capped reasoning from one that ended just short of the cap."""
    budget = e.get("thinking_budget_used", budget) if budget > 0 else 0
    return budget > 0 and e["reasoning_tokens"] >= budget - 1


def _usable(e: dict | None, max_tokens: int) -> bool:
    """A cached response is reused unless it was cut off at a lower token cap than the current one."""
    return e is not None and not (e["finish_reason"] == "length" and e["max_tokens"] < max_tokens)


def get_qwen_responses(records: list[Record], templates: list[str], thinkings: list[str], cache_path: Path, server: str, concurrency: int, max_tokens: int, client=None, budget: int = 0, force: bool = False) -> dict[tuple[str, str, int], dict]:
    """The chat model's response for every (template, thinking, row), from the cache or requested from the server and added to it. With thinking on, the reasoning is limited to `budget` tokens (0: no limit).

    The server is contacted only if something is missing. A given client (a fake, in the tests) is used instead. Stops with a ScoreError listing any response that has no answer (usually cut off by max_tokens while thinking).
    """
    model = client.model if client is not None else serving.endpoint(server)[1]
    cache = load_cache(cache_path)
    jobs = [(rec, t, th) for t in templates for th in thinkings for rec in records]
    todo = [(rec, t, th) for rec, t, th in jobs if not _usable(cache.get(qwen_key(model, qwen_prompt(rec, t), th, budget)), max_tokens)]

    def unanswered():
        return [(rec, t, th) for rec, t, th in jobs if th == "on" and (e := cache.get(qwen_key(model, qwen_prompt(rec, t), th, budget))) and e["answer"] is None]

    if todo or (force and unanswered()):
        if client is None:
            print(f"{len(todo)} of {len(jobs)} prompts have no cached response from {server}.", flush=True)
            _require(server)
        c = client or QwenClient(server)
        if todo:
            try:
                _request_responses(todo, c, cache, cache_path, server, concurrency, max_tokens, budget)
            except ScoreError as exc:
                if not force:
                    raise
                failed_requests = exc  # the answers for what did finish are still forced below
            else:
                failed_requests = None
        else:
            failed_requests = None
        if force and (left := unanswered()):
            print(f"Forcing answers for {len(left)} responses from {server} that ended with none...", flush=True)
            with open(cache_path, "a", encoding="utf-8") as f:
                for rec, t, th in left:
                    k = qwen_key(model, qwen_prompt(rec, t), th, budget)
                    e = force_entry(c, cache[k], qwen_prompt(rec, t), budget or max_tokens)
                    cache[k] = e
                    f.write(json.dumps(e) + "\n")
                    f.flush()
                    print(f"  row {rec.row}: forced after {e['forced_trace_tokens']} reasoning tokens, answer {e['answer']} {e['probability']}", flush=True)
        if failed_requests is not None:
            raise failed_requests
    out = {(t, th, rec.row): cache[qwen_key(model, qwen_prompt(rec, t), th, budget)] for rec, t, th in jobs}
    failed = [e for e in out.values() if e["answer"] is None]
    if failed:
        listed = "; ".join(f"row {e['row']} ({e['template']}, thinking {e['thinking']}, finish {e['finish_reason']})" for e in failed[:10])
        raise ScoreError(
            f"{len(failed)} responses from {server} have no answer: {listed}{' ...' if len(failed) > 10 else ''}. "
            f"The other responses are cached. A higher --qwen_max_tokens (current {max_tokens}) requests only these again."
        )
    return out


def _request_responses(todo: list, client, cache: dict, cache_path: Path, server: str, concurrency: int, max_tokens: int, budget: int) -> None:
    print(f"Requesting {len(todo)} responses from {server} ({concurrency} at a time)...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    errors = []
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as pool:
        futures = {pool.submit(_ask_row, client, rec, t, th, max_tokens, budget): (rec, t, th) for rec, t, th in todo}
        for done, fut in enumerate(as_completed(futures), start=1):
            # a failed request is reported and the others go on, so their responses are still cached
            try:
                e = fut.result()
            except Exception as exc:
                rec, t, th = futures[fut]
                errors.append(f"row {rec.row} ({t}, thinking {th}): {exc}")
                print(f"  failed: {errors[-1]}", flush=True)
                continue
            cache[e["key"]] = e
            f.write(json.dumps(e) + "\n")
            f.flush()
            if done % 10 == 0 or done == len(todo):
                mins = (time.monotonic() - start) / 60
                print(f"  {done}/{len(todo)} after {mins:.1f} min ({done / max(mins, 1e-9):.1f} per min)", flush=True)
    if errors:
        raise ScoreError(f"{len(errors)} of {len(todo)} requests to {server} failed; the others are cached. " + "; ".join(errors[:10]) + (" ..." if len(errors) > 10 else ""))


# ---------------------------------------------------------------- runs


@dataclass(frozen=True)
class Run:
    predictor: str          # "base", "replicats", "markets" or "chat"
    subset: str
    data_type: str
    template: str = ""      # base and chat
    mode: str = ""          # base only
    ending: str = ""        # base only: "yes_no"
    thinking: str = ""      # chat only: "on" or "off"
    qwen_score: str = ""    # chat only: "stated" or "token"
    server: str = ""        # base and chat: model name in models.toml

    def filename(self, human_accuracy: str) -> str:
        if self.predictor == "base":
            return f"{self.server}_{self.subset}_{self.data_type}_{self.template}_yesno_{self.mode}.csv"
        if self.predictor == "chat":
            think = "think" if self.thinking == "on" else "nothink"
            return f"{self.server}_{self.subset}_{self.data_type}_{self.template}_{think}_{self.qwen_score}_{human_accuracy}.csv"
        return f"{self.predictor}_{self.subset}_{self.data_type}_{human_accuracy}.csv"


def human_estimate(rec: Record, predictor: str) -> float | None:
    return rec.replicats if predictor == "replicats" else rec.markets


def human_applicable(rows: list[Record], predictor: str) -> bool:
    """Human predictors apply only to selections where every row has an estimate."""
    return bool(rows) and all(human_estimate(r, predictor) is not None for r in rows)


def plan_runs(cfg: dict, records: list[Record]) -> list[Run]:
    """Every combination of enabled options, skipping human runs that are not applicable."""
    runs = []
    templates = [t for t in ALL_TEMPLATES if cfg[f"template_{t}"]]
    if cfg["predictor_base"]:
        modes = [m for m in MODES if cfg[f"mode_{m}"]]
        endings = list(ENDINGS)
        runs += [
            Run("base", s, dt, t, m, ending=e, server=sv)
            for sv in cfg["base_servers"] for s in cfg["subsets"] for dt in cfg["data_types"] for e in endings for t in templates for m in modes
        ]
    for h in HUMANS:
        if cfg[f"predictor_{h}"]:
            runs += [Run(h, s, dt) for s in cfg["subsets"] for dt in cfg["data_types"] if human_applicable(select(records, s, dt), h)]
    if cfg["predictor_chat"]:
        thinkings = [th for th in THINKING if cfg[f"qwen_thinking_{th}"]]
        scores = [q for q in QWEN_SCORES if cfg[f"qwen_score_{q}"]]
        runs += [
            Run("chat", s, dt, t, thinking=th, qwen_score=q, server=sv)
            for sv in cfg["chat_servers"] for s in cfg["subsets"] for dt in cfg["data_types"] for t in templates for th in thinkings for q in scores
        ]
    return runs


def run_olmo(run: Run, rows: list[Record], ppl: dict) -> tuple[list[dict], dict]:
    """Score L = log p(" yes") - log p(" no") = log PPL(" no") - log PPL(" yes"), rounded to SCORE_DECIMALS places."""
    w = weights(rows)
    pf = [ppl[(run.template, r.row, " no")] for r in rows]
    ps = [ppl[(run.template, r.row, " yes")] for r in rows]
    d = [round(math.log(f) - math.log(s), SCORE_DECIMALS) for f, s in zip(pf, ps)]
    succ = [r.succeeded for r in rows]
    fitted = ""
    if run.mode == "fitted":
        fitted = fit_offset(d, succ, w)
        offsets = [fitted] * len(rows)
    else:
        offsets = loo_offsets(d, succ, w, [r.paper_id for r in rows])
    out = []
    for r, wi, f, s, di, b in zip(rows, w, pf, ps, d, offsets):
        final = di + b
        pred = "succeeded" if final > 0 else "failed"
        out.append({
            "row": r.row, "claim_id": r.claim_id, "paper": r.paper, "data_type": r.data_type, "weight": wi,
            "ppl_no": f, "ppl_yes": s, "score": di, "offset": b, "final_score": final,
            "prediction": pred, "correct_answer": r.correct_answer, "correct": int(pred == r.correct_answer),
        })
    summary = {
        "weighted_auc": weighted_auc(d, succ, w),
        "fitted_offset": fitted,
        "weighted_accuracy": weighted_accuracy(w, [o["correct"] for o in out]),
    }
    return out, summary


def run_human(run: Run, rows: list[Record], human_accuracy: str) -> tuple[list[dict], dict]:
    w = weights(rows)
    p = [human_estimate(r, run.predictor) for r in rows]
    out = [{
        "row": r.row, "claim_id": r.claim_id, "paper": r.paper, "data_type": r.data_type, "weight": wi,
        "estimate": pi, "prediction": human_prediction(pi), "correct_answer": r.correct_answer,
        "credit": human_credit(pi, r.succeeded, human_accuracy),
    } for r, wi, pi in zip(rows, w, p)]
    summary = {
        "weighted_auc": weighted_auc(p, [r.succeeded for r in rows], w),
        "fitted_offset": "",
        "weighted_accuracy": weighted_accuracy(w, [o["credit"] for o in out]),
    }
    return out, summary


def run_qwen(run: Run, rows: list[Record], responses: dict, human_accuracy: str) -> tuple[list[dict], dict]:
    """The prediction is the model's yes or no. The score (for AUC and hedged credit) is the stated probability or the token probability of yes."""
    w = weights(rows)
    out, x = [], []
    for r, wi in zip(rows, w):
        e = responses[(run.template, run.thinking, r.row)]
        stated = e["probability"] / 100 if e["probability"] is not None else math.nan
        if run.qwen_score == "stated" and math.isnan(stated):
            raise ScoreError(f"{run.server} has no stated probabilities; only its token score can be run")
        token, note = token_probability(e["answer_top_logprobs"]) if e["answer_top_logprobs"] else (math.nan, "no logprobs")
        x.append(stated if run.qwen_score == "stated" else token)
        pred = "succeeded" if e["answer"] == "yes" else "failed"
        if human_accuracy == "threshold":
            credit = float(pred == r.correct_answer)
        elif human_accuracy == "hedged":
            credit = x[-1] if r.succeeded else 1 - x[-1]
        else:
            raise ScoreError(f"human_accuracy must be one of {HUMAN_ACCURACY}, not {human_accuracy!r}")
        out.append({
            "row": r.row, "claim_id": r.claim_id, "paper": r.paper, "data_type": r.data_type, "weight": wi,
            "answer": e["answer"], "stated_probability": stated, "token_probability": token, "token_note": note,
            "score": x[-1], "prediction": pred, "correct_answer": r.correct_answer, "credit": credit,
            "answer_disagrees": "" if math.isnan(stated) else int((e["answer"] == "yes" and stated < 0.5) or (e["answer"] == "no" and stated > 0.5)),
        })
    summary = {
        "weighted_auc": weighted_auc(x, [r.succeeded for r in rows], w),
        "fitted_offset": "",
        "weighted_accuracy": weighted_accuracy(w, [o["credit"] for o in out]),
    }
    return out, summary


def execute(run: Run, records: list[Record], cfg: dict, ppl: dict | None, responses: dict | None = None) -> tuple[list[dict], dict]:
    """Per-row output and the summary line for one run. ppl and responses are keyed by server name."""
    rows = select(records, run.subset, run.data_type)
    if not rows:
        raise ScoreError(f"no rows selected for subset {run.subset!r}, data type {run.data_type!r}")
    if run.predictor == "base":
        out, s = run_olmo(run, rows, ppl[run.server])
    elif run.predictor == "chat":
        out, s = run_qwen(run, rows, responses[run.server], cfg["human_accuracy"])
    else:
        out, s = run_human(run, rows, cfg["human_accuracy"])
    line = {
        "predictor": run.predictor, "server": run.server, "subset": run.subset, "data_type": run.data_type,
        "template": run.template, "mode": run.mode, "olmo_ending": run.ending, "thinking": run.thinking, "qwen_score": run.qwen_score,
        "human_accuracy": "" if run.predictor == "base" else cfg["human_accuracy"],
        "n_rows": len(rows), "n_papers": len({r.paper_id for r in rows}), "sum_weights": sum(weights(rows)),
        **s,
    }
    return out, line


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        wr.writerows(rows)


def print_summary(lines: list[dict]) -> None:
    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.6f}"
        return str(v)
    cols = [c for c in SUMMARY_COLUMNS if any(line[c] != "" for line in lines)]
    table = [cols] + [[fmt(line[c]) for c in cols] for line in lines]
    widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
    for r in table:
        print("  ".join(v.ljust(wd) for v, wd in zip(r, widths)).rstrip())


# ---------------------------------------------------------------- configuration handling


def _bool(text: str) -> bool:
    t = text.lower()
    if t in ("true", "yes", "1"):
        return True
    if t in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, not {text!r}")


def parse_config(argv: list[str] | None) -> dict:
    """CONFIG with any command-line overrides, validated."""
    ap = argparse.ArgumentParser(description="Base and chat models versus human forecasters on SCORE replication outcomes.")
    for name in BOOL_OPTIONS:
        ap.add_argument(f"--{name}", type=_bool, metavar="true|false")
    ap.add_argument("--human_accuracy", choices=HUMAN_ACCURACY)
    ap.add_argument("--subsets", nargs="+", choices=SUBSETS)
    ap.add_argument("--data_types", nargs="+", choices=DATA_TYPES)
    for name in ("base_servers", "chat_servers"):
        ap.add_argument(f"--{name}", nargs="+")
    for name in ("data_file", "out_dir", "cache_file", "qwen_cache_file"):
        ap.add_argument(f"--{name}")
    for name in ("concurrency", "qwen_concurrency", "qwen_max_tokens", "qwen_thinking_budget"):
        ap.add_argument(f"--{name}", type=int)
    args = vars(ap.parse_args(argv))
    cfg = dict(CONFIG)
    cfg.update({k: v for k, v in args.items() if v is not None})
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    for name in BOOL_OPTIONS:
        if not isinstance(cfg[name], bool):
            raise ScoreError(f"{name} must be true or false")
    if cfg["human_accuracy"] not in HUMAN_ACCURACY:
        raise ScoreError(f"human_accuracy must be one of {HUMAN_ACCURACY}")
    for name, allowed in (("subsets", SUBSETS), ("data_types", DATA_TYPES)):
        vals = cfg[name]
        if not vals or any(v not in allowed for v in vals) or len(set(vals)) != len(vals):
            raise ScoreError(f"{name} must be a non-empty list of distinct values from {allowed}")
    for name in ("concurrency", "qwen_concurrency", "qwen_max_tokens"):
        if cfg[name] < 1:
            raise ScoreError(f"{name} must be at least 1")
    if cfg["qwen_thinking_budget"] < 0:
        raise ScoreError("qwen_thinking_budget must be 0 (no budget) or more")
    if cfg["qwen_thinking_budget"] and cfg["qwen_max_tokens"] < cfg["qwen_thinking_budget"] + 64:
        raise ScoreError("qwen_max_tokens must leave at least 64 tokens for the answer after qwen_thinking_budget")
    for name, kind in (("base_servers", "base"), ("chat_servers", "chat")):
        vals = cfg[name]
        if not isinstance(vals, list) or len(set(vals)) != len(vals):
            raise ScoreError(f"{name} must be a list of distinct model names")
        for sv in vals:
            try:
                k = serving.kind(sv)
            except serving.ServingError as e:
                raise ScoreError(str(e)) from None
            if k != kind:
                raise ScoreError(f"{name}: {sv} is a {k} model; {name} takes models of kind {kind}")


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- main


def _fake(given, server: str):
    """A test double for one server: given may be one fake for every server, or a dict by server name."""
    return given.get(server) if isinstance(given, dict) else given


def main(argv: list[str] | None = None, scorer=None, qwen_client=None) -> int:
    try:
        cfg = parse_config(argv)
        records = load_data(_path(cfg["data_file"]))
        runs = plan_runs(cfg, records)
        if not runs:
            print("No runs: every enabled combination is switched off or not applicable.")
            return 0
        ppl = {}
        for sv in dict.fromkeys(r.server for r in runs if r.predictor == "base"):
            sv_runs = [r for r in runs if r.predictor == "base" and r.server == sv]
            templates = sorted({r.template for r in sv_runs}, key=ALL_TEMPLATES.index)
            endings = sorted({r.ending for r in sv_runs}, key=ENDINGS.index)
            ppl[sv] = get_perplexities(records, templates, _path(cfg["cache_file"]), sv, cfg["concurrency"], _fake(scorer, sv), endings)
        responses = {}
        for sv in dict.fromkeys(r.server for r in runs if r.predictor == "chat"):
            sv_runs = [r for r in runs if r.predictor == "chat" and r.server == sv]
            # only the rows some chat run uses are requested, in data-file order
            needed = {rec.row for r in sv_runs for rec in select(records, r.subset, r.data_type)}
            responses[sv] = get_qwen_responses(
                [rec for rec in records if rec.row in needed],
                sorted({r.template for r in sv_runs}, key=ALL_TEMPLATES.index),
                sorted({r.thinking for r in sv_runs}, key=THINKING.index),
                _path(cfg["qwen_cache_file"]), sv, cfg["qwen_concurrency"], cfg["qwen_max_tokens"], _fake(qwen_client, sv), cfg["qwen_thinking_budget"], cfg["qwen_force_answer"],
            )
        out_dir = _path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for run in runs:
            out, line = execute(run, records, cfg, ppl, responses)
            columns = {"base": OLMO_YES_NO_FILE_COLUMNS, "chat": QWEN_FILE_COLUMNS}.get(run.predictor, HUMAN_FILE_COLUMNS)
            write_csv(out_dir / run.filename(cfg["human_accuracy"]), columns, out)
            lines.append(line)
        write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS, lines)
    except (ScoreError, serving.ServingError) as e:
        print(f"score.py: {e}", file=sys.stderr)
        return 2
    print_summary(lines)
    print(f"\n{len(runs)} runs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
