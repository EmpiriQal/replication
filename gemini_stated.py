"""Gemini 3.1 Pro and Gemini 3 Flash on Vertex AI, asked whether each SCORE replication succeeded, scored by their stated probability.

Each claim gets one chat request with score.py's chat prompt (template with the original finding, ending in the "Answer: yes or no / Probability: 0 to 100" instruction), with temperature 0, at most 8,192 tokens for reasoning and answer together, and reasoning capped at 2,500 tokens. Gemini returns no log-probabilities, so only the stated probability is scored. There is no answer-format constraint, so a reply is read by finding "Answer:" with yes or no and "Probability:" with a whole number from 0 to 100 anywhere in it; `strict` records whether the reply was exactly the required format. Replies are cached in results/gemini_stated.jsonl, keyed by model, template, prompt and settings, and cached ones are not requested again.

Results, for each set (all, human, high-power) with paper weights: weighted AUC of the stated probability and weighted accuracy of the yes or no answer, with the humans and Qwen3.8 27B (stated and token scores, reasoning off) for comparison. Rows a model did not answer readably are left out of its numbers and counted.

Usage (needs gcloud application-default credentials and GOOGLE_CLOUD_PROJECT; see gemini.py):
  uv run python gemini_stated.py --limit 5      a trial on the first 5 claims
  uv run python gemini_stated.py                all 274 claims, then the results
  uv run python gemini_stated.py --report       the results from the cache only, no requests
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import score as S
from gemini import GeminiClient, GeminiError

MODELS = ("gemini-3.1-pro-preview", "gemini-3-flash-preview")
TEMPLATE = "with_original_finding"
TEMPERATURE = 0
MAX_TOKENS = 8192  # reasoning and answer together
# reasoning cap per model (part of the cache key)
REASONING_BUDGET = {"gemini-3.1-pro-preview": 2500, "gemini-3-flash-preview": 2500}
OUT = S.ROOT / "results" / "gemini_stated.jsonl"
LENIENT = re.compile(r"Answer:\W*(yes|no)\b.*?Probability:\W*(100|[1-9]?[0-9])\b", re.IGNORECASE | re.DOTALL)
QWEN27_MODEL = "Qwen/Qwen3.8-27B-FP8"


def key(model: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, TEMPLATE, prompt, TEMPERATURE, MAX_TOKENS, REASONING_BUDGET.get(model)]).encode()).hexdigest()


def parse(text: str) -> tuple[str | None, int | None, bool]:
    """(answer, probability, strict): strict if the reply is exactly score.py's format; otherwise the first Answer and Probability found anywhere, or (None, None, False)."""
    strict = S.parse_answer(text)
    if strict:
        return strict[0], strict[1], True
    m = LENIENT.search(text or "")
    return (m.group(1).lower(), int(m.group(2)), False) if m else (None, None, False)


def load() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    return {e["key"]: e for e in map(json.loads, OUT.read_text().splitlines())}


def request(models, records, concurrency: int, client_for=GeminiClient) -> int:
    cache = load()
    todo = [(m, r) for m in models for r in records if key(m, S.qwen_prompt(r, TEMPLATE)) not in cache]
    if not todo:
        print("Nothing to request: every reply is cached.")
        return 0
    clients = {m: client_for(m) for m in models}
    print(f"{len(todo)} requests ({concurrency} at a time)...", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(concurrency) as pool, open(OUT, "a", encoding="utf-8") as f:
        futures = {pool.submit(clients[m].ask, S.qwen_prompt(r, TEMPLATE), TEMPERATURE, MAX_TOKENS, REASONING_BUDGET.get(m)): (m, r) for m, r in todo}
        for done, fut in enumerate(as_completed(futures), start=1):
            m, r = futures[fut]
            try:
                reply = fut.result()
            except GeminiError as e:
                failed += 1
                print(f"  failed: {m} row {r.row}: {e}", file=sys.stderr, flush=True)
                continue
            answer, prob, strict = parse(reply["text"])
            e = {"key": key(m, S.qwen_prompt(r, TEMPLATE)), "model": m, "template": TEMPLATE, "row": r.row, "claim_id": r.claim_id,
                 "reasoning_budget": REASONING_BUDGET.get(m), "answer": answer, "probability": prob, "strict": strict, **reply}
            f.write(json.dumps(e) + "\n")
            f.flush()
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    if failed:
        print(f"{failed} requests failed; run again to retry them (cached replies are kept).", file=sys.stderr)
    return 1 if failed else 0


def metrics(rows, score_of, answer_of) -> dict:
    """Weighted AUC of the score and weighted accuracy of the answer, over the rows that have both (weights from those rows)."""
    keep = [r for r in rows if score_of(r) is not None and answer_of(r) is not None]
    if not keep:
        return {"n": 0, "auc": math.nan, "acc": math.nan, "missing": len(rows)}
    w = S.weights(keep)
    auc = S.weighted_auc([score_of(r) for r in keep], [r.succeeded for r in keep], w)
    acc = S.weighted_accuracy(w, [float((answer_of(r) == "yes") == r.succeeded) for r in keep])
    return {"n": len(keep), "auc": auc, "acc": acc, "missing": len(rows) - len(keep)}


def report(models, records) -> None:
    cache = load()
    qc = S.load_cache(S.ROOT / "results" / "qwen_responses.jsonl")
    pred: dict[str, tuple] = {}

    def stated(entry):
        return lambda r: None if (e := entry(r)) is None or e["probability"] is None else e["probability"] / 100

    def token(entry):
        return lambda r: None if (e := entry(r)) is None else S.token_probability(e["answer_top_logprobs"])[0]

    def answer(entry):
        return lambda r: None if (e := entry(r)) is None else e["answer"]

    for m in models:
        gem = lambda r, m=m: cache.get(key(m, S.qwen_prompt(r, TEMPLATE)))
        pred[f"{m}, stated"] = (stated(gem), answer(gem))
    qwen27 = lambda r: qc.get(S.qwen_key(QWEN27_MODEL, S.qwen_prompt(r, TEMPLATE), "off"))
    pred["Qwen3.8 27B, stated"] = (stated(qwen27), answer(qwen27))
    pred["Qwen3.8 27B, token"] = (token(qwen27), answer(qwen27))
    pred["repliCATS"] = (lambda r: r.replicats, lambda r: None if r.replicats is None else ("yes" if r.replicats >= 0.5 else "no"))
    pred["Markets"] = (lambda r: r.markets, lambda r: None if r.markets is None else ("yes" if r.markets >= 0.5 else "no"))

    for subset in S.SUBSETS:
        rows = S.select(records, subset, "all")
        print(f"\n{subset} ({len(rows)} claims; weighted AUC, weighted accuracy of the yes or no answer, rows without a readable answer)")
        for name, (sc, an) in pred.items():
            if name in ("repliCATS", "Markets") and not S.human_applicable(rows, name.lower()):
                continue
            x = metrics(rows, sc, an)
            if x["n"]:
                print(f"  {name:32s} AUC {x['auc']:.3f}  accuracy {x['acc']:.3f}  n {x['n']:3d}" + (f"  missing {x['missing']}" if x["missing"] else ""))

    print("\nPer model: replies, strict format, unreadable, answers yes, finish reasons, most common stated probabilities, tokens")
    for m in models:
        es = [cache[k] for r in records if (k := key(m, S.qwen_prompt(r, TEMPLATE))) in cache]
        if not es:
            continue
        probs = collections.Counter(e["probability"] for e in es if e["probability"] is not None).most_common(6)
        finish = collections.Counter(e["finish_reason"] for e in es)
        tok = {f: sum(e[f] or 0 for e in es) for f in ("prompt_tokens", "reply_tokens", "reasoning_tokens")}
        print(f"  {m}: {len(es)} replies, {sum(e['strict'] for e in es)} strict, {sum(e['answer'] is None for e in es)} unreadable, "
              f"{sum(e['answer'] == 'yes' for e in es)} yes; finish {dict(finish)}; probabilities {probs}; "
              f"tokens: prompt {tok['prompt_tokens']:,}, reply {tok['reply_tokens']:,}, reasoning {tok['reasoning_tokens']:,} "
              f"(mean reasoning {tok['reasoning_tokens'] / len(es):.0f})")


def main(argv: list[str] | None = None, client_for=GeminiClient) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N claims (a trial)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--report", action="store_true", help="results from the cache only, no requests")
    args = ap.parse_args(argv)
    records = S.load_data(S.ROOT / "data" / "SCORE_replicated_claims.csv")
    if args.limit:
        records = records[: args.limit]
    status = 0 if args.report else request(args.models, records, args.concurrency, client_for)
    report(args.models, records)
    return status


if __name__ == "__main__":
    sys.exit(main())
