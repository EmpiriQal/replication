"""Qwen with thinking on and a 32,000-token reasoning cap, for the high-power rows (with original finding) that did not finish reasoning within 16,384 tokens.

The 16k run (score.py, --qwen_thinking_budget 16384) left 26 of the 57 high-power rows unfinished: 25 hit the cap and one (row 271) ended without answering. This script asks those rows again with a cap of 32,000 (the server's context is 32,768 and the longest prompt is 657 tokens), using score.py's own prompt and client, so the request is the same as score.py's apart from the cap.

Each response is flagged:
- hit_cap: the reasoning used the whole budget, so vLLM ended it and Qwen had to answer.
- forced: the response ended with no answer, so a second request sent the prompt, the reasoning so far and the end-of-reasoning marker, with the same answer format enforced, and Qwen had to answer.
- stuck: the last part of the reasoning repeats one line over and over (a loop), whichever way it ended.

Responses go to results/thinking32k_responses.jsonl (rows already there are not asked again); progress to stdout. score.py and its caches are not changed. Missing responses are requested from qwen27 at its URL in models.toml.

Usage: uv run python thinking32k.py
"""

import collections
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np

import score as S
import serving

SERVER = "qwen27"
MODEL = "Qwen/Qwen3.8-27B-FP8"
TEMPLATE = "with_original_finding"
OLD_BUDGET = 16384
BUDGET = 32000
MAX_TOKENS = BUDGET + 64
# only a few requests run at once: each can reach 32k tokens, and requests queued at the server would count against the client's timeout
CONCURRENCY = 4
OUT = S.ROOT / "results/thinking32k_responses.jsonl"


def stuck(reasoning: str) -> bool:
    """True if one non-empty line makes up at least half of the last 60 non-empty lines (and occurs 10 times or more)."""
    lines = [l.strip() for l in reasoning.splitlines() if l.strip()][-60:]
    if not lines:
        return False
    n = collections.Counter(lines).most_common(1)[0][1]
    return n >= 10 and n >= len(lines) / 2


class Forcer:
    """Second request for a response that ended without an answer: the templated prompt, the reasoning so far and </think>, then the answer under the same format."""

    def __init__(self, server: str):
        url, self.model = serving.endpoint(server)
        self.base = url.removesuffix("/v1")
        self.client = httpx.Client(timeout=3600)

    def _post(self, path: str, body: dict) -> dict:
        r = self.client.post(f"{self.base}{path}", json=body)
        if r.status_code != 200:
            raise S.ScoreError(f"POST {path} returned {r.status_code}: {r.text[:300]}")
        return r.json()

    def answer(self, prompt: str, reasoning: str) -> dict:
        # the generation prompt already ends with "<think>\n" (checked against the server's template)
        head = self._post("/tokenize", {"model": self.model, "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": True}, "add_generation_prompt": True})["tokens"]
        tail = self._post("/tokenize", {"model": self.model, "prompt": reasoning.rstrip("\n") + "\n</think>\n\n", "add_special_tokens": False})["tokens"]
        ids = head + tail
        room = 32768 - len(ids)
        if room < 32:
            # a trace that filled the context: keep its start and drop reasoning from the end until the answer fits
            ids = head + tail[: len(tail) - (32 - room) - 8] + self._post("/tokenize", {"model": self.model, "prompt": "\n</think>\n\n", "add_special_tokens": False})["tokens"]
        out = self._post("/v1/completions", {
            "model": self.model, "prompt": ids, "max_tokens": 32, "temperature": S.QWEN_TEMPERATURE,
            "logprobs": S.QWEN_TOP_LOGPROBS, "structured_outputs": {"regex": S.QWEN_FORMAT},
        })
        ch = out["choices"][0]
        lp = ch["logprobs"]
        answer_top = None
        for tok, top in zip(lp["tokens"], lp["top_logprobs"]):
            if tok.strip().lower() in ("yes", "no"):
                answer_top = [[t, v] for t, v in top.items()]
                break
        return {"content": ch["text"], "finish_reason": ch["finish_reason"], "answer_top_logprobs": answer_top, "prompt_tokens": len(ids)}


def ask(client: S.QwenClient, forcer: Forcer, rec: S.Record) -> dict:
    prompt = S.qwen_prompt(rec, TEMPLATE)
    t0 = time.time()
    resp = client.ask(prompt, "on", MAX_TOKENS, BUDGET)
    e = {
        "template": TEMPLATE, "thinking": "on", "row": rec.row, "claim_id": rec.claim_id,
        "thinking_budget": BUDGET, "max_tokens": MAX_TOKENS, **resp,
        # the budget includes the </think> token vLLM inserts, so a capped reasoning has BUDGET - 1 tokens
        "hit_cap": S.reasoning_capped(resp, BUDGET), "forced": False, "stuck": stuck(resp["reasoning"]),
        "first_content": None, "first_finish_reason": None,
    }
    if S.parse_answer(resp["content"]) is None:
        f = forcer.answer(prompt, resp["reasoning"])
        e.update(forced=True, first_content=resp["content"], first_finish_reason=resp["finish_reason"],
                 content=f["content"], finish_reason=f["finish_reason"], answer_top_logprobs=f["answer_top_logprobs"], forced_prompt_tokens=f["prompt_tokens"])
    parsed = S.parse_answer(e["content"])
    e["answer"], e["probability"] = parsed if parsed else (None, None)
    e["minutes"] = round((time.time() - t0) / 60, 1)
    return e


def todo_rows(records: list[S.Record], cache: dict) -> list[S.Record]:
    """High-power rows whose 16k response hit the cap or has no answer."""
    out = []
    for r in S.select(records, "highpower", "all"):
        e = S.unforced(cache[S.qwen_key(MODEL, S.qwen_prompt(r, TEMPLATE), "on", OLD_BUDGET)])
        if e["answer"] is None or S.reasoning_capped(e, OLD_BUDGET):
            out.append(r)
    return out


def fetch(records: list[S.Record], cache: dict) -> dict[int, dict]:
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            have[e["row"]] = e
    rows = todo_rows(records, cache)
    # the file also holds row 189, asked in error by an earlier version that took completion tokens (reasoning and answer) at or over 16,384 as the cap; its 16k reasoning ended at 16,379 tokens, so its 32k response is not used
    have = {r.row: have[r.row] for r in rows if r.row in have}
    todo = [r for r in rows if r.row not in have]
    if todo:
        if not serving.is_ready(SERVER):
            sys.exit(f"{SERVER} is not answering at its URL in models.toml.")
        client, forcer = S.QwenClient(SERVER), Forcer(SERVER)
        print(f"Requesting {len(todo)} Qwen responses from {SERVER} with a {BUDGET}-token reasoning cap ({CONCURRENCY} at a time)...", flush=True)
        t0 = time.time()
        with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(CONCURRENCY) as pool:
            futures = [pool.submit(ask, client, forcer, r) for r in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                e = fut.result()
                f.write(json.dumps(e) + "\n")
                f.flush()
                have[e["row"]] = e
                print(f"  {i}/{len(todo)} after {(time.time() - t0) / 60:.1f} min: row {e['row']}, {e['completion_tokens']} tokens, answer {e['answer']} {e['probability']}"
                      f"{', hit cap' if e['hit_cap'] else ''}{', forced' if e['forced'] else ''}{', stuck' if e['stuck'] else ''}", flush=True)
    return have


def main() -> None:
    records = S.load_data(S.ROOT / "data/SCORE_replicated_claims.csv")
    cache = S.load_cache(S.ROOT / "results/qwen_responses.jsonl")
    new = fetch(records, cache)
    rows = S.select(records, "highpower", "all")

    def old(r, th, b):
        return S.unforced(cache[S.qwen_key(MODEL, S.qwen_prompt(r, TEMPLATE), th, b)])

    def tok(e):
        return S.token_probability(e["answer_top_logprobs"])[0] if e["answer_top_logprobs"] else math.nan

    # the complete thinking set: the 32k response where the 16k one did not finish, otherwise the 16k one (its reasoning ended before the cap)
    full = {r.row: new.get(r.row) or old(r, "on", OLD_BUDGET) for r in rows}
    flags = {r.row: (S.reasoning_capped(new[r.row], BUDGET), new[r.row]["forced"], new[r.row]["stuck"]) if r.row in new else (False, False, stuck(full[r.row]["reasoning"])) for r in rows}
    print(f"\n{len(rows)} rows: {len(new)} from the 32k run, {len(rows) - len(new)} finished within 16k")
    print(f"hit 32k cap {sum(f[0] for f in flags.values())}, forced {sum(f[1] for f in flags.values())}, stuck {sum(f[2] for f in flags.values())}, no answer {sum(e['answer'] is None for e in full.values())}")
    toks = [e["completion_tokens"] for e in full.values()]
    print(f"completion tokens: median {int(np.median(toks))}, max {max(toks)}")

    print("\nHigh-power, with original finding: weighted AUC (stated, token), yes answers, accuracy")
    sets = {
        "all": rows,
        "excluding capped, forced or stuck": [r for r in rows if not any(flags[r.row])],
    }
    for name, rs in sets.items():
        rs = [r for r in rs if full[r.row]["answer"]]
        w, y = S.weights(rs), [r.succeeded for r in rs]
        print(f"  {name} (n={len(rs)})")
        for label, es in (("thinking off", [old(r, "off", 0) for r in rs]), ("thinking 4k", [old(r, "on", 4096) for r in rs]), ("thinking 32k", [full[r.row] for r in rs])):
            x = [tok(e) for e in es]
            print(f"    {label:13} stated {S.weighted_auc([e['probability'] for e in es], y, w):.3f}  token {S.weighted_auc(x, y, w):.3f}  yes {sum(e['answer'] == 'yes' for e in es):2d}  "
                  f"acc {S.weighted_accuracy(w, [float((e['answer'] == 'yes') == s) for e, s in zip(es, y)]):.3f}  near certain {np.nanmean([v > 0.99 or v < 0.01 for v in x]):.2f}")
        for h in S.HUMANS:
            print(f"    {h:13} {S.weighted_auc([getattr(r, h) for r in rs], y, w):.3f}  acc {S.weighted_accuracy(w, [float((getattr(r, h) >= 0.5) == s) for r, s in zip(rs, y)]):.3f}")


if __name__ == "__main__":
    main()
