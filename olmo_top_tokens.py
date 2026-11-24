"""Olmo's top 20 next tokens after "Answer:" with the yes/no ending, unconstrained, for every claim and both templates.

Prompts come from score.py's own `context` (ending "yes_no") and are tokenized by the server as score.py does. Responses are saved to results/olmo_top_tokens.jsonl (rows already there are not requested again); the analysis is printed. Missing lists are requested from olmo32 at its URL in models.toml.

Usage: uv run python olmo_top_tokens.py
"""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import score as S
import serving

SERVER = "olmo32"
OUT = S.ROOT / "results/olmo_top_tokens.jsonl"
TOP = 20
CONCURRENCY = 32
SCORED = {" yes": "yes", " no": "no"}
SYNONYMS = {"not", "probably", "unlikely", "likely", "maybe", "possibly", "uncertain", "yeah", "nope", "true", "false", "neither", "partially", "partly", "mixed", "unclear", "unknown", "n/a", "na"}
GROUPS = ("scored yes", "scored no", "yes variant", "no variant", "synonym", "other")


def group(tok: str) -> str:
    if tok in SCORED:
        return "scored " + SCORED[tok]
    w = tok.strip().lower()
    if w in ("yes", "no"):
        return w + " variant"
    if w.strip(".,:;!") in SYNONYMS:
        return "synonym"
    return "other"


def request(scorer: S.OlmoScorer, rec: S.Record, template: str) -> dict:
    ids = scorer.tokenize(S.context(rec, template, "yes_no"))
    out = scorer._post("/v1/completions", {
        "model": scorer.model, "prompt": ids, "max_tokens": 1, "temperature": 0,
        "logprobs": TOP, "return_tokens_as_token_ids": True,
    })
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]
    tok_ids = [int(k.removeprefix("token_id:")) for k in top]
    # decode each ID separately, so tokens that print the same are kept apart
    return {
        "template": template, "row": rec.row, "claim_id": rec.claim_id, "n_context_tokens": len(ids),
        "top": [[i, scorer._post("/detokenize", {"model": scorer.model, "tokens": [i]})["prompt"], top[f"token_id:{i}"]] for i in tok_ids],
    }


def fetch(records: list[S.Record]) -> dict[tuple[str, int], dict]:
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            have[(e["template"], e["row"])] = e
    todo = [(r, t) for t in S.TEMPLATES for r in records if (t, r.row) not in have]
    if todo:
        if not serving.is_ready(SERVER):
            sys.exit(f"{SERVER} is not answering at its URL in models.toml.")
        scorer = S.OlmoScorer(SERVER)
        print(f"Requesting {len(todo)} top-{TOP} lists from {SERVER}...", flush=True)
        with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(CONCURRENCY) as pool:
            for e in pool.map(lambda job: request(scorer, *job), todo):
                f.write(json.dumps(e) + "\n")
                have[(e["template"], e["row"])] = e
    return have


def masses(e: dict) -> dict[str, float]:
    m = dict.fromkeys(GROUPS, 0.0)
    for _, tok, lp in e["top"]:
        m[group(tok)] += math.exp(lp)
    m["all yes"] = m["scored yes"] + m["yes variant"]
    m["all no"] = m["scored no"] + m["no variant"]
    m["outside top 20"] = 1 - sum(m[g] for g in GROUPS)
    return m


def main() -> None:
    records = S.load_data(S.ROOT / "data/SCORE_replicated_claims.csv")
    tops = fetch(records)
    ppl = S.load_cache(S.ROOT / "results/olmo_perplexities.jsonl")
    model = "allenai/Olmo-3-1125-32B"

    print("\n1. Tokens in the top 20 (both templates, 548 lists)")
    seen: dict[str, list[float]] = {}
    for e in tops.values():
        for _, tok, lp in e["top"]:
            seen.setdefault(tok, []).append(math.exp(lp))
    print(f"{'token':>14} {'group':12} {'lists':>5} {'median p':>9} {'max p':>9}")
    for tok, ps in sorted(seen.items(), key=lambda x: (GROUPS.index(group(x[0])), -len(x[1]))):
        print(f"{tok!r:>14} {group(tok):12} {len(ps):5d} {np.median(ps):9.2e} {max(ps):9.2e}")

    print("\n2. Probability per claim, median [min, max]")
    for t in S.TEMPLATES:
        ms = [masses(tops[(t, r.row)]) for r in records]
        print(f"  {t}")
        for g in (*GROUPS, "all yes", "all no", "outside top 20"):
            v = [m[g] for m in ms]
            print(f"    {g:15} {np.median(v):.3g} [{min(v):.3g}, {max(v):.3g}]")

    print("\n3. Weighted AUC: L = log p(' yes') - log p(' no') against class score log(all yes) - log(all no)")
    worst = 0.0
    for t in S.TEMPLATES:
        for sub in S.SUBSETS:
            rows = S.select(records, sub, "all")
            w, y = S.weights(rows), [r.succeeded for r in rows]
            L, C = [], []
            for r in rows:
                ctx = S.context(r, t, "yes_no")
                lyes, lno = (ppl[S.cache_key(model, ctx, c)]["continuation_logprobs"][0] for c in (" yes", " no"))
                top = {tok: lp for _, tok, lp in tops[(t, r.row)]["top"]}
                worst = max(worst, abs(top[" yes"] - lyes), abs(top[" no"] - lno))
                m = masses(tops[(t, r.row)])
                L.append(lyes - lno)
                C.append(math.log(m["all yes"]) - math.log(m["all no"]))
            print(f"  {t:22} {sub:9} L {S.weighted_auc(L, y, w):.3f}  classes {S.weighted_auc(C, y, w):.3f}  rank corr {np.corrcoef(np.argsort(np.argsort(L)), np.argsort(np.argsort(C)))[0, 1]:.3f}")
    print(f"\nCheck: largest difference between ' yes' / ' no' log-probabilities here and in olmo_perplexities.jsonl: {worst:.2e}")


if __name__ == "__main__":
    main()
