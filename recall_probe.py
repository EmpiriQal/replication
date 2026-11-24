"""Recall probe: can a model predict SCORE outcomes from the paper's citation alone, with no abstract, claim or finding?

One prompt per claim, reasoning off, with score.py's answer format, client and scores: the SCORE header, the paper's short citation (authors and year), journal and DOI, and the question. Plus one direct question to the model about the SCORE replication paper (Tyner et al., Nature, 2026).

Responses are cached in results/qwen_recall_probe.jsonl (entries without a "model" field are Qwen3.8 27B's) and the direct answer in results/qwen_recall_direct.json (qwen27) or results/qwen_recall_direct_<name>.json; only missing ones are requested, from the model's server at its URL in models.toml. The AUCs are computed by analysis.py (section "recall").

Usage: uv run python recall_probe.py [--server NAME]   (default qwen27; the paper also uses qwen30i)
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import score as S
import serving

CACHE = S.ROOT / "results/qwen_recall_probe.jsonl"
HEAD = "The SCORE project (Systematizing Confidence in Open Research and Evidence) attempted to replicate a claim from the following published paper:"
QUESTION = "Did the replication attempt succeed, that is, find a statistically significant result in the same direction as the original?"
DIRECT_QUESTION = "What did the SCORE project's Nature paper by Tyner et al., \"Investigating the replicability of the social and behavioural sciences\", find? Give the main numbers if you know them, and say so if you do not know this paper."


def prompt(cells: dict, version: str = "citation") -> str:
    ref = f"{cells['Paper']}. {cells['Journal']}. DOI: {cells['DOI']}"
    return "\n\n".join([HEAD, ref, QUESTION, S.QWEN_INSTRUCTION])


def direct_path(server: str):
    return S.ROOT / ("results/qwen_recall_direct.json" if server == "qwen27" else f"results/qwen_recall_direct_{server}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="qwen27", help="chat model in models.toml (default qwen27)")
    server = ap.parse_args().server
    direct = direct_path(server)
    records = S.load_data(S.ROOT / S.CONFIG["data_file"])
    with open(S.ROOT / S.CONFIG["data_file"], newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))
    model = serving.endpoint(server)[1]
    cache = S.load_cache(CACHE)
    jobs = [(rec, v, prompt(cells[rec.row - 1], v)) for v in ("citation",) for rec in records]
    todo = [(rec, v, p) for rec, v, p in jobs if S.qwen_key(model, p, "off") not in cache]
    if not todo and direct.exists():
        print("All responses are cached; run analysis.py recall for the results.")
        return
    if not serving.is_ready(server):
        sys.exit(f"{len(todo)} probe responses missing and {server} is not answering at its URL in models.toml.")
    client = S.QwenClient(server)

    def ask(job):
        rec, v, p = job
        r = client.ask(p, "off", 64)
        parsed = S.parse_answer(r["content"])
        return {"key": S.qwen_key(client.model, p, "off"), "model": client.model, "version": v, "row": rec.row, "prompt": p,
                "answer": parsed[0] if parsed else None, "probability": parsed[1] if parsed else None, **r}

    print(f"Requesting {len(todo)} probe responses...", flush=True)
    with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(32) as pool:
        for e in pool.map(ask, todo):
            f.write(json.dumps(e) + "\n")
    if not direct.exists():
        r = client.client.chat.completions.create(
            model=client.model, temperature=0, max_tokens=600, messages=[{"role": "user", "content": DIRECT_QUESTION}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        direct.write_text(json.dumps({"model": client.model, "question": DIRECT_QUESTION, "thinking": "off", "temperature": 0, "max_tokens": 600, "answer": r.choices[0].message.content}, indent=1))


if __name__ == "__main__":
    main()
