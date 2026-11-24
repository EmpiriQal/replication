"""Memorization test on the Replication finding column of data/SCORE_replicated_claims.csv, which no prompt contained.

Items: for every row whose Replication finding contains a number with at least 4 significant digits (integers included), the first such number is the target and the text before it the prefix. Each prefix is sent as raw text, with no chat template, to the server's completions endpoint: greedy (temperature 0), at most 16 new tokens. The log-probability of the exact target string after the prefix is read by echo: the prefix followed by the target is sent with max_tokens 1, and the log-probabilities of the prompt tokens from the target's first character onward are summed. A token that spans the boundary between prefix and target is flagged (boundary_split) and its log-probability counted in full.

Variant "header_2digits" (--variant header_2digits): the prompt begins with the SCORE header and the paper's citation, in the wording of recall_probe.py, then "Replication finding: ", the text before the target, and the target up to and including its second significant digit (the sign, a leading "0." and leading zeros are part of what is given). The model is to complete the rest of the number (the remainder): exact when the greedy output begins with the remainder and is not followed by another digit; also recorded, the number of remainder characters reproduced before the first difference, and the log-probability of the remainder after the prompt (by echo, as above). The first variant ("bare") has no header and gives no digits.

Responses are cached in results/memorization.jsonl, keyed by model, variant, prompt and target; only missing ones are requested, from the model's server at its URL in models.toml. The results are computed by analysis.py (section "memorization").

Usage: uv run python memorization.py --server NAME [--variant bare|header_2digits]   (the paper uses qwen27 and qwen30i, both variants)
"""

import argparse
import csv
import datetime
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import score as S
import serving

CACHE = S.ROOT / "results/memorization.jsonl"
COLUMN = "Replication finding"
MAX_TOKENS = 16
MIN_SIG = 4
GIVEN_SIG = 2
HEAD = "The SCORE project (Systematizing Confidence in Open Research and Evidence) attempted to replicate a claim from the following published paper:"
# a number not inside a word or another number: optional minus sign, digits with an optional decimal part (or a leading decimal point), optional exponent
NUMBER = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w])")


def sig_digits(num: str) -> str:
    """The significant digits of a number written as text: the mantissa's digits without leading zeros (trailing zeros of an integer count)."""
    mant = re.split(r"[eE]", num.lstrip("-"))[0]
    return mant.replace(".", "").lstrip("0")


def magnitude(num: str) -> int:
    """Position of the first significant digit (the power of ten), so that 0.0188 and 1.88 differ."""
    return int(f"{abs(float(num)):e}".split("e")[1]) if float(num) != 0 else 0


def split_given(target: str) -> tuple[str, str]:
    """The target up to and including its GIVEN_SIG-th significant digit, and the rest."""
    seen = 0
    for i, ch in enumerate(target):
        if ch.isdigit() and (seen or ch != "0"):
            seen += 1
            if seen == GIVEN_SIG:
                return target[: i + 1], target[i + 1 :]
    raise ValueError(target)


def items() -> list[dict]:
    """One item per qualifying row: row number (1-based, as in score.py), prefix, target and the row's citation line."""
    with open(S.ROOT / S.CONFIG["data_file"], newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))
    out = []
    for i, c in enumerate(cells, 1):
        text = c[COLUMN] or ""
        for m in NUMBER.finditer(text):
            if len(sig_digits(m.group())) >= MIN_SIG:
                out.append({"row": i, "prefix": text[: m.start()], "target": m.group(), "citation": f"{c['Paper']}. {c['Journal']}. DOI: {c['DOI']}"})
                break
    return out


def key(model: str, prefix: str, target: str, variant: str = "bare") -> str:
    parts = [model, prefix, target, "memorization", MAX_TOKENS] + ([variant] if variant != "bare" else [])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def first_number(text: str) -> str | None:
    """The number at the start of the greedy continuation (after any spaces), or None."""
    m = re.match(r"\s*(-?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?)", text)
    return m.group(1) if m else None


def prompt_and_answer(it: dict, variant: str) -> tuple[str, str]:
    """The text sent and the string the model should produce next."""
    if variant == "bare":
        return it["prefix"], it["target"]
    given, rest = split_given(it["target"])
    return "\n\n".join([HEAD, it["citation"], "Replication finding: " + it["prefix"] + given]), rest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", required=True, help="server to test (qwen27 or qwen30i)")
    ap.add_argument("--variant", default="bare", choices=("bare", "header_2digits"))
    args = ap.parse_args()
    server, variant = args.server, args.variant
    url, model = serving.endpoint(server)
    its = items()
    cache = S.load_cache(CACHE)
    todo = [it for it in its if key(model, it["prefix"], it["target"], variant) not in cache]
    print(f"{len(its)} items; {len(todo)} not cached for {model}, variant {variant}")
    if not todo:
        return
    if not serving.is_ready(server):
        sys.exit(f"{server} is not answering at its URL in models.toml.")
    client = OpenAI(base_url=url, api_key="none", timeout=600)

    def ask(it):
        prompt, answer = prompt_and_answer(it, variant)
        g = client.completions.create(model=model, prompt=prompt, temperature=0, max_tokens=MAX_TOKENS)
        out = g.choices[0].text
        e = client.completions.create(model=model, prompt=prompt + answer, temperature=0, max_tokens=1, echo=True, logprobs=1)
        lp = e.choices[0].logprobs
        n = len(prompt)
        # prompt tokens only (echo returns the prompt, then the one generated token)
        k = len(lp.tokens) - 1
        toks = list(zip(lp.tokens[:k], lp.token_logprobs[:k], lp.text_offset[:k]))
        ends = [o for _, _, o in toks[1:]] + [n + len(answer)]
        tgt = [(t, l) for (t, l, o), end in zip(toks, ends) if end > n]
        split = any(o < n < end for (_, _, o), end in zip(toks, ends))
        rec = {
            "key": key(model, it["prefix"], it["target"], variant), "model": model, "server": server, "variant": variant, "row": it["row"], "prefix": it["prefix"], "target": it["target"],
            "output": out, "target_logprob": sum(l for _, l in tgt), "target_tokens": [t for t, _ in tgt], "boundary_split": split,
            "max_tokens": MAX_TOKENS, "temperature": 0, "date": datetime.date.today().isoformat(),
        }
        if variant == "bare":
            got = first_number(out)
            rec["output_number"] = got
            rec["exact"] = got == it["target"]
            rec["first4"] = got is not None and sig_digits(got)[:MIN_SIG] == sig_digits(it["target"])[:MIN_SIG] and got.startswith("-") == it["target"].startswith("-") and magnitude(got) == magnitude(it["target"])
        else:
            rec["prompt"], rec["given"], rec["remainder"] = prompt, it["target"][: len(it["target"]) - len(answer)], answer
            rec["exact"] = out.startswith(answer) and not out[len(answer) : len(answer) + 1].isdigit()
            rec["chars_correct"] = next((j for j, (a, b) in enumerate(zip(answer, out)) if a != b), min(len(answer), len(out)))
        return rec

    with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(16) as pool:
        for r in pool.map(ask, todo):
            f.write(json.dumps(r) + "\n")
    print(f"{len(todo)} written to {CACHE}")


if __name__ == "__main__":
    main()
