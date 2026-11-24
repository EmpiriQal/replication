"""Every number in the paper and its Supplementary Information, recomputed from the cached model outputs. Sends no requests to any server.

Inputs: data/SCORE_replicated_claims.csv; data/individual_estimates.csv (individual_humans.py); data/leaked_claims.csv (leak_search.py); results/olmo_perplexities.jsonl (score.py, base models); results/qwen_responses.jsonl (score.py, chat models); results/thinking32k_responses.jsonl (thinking32k.py); results/olmo_top_tokens.jsonl (olmo_top_tokens.py); results/qwen_recall_probe.jsonl and results/qwen_recall_direct*.json (recall_probe.py); results/memorization.jsonl (memorization.py); results/plausibility_demo.jsonl (plausibility_demo.py); results/gemini_stated.jsonl (gemini_stated.py); results/thinking16k_run.log, results/thinking32k_run.log and results/run_new_models.log (run times). The ablation section also writes results/ablation_scores.csv, and the individual_tests section results/forecaster_comparisons.csv. The ceiling section is a simulation and reads no inputs.

Usage: uv run python analysis.py [section ...]   (no section: all of them; sections are listed by --list)
"""

import collections
import csv
import itertools
import json
import math
import re
import sys

import numpy as np
from scipy import stats

import plausibility_demo as PD
import score as S

OLMO = "allenai/Olmo-3-1125-32B"
QWEN = "Qwen/Qwen3.8-27B-FP8"
OLMO_THINK = "allenai/Olmo-3.1-32B-Think"
# the model set: (label, model ID, parameters in billions as listed by Hugging Face for the checkpoint); base models read by L, chat models by the token and stated scores
BASE_MODELS = (
    ("Olmo 3 7B", "allenai/Olmo-3-1025-7B", 7.30), ("Olmo 3 32B", OLMO, 32.23),
    ("Qwen3.5 0.8B", "Qwen/Qwen3.5-0.8B-Base", 0.87), ("Qwen3.5 2B", "Qwen/Qwen3.5-2B-Base", 2.27), ("Qwen3.5 4B", "Qwen/Qwen3.5-4B-Base", 4.66),
    ("Qwen3.5 9B", "Qwen/Qwen3.5-9B-Base", 9.65), ("Qwen3 30B-A3B", "Qwen/Qwen3-30B-A3B-Base", 30.53),
)
CHAT_MODELS = (
    ("Olmo 3 7B Instruct", "allenai/Olmo-3-7B-Instruct", 7.30), ("Olmo 3.1 32B Instruct", "allenai/Olmo-3.1-32B-Instruct", 32.23),
    ("Qwen3.5 0.8B", "Qwen/Qwen3.5-0.8B", 0.87), ("Qwen3.5 2B", "Qwen/Qwen3.5-2B", 2.27), ("Qwen3.5 4B", "Qwen/Qwen3.5-4B", 4.66),
    ("Qwen3.5 9B", "Qwen/Qwen3.5-9B", 9.65), ("Qwen3 30B-A3B Instruct", "Qwen/Qwen3-30B-A3B-Instruct-2507", 30.53), ("Qwen3.8 27B", QWEN, 27.78),
)
# base and chat models of the same family and size, in BASE_MODELS and CHAT_MODELS order
PAIRS = tuple(zip(BASE_MODELS, CHAT_MODELS[:7]))
F = "with_original_finding"
C = "claim_only"
BOOT = 10000  # bootstrap resamples, everywhere
R = S.ROOT / "results"

RECS = S.load_data(S.ROOT / "data/SCORE_replicated_claims.csv")
BY_ROW = {r.row: r for r in RECS}
PPL = S.load_cache(R / "olmo_perplexities.jsonl")
QC = S.load_cache(R / "qwen_responses.jsonl")


def sel(subset: str, data_type: str = "all") -> list[S.Record]:
    return S.select(RECS, subset, data_type)


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------- model outputs


def ppl(rec: S.Record, template: str, ending: str, cont: str, model: str = OLMO) -> float:
    return PPL[S.cache_key(model, S.context(rec, template, ending), cont)]["ppl"]






def olmo_l(rows, template, model=OLMO):
    """L = log p(" yes") - log p(" no") with the yes/no ending (any base model)."""
    return np.round(arr([math.log(ppl(r, template, "yes_no", " no", model)) - math.log(ppl(r, template, "yes_no", " yes", model)) for r in rows]), S.SCORE_DECIMALS)


def qwen(rec: S.Record, template: str, thinking: str, budget: int = 0, model: str = QWEN) -> dict:
    # as first generated: answers forced later by score.py (qwen_force_answer) are not part of the analyses here
    return S.unforced(QC[S.qwen_key(model, S.qwen_prompt(rec, template), thinking, budget if thinking == "on" else 0)])


def chat_scores(rows, model, thinking="off"):
    """Token score, stated score and responses of any chat model, with original finding (thinking on only for Olmo Think, with no budget)."""
    es = [qwen(r, F, thinking, 0, model) for r in rows]
    return arr([token(e) for e in es]), arr([e["probability"] / 100 for e in es]), es


def token(e: dict) -> float:
    return S.token_probability(e["answer_top_logprobs"])[0]


def qwen_scores(rows, template, thinking, budget=4096):
    es = [qwen(r, template, thinking, budget) for r in rows]
    return arr([token(e) for e in es]), arr([e["probability"] / 100 for e in es]), es


def complete_reasoning() -> dict[int, dict]:
    """High-power rows, with original finding: the 32k response where the 16k one did not finish (reasoning capped or no answer), otherwise the 16k one. Returns that and the 32k responses used."""
    unfinished = {r.row for r in sel("highpower") if (e := qwen(r, F, "on", 16384))["answer"] is None or S.reasoning_capped(e, 16384)}
    new = {}
    for line in (R / "thinking32k_responses.jsonl").read_text().splitlines():
        e = json.loads(line)
        # the file also holds row 189, asked in error (its 16k reasoning ended at 16,379 tokens, short of the cap); only the unfinished rows are used
        if e["row"] in unfinished:
            new[e["row"]] = e
    return {r.row: new.get(r.row) or qwen(r, F, "on", 16384) for r in sel("highpower")}, new


def humans(rows):
    return arr([r.replicats for r in rows]), arr([r.markets for r in rows])


def outcomes(rows):
    return np.array([r.succeeded for r in rows])


# ---------------------------------------------------------------- statistics


def auc(x, rows):
    return S.weighted_auc(x, outcomes(rows), S.weights(rows))


def ranks(x) -> np.ndarray:
    """Ranks with ties given their average rank."""
    x = arr(x)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def _claim_resample(rng, y):
    """Claims drawn with replacement; a draw with only one outcome is redrawn."""
    while True:
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and not y[i].all():
            return i


def _paper_resample(rng, groups, y):
    """Papers drawn with replacement, each keeping all its rows; a draw with only one outcome is redrawn."""
    while True:
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if y[i].any() and not y[i].all():
            return i


def boot_aucs(scores: dict, rows) -> dict:
    """Bootstrap over claims (all weights are 1 in the human and high-power sets): AUC of each score in each of BOOT resamples (seed 0), the same resamples for all, so differences are paired. A resample with only one outcome is redrawn."""
    y, w = outcomes(rows), arr(S.weights(rows))
    assert (w == 1).all()
    rng = np.random.default_rng(0)
    out = {k: [] for k in scores}
    for _ in range(BOOT):
        i = _claim_resample(rng, y)
        for k, x in scores.items():
            out[k].append(S.weighted_auc(arr(x)[i], y[i], w[i]))
    return {k: arr(v) for k, v in out.items()}


def boot_aucs_papers(scores: dict, rows) -> dict:
    """Bootstrap over papers for selections where papers have several rows (all 274): each resample draws papers with replacement and keeps all their rows at their weights, so each drawn paper counts once. Paired and redrawn as in boot_aucs."""
    y, w = outcomes(rows), arr(S.weights(rows))
    by_paper = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_paper[r.paper_id].append(i)
    groups = list(by_paper.values())
    rng = np.random.default_rng(0)
    out = {k: [] for k in scores}
    for _ in range(BOOT):
        i = _paper_resample(rng, groups, y)
        for k, x in scores.items():
            out[k].append(S.weighted_auc(arr(x)[i], y[i], w[i]))
    return {k: arr(v) for k, v in out.items()}


def boot_any(scores: dict, rows) -> dict:
    return boot_aucs(scores, rows) if (arr(S.weights(rows)) == 1).all() else boot_aucs_papers(scores, rows)


def ci(v) -> str:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def ci_signed(v) -> str:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def offset_accuracy(x, rows, mode: str) -> float:
    """Weighted accuracy of (x + b > 0) with b fitted by score.py's rule, on all rows ("fitted") or leaving each paper out ("loo")."""
    y, w = outcomes(rows), S.weights(rows)
    if mode == "fitted":
        b = [S.fit_offset(x, y, w)] * len(rows)
    else:
        b = S.loo_offsets(x, y, w, [r.paper_id for r in rows])
    return S.weighted_accuracy(w, (arr(x) + arr(b) > 0) == y)


def answer_accuracy(es, rows) -> float:
    return S.weighted_accuracy(S.weights(rows), [float((e["answer"] == "yes") == r.succeeded) for e, r in zip(es, rows)])


def human_accuracy(p, rows) -> float:
    return S.weighted_accuracy(S.weights(rows), (arr(p) >= 0.5) == outcomes(rows))


def logit(p):
    p = np.clip(arr(p), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def header(title: str) -> None:
    print(f"\n{'=' * 8} {title}")


# ---------------------------------------------------------------- sections




def baseline():
    header("Always predicting the more common outcome: accuracy (paper-weighted success rate)")
    for subset in S.SUBSETS:
        cells = []
        for dt in S.DATA_TYPES:
            rows = sel(subset, dt)
            rate = S.weighted_accuracy(S.weights(rows), outcomes(rows))
            cells.append(f"{dt} {max(rate, 1 - rate):.3f} ({rate:.3f})")
        print(f"  {subset:9} " + "   ".join(cells))








def qwen_off():
    header("Qwen, thinking off: AUC by set and data type (with original finding)")
    for subset in S.SUBSETS:
        for dt in S.DATA_TYPES:
            rows = sel(subset, dt)
            tk, st, _ = qwen_scores(rows, F, "off")
            line = f"  {subset:9} {dt:9} n={len(rows):3}  token {auc(tk, rows):.3f}  stated {auc(st, rows):.3f}"
            if subset != "all":
                rc, mk = humans(rows)
                line += f"  repliCATS {auc(rc, rows):.3f}  Markets {auc(mk, rows):.3f}"
            print(line)
    print("  claim only, token AUC: " + "  ".join(f"{s} {auc(qwen_scores(sel(s), C, 'off')[0], sel(s)):.3f}" for s in S.SUBSETS))
    for t in S.TEMPLATES:
        for s in S.SUBSETS:
            rows = sel(s)
            tk, st, es = qwen_scores(rows, t, "off")
            print(f"  {t:22} {s:9} accuracy {answer_accuracy(es, rows):.3f}  says yes {sum(e['answer'] == 'yes' for e in es)} of {len(rows)}")
    _, st, es = qwen_scores(RECS, F, "off")
    print("  stated values with original finding, after 'no':", collections.Counter(e["probability"] for e in es if e["answer"] == "no").most_common(6),
          " after 'yes':", collections.Counter(e["probability"] for e in es if e["answer"] == "yes").most_common(4))
    for s in ("human", "highpower"):
        rows = sel(s)
        tk, st, es = qwen_scores(rows, F, "off")
        no = [i for i, e in enumerate(es) if e["answer"] == "no"]
        print(f"  {s}: rank correlation stated with token {spearman(st, tk):.2f}; within 'no' answers {spearman(st[no], tk[no]):.2f}")
        print(f"  {s}: median p(yes) {np.median(tk):.3f}; success rate {outcomes(rows).mean():.3f}")


def qwen_agreement():
    header("Qwen (thinking off, token) against the humans, with original finding")
    for s in ("human", "highpower"):
        rows = sel(s)
        y = outcomes(rows)
        q = qwen_scores(rows, F, "off")[0]
        rc, mk = humans(rows)
        h = (rc + mk) / 2
        print(f"  {s} (n={len(rows)}): rank correlation Qwen-repliCATS {spearman(q, rc):.2f}, Qwen-Markets {spearman(q, mk):.2f}, repliCATS-Markets {spearman(rc, mk):.2f}")

        # rank credit: a claim's percentile in the predictor's own ranking, counted from the correct end
        def rank_credit(x):
            p = ranks(x) / (len(x) - 1)
            return np.where(y, p, 1 - p)

        hedged = np.where(y, h, 1 - h)
        print(f"      Qwen rank credit with human hedged credit (Pearson) {np.corrcoef(rank_credit(q), hedged)[0, 1]:.2f}")
        qr, hr = rank_credit(q) > 0.5, rank_credit(h) > 0.5
        print(f"      on the correct side of the median: both {np.sum(qr & hr)}, Qwen only {np.sum(qr & ~hr)}, humans only {np.sum(~qr & hr)}, neither {np.sum(~qr & ~hr)}")
        print(f"      AUC of rank average of Qwen and average human {auc(ranks(q) + ranks(h), rows):.3f}")
        b = boot_aucs({"Qwen": q, "repliCATS": rc, "Markets": mk}, rows)
        print(f"      Qwen - repliCATS {auc(q, rows) - auc(rc, rows):+.3f} {ci_signed(b['Qwen'] - b['repliCATS'])}  Qwen - Markets {auc(q, rows) - auc(mk, rows):+.3f} {ci_signed(b['Qwen'] - b['Markets'])}")


def olmo_yesno():
    header("Olmo with the yes/no ending, score L: AUC with original finding")
    for subset in S.SUBSETS:
        for dt in S.DATA_TYPES:
            rows = sel(subset, dt)
            line = f"  {subset:9} {dt:9} n={len(rows):3}  L {auc(olmo_l(rows, F), rows):.3f}  Qwen token {auc(qwen_scores(rows, F, 'off')[0], rows):.3f}"
            if subset != "all":
                rc, mk = humans(rows)
                line += f"  repliCATS {auc(rc, rows):.3f}  Markets {auc(mk, rows):.3f}"
            print(line)
    print("  claim only: " + "  ".join(f"{s} L {auc(olmo_l(sel(s), C), sel(s)):.3f}" for s in S.SUBSETS))
    for t in S.TEMPLATES:
        print(f"  leave-one-paper-out accuracy, {t}: " + "  ".join(f"{s} L {offset_accuracy(olmo_l(sel(s), t), sel(s), 'loo'):.3f}" for s in S.SUBSETS))
    header("Median probabilities of the answer words over the 274 rows")
    for ending, (pos, neg) in (("yes_no", (" yes", " no")),):
        for t in S.TEMPLATES:
            pp = arr([1 / ppl(r, t, ending, pos) for r in RECS])
            pn = arr([1 / ppl(r, t, ending, neg) for r in RECS])
            sh = pp / (pp + pn)
            print(f"  {ending:9} {t:22} p({pos.strip()}) {np.median(pp):.4f}  p({neg.strip()}) {np.median(pn):.3f}  mass {np.median(pp + pn):.3f}  share of positive {sh.min():.3f} to {sh.max():.3f}")


def variants():
    header("Answer-word variants: Qwen thinking off (constrained output) and Olmo yes/no ending (unconstrained top 20)")
    es = [qwen(r, t, "off") for t in S.TEMPLATES for r in RECS]
    finite = collections.defaultdict(list)
    masked = set()
    sums = []
    for e in es:
        sums.append(sum(math.exp(lp) for _, lp in e["answer_top_logprobs"] if lp > -1000))
        for tok, lp in e["answer_top_logprobs"]:
            (finite[tok].append(math.exp(lp)) if lp > -1000 else masked.add(tok))
    print(f"  Qwen: {len(es)} responses; tokens with a finite probability: " + ", ".join(f"{t!r} in {len(v)} (median {np.median(v):.2e}, max {max(v):.2e})" for t, v in finite.items()))
    print(f"  Qwen: masked tokens (-9999): {sorted(masked)}; sum of finite probabilities {min(sums):.7f} to {max(sums):.7f}")

    def summed(e):
        y = sum(math.exp(lp) for t, lp in e["answer_top_logprobs"] if lp > -1000 and t.strip() in ("yes", "y", "ye"))
        n = sum(math.exp(lp) for t, lp in e["answer_top_logprobs"] if lp > -1000 and t.strip() in ("no", "n"))
        return 1 / (1 + math.exp(-round(math.log(y) - math.log(n), S.SCORE_DECIMALS)))

    for t in S.TEMPLATES:
        print(f"  Qwen {t}: AUC current / summed variants: " + "  ".join(f"{s} {auc([token(qwen(r, t, 'off')) for r in sel(s)], sel(s)):.3f} / {auc([summed(qwen(r, t, 'off')) for r in sel(s)], sel(s)):.3f}" for s in S.SUBSETS))
    top = {}
    for line in (R / "olmo_top_tokens.jsonl").read_text().splitlines():
        e = json.loads(line)
        top[(e["template"], e["row"])] = {tok: lp for _, tok, lp in e["top"]}
    worst = max(abs(top[(t, r.row)][" yes"] + math.log(ppl(r, t, "yes_no", " yes"))) for t in S.TEMPLATES for r in RECS)
    print(f"  Olmo: largest difference of top-20 log p(' yes') from the perplexity cache: {worst:.1e}")

    def mass(d, words):
        return sum(math.exp(lp) for tok, lp in d.items() if tok.strip().lower() in words and tok not in (" yes", " no"))

    for t in S.TEMPLATES:
        ds = [top[(t, r.row)] for r in RECS]
        yv = arr([mass(d, ("yes",)) for d in ds])
        nv = arr([mass(d, ("no",)) for d in ds])
        print(f"  Olmo {t}: yes variants median {np.median(yv):.3f} [{yv.min():.3f}, {yv.max():.3f}]  no variants median {np.median(nv):.3f} [{nv.min():.3f}, {nv.max():.3f}]  outside top 20 median {np.median([1 - sum(math.exp(v) for v in d.values()) for d in ds]):.4f}")
        for w in (" yes", " no"):
            v = arr([math.exp(d[w]) for d in ds])
            print(f"  Olmo {t}: {w!r} median {np.median(v):.3f} [{v.min():.3f}, {v.max():.3f}]")
        for w in (" not", " Not", " Nope"):
            v = [math.exp(d[w]) for d in ds if w in d]
            print(f"  Olmo {t}: {w!r} in {len(v)} of {len(ds)} lists" + (f", max {max(v):.1e}" if v else ""))
        cls_all = [math.log(math.exp(d[" yes"]) + mass(d, ("yes",))) - math.log(math.exp(d[" no"]) + mass(d, ("no",))) for d in ds]
        print(f"  Olmo {t}: rank correlation of classes with L over 274 {spearman(cls_all, olmo_l(RECS, t)):.3f}")
        for s in S.SUBSETS:
            rows = sel(s)
            cls = [round(math.log(math.exp(top[(t, r.row)][" yes"]) + mass(top[(t, r.row)], ("yes",))) - math.log(math.exp(top[(t, r.row)][" no"]) + mass(top[(t, r.row)], ("no",))), S.SCORE_DECIMALS) for r in rows]
            print(f"      {s:9} AUC L {auc(olmo_l(rows, t), rows):.3f}  classes {auc(cls, rows):.3f}")


RECALL_MODELS = {"Qwen3.8 27B": QWEN, "Qwen3 30B-A3B Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507"}


def recall():
    """Recall probe, reasoning off: token AUC from the citation alone, per model and set, with bootstrap 95% intervals. Entries without a "model" field are Qwen3.8 27B's."""
    header("Recall probe (thinking off): token AUC with bootstrap 95% CI")
    probe = {}
    for line in (R / "qwen_recall_probe.jsonl").read_text().splitlines():
        e = json.loads(line)
        probe[(e.get("model", QWEN), e["version"], int(e["row"]))] = e
    have = {m for m, _, _ in probe}
    models = {k: m for k, m in RECALL_MODELS.items() if m in have}
    for label, m in models.items():
        print(f"  {label}")
        for v in ("citation",):
            for s in S.SUBSETS:
                rows = sel(s)
                es = [probe[(m, v, r.row)] for r in rows]
                x = arr([token(e) for e in es])
                extra = f" {ci(boot_aucs({'x': x}, rows)['x'])}" if s != "all" else ""
                print(f"      {v:8} {s:9} n={len(rows):3}  token AUC {auc(x, rows):.3f}{extra}  says yes {sum(e['answer'] == 'yes' for e in es)}")
    for p in sorted((R).glob("qwen_recall_direct*.json")):
        d = json.loads(p.read_text())
        print(f"  direct question, {d.get('model', QWEN)}:\n" + "\n".join("      " + l for l in d["answer"].splitlines()))


def _paired_logprob(a, b) -> str:
    d = arr([x["target_logprob"] - y["target_logprob"] for x, y in zip(a, b)])
    t = stats.ttest_rel([x["target_logprob"] for x in a], [y["target_logprob"] for y in b])
    w = stats.wilcoxon(d)
    same_tok = sum(x["target_tokens"] == y["target_tokens"] for x, y in zip(a, b))
    return (f"log-probability, Qwen3.8 minus Qwen3 30B: mean {d.mean():+.3f} (sd {d.std(ddof=1):.3f}, median {np.median(d):+.3f}); Qwen3.8 higher on {int((d > 0).sum())} of {len(d)}"
            f"; paired t({len(d) - 1}) = {t.statistic:.2f}, p {t.pvalue:.3g}; Wilcoxon signed-rank p {w.pvalue:.3g}; tokenized identically in {same_tok} of {len(d)}")


def _mcnemar(a, b, f) -> str:
    only_a, only_b = sum(x[f] and not y[f] for x, y in zip(a, b)), sum(y[f] and not x[f] for x, y in zip(a, b))
    p = stats.binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return f"{f}: both {sum(x[f] and y[f] for x, y in zip(a, b))}, Qwen3.8 only {only_a}, Qwen3 30B only {only_b}; exact McNemar p {p:.3f}"


def memorization():
    """Memorization test (memorization.py), on the targets with a decimal point, in two variants. Bare: the report text before the target, no header; per model, greedy outputs reproducing the target exactly and matching its first 4 significant digits, and the mean log-probability of the target. Header and 2 digits: the SCORE header and citation, the report text and the target's first 2 significant digits; per model, exact completions of the remainder, characters of the remainder reproduced before the first difference, and the mean log-probability of the remainder. The two models compared item by item: exact McNemar tests on the matches; on the log-probabilities (and on the characters reproduced) the mean paired difference with a paired t-test and a Wilcoxon signed-rank test."""
    header("Memorization test on the Replication finding column")
    es = collections.defaultdict(dict)
    for line in (R / "memorization.jsonl").read_text().splitlines():
        e = json.loads(line)
        es[(e["model"], e.get("variant", "bare"))][(e["row"], e["prefix"], e["target"])] = e
    ma, mb = QWEN, "Qwen/Qwen3-30B-A3B-Instruct-2507"
    models = (("Qwen3.8 27B", ma), ("Qwen3 30B-A3B Instruct", mb))
    for variant in ("bare", "header_2digits"):
        if not all((m, variant) in es for _, m in models):
            print(f"  variant {variant}: not run for both models")
            continue
        both = sorted(set(es[(ma, variant)]) & set(es[(mb, variant)]))
        # integer targets (sample sizes, years) can be worked out or guessed from the prefix, so only targets with a decimal point (or an exponent) are analysed
        keys = [k for k in both if re.search(r"[.eE]", k[2])]
        dates = sorted({e["date"] for _, m in models for e in es[(m, variant)].values()})
        print(f"  variant {variant}: items run in both models {len(both)}; analysed (decimal targets) {len(keys)}, integer targets dropped {len(both) - len(keys)}; run dates {dates}")
        for label, m in models:
            d = [es[(m, variant)][k] for k in keys]
            lp = arr([e["target_logprob"] for e in d])
            nt = arr([len(e["target_tokens"]) for e in d])
            line = f"      {label:24} n={len(d)}  exact {sum(e['exact'] for e in d)}"
            if variant == "bare":
                line += f"  first 4 significant digits {sum(e['first4'] for e in d)}  greedy output starts with no number {sum(e['output_number'] is None for e in d)}"
            else:
                cc = np.array([e["chars_correct"] for e in d])
                nr = arr([len(e["remainder"]) for e in d])
                line += f"  remainder characters median {np.median(nr):.0f} mean {nr.mean():.2f}  characters correct before the first difference: mean {cc.mean():.2f}, counts by number {np.bincount(cc).tolist()}"
            line += f"  mean log-probability {lp.mean():.3f} (median {np.median(lp):.3f}, sd {lp.std(ddof=1):.3f})  tokens mean {nt.mean():.2f}  per token {(lp / nt).mean():.3f}  boundary splits {sum(e['boundary_split'] for e in d)}"
            print(line)
            for e in d:
                if e["exact"] or e.get("first4") or e.get("chars_correct", 0) >= 2:
                    shown = f"given {e['given']}, remainder {e['remainder']}, output {e['output'][:12]!r}" if variant != "bare" else f"target {e['target']}, output {e['output_number']}"
                    print(f"          row {e['row']}: {shown}, log-probability {e['target_logprob']:.3f}")
        a = [es[(ma, variant)][k] for k in keys]
        b = [es[(mb, variant)][k] for k in keys]
        for f in (("exact", "first4") if variant == "bare" else ("exact",)):
            print("      " + _mcnemar(a, b, f))
        if variant != "bare":
            d = arr([x["chars_correct"] - y["chars_correct"] for x, y in zip(a, b)])
            nz = d[d != 0]
            t = stats.ttest_rel([x["chars_correct"] for x in a], [y["chars_correct"] for y in b])
            w = stats.wilcoxon(d) if len(nz) else None
            print(f"      characters correct, Qwen3.8 minus Qwen3 30B: mean {d.mean():+.3f}; Qwen3.8 more on {int((d > 0).sum())}, fewer on {int((d < 0).sum())}, equal on {int((d == 0).sum())}"
                  f"; paired t({len(d) - 1}) = {t.statistic:.2f}, p {t.pvalue:.3g}; Wilcoxon signed-rank p {w.pvalue if w else float('nan'):.3g}")
        print("      " + _paired_logprob(a, b))


def thinking_4k():
    header("Qwen thinking on, 4,096-token cap: AUC, yes answers, accuracy")
    es_all = [qwen(r, t, "on", 4096) for t in S.TEMPLATES for r in RECS]
    toks = [e["completion_tokens"] for e in es_all]
    print(f"  {len(es_all)} responses; completion tokens {min(toks)} to {max(toks)}")
    for t in S.TEMPLATES:
        for s in S.SUBSETS:
            for dt in S.DATA_TYPES:
                rows = sel(s, dt)
                tk0, st0, e0 = qwen_scores(rows, t, "off")
                tk1, st1, e1 = qwen_scores(rows, t, "on", 4096)
                line = (f"  {t:22} {s:9} {dt:9} n={len(rows):3}  off stated {auc(st0, rows):.3f} token {auc(tk0, rows):.3f} yes {sum(e['answer'] == 'yes' for e in e0):3} acc {answer_accuracy(e0, rows):.3f}"
                        f" | on stated {auc(st1, rows):.3f} token {auc(tk1, rows):.3f} yes {sum(e['answer'] == 'yes' for e in e1):3} acc {answer_accuracy(e1, rows):.3f}")
                if s != "all":
                    rc, mk = humans(rows)
                    line += f" | humans {auc(rc, rows):.3f} {auc(mk, rows):.3f}"
                print(line)
        tk, st, es = qwen_scores(RECS, t, "on", 4096)
        print(f"  {t}: token near certain {np.mean((tk > 0.99) | (tk < 0.01)):.2f}; stated values {collections.Counter(e['probability'] for e in es).most_common(6)}")
        for s in ("human", "highpower"):
            rows = sel(s)
            h = sum(humans(rows)) / 2
            tk1, st1, _ = qwen_scores(rows, t, "on", 4096)
            tk0 = qwen_scores(rows, t, "off")[0]
            print(f"      {s}: rank correlation stated-token (on) {spearman(st1, tk1):.2f}; token on with average human {spearman(tk1, h):.2f}, token off with average human {spearman(tk0, h):.2f}; token on with token off {spearman(tk1, tk0):.2f}")
    groups = collections.defaultdict(list)
    for line in (R / "qwen_responses.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["template"] == C and e["thinking"] == "on" and e.get("thinking_budget") == 4096:
            groups[e["key"]].append((e["answer"], e["probability"]))
    multi = [v for v in groups.values() if len(v) > 1]
    print(f"  claim only, identical prompts asked more than once: {len(multi)} groups; stated probability differs in {sum(len({p for _, p in v}) > 1 for v in multi)}, answer differs in {sum(len({a for a, _ in v}) > 1 for v in multi)}")




def thinking_complete():
    header("Qwen with complete reasoning, high-power set, with original finding")
    full, new = complete_reasoning()
    rows = sel("highpower")
    old16 = [qwen(r, F, "on", 16384) for r in rows]
    print(f"  16k run: tokens median {int(np.median([e['completion_tokens'] for e in old16]))}, reasoning at cap {sum(S.reasoning_capped(e, 16384) for e in old16)}, no answer {sum(e['answer'] is None for e in old16)}")
    cap = {r: S.reasoning_capped(e, 32000) for r, e in new.items()}
    print(f"  32k run: {len(new)} rows; hit cap {sum(cap.values())} {sorted(r for r, v in cap.items() if v)}; forced {sorted(r for r, e in new.items() if e['forced'])}; stuck {sorted(r for r, e in new.items() if e['stuck'])}")
    rt = [e.get("reasoning_tokens") or e["completion_tokens"] for e in full.values()]
    print(f"  complete set: reasoning tokens median {int(np.median(rt))}, range {min(rt)} to {max(rt)}; row 271 at 32k: {new[271]['completion_tokens']} tokens, forced {new[271]['forced']}")
    flagged = {r for r in new if cap[r] or new[r]["forced"] or new[r]["stuck"]}
    for name, rs in (("all", rows), ("excluding capped, forced or stuck", [r for r in rows if r.row not in flagged])):
        y = outcomes(rs)
        x = {
            "off token": arr([token(qwen(r, F, "off")) for r in rs]),
            "4k stated": arr([qwen(r, F, "on", 4096)["probability"] for r in rs]),
            "4k token": arr([token(qwen(r, F, "on", 4096)) for r in rs]),
            "complete stated": arr([full[r.row]["probability"] for r in rs]),
            "complete token": arr([token(full[r.row]) for r in rs]),
        }
        rc, mk = humans(rs)
        x["repliCATS"], x["Markets"] = rc, mk
        b = boot_aucs(x, rs)
        print(f"  {name} (n={len(rs)}, success rate {y.mean():.3f}):")
        for k, v in x.items():
            print(f"      {k:16} AUC {auc(v, rs):.3f} {ci(b[k])}")
        for label, es in (("off", [qwen(r, F, "off") for r in rs]), ("4k", [qwen(r, F, "on", 4096) for r in rs]), ("complete", [full[r.row] for r in rs])):
            tk = arr([token(e) for e in es])
            print(f"      {label:9} yes {sum(e['answer'] == 'yes' for e in es):2}  accuracy {answer_accuracy(es, rs):.3f}  token near certain {np.mean((tk > 0.99) | (tk < 0.01)):.2f}")
        print(f"      humans accuracy {human_accuracy(rc, rs):.3f} {human_accuracy(mk, rs):.3f}")
        if name == "all":
            for a, h in (("complete stated", "off token"), ("complete stated", "repliCATS"), ("complete stated", "Markets"), ("complete stated", "4k stated"), ("complete token", "off token"), ("off token", "repliCATS"), ("off token", "Markets")):
                d = b[a] - b[h]
                print(f"      {a} - {h}: {auc(x[a], rs) - auc(x[h], rs):+.3f} {ci_signed(d)}, two-sided p {2 * min(np.mean(d >= 0), np.mean(d <= 0)):.3f}")
            h = (rc + mk) / 2
            print(f"      rank correlation with average human: complete stated {spearman(x['complete stated'], h):.2f}, off token {spearman(x['off token'], h):.2f}; complete stated with off token {spearman(x['complete stated'], x['off token']):.2f}")
            print(f"      complete stated values {collections.Counter(int(v) for v in x['complete stated']).most_common(10)}")


TRACE_PATTERNS = {
    "p-value": r"p\s*[=<]|p-value", "sample size": r"sample size|\bN\s*=|participants", "effect size": r"effect size|cohen|\bd\s*=",
    "names SCORE, Many Labs or the Reproducibility Project": r"\bSCORE\b|Many Labs|Reproducibility Project|Open Science Collaboration", "'I recall'": r"I recall",
}


def _trace_counts(es) -> str:
    return ", ".join(f"{k} {sum(bool(re.search(p, e['reasoning'], re.I)) for e in es)}" for k, p in TRACE_PATTERNS.items())


def traces():
    header("Content of the reasoning traces, with original finding")
    full, _ = complete_reasoning()
    for label, es in (("4k, all 274", [qwen(r, F, "on", 4096) for r in RECS]), ("16k, high-power (answered)", [e for e in (qwen(r, F, "on", 16384) for r in sel("highpower")) if e["answer"]]),
                      ("complete, high-power", list(full.values())), ("Olmo Think, high-power", chat_scores(sel("highpower"), OLMO_THINK, "on")[2])):
        print(f"  {label} (n={len(es)}): " + _trace_counts(es))


def cost():
    header("Cost: thinking off against complete reasoning, high-power set, with original finding")
    rows = sel("highpower")
    full, _ = complete_reasoning()
    off = [qwen(r, F, "off") for r in rows]
    g_off = [e["completion_tokens"] for e in off]
    g_on = [full[r.row]["completion_tokens"] for r in rows]
    forced = [full[r.row].get("forced_prompt_tokens", 0) for r in rows]
    # prompt lengths with each chat template (thinking on adds a system message), measured with the server's tokenizer on 28 October 2026
    p_off, p_on = 25104, 27384
    w_flops = 2 * 27e9
    a_flops = 4 * 24 * 256 * 16
    attn_off = a_flops * sum(g * (p_off / len(rows) + g / 2) for g in g_off)
    attn_on = a_flops * sum(g * (p_on / len(rows) + g / 2) for g in g_on)
    print(f"  prompt tokens: off {p_off} (mean {p_off / len(rows):.0f}), on {p_on} (mean {p_on / len(rows):.0f})")
    print(f"  generated tokens: off {sum(g_off)} (mean {np.mean(g_off):.1f}); complete {sum(g_on)} (mean {np.mean(g_on):.0f}, median {np.median(g_on):.0f}, range {min(g_on)} to {max(g_on)}); forced prompts {sum(forced)}")
    t_off = p_off + sum(g_off)
    t_on = p_on + sum(g_on) + sum(forced) + 10 * sum(1 for f in forced if f)
    f_off = w_flops * t_off + attn_off
    f_on = w_flops * t_on + attn_on
    print(f"  tokens processed: off {t_off}, complete {t_on}; FLOPs, weights: off {w_flops * t_off:.3e}, complete {w_flops * t_on:.3e}; attention during generation: off {attn_off:.2e}, complete {attn_on:.2e}; total off {f_off:.3e}, complete {f_on:.3e}; ratio {f_on / f_off:.1f}")
    for name in ("thinking16k_run.log", "thinking32k_run.log"):
        last = [l for l in (R / name).read_text().splitlines() if "/57 after" in l or "/27 after" in l][-1]
        print(f"  {name}: {last.strip()}")
    gen16 = sum(qwen(r, F, "on", 16384)["completion_tokens"] for r in rows)
    min16 = float(re.search(r"after ([\d.]+) min", [l for l in (R / "thinking16k_run.log").read_text().splitlines() if "/57 after" in l][-1]).group(1))
    rate = gen16 / (min16 * 60)
    # the 57 thinking-off requests sent at once took 18 seconds (timed by hand on 28 October 2026)
    t_off_s = 18
    print(f"  16k run generated {gen16} tokens in {min16} min ({rate:.1f} tokens per second); complete set at that rate {sum(g_on) / rate / 3600:.2f} hours, {sum(g_on) / rate / t_off_s:.0f} times the {t_off_s} seconds without reasoning")


def _confidence_rows(subset, name):
    """Score, outcome, weight and paper for one predictor; Olmo uses score.py's leave-one-paper-out offsets."""
    rows = sel(subset)
    if name == "Qwen token":
        x = logit(qwen_scores(rows, F, "off")[0])
    elif name == "Olmo L":
        x = olmo_l(rows, F)
    elif name == "repliCATS":
        x = humans(rows)[0]
    elif name == "Markets":
        x = humans(rows)[1]
    return rows, x


def _pct(x, v):
    return ((x < v).sum() + 0.5 * (x == v).sum()) / len(x)


def confidence_stats(rows, pred, conf) -> tuple[float, list[float], float, np.ndarray]:
    """For predictions pred with confidence conf: accuracy, accuracy in the low, middle and high confidence terciles, the AUC of confidence for a correct prediction, and that AUC in each of BOOT resamples of claims (seed 0)."""
    y, w = outcomes(rows), arr(S.weights(rows))
    correct = pred == y
    o = np.argsort(conf, kind="stable")
    g = np.empty(len(conf), int)
    for k, idx in enumerate(np.array_split(o, 3)):
        g[idx] = k
    accs = [S.weighted_accuracy(w[g == k], correct[g == k]) for k in range(3)]
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(BOOT):
        i = _claim_resample(rng, correct)
        vals.append(S.weighted_auc(conf[i], correct[i], w[i]))
    return S.weighted_accuracy(w, correct), accs, S.weighted_auc(conf, correct, w), arr(vals)


def _conf_report(label, rows, pred, conf):
    acc, accs, a, vals = confidence_stats(rows, pred, conf)
    print(f"    {label:44} acc {acc:.3f} | {accs[0]:.3f} / {accs[1]:.3f} / {accs[2]:.3f} | AUC {a:.3f} {ci(vals)}")


def _loo(x, rows):
    thr = -arr(S.loo_offsets(x, outcomes(rows), S.weights(rows), [r.paper_id for r in rows]))
    return x > thr, arr([abs(_pct(x, s) - _pct(x, t)) for s, t in zip(x, thr)])


def confidence():
    header("Confidence and accuracy, with original finding: accuracy by confidence tercile (low / mid / high), AUC of confidence for a correct prediction")
    for subset in ("human", "highpower"):
        print(f"  {subset}")
        for name in ("Qwen token", "Olmo L", "repliCATS", "Markets"):
            rows, x = _confidence_rows(subset, name)
            _conf_report(name + ", LOO threshold", rows, *_loo(x, rows))
        for name in ("repliCATS", "Markets"):
            rows, x = _confidence_rows(subset, name)
            _conf_report(name + ", simple", rows, x >= 0.5, np.abs(x - 0.5))
    header("Qwen's stated probability as confidence")
    full, _ = complete_reasoning()
    for subset, conds in (("human", ("off", "4k")), ("highpower", ("off", "4k", "complete"))):
        rows = sel(subset)
        print(f"  {subset}")
        for c in conds:
            if c == "complete":
                es = [full[r.row] for r in rows]
            else:
                es = [qwen(r, F, "off" if c == "off" else "on", 4096) for r in rows]
            st, tk = arr([e["probability"] / 100 for e in es]), logit([token(e) for e in es])
            print(f"   {c}: {len(np.unique(st))} distinct stated values")
            _conf_report("stated, LOO threshold", rows, *_loo(st, rows))
            _conf_report("stated, simple", rows, st >= 0.5, np.abs(st - 0.5))
            _conf_report("stated |p - 0.5| for the token LOO predictions", rows, _loo(tk, rows)[0], np.abs(st - 0.5))


def _tie_bounds(x, rows):
    """Weighted AUC with ties counted as wins (upper bound) and as losses (lower bound)."""
    w, succ = arr(S.weights(rows)), np.array([r.succeeded for r in rows])
    gt = x[succ][:, None] > x[~succ][None, :]
    eq = x[succ][:, None] == x[~succ][None, :]
    ww = w[succ][:, None] * w[~succ][None, :]
    return (ww * gt).sum() / ww.sum(), (ww * (gt | eq)).sum() / ww.sum()


def _qwen_raw_log_odds(rows):
    """Qwen's token score as log odds without rounding (floating-point noise left in), for comparison only."""
    out = []
    for r in rows:
        lps = {"yes": [], "no": []}
        for tok, lp in qwen(r, F, "off")["answer_top_logprobs"]:
            if tok.strip().lower() in lps:
                lps[tok.strip().lower()].append(lp)
        out.append(float(np.logaddexp.reduce(lps["yes"]) - np.logaddexp.reduce(lps["no"])))
    return arr(out)


def precision():
    header("Numerical precision: bfloat16 logits round the yes/no log-odds; scores are rounded to 6 decimals so that equal scores tie")
    rng = np.random.default_rng(0)
    scores = {
        "Olmo L": (lambda rows: olmo_l(rows, F), lambda rows: arr([math.log(ppl(r, F, "yes_no", " no")) - math.log(ppl(r, F, "yes_no", " yes")) for r in rows])),
        "Qwen token log odds": (lambda rows: logit(qwen_scores(rows, F, "off")[0]), _qwen_raw_log_odds),
    }
    for name, (fn, raw_fn) in scores.items():
        raw = raw_fn(RECS)
        v = np.unique(fn(RECS))
        print(f"  {name} (with finding, thinking off): {len(v)} distinct values over 274 claims ({len(np.unique(raw))} before rounding to 6 decimals); smallest step {np.diff(v).min():g};"
              f" largest distance from a multiple of the smallest step {np.abs(raw - np.round(raw / np.diff(v).min()) * np.diff(v).min()).max():.1e}")
        step = np.diff(v).min()
        for subset in S.SUBSETS:
            rows = sel(subset)
            x = fn(rows)
            lo, hi = _tie_bounds(x, rows)
            spread = [auc(x + rng.uniform(-step / 2, step / 2, len(x)), rows) for _ in range(500)]
            print(f"    {subset:9} AUC {auc(x, rows):.3f} (with the noise left in {auc(raw_fn(rows), rows):.3f}); ties all broken against or for {lo:.3f} to {hi:.3f};"
                  f" spread within +/-{step / 2:g} (500 draws, seed 0): mean {np.mean(spread):.3f} [{np.percentile(spread, 2.5):.3f}, {np.percentile(spread, 97.5):.3f}]")
        rows = sel("highpower", "Secondary")
        print(f"    highpower Secondary AUC {auc(fn(rows), rows):.3f} (with the noise left in {auc(raw_fn(rows), rows):.3f})")
        if name == "Olmo L":
            # the leave-one-paper-out threshold can fall between copies of the same value when the noise is left in (the Olmo L line of Supplementary Section S13, human set)
            rows = sel("human")
            _conf_report("human, LOO threshold, exact ties", rows, *_loo(fn(rows), rows))
            _conf_report("human, LOO threshold, noise left in", rows, *_loo(raw_fn(rows), rows))


def _model_scores(rows) -> dict:
    """Every model in the set, with original finding and no reasoning: base models by L, chat models by the token and stated scores."""
    x = {}
    for label, m, _ in BASE_MODELS:
        x[f"base {label}"] = olmo_l(rows, F, m)
    for label, m, _ in CHAT_MODELS:
        x[f"chat {label}, token"], x[f"chat {label}, stated"], _ = chat_scores(rows, m)
    return x


def model_set():
    header("Model set, with original finding, no reasoning: AUC by set and data type")
    cols = [(s, dt) for s in S.SUBSETS for dt in S.DATA_TYPES]
    print("  " + " " * 34 + "  ".join(f"{s[:5]}/{dt[:3]}" for s, dt in cols))
    per_set = {c: _model_scores(sel(*c)) for c in cols}
    for k in per_set[cols[0]]:
        print(f"  {k:34}" + "  ".join(f"{auc(per_set[c][k], sel(*c)):9.3f}" for c in cols))
    for s, dt in cols:
        rows = sel(s, dt)
        if s != "all":
            print(f"  humans {s}/{dt}: " + "  ".join(f"{n} {auc(p, rows):.3f}" for n, p in zip(("repliCATS", "Markets"), humans(rows))))
    header("Model set: AUC with bootstrap 95% CI (all 274: resampling papers; human and high-power: resampling claims)")
    for s in S.SUBSETS:
        rows = sel(s)
        x = per_set[(s, "all")]
        b = boot_any(x, rows)
        print(f"  {s} (n={len(rows)})")
        for k, v in x.items():
            print(f"      {k:34} {auc(v, rows):.3f} {ci(b[k])}")
    header("Qwen3 30B-A3B Instruct against Qwen3.8 27B and the humans: paired differences in AUC [95% CI]")
    q30, q38 = "chat Qwen3 30B-A3B Instruct, token", "chat Qwen3.8 27B, token"
    for s in S.SUBSETS:
        rows = sel(s)
        x = {k: per_set[(s, "all")][k] for k in (q30, q38)}
        if s != "all":
            x["repliCATS"], x["Markets"] = humans(rows)
        b = boot_any(x, rows)
        print(f"  {s} (n={len(rows)})")
        for h in [k for k in x if k != q30]:
            print(f"      Qwen3 30B-A3B Instruct token - {h}: {auc(x[q30], rows) - auc(x[h], rows):+.3f} {ci_signed(b[q30] - b[h])}")
        print(f"      rank corr Qwen3 30B-A3B Instruct token with Qwen3.8 token {spearman(x[q30], x[q38]):.2f}")
    header("Model set: accuracy (base: offset fitted leaving each paper out; chat: its yes or no), yes answers, and rank correlation with the average human")
    for s in S.SUBSETS:
        rows = sel(s)
        h = sum(humans(rows)) / 2 if s != "all" else None
        print(f"  {s} (n={len(rows)}, always-failed baseline {1 - S.weighted_accuracy(S.weights(rows), outcomes(rows)):.3f})")
        for label, m, _ in BASE_MODELS:
            l = olmo_l(rows, F, m)
            print(f"      base {label:28} acc LOO {offset_accuracy(l, rows, 'loo'):.3f}" + (f"  rank corr with average human {spearman(l, h):.2f}" if h is not None else ""))
        for label, m, _ in CHAT_MODELS:
            tk, st, es = chat_scores(rows, m)
            line = f"      chat {label:28} acc {answer_accuracy(es, rows):.3f}  says yes {sum(e['answer'] == 'yes' for e in es):3}  median p(yes) {np.median(tk):.3f}  token near certain {np.mean((tk > 0.99) | (tk < 0.01)):.2f}"
            if h is not None:
                line += f"  rank corr with average human: token {spearman(tk, h):.2f}, stated {spearman(st, h):.2f}"
            print(line)
    header("Chat models: stated values (all 274) and distinct token scores")
    for label, m, _ in CHAT_MODELS:
        tk, st, es = chat_scores(RECS, m)
        print(f"  {label:28} {len(np.unique(tk))} distinct token scores; stated {collections.Counter(e['probability'] for e in es).most_common(5)}; rank corr stated with token {spearman(st, tk):.2f}")


def base_chat():
    header("Base against chat, same family and size, with original finding: AUC and paired difference (chat token minus base L) [95% CI]")
    for s in S.SUBSETS:
        rows = sel(s)
        print(f"  {s} (n={len(rows)})")
        for (bl, bm, _), (cl, cm, _) in PAIRS:
            tk, st, _ = chat_scores(rows, cm)
            x = {"base": olmo_l(rows, F, bm), "token": tk, "stated": st}
            b = boot_any(x, rows)
            print(f"      {bl:14} base L {auc(x['base'], rows):.3f}  chat token {auc(tk, rows):.3f}  stated {auc(st, rows):.3f}  token - base {auc(tk, rows) - auc(x['base'], rows):+.3f} {ci_signed(b['token'] - b['base'])}"
                  f"  rank corr base with token {spearman(x['base'], tk):.2f}")
    header("Scale: AUC on all 274 against parameters")
    for kind, models in (("base L", BASE_MODELS), ("chat token", CHAT_MODELS)):
        for label, m, n in sorted(models, key=lambda t: t[2]):
            x = olmo_l(RECS, F, m) if kind == "base L" else chat_scores(RECS, m)[0]
            print(f"  {kind:10} {label:24} {n:6.2f}B  {auc(x, RECS):.3f}")


def olmo_think():
    header("Olmo 3.1 32B Think (complete reasoning) against Olmo 3.1 32B Instruct, high-power set, with original finding")
    rows = sel("highpower")
    i_tk, i_st, i_es = chat_scores(rows, "allenai/Olmo-3.1-32B-Instruct")
    t_tk, t_st, t_es = chat_scores(rows, OLMO_THINK, "on")
    full, _ = complete_reasoning()
    rc, mk = humans(rows)
    x = {
        "Instruct token": i_tk, "Instruct stated": i_st, "Think token": t_tk, "Think stated": t_st, "Olmo 32B base L": olmo_l(rows, F),
        "Qwen3.8 no reasoning token": qwen_scores(rows, F, "off")[0], "Qwen3.8 no reasoning stated": qwen_scores(rows, F, "off")[1],
        "Qwen3.8 complete stated": arr([full[r.row]["probability"] for r in rows]),
        "Qwen3.8 complete token": arr([token(full[r.row]) for r in rows]), "repliCATS": rc, "Markets": mk,
    }
    b = boot_aucs(x, rows)
    for k, v in x.items():
        print(f"  {k:28} AUC {auc(v, rows):.3f} {ci(b[k])}")
    for a, h in (("Think token", "Instruct token"), ("Think stated", "Instruct stated"), ("Think stated", "Instruct token"), ("Think token", "Olmo 32B base L"),
                 ("Think stated", "repliCATS"), ("Think stated", "Markets"), ("Think token", "repliCATS"), ("Think token", "Markets"),
                 ("Instruct token", "repliCATS"), ("Instruct token", "Markets"), ("Think stated", "Qwen3.8 complete stated")):
        print(f"  {a} - {h}: {auc(x[a], rows) - auc(x[h], rows):+.3f} {ci_signed(b[a] - b[h])}")
    # interaction of score and reasoning: (stated - token with reasoning) - (stated - token without), per model and the mean of the two
    terms = {"Qwen3.8": ("Qwen3.8 complete stated", "Qwen3.8 complete token", "Qwen3.8 no reasoning stated", "Qwen3.8 no reasoning token"),
             "Olmo 3.1 32B": ("Think stated", "Think token", "Instruct stated", "Instruct token")}
    point = {m: (auc(x[rs], rows) - auc(x[rt], rows)) - (auc(x[os], rows) - auc(x[ot], rows)) for m, (rs, rt, os, ot) in terms.items()}
    boot = {m: (b[rs] - b[rt]) - (b[os] - b[ot]) for m, (rs, rt, os, ot) in terms.items()}
    point["both, pooled"], boot["both, pooled"] = np.mean(list(point.values())), np.mean(list(boot.values()), axis=0)
    for m in point:
        d = boot[m]
        print(f"  interaction of score and reasoning, {m}: {point[m]:+.3f} {ci_signed(d)}, two-sided p {2 * min(np.mean(d >= 0), np.mean(d <= 0)):.3f}")
    # the effects behind the interaction, on the same resamples: reasoning's effect on each score (simple effects), and the 2 x 2 main effects (each averaged over the other factor)
    pv = lambda d: 2 * min(np.mean(d >= 0), np.mean(d <= 0))
    for m, (rs, rt, os, ot) in terms.items():
        rows_ = [
            ("effect of reasoning on the stated score", auc(x[rs], rows) - auc(x[os], rows), b[rs] - b[os]),
            ("effect of reasoning on the token score", auc(x[rt], rows) - auc(x[ot], rows), b[rt] - b[ot]),
            ("main effect of reasoning (mean of the two scores)", (auc(x[rs], rows) + auc(x[rt], rows) - auc(x[os], rows) - auc(x[ot], rows)) / 2, (b[rs] + b[rt] - b[os] - b[ot]) / 2),
            ("main effect of score, stated minus token (mean with and without reasoning)", (auc(x[rs], rows) + auc(x[os], rows) - auc(x[rt], rows) - auc(x[ot], rows)) / 2, (b[rs] + b[os] - b[rt] - b[ot]) / 2),
        ]
        for name, v, d in rows_:
            print(f"  {m}: {name}: {v:+.3f} {ci_signed(d)}, two-sided p {pv(d):.3f}")
    # Qwen3.8 with reasoning capped at 4,096 tokens, against no reasoning, on the same resamples
    k4 = {"4k stated": arr([qwen(r, F, "on", 4096)["probability"] for r in rows]), "4k token": arr([token(qwen(r, F, "on", 4096)) for r in rows]),
          "off stated": x["Qwen3.8 no reasoning stated"], "off token": x["Qwen3.8 no reasoning token"]}
    b4 = boot_aucs(k4, rows)
    for sc in ("stated", "token"):
        d = b4[f"4k {sc}"] - b4[f"off {sc}"]
        print(f"  Qwen3.8: effect of 4k reasoning on the {sc} score: {auc(k4[f'4k {sc}'], rows) - auc(k4[f'off {sc}'], rows):+.3f} {ci_signed(d)}, two-sided p {pv(d):.3f}")
    d = (b4["4k stated"] - b4["4k token"]) - (b4["off stated"] - b4["off token"])
    i4 = (auc(k4["4k stated"], rows) - auc(k4["4k token"], rows)) - (auc(k4["off stated"], rows) - auc(k4["off token"], rows))
    print(f"  Qwen3.8: interaction of score and 4k reasoning: {i4:+.3f} {ci_signed(d)}, two-sided p {pv(d):.3f}")
    for label, tk, es in (("Instruct", i_tk, i_es), ("Think", t_tk, t_es)):
        print(f"  {label:8} says yes {sum(e['answer'] == 'yes' for e in es):2}  accuracy {answer_accuracy(es, rows):.3f}  token near certain {np.mean((tk > 0.99) | (tk < 0.01)):.2f}"
              f"  stated values {collections.Counter(e['probability'] for e in es).most_common(8)}")
    h = (rc + mk) / 2
    print(f"  rank correlation with average human: Think token {spearman(t_tk, h):.2f}, Think stated {spearman(t_st, h):.2f}, Instruct token {spearman(i_tk, h):.2f}; Think stated with Instruct token {spearman(t_st, i_tk):.2f}")
    ct = [e["completion_tokens"] for e in t_es]
    print(f"  Think: finish {collections.Counter(e['finish_reason'] for e in t_es)}; generated tokens median {int(np.median(ct))}, mean {np.mean(ct):.0f}, range {min(ct)} to {max(ct)}, total {sum(ct)}; answer cap {t_es[0]['max_tokens']}")
    finite = collections.Counter(sum(lp > -1000 for _, lp in e["answer_top_logprobs"]) for e in t_es)
    print(f"  Think: finite log-probabilities among the top 20 at the answer word {dict(finite)}; answers beginning with a blank line {sum(e['content'].startswith(chr(10)) for e in t_es)}")
    print(f"  Think traces: {_trace_counts(t_es)}")
    last = [l for l in (R / "run_new_models.log").read_text().splitlines() if "/57 after" in l][-1]
    print(f"  run time (run_new_models.log, 12 requests at a time): {last.strip()}")


def _slope(s, a):
    """OLS slope of a on s."""
    s = s - s.mean()
    return float((s * (a - a.mean())).sum() / (s**2).sum())


def _t_sf(t, df):
    """Upper tail of Student's t beyond |t|, by numerical integration of the density."""
    z = np.linspace(abs(t), abs(t) + 2000, 2_000_001)
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return float(np.trapezoid(c * (1 + z**2 / df) ** (-(df + 1) / 2), z))


def _boot_p(v) -> float:
    """Two-sided bootstrap p-value for a statistic being zero: twice the smaller tail share."""
    return min(1.0, 2 * min(float((v <= 0).mean()), float((v >= 0).mean())))


def _ols_p(s, a) -> float:
    """Two-sided p of the OLS slope, t test with n - 2 degrees of freedom."""
    n, beta = len(s), _slope(s, a)
    resid = a - a.mean() - beta * (s - s.mean())
    se = math.sqrt((resid**2).sum() / (n - 2) / ((s - s.mean()) ** 2).sum())
    return 2 * _t_sf(beta / se, n - 2)


def _spearman_perm_p(s, a) -> float:
    """Two-sided permutation p of Spearman's rho: every ordering for up to 8 models, otherwise 200,000 random orderings (seed 0)."""
    rs, ra = ranks(s), ranks(a)
    r0 = abs(np.corrcoef(rs, ra)[0, 1])
    n = len(s)
    if n <= 8:
        perms = (np.array(p) for p in itertools.permutations(range(n)))
    else:
        rng = np.random.default_rng(0)
        perms = (rng.permutation(n) for _ in range(200_000))
    hits = tot = 0
    for p in perms:
        tot += 1
        hits += abs(np.corrcoef(rs, ra[p])[0, 1]) >= r0 - 1e-12
    return hits / tot


def size():
    header("Model size and AUC, high-power set, with original finding, no reasoning (base L, chat token score)")
    rows = sel("highpower")
    x, n_params = {}, {}
    for label, m, n in BASE_MODELS:
        x[f"base {label}"], n_params[f"base {label}"] = olmo_l(rows, F, m), n
    for label, m, n in CHAT_MODELS:
        x[f"chat {label}"], n_params[f"chat {label}"] = chat_scores(rows, m)[0], n
    b = boot_aucs(x, rows)
    pt = {k: auc(v, rows) for k, v in x.items()}
    for k in sorted(x, key=lambda k: (k[:4], n_params[k])):
        print(f"  {k:30} {n_params[k]:6.2f}B  AUC {pt[k]:.3f} {ci(b[k])}")
    for name, keys in (("base L, 7 models", [k for k in x if k.startswith("base")]), ("chat token, 8 models", [k for k in x if k.startswith("chat")]), ("all 15", list(x))):
        s = arr([n_params[k] for k in keys])
        a = arr([pt[k] for k in keys])
        bt = np.column_stack([b[k] for k in keys])
        print(f"  {name}")
        for scale, sv in (("log10 parameters", np.log10(s)), ("parameters / 10B", s / 10)):
            bs = arr([_slope(sv, r) for r in bt])
            print(f"      linear on {scale:16}: slope {_slope(sv, a):+.3f} {ci_signed(bs)}  bootstrap p {_boot_p(bs):.4f}  model-level t({len(s) - 2}) p {_ols_p(sv, a):.4f}")
        br = arr([spearman(s, r) for r in bt])
        print(f"      ranked: Spearman rho {spearman(s, a):+.2f} {ci_signed(br)}  bootstrap p {_boot_p(br):.4f}  permutation p {_spearman_perm_p(s, a):.4f}")






INDIVIDUAL_DRAWS = 5000
INDIVIDUAL_SOURCES = {"repliCATS_R1": "repliCATS individual, Round 1", "RM_survey": "Replication Markets individual, survey"}


def individual_estimates() -> dict[str, dict[int, np.ndarray]]:
    """Each source's individual estimates (as probabilities) for each data-file row, from data/individual_estimates.csv."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(S.ROOT / "data/individual_estimates.csv") as f:
        next(f)
        for line in f:
            row, _, source, _, est = line.rstrip("\n").split(",")
            out[source][int(row)].append(float(est) / 100)
    return {s: {r: arr(v) for r, v in d.items()} for s, d in out.items()}


def _padded(rows, est):
    """Estimates of the rows as one array, padded with NaN, and the number of raters of each row."""
    k = np.array([len(est[r.row]) for r in rows])
    e = np.full((len(rows), k.max()), np.nan)
    for j, r in enumerate(rows):
        e[j, :k[j]] = est[r.row]
    return e, k


def _draw_raters(rng, e, k, i):
    """One rater drawn at random for each claim index in i."""
    return e[i, (rng.random(len(i)) * k[i]).astype(int)]


def _p_le_zero(d) -> str:
    p = np.mean(d <= 0)
    return f"{p:.4f}" if p > 0 else f"< {1 / len(d):.4f}"


def typical_fixed(kept, est) -> np.ndarray:
    """AUC of a typical individual on the claims as they are: INDIVIDUAL_DRAWS draws of one rater per claim (seed 0)."""
    e, k = _padded(kept, est)
    y, ones, every = outcomes(kept), np.ones(len(kept)), np.arange(len(kept))
    rng = np.random.default_rng(0)
    return arr([S.weighted_auc(_draw_raters(rng, e, k, every), y, ones) for _ in range(INDIVIDUAL_DRAWS)])


def typical_single(kept, est, xm: dict) -> tuple[np.ndarray, dict]:
    """Claims and raters resampled: in each of BOOT resamples of claims (seed 0), the AUC of one rater drawn per resampled claim, and the AUC of each score in xm on the same resample."""
    e, k = _padded(kept, est)
    y, ones = outcomes(kept), np.ones(len(kept))
    rng = np.random.default_rng(0)
    bi, bm = [], {m: [] for m in xm}
    for _ in range(BOOT):
        i = _claim_resample(rng, y)
        bi.append(S.weighted_auc(_draw_raters(rng, e, k, i), y[i], ones))
        for m, v in xm.items():
            bm[m].append(S.weighted_auc(v[i], y[i], ones))
    return arr(bi), {m: arr(v) for m, v in bm.items()}


def typical_averaged(kept, est, xm: dict) -> tuple[np.ndarray, dict]:
    """The typical individual averaged over raters: in each of BOOT resamples of claims (seed 0), the mean AUC over PANEL_INNER_DRAWS draws of one rater per resampled claim, and the AUC of each score in xm on the same resample."""
    e, k = _padded(kept, est)
    y, ones = outcomes(kept), np.ones(len(kept))
    rng = np.random.default_rng(0)
    avg, bm = [], {m: [] for m in xm}
    for _ in range(BOOT):
        i = _claim_resample(rng, y)
        draws = e[i[None, :], (rng.random((PANEL_INNER_DRAWS, len(i))) * k[i]).astype(int)]
        avg.append(auc_rows(draws, y[i]).mean())
        for m, v in xm.items():
            bm[m].append(S.weighted_auc(v[i], y[i], ones))
    return arr(avg), {m: arr(v) for m, v in bm.items()}


def individuals():
    """Qwen3.8 27B and Qwen3 30B-A3B Instruct (token score, with original finding) against a typical individual human and against the group aggregates. Typical individual: one rater drawn at random per claim. Paired bootstrap: 10,000 resamples of claims (redrawn if one outcome is absent), with one rater drawn per resampled claim, the same draws for all models."""
    est = individual_estimates()
    models = {"Qwen3.8 27B": lambda rows: chat_scores(rows, QWEN)[0], "Qwen3 30B-A3B Instruct": lambda rows: chat_scores(rows, "Qwen/Qwen3-30B-A3B-Instruct-2507")[0]}
    for s in ("human", "highpower"):
        for dt in S.DATA_TYPES:
            rows = sel(s, dt)
            assert (arr(S.weights(rows)) == 1).all()
            x = {m: f(rows) for m, f in models.items()}
            header(f"{s}/{dt} (n={len(rows)}): models against the group aggregates; difference model minus aggregate, mean of the paired bootstrap [95% CI], one-sided p (share of differences at or below 0)")
            y = outcomes(rows)
            agg = dict(zip(("repliCATS group", "Markets market"), humans(rows)))
            rng = np.random.default_rng(0)
            b = {k: [] for k in [*x, *agg]}
            for _ in range(BOOT):
                i = _claim_resample(rng, y)
                for k, v in {**x, **agg}.items():
                    b[k].append(S.weighted_auc(v[i], y[i], np.ones(len(i))))
            b = {k: arr(v) for k, v in b.items()}
            for k, v in {**x, **agg}.items():
                print(f"      {k:40} AUC {auc(v, rows):.3f} {ci(b[k])}")
            for m in x:
                for h in agg:
                    d = b[m] - b[h]
                    print(f"      {m} - {h}: {auc(x[m], rows) - auc(agg[h], rows):+.3f}, bootstrap mean {d.mean():+.3f} {ci_signed(d)}  p {_p_le_zero(d)}")
            for src, name in INDIVIDUAL_SOURCES.items():
                kept = [r for r in rows if r.row in est[src]]
                k = _padded(kept, est[src])[1]
                header(f"{s}/{dt}: models against a typical {name} (claims with estimates: {len(kept)} of {len(rows)}; raters per claim {k.min()} to {k.max()}, median {np.median(k):g})")
                typical = typical_fixed(kept, est[src])
                print(f"      typical individual, claims fixed ({INDIVIDUAL_DRAWS} rater draws): AUC mean {typical.mean():.3f} {ci(typical)}")
                xm = {m: f(kept) for m, f in models.items()}
                bi, bm = typical_single(kept, est[src], xm)
                print(f"      typical individual, claims and raters resampled: AUC mean {bi.mean():.3f} {ci(bi)}")
                for m, v in xm.items():
                    d = bm[m] - bi
                    print(f"      {m:24} AUC {auc(v, kept):.3f} {ci(bm[m])}; minus typical individual: {auc(v, kept) - typical.mean():+.3f}, bootstrap mean {d.mean():+.3f} {ci_signed(d)}  p {_p_le_zero(d)}")


# ---------------------------------------------------------------- prompt ablation (Supplementary Section S12)

ABLATION_MODELS = {"Qwen3.8 27B": QWEN, "Qwen3 30B-A3B Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507"}
# condition: score.py template
ABLATION_CONDITIONS = {"full": F, "no finding": C, "no abstract": "no_abstract", "claim alone": "claim_alone"}
ABLATION_SETS = (("all", "all"), ("human", "all"), ("highpower", "all"), ("human", "New"), ("human", "Secondary"), ("highpower", "New"), ("highpower", "Secondary"))
ABLATION_FILE_COLUMNS = (
    "row", "claim_id", "paper", "data_type", "human_set", "highpower_set", "succeeded", "model", "condition", "template",
    "token_probability", "answer", "stated_probability",
)


def ablation():
    """Qwen3.8 27B and Qwen3 30B-A3B Instruct, reasoning off, token score, with four templates: full (abstract, claim, original finding), no finding (claim only), no abstract, and the claim alone. AUC with 95% intervals, paired differences (full minus each ablated condition; each condition minus the human aggregates on the human and high-power sets) with one-sided p, and the interaction of original finding by abstract, (full minus no finding) minus (no abstract minus claim alone), with two-sided p, from 10,000 resamples per set (seed 0; claims resampled on the human and high-power sets, papers on all 274; redrawn if one outcome is absent), the same resamples for every predictor in a set. Writes the per-claim scores to results/ablation_scores.csv."""
    es = {(m, c): {r.row: qwen(r, t, "off", 0, mid) for r in RECS} for m, mid in ABLATION_MODELS.items() for c, t in ABLATION_CONDITIONS.items()}
    with open(R / "ablation_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ABLATION_FILE_COLUMNS)
        for (m, c), d in es.items():
            for r in RECS:
                e = d[r.row]
                w.writerow([r.row, r.claim_id, r.paper, r.data_type, int(r.human_set), int(r.highpower), int(r.succeeded), m, c, ABLATION_CONDITIONS[c],
                            f"{token(e):.6g}", e["answer"], e["probability"] / 100])
    header("Prompt ablation: answers (all 274)")
    for (m, c), d in es.items():
        x = arr([token(e) for e in d.values()])
        yes = sum(e["answer"] == "yes" for e in d.values())
        near = np.mean((x > 0.99) | (x < 0.01))
        prompts = len({S.qwen_prompt(r, ABLATION_CONDITIONS[c]) for r in RECS})
        print(f"      {m:24} {c:12} says yes {yes:3} of 274; median p(yes) {np.median(x):.3f}; near certain {near:.0%}; distinct prompts {prompts}; finish reasons {dict(collections.Counter(e['finish_reason'] for e in d.values()))}")
    for s, dt in ABLATION_SETS:
        rows = sel(s, dt)
        y, wt = outcomes(rows), arr(S.weights(rows))
        x = {(m, c): arr([token(es[(m, c)][r.row]) for r in rows]) for m in ABLATION_MODELS for c in ABLATION_CONDITIONS}
        hum = dict(zip(("repliCATS", "Markets"), humans(rows))) if s != "all" else {}
        rng = np.random.default_rng(0)
        if s == "all":
            by_paper = collections.defaultdict(list)
            for j, r in enumerate(rows):
                by_paper[r.paper_id].append(j)
            groups = list(by_paper.values())
            draw = lambda: _paper_resample(rng, groups, y)
        else:
            assert (wt == 1).all()
            draw = lambda: _claim_resample(rng, y)
        b = {k: [] for k in [*x, *hum]}
        for _ in range(BOOT):
            i = draw()
            for k, v in {**x, **hum}.items():
                b[k].append(S.weighted_auc(v[i], y[i], wt[i]))
        b = {k: arr(v) for k, v in b.items()}
        point = {k: auc(v, rows) for k, v in {**x, **hum}.items()}
        header(f"Prompt ablation, {s}/{dt} (n={len(rows)}, {'papers' if s == 'all' else 'claims'} resampled): AUC [95% CI]; differences: point [95% CI], one-sided p (share of resampled differences at or below 0)")
        for k in b:
            print(f"      {' '.join(k) if isinstance(k, tuple) else k:40} AUC {point[k]:.3f} {ci(b[k])}")
        for m in ABLATION_MODELS:
            for c in list(ABLATION_CONDITIONS)[1:]:
                d = b[(m, "full")] - b[(m, c)]
                print(f"      {m} full - {c}: {point[(m, 'full')] - point[(m, c)]:+.3f} {ci_signed(d)}  p {_p_le_zero(d)}  (bootstrap mean {d.mean():+.3f}; rank correlation {spearman(x[(m, 'full')], x[(m, c)]):.2f})")
        for m in ABLATION_MODELS:
            with_abstract = b[(m, "full")] - b[(m, "no finding")]
            without = b[(m, "no abstract")] - b[(m, "claim alone")]
            inter = with_abstract - without
            p_without, p_inter = point[(m, "no abstract")] - point[(m, "claim alone")], (point[(m, "full")] - point[(m, "no finding")]) - (point[(m, "no abstract")] - point[(m, "claim alone")])
            print(f"      {m} finding by abstract: full - no finding two-sided p {_boot_p(with_abstract):.4f}; no abstract - claim alone {p_without:+.3f} {ci_signed(without)}  two-sided p {_boot_p(without):.4f}; "
                  f"interaction {p_inter:+.3f} {ci_signed(inter)}  two-sided p {_boot_p(inter):.4f}")
        if dt == "all":
            for m in ABLATION_MODELS:
                for c in ABLATION_CONDITIONS:
                    for h in hum:
                        d = b[(m, c)] - b[h]
                        print(f"      {m} {c} - {h}: {point[(m, c)] - point[h]:+.3f} {ci_signed(d)}  p {_p_le_zero(d)}  (bootstrap mean {d.mean():+.3f})")


# ---------------------------------------------------------------- models against individual forecasters: averaged and per-forecaster tests (Supplementary Sections S9 and S10)

PANEL_MODELS = {"Qwen3.8 27B": QWEN, "Qwen3 30B-A3B Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507"}
PANEL_SETS = (("highpower", "all"), ("human", "all"), ("highpower", "New"), ("highpower", "Secondary"), ("human", "New"), ("human", "Secondary"))
PANEL_INNER_DRAWS = 200
PANEL_MIN_CLAIMS = 5
PANEL_MIN_FORECASTERS_SPLIT = 20
PANEL_FILE_COLUMNS = ("team", "forecaster_id", "set", "model", "n_claims", "n_successes", "n_failures", "forecaster_auc", "model_auc")


def auc_rows(x, y) -> np.ndarray:
    """AUC of each row of x (all weights 1, ties counting half) against outcomes y, for many rows at once. Values are replaced by integer codes and each row is offset so that rows do not overlap, so one sorted search counts, for every success, the failures below and equal to it in its own row."""
    y = np.asarray(y, dtype=bool)
    codes = np.unique(x, return_inverse=True)[1].reshape(x.shape)
    m, n0, n1 = x.shape[0], int((~y).sum()), int(y.sum())
    z = codes + (np.arange(m) * (codes.max() + 1))[:, None]
    neg = np.sort(z[:, ~y], axis=None)
    pos = z[:, y]
    before = (np.arange(m) * n0)[:, None]
    lt = np.searchsorted(neg, pos, "left") - before
    le = np.searchsorted(neg, pos, "right") - before
    return (lt + 0.5 * (le - lt)).sum(axis=1) / (n0 * n1)


def _forecasters(rows, raw):
    """The survey forecasters with at least PANEL_MIN_CLAIMS rated claims among rows, including a success and a failure: (rater, indices of their claims in rows, their estimates)."""
    by = collections.defaultdict(list)
    for j, r in enumerate(rows):
        for rater, v in raw.get(r.row, ()):
            by[rater].append((j, v))
    y = outcomes(rows)
    out = []
    for rater, jv in sorted(by.items()):
        j = np.array([a for a, _ in jv])
        if len(j) >= PANEL_MIN_CLAIMS and y[j].any() and not y[j].all():
            out.append((rater, j, arr([v for _, v in jv])))
    return out


def _win(x, ys):
    """Pairwise credit of each success over each failure (1 higher, 0.5 tied, 0 lower)."""
    xp, xn = x[ys], x[~ys]
    return (xp[:, None] > xn[None, :]) + 0.5 * (xp[:, None] == xn[None, :])


def survey_estimates() -> dict[int, list]:
    """The Replication Markets survey estimates of each data-file row, as (rater, probability), from data/individual_estimates.csv."""
    raw = collections.defaultdict(list)
    with open(S.ROOT / "data/individual_estimates.csv") as f:
        next(f)
        for line in f:
            row, _, source, rater, v = line.rstrip("\n").split(",")
            if source == "RM_survey":
                raw[int(row)].append((rater, float(v) / 100))
    return raw


def forecaster_test(rows, fc, x) -> dict:
    """A model's scores x (one per row) against each forecaster of fc (from _forecasters) on that forecaster's claims. Returns each forecaster's AUC and the model's AUC on the same claims, each forecaster's number of success-failure pairs, the point values (share of forecasters below the model, mean difference with equal weights, mean difference weighted by pairs), and the two-way bootstrap of the three (BOOT resamples, seed 0, the share as its excess over 0.5: claims and forecasters drawn independently with replacement; a drawn forecaster whose resampled claims lack an outcome is skipped)."""
    y = outcomes(rows)
    ys = [y[j] for _, j, _ in fc]
    wf = [_win(v, yj) for (_, j, v), yj in zip(fc, ys)]
    sidx = [j[yj] for (_, j, _), yj in zip(fc, ys)]
    fidx = [j[~yj] for (_, j, _), yj in zip(fc, ys)]
    pairs = arr([len(a) * len(b) for a, b in zip(sidx, fidx)])
    fa = arr([w.mean() for w in wf])
    wm = [_win(x[j], yj) for (_, j, _), yj in zip(fc, ys)]
    ma = arr([w.mean() for w in wm])
    d = ma - fa
    point = ((d > 0) + 0.5 * (d == 0)).mean(), d.mean(), (pairs * d).sum() / pairs.sum()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(BOOT):
        while True:
            c = np.bincount(rng.integers(0, len(rows), len(rows)), minlength=len(rows)).astype(float)
            num_m, num_f, pw = [], [], []
            for f_ in rng.integers(0, len(fc), len(fc)):
                cs, cf = c[sidx[f_]], c[fidx[f_]]
                if cs.sum() == 0 or cf.sum() == 0:
                    continue
                den = cs.sum() * cf.sum()
                num_m.append(cs @ wm[f_] @ cf)
                num_f.append(cs @ wf[f_] @ cf)
                pw.append(den)
            if pw:
                break
        num_m, num_f, pw = arr(num_m), arr(num_f), arr(pw)
        db = (num_m - num_f) / pw
        bs.append(((num_m > num_f) + 0.5 * (num_m == num_f)).mean() - 0.5)
        bs.append(db.mean())
        bs.append((pw * db).sum() / pw.sum())
    return {"forecaster_auc": fa, "model_auc": ma, "pairs": pairs, "point": point, "boot": arr(bs).reshape(-1, 3)}


def individual_tests():
    """Qwen3.8 27B and Qwen3 30B-A3B Instruct (token score, with original finding, no reasoning) against individual forecasters. (1) The typical individual averaged over raters: point difference as in the individuals section (5,000 rater draws on the original claims, seed 0); 10,000 claim resamples (seed 0, redrawn if one outcome is absent), in each the typical individual's AUC is the mean over 200 draws of one rater per resampled claim. The single-draw test of the individuals section is recomputed alongside. (2) The model against each Replication Markets survey forecaster with at least 5 rated claims in the set, including a success and a failure, on that forecaster's claims; two-way bootstrap (10,000 resamples, seed 0: claims and forecasters drawn independently with replacement; a drawn forecaster whose resampled claims lack an outcome is skipped). Writes results/forecaster_comparisons.csv."""
    est = individual_estimates()
    rng = np.random.default_rng(1)
    for n in (7, 40, 135):
        yt = rng.random(n) < 0.5
        yt[:2] = (True, False)
        xt = rng.integers(0, 6, (5, n)) / 5
        assert np.allclose(auc_rows(xt, yt), [S.weighted_auc(r, yt, np.ones(n)) for r in xt])
    for s, dt in PANEL_SETS:
        rows = sel(s, dt)
        assert (arr(S.weights(rows)) == 1).all()
        for src, name in INDIVIDUAL_SOURCES.items():
            kept = [r for r in rows if r.row in est[src]]
            xm = {m: chat_scores(kept, mid)[0] for m, mid in PANEL_MODELS.items()}
            pm = {m: auc(v, kept) for m, v in xm.items()}
            header(f"{s}/{dt}: models against a typical {name}, single draw and averaged over raters (claims with estimates: {len(kept)} of {len(rows)})")
            typical = typical_fixed(kept, est[src])
            print(f"      typical individual, claims fixed ({INDIVIDUAL_DRAWS} rater draws): AUC mean {typical.mean():.3f} {ci(typical)}")
            # single draw per resample: the procedure of the individuals section, same seed
            bi, bm = typical_single(kept, est[src], xm)
            single = {m: bm[m] - bi for m in xm}
            # averaged over PANEL_INNER_DRAWS raters per resample
            avg, bm = typical_averaged(kept, est[src], xm)
            averaged = {m: bm[m] - avg for m in xm}
            print(f"      typical individual averaged over {PANEL_INNER_DRAWS} raters per resample: AUC mean {np.mean(avg):.3f} {ci(avg)}")
            for m in xm:
                print(f"      {m:24} AUC {pm[m]:.3f}; minus typical individual {pm[m] - typical.mean():+.3f}; single draw {ci_signed(single[m])} p {_p_le_zero(single[m])}; averaged {ci_signed(averaged[m])} p {_p_le_zero(averaged[m])} (bootstrap mean {averaged[m].mean():+.3f})")
    raw = survey_estimates()
    lines = []
    for s, dt in PANEL_SETS:
        rows = [r for r in sel(s, dt) if r.row in est["RM_survey"]]
        y = outcomes(rows)
        fc = _forecasters(rows, raw)
        if dt != "all" and len(fc) < PANEL_MIN_FORECASTERS_SPLIT:
            header(f"{s}/{dt}: {len(fc)} qualifying survey forecasters, fewer than {PANEL_MIN_FORECASTERS_SPLIT}; not tested")
            continue
        xm = {m: chat_scores(rows, mid)[0] for m, mid in PANEL_MODELS.items()}
        header(f"{s}/{dt}: models against each Replication Markets survey forecaster with at least {PANEL_MIN_CLAIMS} claims and both outcomes ({len(fc)} forecasters, {len(rows)} claims)")
        ys = [y[j] for _, j, _ in fc]
        pairs = arr([int(yj.sum()) * int((~yj).sum()) for yj in ys])
        print(f"      claims per forecaster {min(len(j) for _, j, _ in fc)} to {max(len(j) for _, j, _ in fc)}, median {np.median([len(j) for _, j, _ in fc]):g}; pairs {pairs.min():g} to {pairs.max():g}, median {np.median(pairs):g}")
        for m, x in xm.items():
            t = forecaster_test(rows, fc, x)
            fa, ma, point, bs = t["forecaster_auc"], t["model_auc"], t["point"], t["boot"]
            d = ma - fa
            for (rater, j, _), yj, a, b in zip(fc, ys, fa, ma):
                lines.append(["Replication Markets survey", rater, f"{s}/{dt}", m, len(j), int(yj.sum()), int((~yj).sum()), f"{a:.6f}", f"{b:.6f}"])
            print(f"      {m}: share of forecasters below the model {point[0]:.3f} (minus 0.5: {point[0] - 0.5:+.3f} {ci_signed(bs[:, 0])} p {_p_le_zero(bs[:, 0])}); "
                  f"mean difference, equal weights {point[1]:+.3f} {ci_signed(bs[:, 1])} p {_p_le_zero(bs[:, 1])}; weighted by pairs {point[2]:+.3f} {ci_signed(bs[:, 2])} p {_p_le_zero(bs[:, 2])}; "
                  f"forecaster AUC mean {fa.mean():.3f}, model AUC mean {ma.mean():.3f}; ties {int((d == 0).sum())}")
    with open(R / "forecaster_comparisons.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PANEL_FILE_COLUMNS)
        w.writerows(lines)


# ---------------------------------------------------------------- simulated ceiling (no cached outputs; a simulation)

SIM_Z = stats.norm.isf(0.025)  # two-sided alpha = .05


def _sim_auc(score, y):
    r = stats.rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _sim_power(d, n_per_group):
    se = np.sqrt(2.0 / n_per_group)
    return stats.norm.sf(SIM_Z - d / se)  # significant in the positive direction


def _simulate(pi, mean_d, tau, k=1, n_studies=400_000, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.random(n_studies) < pi
    delta = np.where(real, np.abs(rng.normal(0, mean_d * np.sqrt(np.pi / 2), n_studies)), 0.0)
    n_total = np.clip(np.exp(rng.normal(np.log(150), 1.0, n_studies)), 20, 5000)
    n_orig = n_total / 2
    se_orig = np.sqrt(2.0 / n_orig)
    # flexible analysis: the best of k independent analyses is reported (k = 1: none)
    d_obs = rng.normal(delta[:, None], se_orig[:, None], (n_studies, k)).max(axis=1)
    published = d_obs / se_orig > SIM_Z  # publication bias: only significant positive results
    delta, real, d_obs = delta[published], real[published], d_obs[published]
    # replication sample: 80% power to detect half the published effect
    half = d_obs / 2
    n_rep = 2 * ((SIM_Z + stats.norm.isf(0.2)) / half) ** 2
    n_rep = np.clip(n_rep, 10, 1e6)
    shift = rng.normal(0, tau, len(delta)) if tau > 0 else np.zeros(len(delta))
    delta_rep = delta + shift
    d_rep = rng.normal(delta_rep, np.sqrt(2.0 / n_rep))
    success = (d_rep / np.sqrt(2.0 / n_rep) > SIM_Z).astype(int)
    if tau > 0:
        grid = np.linspace(-4, 4, 41)
        w = stats.norm.pdf(grid); w /= w.sum()
        ceiling_score = sum(wi * _sim_power(delta + g * tau, n_rep) for g, wi in zip(grid, w))
    else:
        ceiling_score = _sim_power(delta, n_rep)
    return dict(n=len(success), success_rate=success.mean(), real_share=real.mean(), auc_ceiling=_sim_auc(ceiling_score, success))


def ceiling():
    """Simulated ceiling on predicting replication outcomes. Original two-group studies are run on true effects (a share pi of them real), and only significant results in the positive direction are published (publication bias). Each published finding is replicated with a sample giving exactly 80% power to detect half the published effect (the boundary of SCORE's high-power definition). The replication population may differ from the original: its true effect is the original true effect plus N(0, tau). The idealized forecaster knows the original true effect and the replication's sample size but not the population shift; its score is the replication's power averaged over the shift. The AUC of this score against the simulated replication outcome is the ceiling. 108 settings, 200,000 original studies each, seed 0."""
    header("simulated ceiling")
    res = []
    for pi, mean_d, tau, k in itertools.product([0.1, 0.2, 0.3, 0.5], [0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [1, 3, 5]):
        r = _simulate(pi, mean_d, tau, k, n_studies=200_000)
        res.append((pi, mean_d, tau, k, r))
    print("Settings whose replication success rate is 35% to 55% (SCORE high-power set: 44%):")
    sel_ = [x for x in res if 0.35 <= x[4]["success_rate"] <= 0.55]
    for pi, mean_d, tau, k, r in sel_:
        print(f"pi={pi:.1f} mean_d={mean_d:.1f} tau={tau:.1f} k={k}  real share={r['real_share']:.2f}  success={r['success_rate']:.2f}  ceiling={r['auc_ceiling']:.3f}")
    for tau in [0.0, 0.1, 0.2]:
        v = np.array([x[4]["auc_ceiling"] for x in sel_ if x[2] == tau])
        print("tau=%.1f n=%d median %.3f min %.3f max %.3f" % (tau, len(v), np.median(v), v.min(), v.max()))
    v = np.array([x[4]["auc_ceiling"] for x in sel_])
    print("all n=%d median %.3f min %.3f max %.3f" % (len(v), np.median(v), v.min(), v.max()))
    pub = [x[4]["n"] for x in sel_]
    print(f"published studies per kept setting (of 200,000): {min(pub):,} to {max(pub):,}")


def _consistency_runs():
    """(label, rows, responses) for every chat run whose stated probability is reported: Qwen3.8 27B and Qwen3 30B-A3B Instruct without reasoning (four templates), Qwen3.8 27B with 4k reasoning (two templates) and with complete reasoning (high-power), the other chat models without reasoning, and Olmo 3.1 32B Think (all 274, with the one forced answer)."""
    runs = []
    for m, mid in ABLATION_MODELS.items():
        for c, t in ABLATION_CONDITIONS.items():
            runs.append((f"{m}, no reasoning, {c}", RECS, [qwen(r, t, "off", 0, mid) for r in RECS]))
    for c, t in (("full", F), ("no finding", C)):
        runs.append((f"Qwen3.8 27B, 4k reasoning, {c}", RECS, [qwen(r, t, "on", 4096) for r in RECS]))
    full, _ = complete_reasoning()
    hp = sel("highpower")
    runs.append(("Qwen3.8 27B, complete reasoning, full", hp, [full[r.row] for r in hp]))
    for label, m, _ in CHAT_MODELS:
        if m not in ABLATION_MODELS.values():
            runs.append((f"{label}, no reasoning, full", RECS, [qwen(r, F, "off", 0, m) for r in RECS]))
    # with row 106's forced answer (Section 6): its reasoning reached the 60,000-token limit with no answer
    runs.append(("Olmo 3.1 32B Think, complete reasoning, full", RECS, [QC[S.qwen_key(OLMO_THINK, S.qwen_prompt(r, F), "on", 0)] for r in RECS]))
    return runs


def consistency():
    """Agreement between the yes or no and the stated probability of success. A response is inconsistent when it says yes with a probability below 50, or no with a probability above 50 (50 agrees with either). The stated AUC is shown as scored (the number as the probability of success) and with each inconsistent number reread as confidence in the answer given (100 minus it), on all claims the run covers and on the high-power set."""
    header("Consistency of the yes or no with the stated probability")
    hp_rows = {r.row for r in sel("highpower")}
    for label, rows, es in _consistency_runs():
        ok = [(r, e) for r, e in zip(rows, es) if e["answer"] is not None and e["probability"] is not None]
        yes_low = [(r, e) for r, e in ok if e["answer"] == "yes" and e["probability"] < 50]
        no_high = [(r, e) for r, e in ok if e["answer"] == "no" and e["probability"] > 50]
        bad = {r.row for r, _ in yes_low + no_high}
        fifty = sum(e["probability"] == 50 for _, e in ok)
        line = f"  {label:52} n={len(ok):3}  inconsistent {len(bad):2} ({len(bad) / len(ok):.1%}): yes below 50 {len(yes_low):2}, no above 50 {len(no_high):2}; exactly 50 {fifty:2}"
        if bad:
            line += "  values " + str(sorted(collections.Counter((e["answer"], e["probability"]) for _, e in yes_low + no_high).items()))
        print(line)
        for name, keep in (("all", lambda r: True), ("high-power", lambda r: r.row in hp_rows)):
            sub = [(r, e) for r, e in ok if keep(r)]
            if not bad or (name == "all" and len(sub) == len(hp_rows)):
                continue
            rs = [r for r, _ in sub]
            st = arr([e["probability"] / 100 for _, e in sub])
            re_ = arr([(100 - e["probability"]) / 100 if r.row in bad else e["probability"] / 100 for r, e in sub])
            print(f"      {name:10} n={len(rs):3} inconsistent {sum(r.row in bad for r in rs):2}  stated AUC as scored {auc(st, rs):.3f}, reread {auc(re_, rs):.3f}")


def plausibility():
    """Supplementary Section S21: the made-up papers of plausibility_demo.py (implausible or obvious claims, each with an original p-value of .001 and of .030), Qwen3.8 27B with reasoning off."""
    header("Plausibility demonstration (Supplementary Section S21): Qwen3.8 27B, reasoning off, with original finding")
    cache = S.load_cache(R / "plausibility_demo.jsonl")
    recs = PD.records()
    responses = {(PD.TEMPLATE, "off", rec.row): cache[S.qwen_key(QWEN, S.qwen_prompt(rec, PD.TEMPLATE), "off")] for rec in recs}
    lines = PD.table(responses)
    PD.report(lines)
    print("\nMean log odds of yes (token score):")
    for kind in PD.KINDS:
        m = [float(np.mean(logit([ln["token_probability"] for ln in lines if ln["kind"] == kind and ln["p_level"] == level]))) for level in PD.P_LEVELS]
        print(f"  {kind:<12} p=.001 {m[0]:+.3f}   p=.030 {m[1]:+.3f}   difference {m[0] - m[1]:+.3f}")
    for level in PD.P_LEVELS:
        m = [float(np.mean(logit([ln["token_probability"] for ln in lines if ln["kind"] == kind and ln["p_level"] == level]))) for kind in PD.KINDS]
        print(f"  obvious minus implausible at {level}: {m[1] - m[0]:+.3f}")
    print("\nInteraction of kind with p-value (each paper's change from p=.030 to p=.001, obvious minus implausible; t is two-sided):")
    scales = (("token score", lambda ln: ln["token_probability"]), ("log odds", lambda ln: PD.logit(ln["token_probability"])), ("stated probability", lambda ln: ln["stated_probability"]))
    for name, value in scales:
        r = PD.interaction(lines, value)
        print(f"  {name}:")
        for kind in PD.KINDS:
            ch = r["changes"][kind]
            print(f"    {kind:<12} changes {', '.join(f'{c:+.3f}' for c in ch)}   mean {np.mean(ch):+.3f}")
        print(f"    interaction {r['difference']:+.3f}   exact permutation p {r['perm_count']} of {r['perm_total']} = {r['perm_count'] / r['perm_total']:.3f}")
        if r["t"] is None:
            print("    no t test: the changes of one kind do not vary")
        else:
            print(f"    t({r['df']}) = {r['t']:.2f}, p = {2 * stats.t.sf(abs(r['t']), r['df']):.4f}   F(1, {r['df']}) = {r['t'] ** 2:.2f}   Welch t({r['welch_df']:.1f}) = {r['welch_t']:.2f}, p = {2 * stats.t.sf(abs(r['welch_t']), r['welch_df']):.4f}")
    print("Answers of yes: " + ", ".join(f"{kind} {level} {sum(ln['answer'] == 'yes' for ln in lines if ln['kind'] == kind and ln['p_level'] == level)} of 4" for kind in PD.KINDS for level in PD.P_LEVELS))
    print(f"Responses with yes or no missing from the top {S.QWEN_TOP_LOGPROBS} tokens: {sum(bool(ln['token_note']) for ln in lines)} of {len(lines)}")


ORIGINAL_P = re.compile(r"\bp(?:-value)?s?\s*(<=|>=|=|<|>|≤|≥)\s*(0?\.\d+|\d+(?:\.\d+)?)", re.I)


def original_p(rec: S.Record) -> tuple[str, float] | None:
    """The first p-value stated in the original finding, as (sign, value): "p" or "p-value", then =, <, > or their or-equal forms, then a number from 0 to 1. None if the finding states none."""
    for op, v in ORIGINAL_P.findall(rec.finding):
        if 0 <= float(v) <= 1:
            return op, float(v)
    return None


def pvalue_baseline():
    """The original p-value alone as a predictor of replication (smaller p scored as more likely to replicate; an inequality is scored at its bound), on the claims whose original finding states a p-value, against Qwen3.8 27B (token score, with original finding, no reasoning) on the same claims. Paired bootstrap (BOOT resamples, seed 0; claims on the human and high-power sets, papers on all claims); two-sided p-values are twice the smaller tail share, for the baseline against an AUC of 0.5 and for the difference against zero."""
    header("Original p-value alone as a baseline")
    for s in S.SUBSETS:
        every = sel(s)
        rows = [r for r in every if original_p(r)]
        none = [r for r in every if not original_p(r)]
        ps = [original_p(r) for r in rows]
        x = {"p": arr([-v for _, v in ps]), "qwen": qwen_scores(rows, F, "off")[0]}
        b = boot_any(x, rows)
        d = b["qwen"] - b["p"]
        print(f"  {s} (n={len(every)}): p-value stated in {len(rows)} (as an equality {sum(op == '=' for op, _ in ps)}, as an inequality {sum(op != '=' for op, _ in ps)}; distinct values {len({v for _, v in ps})}), not stated in {len(none)}")
        print(f"      replication rate (weighted): with a p-value {np.average(outcomes(rows), weights=S.weights(rows)):.3f}, without {np.average(outcomes(none), weights=S.weights(none)):.3f}")
        print(f"      p-value alone AUC {auc(x['p'], rows):.3f} {ci(b['p'])}, against 0.5: two-sided p {_boot_p(b['p'] - 0.5):.4f}")
        print(f"      Qwen3.8 token AUC {auc(x['qwen'], rows):.3f} {ci(b['qwen'])}")
        print(f"      Qwen3.8 - p-value {auc(x['qwen'], rows) - auc(x['p'], rows):+.3f} {ci_signed(d)}, two-sided p {_boot_p(d):.4f}; rank correlation of the two scores {spearman(x['p'], x['qwen']):.2f}")
        if s == "highpower":
            print(f"      rows without a p-value: {[r.row for r in none]}")
            print(f"      values: {sorted(collections.Counter((op, v) for op, v in ps).items(), key=lambda t: t[0][1])}")


def leaked():
    """The 17 claims whose replication was published before 8 January 2026 (data/leaked_claims.csv, from leak_search.py and reading its candidates): weighted AUC on each set with all claims, without the leaked claims, and within the leaked claims alone (weights recomputed on each selection). Gemini's stated scores are read from results/gemini_stated.jsonl (gemini_stated.py)."""
    import gemini_stated as G
    header("Claims whose outcome was published before 8 January 2026: AUC with all claims, without them, and within them")
    with open(S.ROOT / "data/leaked_claims.csv", newline="", encoding="utf-8") as f:
        leak = {int(r["row"]) for r in csv.DictReader(f)}
    gem = G.load()

    def gemini(model):
        return lambda rows: [gem[G.key(model, S.qwen_prompt(r, G.TEMPLATE))]["probability"] / 100 for r in rows]

    preds = {
        "Qwen3.8 27B token": lambda rows: chat_scores(rows, QWEN)[0],
        "Qwen3 30B-A3B Instruct token": lambda rows: chat_scores(rows, "Qwen/Qwen3-30B-A3B-Instruct-2507")[0],
        "Gemini 3.1 Pro stated": gemini("gemini-3.1-pro-preview"),
        "Gemini 3 Flash stated": gemini("gemini-3-flash-preview"),
        "repliCATS": lambda rows: humans(rows)[0],
        "Markets": lambda rows: humans(rows)[1],
    }
    for s in S.SUBSETS:
        rows = sel(s)
        out = [r for r in rows if r.row not in leak]
        inside = [r for r in rows if r.row in leak]
        print(f"  {s} ({len(rows)} claims; {len(inside)} leaked, {sum(r.succeeded for r in inside)} of them replicated)")
        for name, f in preds.items():
            if name in ("repliCATS", "Markets") and s == "all":
                continue
            a, b, c = auc(f(rows), rows), auc(f(out), out), auc(f(inside), inside)
            print(f"      {name:30} AUC {a:.3f}  without leaked {b:.3f}  change {b - a:+.3f}  within leaked {c:.3f}")


SECTIONS = {
    "baseline": baseline,
    "qwen_off": qwen_off, "qwen_agreement": qwen_agreement, "olmo_yesno": olmo_yesno, "variants": variants, "recall": recall,
    "thinking_4k": thinking_4k, "thinking_complete": thinking_complete, "traces": traces, "cost": cost, "confidence": confidence,
    "precision": precision, "model_set": model_set, "base_chat": base_chat, "olmo_think": olmo_think, "size": size,
    "individuals": individuals, "ablation": ablation, "individual_tests": individual_tests, "ceiling": ceiling, "consistency": consistency, "memorization": memorization,
    "plausibility": plausibility, "pvalue_baseline": pvalue_baseline, "leaked": leaked,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--list"]:
        print(" ".join(SECTIONS))
        sys.exit()
    for name in args or SECTIONS:
        SECTIONS[name]()
