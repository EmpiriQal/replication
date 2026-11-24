"""Plausibility demonstration: does a chat model's prediction of replication follow the claim's plausibility, or only the original p-value?

Eight made-up papers, four with implausible claims (telepathy, precognition, psychokinesis, distant intention) and four with obvious ones (sleep loss, divided attention, practice, alcohol), each crossed with two original p-values (.001 and .030): 16 prompts. Every paper is a two-group experiment with 40 participants per group, so the statistics of the original finding are the same text for all eight papers at a given p-value, and the two prompts of a paper differ only in those statistics. The abstracts state no statistics, like the real ones (1 of the 274 SCORE abstracts gives a p-value; 214 of the original findings do).

The prompt is score.py's with-original-finding chat prompt, the request settings are those of the SCORE runs without reasoning (default server qwen27, Qwen3.8 27B, reasoning off), and the scores are the token score and the stated probability. Responses are cached in results/plausibility_demo.jsonl; the table is also written to results/plausibility_demo.csv.

Usage:
  uv run python plausibility_demo.py --prompts         print the 16 prompts; contacts no server
  uv run python plausibility_demo.py [--server NAME]
      request any missing responses (from the model's server at its URL in models.toml) and print the results
"""

import argparse
import csv
import itertools
import math
import statistics
import sys

import score as S

CACHE = S.ROOT / "results/plausibility_demo.jsonl"
OUT = S.ROOT / "results/plausibility_demo.csv"
TEMPLATE = "with_original_finding"
MAX_TOKENS = 1024  # far above the answer's length; with reasoning off the answer format ends the response after a few tokens
N_PER_GROUP = 40

# statistics of the original finding at each p-value: independent-samples t test, 40 per group (78 degrees of freedom), two-sided p; no effect size is given
P_LEVELS = {
    "p001": "t(78) = 3.42, p = .001",
    "p030": "t(78) = 2.21, p = .030",
}

# each paper: name, kind, abstract, claim tested, and the opening of the original finding (the statistics are appended)
ITEMS = (
    {
        "name": "telepathy",
        "kind": "implausible",
        "abstract": "Whether information can pass from one person to another without any sensory channel remains disputed. We tested the telepathic transmission of pictures under controlled conditions. Eighty adults acted as receivers and were randomly assigned to a sender condition or a no-sender condition. On each of 40 trials, the receiver chose which of four pictures was the target. In the sender condition, a stranger seated in an electromagnetically shielded room in another building concentrated on the target picture throughout the trial; in the no-sender condition, that room was empty. Receivers and experimenters did not know the condition. Receivers in the sender condition identified more targets than receivers in the no-sender condition. These results indicate that the content of one person's thoughts can be received by another person at a distance, without sensory contact.",
        "claim": "Receivers identify more target pictures when a distant sender concentrates on the target than when there is no sender.",
        "finding": "Receivers in the sender condition identified more target pictures than receivers in the no-sender condition",
    },
    {
        "name": "precognition",
        "kind": "implausible",
        "abstract": "Memory is assumed to depend only on events that precede the test. We examined whether studying material after a test can improve performance on that test. Eighty undergraduates saw a list of 48 common nouns once and then completed a free recall test. After the recall sheets had been collected, a computer randomly assigned each participant to a later-study condition, in which the participant studied the same list for a further 10 minutes, or to a no-study condition, in which the participant completed an unrelated task for 10 minutes. Participants who went on to study the list after the test had recalled more words on the test than participants who did not. Because the conditions were assigned only after recall was complete, these results indicate that future study reaches back in time to strengthen present memory.",
        "claim": "Studying a word list after a recall test improves performance on that earlier test.",
        "finding": "Participants in the later-study condition had recalled more words on the test than participants in the no-study condition",
    },
    {
        "name": "psychokinesis",
        "kind": "implausible",
        "abstract": "We tested whether conscious intention can influence a physical random process directly. Eighty adults each sat alone for 20 minutes in front of a sealed electronic random number generator, which produced zeros and ones with equal probability from the timing of radioactive decay and had no keyboard, switch or other input. Participants were randomly assigned to an intention condition, in which they were asked to will the device to produce more ones, or to a control condition, in which they were asked to watch the device without any intention. The proportion of ones produced during the session was higher in the intention condition than in the control condition. These results indicate that mental intention alone, without physical contact, can bias the output of a random physical system.",
        "claim": "Mental intention to produce ones raises the proportion of ones that a random number generator produces.",
        "finding": "The proportion of ones produced by the random number generator was higher in the intention condition than in the control condition",
    },
    {
        "name": "distant intention",
        "kind": "implausible",
        "abstract": "Many traditions hold that one person's focused intention can benefit another person at a distance. We tested this claim in a double-blind experiment. Eighty adults with elevated test anxiety were randomly assigned to a distant intention condition or a control condition. For each participant in the distant intention condition, a practitioner located 800 kilometres away, who never met or contacted the participant and was given only a first name and a photograph, directed calming intention toward the participant for 15 minutes each evening for two weeks. Nothing was done for participants in the control condition. Participants did not know that conditions existed, and the experimenters did not know the assignments. After two weeks, participants in the distant intention condition reported lower anxiety than participants in the control condition. These results indicate that calming intention directed at a person from afar reduces that person's anxiety.",
        "claim": "Calming intention directed at a person from a distance, without that person's knowledge, lowers the person's anxiety.",
        "finding": "Participants in the distant intention condition reported lower anxiety after two weeks than participants in the control condition",
    },
    {
        "name": "sleep loss",
        "kind": "obvious",
        "abstract": "Sleep loss is common in modern working life. We examined the effect of one night without sleep on sustained attention. Eighty healthy adults were randomly assigned to a sleep deprivation condition, in which they remained awake for 24 hours under supervision in the laboratory, or to a rested condition, in which they slept a normal night at home. The next morning, all participants completed a 10-minute vigilance task, pressing a button as quickly as possible each time a light appeared on a screen. Participants in the sleep deprivation condition had slower mean reaction times than participants in the rested condition. These results indicate that a single night without sleep slows responses on a simple task that requires sustained attention.",
        "claim": "A night without sleep slows reaction times on a vigilance task.",
        "finding": "Participants in the sleep deprivation condition had slower mean reaction times on the vigilance task than participants in the rested condition",
    },
    {
        "name": "divided attention",
        "kind": "obvious",
        "abstract": "People often try to learn new material while doing something else. We examined the effect of dividing attention during study on later memory. Eighty undergraduates studied a list of 48 common nouns presented one at a time. Participants were randomly assigned to a full attention condition, in which they only studied the words, or to a divided attention condition, in which they also listened to a stream of spoken digits and pressed a key whenever three odd digits occurred in a row. After a 5-minute unrelated task, all participants completed a free recall test. Participants in the full attention condition recalled more words than participants in the divided attention condition. These results indicate that performing a demanding second task during study reduces later memory for the studied material.",
        "claim": "Performing a demanding second task while studying a word list lowers later recall of the words.",
        "finding": "Participants in the full attention condition recalled more words than participants in the divided attention condition",
    },
    {
        "name": "practice",
        "kind": "obvious",
        "abstract": "New motor skills are usually performed poorly at first. We examined the effect of practice on an unfamiliar motor task, mirror tracing. Eighty adults who had never performed the task were randomly assigned to a practice condition or a no-practice condition. Participants in the practice condition traced the outline of a star 20 times while seeing their hand only in a mirror; participants in the no-practice condition spent the same time on an unrelated task. All participants then completed one test trial, in which they traced a new shape in the mirror. Participants in the practice condition completed the test trial faster than participants in the no-practice condition. These results indicate that practice improves performance on a new motor task.",
        "claim": "Practice at mirror tracing makes later mirror tracing faster.",
        "finding": "Participants in the practice condition completed the test trial faster than participants in the no-practice condition",
    },
    {
        "name": "alcohol",
        "kind": "obvious",
        "abstract": "Alcohol is involved in a large share of road accidents. We examined the effect of alcohol on lane keeping in a driving simulator. Eighty adult social drinkers were randomly assigned to an alcohol condition, in which they drank a beverage that raised their blood alcohol concentration to 0.08%, or to a placebo condition, in which they drank a beverage with the same taste and smell and no alcohol. Neither the participants nor the experimenters knew the condition. Participants then drove for 20 minutes on a simulated two-lane road, and the variability of the vehicle's lateral position was recorded. Lateral position was more variable in the alcohol condition than in the placebo condition. These results indicate that alcohol at a blood alcohol concentration of 0.08% impairs lane keeping.",
        "claim": "Alcohol at a blood alcohol concentration of 0.08% impairs lane keeping in simulated driving.",
        "finding": "The variability of lateral position was greater in the alcohol condition than in the placebo condition",
    },
)
KINDS = ("implausible", "obvious")


def records() -> list[S.Record]:
    """One Record per paper and p-value (rows 1 to 16, papers in ITEMS order, p = .001 before p = .030), so that score.py's prompt, client and cache apply unchanged. The outcome fields are fillers: these papers were never replicated."""
    out = []
    for i, item in enumerate(ITEMS):
        for j, (level, stats) in enumerate(P_LEVELS.items()):
            out.append(S.Record(
                row=2 * i + j + 1, claim_id=f"demo{i + 1}_{level}", paper_id=f"demo{i + 1}", paper=item["name"], data_type=item["kind"], replicated=level,
                claim=item["claim"], abstract=item["abstract"], finding=f"{item['finding']}, {stats}.",
                paper_weight=1.0, human_set=False, highpower=False, replicats=None, markets=None,
            ))
    return out


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def interaction(lines: list[dict], value) -> dict:
    """Interaction of kind with p-value. The design is mixed: p-value varies within a paper, kind between papers, so the interaction compares each paper's change (value at p = .001 minus value at p = .030) between the kinds, obvious minus implausible. value maps a table line to the number analysed. Returns the changes by kind, the difference of their means, the pooled-variance t (degrees of freedom n1 + n2 - 2; its square is the interaction F of the mixed analysis of variance), Welch's t with its degrees of freedom, and the exact two-sided permutation test: the number of ways of splitting the papers into the two kinds whose absolute difference of means is at least the observed one, out of all ways. The t values are None when a kind's changes do not vary."""
    by = {(ln["paper"], ln["p_level"]): value(ln) for ln in lines}
    first, second = P_LEVELS
    changes = {kind: [by[(it["name"], first)] - by[(it["name"], second)] for it in ITEMS if it["kind"] == kind] for kind in KINDS}
    a, b = changes["obvious"], changes["implausible"]
    diff = statistics.mean(a) - statistics.mean(b)
    out = {"changes": changes, "difference": diff, "df": len(a) + len(b) - 2, "t": None, "welch_t": None, "welch_df": None}
    va, vb = statistics.variance(a), statistics.variance(b)
    if va > 1e-12 and vb > 1e-12:
        pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / out["df"]
        out["t"] = diff / math.sqrt(pooled * (1 / len(a) + 1 / len(b)))
        sa, sb = va / len(a), vb / len(b)
        out["welch_t"] = diff / math.sqrt(sa + sb)
        out["welch_df"] = (sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
    pool = a + b
    splits = list(itertools.combinations(range(len(pool)), len(a)))
    out["perm_total"] = len(splits)
    out["perm_count"] = sum(abs(statistics.mean(pool[i] for i in s) - statistics.mean(pool[i] for i in range(len(pool)) if i not in s)) >= abs(diff) - 1e-9 for s in splits)
    return out


def table(responses: dict) -> list[dict]:
    """One line per prompt: paper, kind, p-value, answer, token score, stated probability."""
    lines = []
    for rec in records():
        e = responses[(TEMPLATE, "off", rec.row)]
        token, note = S.token_probability(e["answer_top_logprobs"])
        lines.append({"row": rec.row, "paper": rec.paper, "kind": rec.data_type, "p_level": rec.replicated, "answer": e["answer"], "token_probability": round(token, 6), "stated_probability": e["probability"] / 100, "token_note": note})
    return lines


def report(lines: list[dict]) -> None:
    by = {(ln["paper"], ln["p_level"]): ln for ln in lines}
    print(f"\n{'paper':<18} {'kind':<12} {'token score':>23} {'stated':>15} {'answer':>10}")
    print(f"{'':<18} {'':<12} {'p=.001':>11} {'p=.030':>11} {'p=.001':>7} {'p=.030':>7} {'.001':>5} {'.030':>4}")
    for item in ITEMS:
        a, b = by[(item["name"], "p001")], by[(item["name"], "p030")]
        print(f"{item['name']:<18} {item['kind']:<12} {a['token_probability']:>11.4f} {b['token_probability']:>11.4f} {a['stated_probability']:>7.2f} {b['stated_probability']:>7.2f} {a['answer']:>5} {b['answer']:>4}")
    for score in ("token_probability", "stated_probability"):
        print(f"\nMean {score.replace('_', ' ')}:")
        for kind in KINDS:
            means = [statistics.mean(ln[score] for ln in lines if ln["kind"] == kind and ln["p_level"] == level) for level in P_LEVELS]
            print(f"  {kind:<12} p=.001 {means[0]:.4f}   p=.030 {means[1]:.4f}")
        low = [ln[score] for ln in lines if ln["kind"] == "implausible" and ln["p_level"] == "p001"]
        high = [ln[score] for ln in lines if ln["kind"] == "obvious" and ln["p_level"] == "p030"]
        wins = sum(o > i for o in high for i in low)
        ties = sum(o == i for o in high for i in low)
        print(f"  obvious at p=.030 above implausible at p=.001 in {wins} of {len(low) * len(high)} pairs ({ties} ties)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="qwen27", help="chat model in models.toml (default qwen27, Qwen3.8 27B)")
    ap.add_argument("--prompts", action="store_true", help="print the 16 prompts and stop; contacts no server")
    args = ap.parse_args()
    recs = records()
    if args.prompts:
        for rec in recs:
            print(f"===== row {rec.row}: {rec.paper} ({rec.data_type}), {rec.replicated} =====\n{S.qwen_prompt(rec, TEMPLATE)}\n")
        return
    try:
        responses = S.get_qwen_responses(recs, [TEMPLATE], ["off"], CACHE, args.server, len(recs), MAX_TOKENS)
    except S.ScoreError as e:
        sys.exit(str(e))
    lines = table(responses)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    print(f"Server {args.server}, template {TEMPLATE}, reasoning off, temperature {S.QWEN_TEMPERATURE}.")
    report(lines)
    print(f"\nWritten to {OUT.relative_to(S.ROOT)}")


if __name__ == "__main__":
    main()
