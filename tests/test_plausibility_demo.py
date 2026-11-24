"""Unit tests for plausibility_demo.py: the stimuli, the table with a fake chat client (no server), and the interaction test."""

from scipy import stats

import plausibility_demo as D
import score as S
from tests.test_score import FakeQwen


def test_design_is_crossed():
    recs = D.records()
    assert [r.row for r in recs] == list(range(1, 17))
    assert sorted((r.data_type, r.replicated) for r in recs) == sorted((k, lv) for k in D.KINDS for lv in D.P_LEVELS for _ in range(4))
    assert len({S.qwen_prompt(r, D.TEMPLATE) for r in recs}) == 16


def test_statistics_are_consistent():
    """Each statistics text has the t and two-sided p of a two-group test with 40 per group, and nothing else."""
    df = 2 * D.N_PER_GROUP - 2
    for level, text in D.P_LEVELS.items():
        t, p = (float(part.split("= ")[1]) for part in text.split(", "))
        assert text.startswith(f"t({df}) = ")
        assert round(2 * stats.t.sf(t, df), 3) == p == int(level[1:]) / 1000


def test_only_the_statistics_differ_within_a_paper():
    recs = D.records()
    a_stats, b_stats = D.P_LEVELS.values()
    for a, b in zip(recs[::2], recs[1::2]):
        assert a.paper == b.paper and (a.abstract, a.claim) == (b.abstract, b.claim)
        assert S.qwen_prompt(a, D.TEMPLATE).replace(a_stats, b_stats) == S.qwen_prompt(b, D.TEMPLATE)
    # no abstract or claim states a p-value or a test statistic, and every paper has 80 participants
    for item in D.ITEMS:
        assert "p =" not in item["abstract"] + item["claim"] and "t(" not in item["abstract"] + item["claim"]
        assert item["abstract"].count("Eighty") == 1


def test_table_from_fake_client(tmp_path, capsys):
    client = FakeQwen()
    responses = S.get_qwen_responses(D.records(), [D.TEMPLATE], ["off"], tmp_path / "cache.jsonl", "unused", 4, D.MAX_TOKENS, client=client)
    lines = D.table(responses)
    assert client.calls == 16 and len(lines) == 16
    assert all(0 <= ln["token_probability"] <= 1 and ln["answer"] in ("yes", "no") for ln in lines)
    D.report(lines)
    out = capsys.readouterr().out
    assert "of 16 pairs" in out and "telepathy" in out


def _lines(changes: dict) -> list[dict]:
    """Table lines whose value is 0 at p = .030 and the given change at p = .001, papers in ITEMS order within each kind."""
    left = {kind: list(v) for kind, v in changes.items()}
    out = []
    for item in D.ITEMS:
        c = left[item["kind"]].pop(0)
        out += [{"paper": item["name"], "kind": item["kind"], "p_level": "p001", "v": c}, {"paper": item["name"], "kind": item["kind"], "p_level": "p030", "v": 0.0}]
    return out


def test_interaction_matches_scipy():
    changes = {"implausible": [0.25, 0.875, -0.5, 0.125], "obvious": [0.75, 2.375, 1.625, 1.25]}
    r = D.interaction(_lines(changes), lambda ln: ln["v"])
    assert r["changes"] == changes and abs(r["difference"] - 1.3125) < 1e-12 and r["df"] == 6
    pooled = stats.ttest_ind(changes["obvious"], changes["implausible"])
    welch = stats.ttest_ind(changes["obvious"], changes["implausible"], equal_var=False)
    assert abs(r["t"] - pooled.statistic) < 1e-9 and abs(r["welch_t"] - welch.statistic) < 1e-9 and abs(r["welch_df"] - welch.df) < 1e-9
    assert (r["perm_count"], r["perm_total"]) == (4, 70)


def test_interaction_complete_separation_and_no_variance():
    """Complete separation gives the smallest permutation count (the split and its mirror); identical changes within a kind give no t."""
    r = D.interaction(_lines({"implausible": [-0.05, 0.0, -0.05, 0.0], "obvious": [0.8, 0.8, 0.8, 0.8]}), lambda ln: ln["v"])
    assert (r["perm_count"], r["perm_total"]) == (2, 70) and r["t"] is None and r["welch_t"] is None
