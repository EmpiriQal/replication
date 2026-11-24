"""Unit tests for score.py. No model server is touched: base and chat models are replaced by fakes."""

import contextlib
import csv
import hashlib
import json
import math

import numpy as np
import pytest

import score
from score import DataError, ScoreError

DATA = score.ROOT / "data" / "SCORE_replicated_claims.csv"
TOL = 1e-6
REAL_QWEN_CLIENT = score.QwenClient  # kept before the fixture below blocks it; used only with its HTTP calls faked


@pytest.fixture(autouse=True)
def no_real_servers(monkeypatch):
    """Unit tests must never send requests to the real servers, even when they are awake."""
    def refuse(*args, **kwargs):
        pytest.fail("a unit test tried to contact a real model server")
    monkeypatch.setattr(score, "OlmoScorer", refuse)
    monkeypatch.setattr(score, "QwenClient", refuse)
    monkeypatch.setattr(score.serving, "require", refuse)


@pytest.fixture(scope="module")
def records():
    return score.load_data(DATA)


def approx(v):
    return pytest.approx(v, abs=TOL)


# ---------------------------------------------------------------- data integrity


def test_integrity_passes_on_data_file(records):
    assert len(records) == 274
    assert len({r.paper_id for r in records}) == 164
    assert sum(r.paper_weight for r in records) == approx(164)


def _table():
    with open(DATA, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _write(path, table):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(table)
    return path


def _set(table, row, column, value):
    table[row][score.COLUMNS.index(column)] = value


def _corrupt_drop_row(t):
    del t[-1]


def _corrupt_empty_field(t):
    _set(t, 5, "Journal", "")


def _corrupt_human_flag(t):
    i = next(i for i in range(1, len(t)) if t[i][score.COLUMNS.index("Human set")] == "no" and t[i][score.COLUMNS.index("repliCATS estimate")] == score.NOT_RATED)
    _set(t, i, "Human set", "yes")


def _corrupt_label(t):
    _set(t, 3, "Replicated", "Maybe")


def _corrupt_estimate_text(t):
    i = next(i for i in range(1, len(t)) if t[i][score.COLUMNS.index("Human set")] == "yes")
    _set(t, i, "repliCATS estimate", "0.6x")


def _corrupt_missing_estimate(t):
    i = next(i for i in range(1, len(t)) if t[i][score.COLUMNS.index("Human high-power set")] == "yes")
    _set(t, i, "Replication Markets estimate", score.NOT_RATED)


def _corrupt_outcome(t):
    i = next(i for i in range(1, len(t)) if t[i][score.COLUMNS.index("Replicated")] == "Yes")
    _set(t, i, "Replicated", "No (n.s.)")


def _corrupt_header(t):
    t[0][0] = "Citation"


def _corrupt_extra_column(t):
    t[7].append("extra")


def _corrupt_paper_weight(t):
    _set(t, 1, "Paper weight", "2")


def _corrupt_duplicate_claim(t):
    _set(t, 2, "Claim ID", t[1][score.COLUMNS.index("Claim ID")])


@pytest.mark.parametrize("corrupt", [
    _corrupt_drop_row, _corrupt_empty_field, _corrupt_human_flag, _corrupt_label, _corrupt_estimate_text,
    _corrupt_missing_estimate, _corrupt_outcome, _corrupt_header, _corrupt_extra_column, _corrupt_paper_weight,
    _corrupt_duplicate_claim,
])
def test_integrity_fails_on_corrupted_copy(tmp_path, corrupt):
    t = _table()
    corrupt(t)
    with pytest.raises(DataError):
        score.load_data(_write(tmp_path / "bad.csv", t))


# ---------------------------------------------------------------- the forecasting paper's own metrics (Mody et al.)


def rank_auc(x, y):
    """AUC from ranks with ties averaged (Mann-Whitney)."""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


PAPER_TABLE = [  # selection, data type, predictor, n, AUC, accuracy, Brier, correlation
    ("human", "New", "replicats", 84, 0.667328, 0.619048, 0.229473, 0.293170),
    ("human", "New", "markets", 84, 0.665910, 0.619048, 0.238887, 0.274262),
    ("human", "Secondary", "replicats", 51, 0.549231, 0.470588, 0.264438, 0.133639),
    ("human", "Secondary", "markets", 51, 0.576154, 0.549020, 0.274333, 0.144388),
    ("highpower", "New", "replicats", 37, 0.755000, 0.648649, 0.207657, 0.400248),
    ("highpower", "New", "markets", 37, 0.745000, 0.729730, 0.212771, 0.403116),
    ("highpower", "Secondary", "replicats", 20, 0.532967, 0.600000, 0.226582, 0.123882),
    ("highpower", "Secondary", "markets", 20, 0.527473, 0.600000, 0.237910, 0.124703),
]


@pytest.mark.parametrize("subset,dt,pred,n,auc,acc,brier,corr", PAPER_TABLE)
def test_forecasting_paper_metrics(records, subset, dt, pred, n, auc, acc, brier, corr):
    rows = score.select(records, subset, dt)
    p = np.array([score.human_estimate(r, pred) for r in rows])
    y = np.array([r.succeeded for r in rows], dtype=float)
    assert len(rows) == n
    assert rank_auc(p, y) == approx(auc)
    assert np.mean((p >= 0.5) == (y == 1)) == approx(acc)
    assert np.mean((p - y) ** 2) == approx(brier)
    assert np.corrcoef(p, y)[0, 1] == approx(corr)


# ---------------------------------------------------------------- human predictors in score.py


SCORE_TABLE = [  # subset, data type, predictor, n, AUC, accuracy (threshold), accuracy (hedged)
    ("human", "all", "replicats", 135, 0.617699, 0.562963, 0.534637),
    ("human", "all", "markets", 135, 0.634497, 0.592593, 0.547073),
    ("human", "New", "replicats", 84, 0.667328, 0.619048, 0.548293),
    ("human", "New", "markets", 84, 0.665910, 0.619048, 0.560414),
    ("human", "Secondary", "replicats", 51, 0.549231, 0.470588, 0.512144),
    ("human", "Secondary", "markets", 51, 0.576154, 0.549020, 0.525098),
    ("highpower", "all", "replicats", 57, 0.731250, 0.631579, 0.566163),
    ("highpower", "all", "markets", 57, 0.723125, 0.684211, 0.578991),
    ("highpower", "New", "replicats", 37, 0.755000, 0.648649, 0.575184),
    ("highpower", "New", "markets", 37, 0.745000, 0.729730, 0.587094),
    ("highpower", "Secondary", "replicats", 20, 0.532967, 0.600000, 0.549475),
    ("highpower", "Secondary", "markets", 20, 0.527473, 0.600000, 0.564000),
]


@pytest.mark.parametrize("subset,dt,pred,n,auc,acc_t,acc_h", SCORE_TABLE)
def test_human_runs(records, subset, dt, pred, n, auc, acc_t, acc_h):
    run = score.Run(pred, subset, dt)
    for mode, acc in (("threshold", acc_t), ("hedged", acc_h)):
        out, line = score.execute(run, records, {"human_accuracy": mode}, None)
        assert line["n_rows"] == n and line["n_papers"] == n and line["sum_weights"] == n
        assert all(o["weight"] == 1 for o in out)
        assert line["weighted_auc"] == approx(auc)
        assert line["weighted_accuracy"] == approx(acc)
        assert line["fitted_offset"] == ""


def test_humans_not_applicable_to_all_subset(records):
    assert not score.human_applicable(score.select(records, "all", "all"), "replicats")
    assert score.human_applicable(score.select(records, "highpower", "Secondary"), "markets")


# ---------------------------------------------------------------- weights and metrics


def _rec(row, paper_id, succeeded=True, data_type="New"):
    return score.Record(row, f"{paper_id}_{row}", paper_id, f"Paper {paper_id}", data_type, "Yes" if succeeded else "No (n.s.)",
                        f"claim {row}", f"abstract {paper_id}", f"finding {row}", 1.0, False, False, None, None)


def test_weights_per_selection():
    rows = [_rec(1, "A"), _rec(2, "A"), _rec(3, "B"), _rec(4, "A"), _rec(5, "C"), _rec(6, "C")]
    w = score.weights(rows)
    assert w == approx([1 / 3, 1 / 3, 1, 1 / 3, 0.5, 0.5])
    assert sum(w) == approx(3)


def test_full_file_weights_match_column_16(records):
    assert score.weights(records) == approx([r.paper_weight for r in records])
    assert sum(score.weights(score.select(records, "all", "New"))) == approx(len({r.paper_id for r in records if r.data_type == "New"}))


def test_weighted_auc_with_ties_and_weights():
    # positives (x=3, w=1), (x=2, w=1); negatives (x=2, w=2), (x=1, w=1): (2 + 1 + 1 + 1) / (2 * 3)
    assert score.weighted_auc([3, 2, 2, 1], [True, False, True, False], [1, 2, 1, 1]) == approx(5 / 6)
    assert score.weighted_auc([1, 2, 3], [False, True, True], [1, 1, 1]) == approx(1.0)
    assert math.isnan(score.weighted_auc([1, 2], [True, True], [1, 1]))


def test_weighted_auc_equals_rank_auc_unweighted():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 60).astype(float)
    y = rng.random(60) < 0.5
    assert score.weighted_auc(x, y, np.ones(60)) == approx(rank_auc(x, y))


def test_human_credit():
    assert score.human_credit(0.5, True, "threshold") == 1
    assert score.human_credit(0.5, False, "threshold") == 0
    assert score.human_credit(0.3, False, "hedged") == approx(0.7)
    assert score.human_credit(0.3, True, "hedged") == approx(0.3)
    assert score.human_prediction(0.5) == "succeeded"


# ---------------------------------------------------------------- offset fitting and modes


def test_fit_offset_simple():
    assert score.fit_offset([1, -1], [True, False], [1, 1]) == 0


def test_fit_offset_extreme_candidates():
    assert score.fit_offset([-3, 3], [False, False], [1, 1]) == -4   # u_1 - 1: everything "failed"
    assert score.fit_offset([-3, 3], [True, True], [1, 1]) == 4      # u_K + 1: everything "succeeded"


def test_fit_offset_tie_smallest_abs():
    # candidates -3, -1, 1, 3; -3 and 1 both reach 2/3
    assert score.fit_offset([-2, 0, 2], [False, True, False], [1, 1, 1]) == 1


def test_fit_offset_tie_smaller_b():
    # candidates -2, 0, 2; -2 and 2 both reach 1/2
    assert score.fit_offset([-1, 1], [True, False], [1, 1]) == -2


def test_fit_offset_uses_weights():
    assert score.fit_offset([-2, 0, 2], [False, True, False], [1, 3, 1]) == 1
    assert score.fit_offset([-2, 0, 2], [False, True, False], [3, 1, 2]) == -3


def test_fit_offset_duplicate_scores():
    # distinct -D values are -1 and 1 only: candidates -2, 0, 2
    assert score.fit_offset([1, 1, -1, -1], [True, True, False, True], [1, 1, 1, 1]) == 0


def test_loo_offsets_by_hand():
    # fold A fits on B, C -> -3; fold B fits on A, C -> tie -3 / 0 -> 0; fold C fits on A, B -> 0
    d, succ, w = [1, -1, 2], [True, False, False], [1, 1, 1]
    assert score.loo_offsets(d, succ, w, ["A", "B", "C"]) == [-3, 0, 0]
    assert score.fit_offset(d, succ, w) == 0


def test_loo_holds_out_whole_paper():
    d, succ, papers = [1, 1.5, -1, 2], [True, True, False, False], ["A", "A", "B", "C"]
    w = [0.5, 0.5, 1, 1]
    off = score.loo_offsets(d, succ, w, papers)
    assert off[0] == off[1] == score.fit_offset([-1, 2], [False, False], [1, 1])
    assert off[2] == score.fit_offset([1, 1.5, 2], [True, True, False], [0.5, 0.5, 1])


def _ppl_for(rows, template, d):
    """Perplexities whose score L = log PPL(" no") - log PPL(" yes") is d."""
    return {**{(template, r.row, " yes"): 1.0 for r in rows}, **{(template, r.row, " no"): math.exp(di) for r, di in zip(rows, d)}}


def test_run_olmo_fitted_and_loo():
    rows = [_rec(1, "A", True), _rec(2, "B", False), _rec(3, "C", False)]
    ppl = _ppl_for(rows, "claim_only", [1, -1, 2])
    out, s = score.run_olmo(score.Run("base", "all", "all", "claim_only", "fitted", ending="yes_no"), rows, ppl)
    assert s["fitted_offset"] == 0 and s["weighted_accuracy"] == approx(2 / 3)
    assert [o["prediction"] for o in out] == ["succeeded", "failed", "succeeded"]
    assert s["weighted_auc"] == approx(0.5)
    out, s = score.run_olmo(score.Run("base", "all", "all", "claim_only", "loo", ending="yes_no"), rows, ppl)
    assert [o["offset"] for o in out] == [-3, 0, 0]
    assert [o["correct"] for o in out] == [0, 1, 0]
    assert s["fitted_offset"] == "" and s["weighted_accuracy"] == approx(1 / 3)
    assert list(out[0]) == list(score.OLMO_YES_NO_FILE_COLUMNS)


# ---------------------------------------------------------------- prompts


def test_prompt_with_original_finding(records):
    r = records[0]
    assert score.context(r, "with_original_finding") == (
        "The following paper was subjected to a replication attempt. Here is the original paper's abstract:\n\n"
        f"{r.abstract}\n\n"
        "And, here is the claim the replicating team tested:\n\n"
        f"{r.claim}\n\n"
        "In particular the original paper found:\n\n"
        f"{r.finding}\n\n"
        "What do you think happened in the replication attempt? Did it find a statistically significant result in the same direction as the original?\n\n"
        + score.QWEN_INSTRUCTION + "\n\nAnswer:"
    )


def test_prompt_claim_only(records):
    r = records[0]
    assert score.context(r, "claim_only") == (
        "The following paper was subjected to a replication attempt. Here is the original paper's abstract:\n\n"
        f"{r.abstract}\n\n"
        "And, here is the claim the replicating team tested:\n\n"
        f"{r.claim}\n\n"
        "What do you think happened in the replication attempt? Did it find a statistically significant result in the same direction as the original?\n\n"
        + score.QWEN_INSTRUCTION + "\n\nAnswer:"
    )


def test_prompt_no_abstract(records):
    r = records[0]
    assert score.qwen_prompt(r, "no_abstract") == (
        "The following paper was subjected to a replication attempt. Here is the claim the replicating team tested:\n\n"
        f"{r.claim}\n\n"
        "In particular the original paper found:\n\n"
        f"{r.finding}\n\n"
        "What do you think happened in the replication attempt? Did it find a statistically significant result in the same direction as the original?\n\n"
        + score.QWEN_INSTRUCTION
    )


def test_prompt_claim_alone(records):
    r = records[0]
    assert score.qwen_prompt(r, "claim_alone") == (
        "The following paper was subjected to a replication attempt. Here is the claim the replicating team tested:\n\n"
        f"{r.claim}\n\n"
        "What do you think happened in the replication attempt? Did it find a statistically significant result in the same direction as the original?\n\n"
        + score.QWEN_INSTRUCTION
    )


def test_ablation_templates_off_by_default(records):
    assert {r.template for r in score.plan_runs(score.CONFIG, records) if r.template} == set(score.TEMPLATES)
    cfg = score.parse_config(["--template_no_abstract", "true", "--template_claim_alone", "true"])
    assert {r.template for r in score.plan_runs(cfg, records) if r.template} == set(score.ALL_TEMPLATES)


def test_prompts_never_contain_replication_finding(records):
    findings = {row[score.COLUMNS.index("Claim ID")]: row[score.COLUMNS.index("Replication finding")] for row in _table()[1:]}
    for r in records:
        for t in score.TEMPLATES:
            ctx = score.context(r, t)
            assert ctx.endswith("\n\nAnswer:")
            assert findings[r.claim_id] not in ctx


# ---------------------------------------------------------------- perplexities and cache


class FakeScorer:
    """Stands in for OlmoScorer: whitespace tokenizer with a BOS token, and deterministic log-probabilities."""

    model = "fake-model"

    def __init__(self):
        self.vocab = {"<bos>": 0}
        self.calls = 0

    def tokenize(self, text):
        words = text.replace("\n", " \n ").split(" ")
        toks = [words[0]] + [" " + w for w in words[1:]]
        return [0] + [self.vocab.setdefault(t, len(self.vocab)) for t in toks]

    def prompt_logprobs(self, ids):
        self.calls += 1
        return [None] + [{str(t): {"logprob": -((t % 7) + 1) / 10, "rank": 1}} for t in ids[1:]]


def test_continuation_logprobs_and_perplexity():
    plp = [None, {"5": {"logprob": -0.1}}, {"6": {"logprob": -0.2}}, {"7": {"logprob": -0.3}}, {"8": {"logprob": -0.5}}]
    lps = score.continuation_logprobs([0, 5, 6], [0, 5, 6, 7, 8], plp)
    assert lps == [-0.3, -0.5]
    assert score.perplexity(lps) == approx(math.exp(0.4))


def test_continuation_logprobs_rejects_broken_prefix():
    with pytest.raises(ScoreError):
        score.continuation_logprobs([0, 5, 6], [0, 5, 9, 7], [None] * 4)
    with pytest.raises(ScoreError):
        score.continuation_logprobs([0, 5, 6], [0, 5, 6], [None] * 3)


def test_get_perplexities_computes_then_reuses(tmp_path, records, monkeypatch):
    cache = tmp_path / "ppl.jsonl"
    fake = FakeScorer()
    ppl = score.get_perplexities(records[:5], ["claim_only"], cache, "olmo32", 4, fake)
    assert fake.calls == 10 and len(ppl) == 10
    e = json.loads(cache.read_text().splitlines()[0])
    assert e["first_token_id"] == 0 and len(e["continuation_token_ids"]) == 1
    assert e["ppl"] == approx(score.perplexity(e["continuation_logprobs"]))
    again = score.get_perplexities(records[:5], ["claim_only"], cache, "olmo32", 4, fake)
    assert fake.calls == 10 and again == ppl
    score.get_perplexities(records[:5], ["claim_only", "with_original_finding"], cache, "olmo32", 4, fake)
    assert fake.calls == 20


def test_cache_skips_truncated_line(tmp_path, records):
    cache = tmp_path / "ppl.jsonl"
    fake = FakeScorer()
    score.get_perplexities(records[:2], ["claim_only"], cache, "olmo32", 2, fake)
    lines = cache.read_text().splitlines()
    cache.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])
    score.get_perplexities(records[:2], ["claim_only"], cache, "olmo32", 2, fake)
    assert fake.calls == 6  # the pair whose second entry was cut short is recomputed (both continuations)


def test_changed_prompt_misses_cache(tmp_path, records):
    import dataclasses
    cache = tmp_path / "ppl.jsonl"
    fake = FakeScorer()
    score.get_perplexities(records[:1], ["claim_only"], cache, "olmo32", 1, fake)
    changed = [dataclasses.replace(records[0], claim=records[0].claim + " extra")]
    score.get_perplexities(changed, ["claim_only"], cache, "olmo32", 1, fake)
    assert fake.calls == 4


class FakeRequire:
    """Stands in for serving.require: records each check, and can refuse like a server that is not answering."""

    def __init__(self, refuse=False):
        self.refuse = refuse
        self.checked = []

    def __call__(self, name):
        self.checked.append(name)
        if self.refuse:
            raise score.serving.ServingError(f"{name} is not answering")


def test_missing_perplexities_check_the_server_first(tmp_path, records, monkeypatch):
    req, fake = FakeRequire(), FakeScorer()
    monkeypatch.setattr(score.serving, "require", req)
    monkeypatch.setattr(score, "OlmoScorer", lambda server: fake)
    monkeypatch.setattr(score.serving, "endpoint", lambda name: ("http://x/v1", fake.model))
    score.get_perplexities(records[:2], ["claim_only"], tmp_path / "ppl.jsonl", "olmo32", 2)
    assert req.checked == ["olmo32"] and fake.calls == 4
    score.get_perplexities(records[:2], ["claim_only"], tmp_path / "ppl.jsonl", "olmo32", 2)  # all cached: not checked again
    assert req.checked == ["olmo32"]


def test_a_server_not_answering_stops_the_run(tmp_path, records, monkeypatch):
    monkeypatch.setattr(score.serving, "require", FakeRequire(refuse=True))
    with pytest.raises(ScoreError, match="olmo32 is not answering"):
        score.get_perplexities(records, ["claim_only"], tmp_path / "ppl.jsonl", "olmo32", 4)
    with pytest.raises(ScoreError, match="qwen27 is not answering"):
        score.get_qwen_responses(records, ["claim_only"], ["off"], tmp_path / "qwen.jsonl", "qwen27", 4, 100)


def test_full_cache_needs_no_server(tmp_path, records, monkeypatch):
    cache = tmp_path / "ppl.jsonl"
    fake = FakeScorer()
    fake.model = score.serving.endpoint("olmo32")[1]
    score.get_perplexities(records, list(score.TEMPLATES), cache, "olmo32", 8, fake)
    assert len(score.get_perplexities(records, list(score.TEMPLATES), cache, "olmo32", 8)) == 274 * 2 * 2


# ---------------------------------------------------------------- runs, configuration and outputs


def test_default_runs(records):
    runs = score.plan_runs(dict(score.CONFIG), records)
    assert sum(r.predictor == "base" and r.server == "olmo32" for r in runs) == 12  # 3 subsets x 2 templates x 2 modes
    assert sum(r.predictor == "replicats" for r in runs) == 2
    assert sum(r.predictor == "markets" for r in runs) == 2
    assert sum(r.predictor == "chat" and r.server == "qwen27" for r in runs) == 24  # 3 subsets x 2 templates x 2 thinking x 2 scores


def test_runs_for_several_servers(records):
    cfg = {**score.CONFIG, "base_servers": ["olmo32", "olmo7b", "qwen9b"], "chat_servers": ["qwen27", "qwen9c"]}
    runs = score.plan_runs(cfg, records)
    assert sum(r.predictor == "base" for r in runs) == 36 and sum(r.predictor == "chat" for r in runs) == 48
    assert {r.server for r in runs if r.predictor == "base"} == {"olmo32", "olmo7b", "qwen9b"}


def test_servers_must_exist_and_be_of_the_right_kind():
    score.validate({**score.CONFIG, "base_servers": ["olmo7b", "qwen08b"], "chat_servers": ["olmo32t"]})
    with pytest.raises(ScoreError, match="not in models.toml"):
        score.validate({**score.CONFIG, "base_servers": ["olmo7"]})
    with pytest.raises(ScoreError, match="qwen9c is a chat model"):
        score.validate({**score.CONFIG, "base_servers": ["qwen9c"]})
    with pytest.raises(ScoreError, match="olmo7b is a base model"):
        score.validate({**score.CONFIG, "chat_servers": ["olmo7b"]})
    with pytest.raises(ScoreError, match="distinct"):
        score.validate({**score.CONFIG, "chat_servers": ["qwen27", "qwen27"]})


def test_config_from_command_line():
    cfg = score.parse_config(["--predictor_base", "false", "--subsets", "human", "highpower", "--data_types", "New", "Secondary", "--human_accuracy", "hedged",
                              "--chat_servers", "qwen27", "qwen9c"])
    assert cfg["predictor_base"] is False and cfg["subsets"] == ["human", "highpower"]
    assert cfg["chat_servers"] == ["qwen27", "qwen9c"] and cfg["base_servers"] == ["olmo32"]
    assert cfg["data_types"] == ["New", "Secondary"] and cfg["human_accuracy"] == "hedged"
    with pytest.raises(SystemExit):
        score.parse_config(["--subsets", "everything"])
    with pytest.raises(ScoreError):
        score.validate({**score.CONFIG, "subsets": []})


def test_main_writes_all_outputs(tmp_path, capsys):
    out = tmp_path / "results"
    argv = ["--out_dir", str(out), "--cache_file", str(tmp_path / "ppl.jsonl"), "--qwen_cache_file", str(tmp_path / "qwen.jsonl"), "--data_types", "all", "New"]
    assert score.main(argv, scorer=FakeScorer(), qwen_client=FakeQwen()) == 0
    files = sorted(p.name for p in out.iterdir())
    olmo = [f for f in files if f.startswith("olmo32_")]
    assert len(olmo) == 24 and "olmo32_highpower_New_with_original_finding_yesno_loo.csv" in olmo
    assert "olmo32_human_all_claim_only_yesno_fitted.csv" in olmo
    assert "replicats_human_New_threshold.csv" in files and "markets_highpower_all_threshold.csv" in files
    assert len([f for f in files if f.startswith(("replicats_", "markets_"))]) == 8
    with open(out / "olmo32_all_all_claim_only_yesno_fitted.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == score.OLMO_YES_NO_FILE_COLUMNS and len(rows) == 274
    assert [int(r["row"]) for r in rows] == list(range(1, 275))
    with open(out / "replicats_human_all_threshold.csv", newline="") as f:
        assert tuple(next(csv.DictReader(f))) == score.HUMAN_FILE_COLUMNS
    with open(out / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    assert tuple(summary[0]) == score.SUMMARY_COLUMNS and len(summary) == 24 + 8 + 48
    assert {(r["predictor"], r["server"]) for r in summary} == {("base", "olmo32"), ("chat", "qwen27"), ("replicats", ""), ("markets", "")}
    with open(out / "olmo32_all_all_with_original_finding_yesno_loo.csv", newline="") as f:
        assert tuple(next(csv.DictReader(f))) == score.OLMO_YES_NO_FILE_COLUMNS
    qwen = [f for f in files if f.startswith("qwen27_")]
    assert len(qwen) == 48 and "qwen27_highpower_New_with_original_finding_think_token_threshold.csv" in qwen
    with open(out / "qwen27_all_all_claim_only_nothink_stated_threshold.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == score.QWEN_FILE_COLUMNS and len(rows) == 274
    assert "weighted_accuracy" in capsys.readouterr().out


def test_main_with_several_servers_keeps_their_results_apart(tmp_path):
    base = {"olmo7b": FakeScorer(), "qwen9b": FakeScorer()}
    base["qwen9b"].model = "fake-qwen-base"
    chat = {"olmo7i": FakeQwen(), "qwen9c": FakeQwen()}
    chat["qwen9c"].model = "fake-qwen-chat"
    out = tmp_path / "r"
    argv = ["--out_dir", str(out), "--cache_file", str(tmp_path / "ppl.jsonl"), "--qwen_cache_file", str(tmp_path / "q.jsonl"),
            "--base_servers", "olmo7b", "qwen9b", "--chat_servers", "olmo7i", "qwen9c", "--subsets", "highpower",
            "--template_claim_only", "false", "--qwen_thinking_on", "false",
            "--predictor_replicats", "false", "--predictor_markets", "false"]
    assert score.main(argv, scorer=base, qwen_client=chat) == 0
    assert [f.calls for f in base.values()] == [548, 548] and [f.calls for f in chat.values()] == [57, 57]  # base perplexities cover all 274 rows, chat responses only the rows used
    files = {p.name for p in out.iterdir()}
    assert {"olmo7b_highpower_all_with_original_finding_yesno_loo.csv", "qwen9b_highpower_all_with_original_finding_yesno_fitted.csv",
            "olmo7i_highpower_all_with_original_finding_nothink_token_threshold.csv", "qwen9c_highpower_all_with_original_finding_nothink_stated_threshold.csv"} <= files
    with open(out / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    assert [r["server"] for r in summary] == ["olmo7b"] * 2 + ["qwen9b"] * 2 + ["olmo7i"] * 2 + ["qwen9c"] * 2


# ---------------------------------------------------------------- chat models


class FakeQwen:
    """Stands in for QwenClient: a deterministic probability per prompt; responses are cut off when max_tokens is below `needs`."""

    model = "fake-qwen"

    def __init__(self, needs=0):
        self.needs = needs
        self.calls = 0
        self.budgets = []

    def ask(self, prompt, thinking, max_tokens, budget=0):
        self.calls += 1
        self.budgets.append(budget)
        if max_tokens < self.needs:
            return {"content": None, "reasoning": "...", "finish_reason": "length", "completion_tokens": max_tokens, "answer_top_logprobs": None}
        p = int(hashlib.sha256((prompt + thinking).encode()).hexdigest(), 16) % 101
        word = "yes" if p >= 50 else "no"
        q = min(max(p / 100, 0.01), 0.99)
        return {
            "content": f"Answer: {word}\nProbability: {p}", "reasoning": "thought" if thinking == "on" else "",
            "finish_reason": "stop", "completion_tokens": 10, "answer_top_logprobs": [[" yes", math.log(q)], [" no", math.log(1 - q)]],
        }


class ForcingQwen(FakeQwen):
    """FakeQwen that is always cut off (as by max_tokens) and can be forced to answer."""

    def __init__(self):
        super().__init__(needs=10**9)
        self.forced = []

    def ask(self, prompt, thinking, max_tokens, budget=0):
        r = super().ask(prompt, thinking, max_tokens, budget)
        return {**r, "content": "long trace", "reasoning": "", "answer_top_logprobs": [[" no", -0.5], [" yes", -1.0]]}  # no end-of-reasoning marker: the trace arrives as content, as with Olmo Think

    def force(self, prompt, trace, cap):
        self.forced.append((trace, cap))
        return {"content": "Answer: no\nProbability: 30", "finish_reason": "stop", "answer_top_logprobs": [[" no", -0.1], [" yes", -2.4]], "trace_tokens": cap}


def test_unanswered_responses_are_forced_once_and_kept(tmp_path, records):
    cache = tmp_path / "qwen.jsonl"
    fake = ForcingQwen()
    resp = score.get_qwen_responses(records[:3], ["with_original_finding"], ["on"], cache, "qwen27", 2, 100, fake, force=True)
    assert fake.calls == 3 and fake.forced == [("long trace", 100)] * 3
    e = resp[("with_original_finding", "on", records[0].row)]
    assert (e["answer"], e["probability"], e["forced"], e["forced_trace_tokens"]) == ("no", 30, True, 100)
    assert (e["reasoning"], e["first_content"], e["first_finish_reason"], e["finish_reason"]) == ("long trace", "long trace", "length", "stop")
    assert score.get_qwen_responses(records[:3], ["with_original_finding"], ["on"], cache, "qwen27", 2, 100, fake, force=True) == resp
    assert fake.calls == 3 and len(fake.forced) == 3  # cached: neither asked nor forced again
    u = score.unforced(e)
    assert (u["answer"], u["content"], u["finish_reason"], u["reasoning"]) == (None, "long trace", "length", "")
    assert u["answer_top_logprobs"] == [[" no", -0.5], [" yes", -1.0]]  # the original's, not the forced answer's
    assert "forced" not in u and score.unforced(u) is u


def test_without_forcing_unanswered_responses_still_stop_the_run(tmp_path, records):
    fake = ForcingQwen()
    with pytest.raises(ScoreError, match="have no answer"):
        score.get_qwen_responses(records[:2], ["with_original_finding"], ["on"], tmp_path / "q.jsonl", "qwen27", 2, 100, fake)
    assert fake.forced == []


def test_qwen_prompt(records):
    r = records[0]
    for t in score.TEMPLATES:
        ctx = score.context(r, t)
        head = ctx.removesuffix(score.QWEN_INSTRUCTION + "\n\nAnswer:")
        assert score.qwen_prompt(r, t) == head + (
            "Answer yes or no, and give the probability from 0 to 100 that the replication succeeded, in exactly this format:\n"
            "Answer: yes or no\n"
            "Probability: a whole number from 0 to 100"
        )
    assert r.finding in score.qwen_prompt(r, "with_original_finding") and r.finding not in score.qwen_prompt(r, "claim_only")


def test_qwen_prompts_never_contain_replication_finding(records):
    findings = {row[score.COLUMNS.index("Claim ID")]: row[score.COLUMNS.index("Replication finding")] for row in _table()[1:]}
    for r in records:
        for t in score.TEMPLATES:
            assert findings[r.claim_id] not in score.qwen_prompt(r, t)


def test_parse_answer():
    assert score.parse_answer("Answer: yes\nProbability: 72") == ("yes", 72)
    assert score.parse_answer("Answer: no\nProbability: 0\n") == ("no", 0)
    assert score.parse_answer("Answer: no\nProbability: 100") == ("no", 100)
    for bad in (None, "", "Answer: maybe\nProbability: 5", "Answer: yes\nProbability: 101", "Answer: yes"):
        assert score.parse_answer(bad) is None


def test_token_probability():
    p, note = score.token_probability([[" no", -0.01], [" yes", -5.01], [" ", -11.3]])
    assert p == approx(1 / (1 + math.exp(5.0))) and note == ""
    p, _ = score.token_probability([["yes", math.log(0.2)], [" yes", math.log(0.2)], [" no", math.log(0.4)]])
    assert p == approx(0.5)  # spellings of the same word are summed
    p, note = score.token_probability([[" yes", -0.1], [" y", -4.0], ["3", -9999.0]])
    assert p == approx(1 / (1 + math.exp(-3.9))) and note == "no missing"  # floor is the lowest unmasked value


def test_get_qwen_responses_computes_then_reuses(tmp_path, records, monkeypatch):
    cache = tmp_path / "qwen.jsonl"
    fake = FakeQwen()
    resp = score.get_qwen_responses(records[:5], ["claim_only"], ["off"], cache, "qwen27", 4, 100, fake)
    assert fake.calls == 5 and len(resp) == 5
    e = resp[("claim_only", "off", 1)]
    assert e["answer"] in ("yes", "no") and 0 <= e["probability"] <= 100 and e["max_tokens"] == 100
    assert score.get_qwen_responses(records[:5], ["claim_only"], ["off"], cache, "qwen27", 4, 100, fake) == resp
    assert fake.calls == 5
    score.get_qwen_responses(records[:5], ["claim_only"], ["off", "on"], cache, "qwen27", 4, 100, fake)
    assert fake.calls == 10


def test_qwen_cut_off_responses_stop_the_run_and_are_retried_with_a_higher_cap(tmp_path, records):
    cache = tmp_path / "qwen.jsonl"
    fake = FakeQwen(needs=500)
    with pytest.raises(ScoreError, match="3 responses from qwen27 have no answer"):
        score.get_qwen_responses(records[:3], ["claim_only"], ["on"], cache, "qwen27", 2, 100, fake)
    with pytest.raises(ScoreError):  # same cap: the cut-off responses are not requested again
        score.get_qwen_responses(records[:3], ["claim_only"], ["on"], cache, "qwen27", 2, 100, fake)
    assert fake.calls == 3
    resp = score.get_qwen_responses(records[:3], ["claim_only"], ["on"], cache, "qwen27", 2, 1000, fake)
    assert fake.calls == 6 and all(e["answer"] for e in resp.values())


class FailingQwen(FakeQwen):
    """FakeQwen whose request for one row fails, as a request rejected by the server does."""

    def __init__(self, bad_prompt):
        super().__init__()
        self.bad_prompt = bad_prompt

    def ask(self, prompt, thinking, max_tokens, budget=0):
        if prompt == self.bad_prompt:
            self.calls += 1
            raise ScoreError("Qwen request failed: 400 Bad Request")
        return super().ask(prompt, thinking, max_tokens, budget)


def test_a_failed_request_does_not_lose_the_others(tmp_path, records):
    cache = tmp_path / "qwen.jsonl"
    fake = FailingQwen(score.qwen_prompt(records[1], "claim_only"))
    with pytest.raises(ScoreError, match=f"1 of 5 requests to qwen27 failed; the others are cached. row {records[1].row} "):
        score.get_qwen_responses(records[:5], ["claim_only"], ["off"], cache, "qwen27", 2, 100, fake)
    assert fake.calls == 5 and len(score.load_cache(cache)) == 4
    fake.bad_prompt = None  # a rerun requests only the failed row
    assert len(score.get_qwen_responses(records[:5], ["claim_only"], ["off"], cache, "qwen27", 2, 100, fake)) == 5
    assert fake.calls == 6


def test_qwen_client_fits_long_prompts_to_the_context(monkeypatch):
    monkeypatch.setattr(score.serving, "endpoint", lambda name: ("http://x/v1", "m"))
    count = {"n": 0}

    class R:
        status_code = 200

        def json(self):
            return {"count": count["n"], "max_model_len": 32768}

    monkeypatch.setattr(score.httpx, "post", lambda *a, **k: R())
    c = REAL_QWEN_CLIENT("qwen27")
    count["n"] = 700
    assert c.fit("p", 32064, 32000) == (32064, 32000)
    count["n"] = 800
    assert c.fit("p", 32064, 32000) == (31968, 31904)
    assert c.fit("p", 32064, 0) == (31968, 0)
    count["n"] = 32768
    with pytest.raises(ScoreError, match="leaves no room"):
        c.fit("p", 100, 0)


def test_reasoning_capped_uses_the_budget_actually_sent():
    assert score.reasoning_capped({"reasoning_tokens": 31903, "thinking_budget_used": 31904}, 32000)
    assert not score.reasoning_capped({"reasoning_tokens": 31903}, 32000)


def test_main_requests_only_rows_qwen_runs_use(tmp_path):
    fake = FakeQwen()
    argv = ["--out_dir", str(tmp_path / "r"), "--qwen_cache_file", str(tmp_path / "q.jsonl"), "--predictor_base", "false",
            "--subsets", "highpower", "--qwen_thinking_on", "false", "--template_claim_only", "false"]
    assert score.main(argv, qwen_client=fake) == 0
    assert fake.calls == 57


def _resp(rows, answers, probs, top_yes):
    return {("claim_only", "off", r.row): {"answer": a, "probability": p, "answer_top_logprobs": [[" yes", math.log(y)], [" no", math.log(1 - y)]]}
            for r, a, p, y in zip(rows, answers, probs, top_yes)}


def test_run_qwen_by_hand():
    rows = [_rec(1, "A", True), _rec(2, "A", False), _rec(3, "B", False)]
    resp = _resp(rows, ["yes", "yes", "no"], [80, 40, 30], [0.9, 0.6, 0.2])
    out, s = score.run_qwen(score.Run("chat", "all", "all", "claim_only", thinking="off", qwen_score="stated"), rows, resp, "threshold")
    assert [o["prediction"] for o in out] == ["succeeded", "succeeded", "failed"]
    assert [o["credit"] for o in out] == [1, 0, 1] and [o["answer_disagrees"] for o in out] == [0, 1, 0]
    assert s["weighted_accuracy"] == approx((0.5 + 0 + 1) / 2)  # weights 0.5, 0.5, 1
    assert s["weighted_auc"] == approx(1.0)
    out, s = score.run_qwen(score.Run("chat", "all", "all", "claim_only", thinking="off", qwen_score="token"), rows, resp, "hedged")
    assert [o["score"] for o in out] == [approx(0.9), approx(0.6), approx(0.2)]
    assert s["weighted_accuracy"] == approx((0.5 * 0.9 + 0.5 * 0.4 + 1 * 0.8) / 2)
    assert list(out[0]) == list(score.QWEN_FILE_COLUMNS)


# ---------------------------------------------------------------- Olmo with the yes/no ending 


def test_olmo_yes_no_context(records):
    for t in score.TEMPLATES:
        assert score.context(records[0], t, "yes_no") == score.qwen_prompt(records[0], t) + "\n\nAnswer:"


def test_run_olmo_yes_no_uses_log_ratio():
    rows = [_rec(1, "A", True), _rec(2, "B", False)]
    ppl = {("claim_only", 1, " yes"): 2.0, ("claim_only", 1, " no"): 8.0, ("claim_only", 2, " yes"): 10.0, ("claim_only", 2, " no"): 5.0}
    out, s = score.run_olmo(score.Run("base", "all", "all", "claim_only", "fitted", ending="yes_no"), rows, ppl)
    assert [o["score"] for o in out] == [approx(math.log(4)), approx(math.log(0.5))]
    assert s["weighted_auc"] == approx(1.0) and s["weighted_accuracy"] == approx(1.0)
    assert list(out[0]) == list(score.OLMO_YES_NO_FILE_COLUMNS)


def test_get_perplexities_yes_no_ending(tmp_path, records):
    fake = FakeScorer()
    ppl = score.get_perplexities(records[:3], ["claim_only"], tmp_path / "ppl.jsonl", "olmo32", 2, fake, ["yes_no"])
    assert fake.calls == 6 and len(ppl) == 6
    assert {c for (_, _, c) in ppl} == {" yes", " no"}


def test_qwen_thinking_budget_is_sent_and_keyed(tmp_path, records):
    cache = tmp_path / "qwen.jsonl"
    fake = FakeQwen()
    score.get_qwen_responses(records[:2], ["claim_only"], ["off", "on"], cache, "qwen27", 1, 5000, fake, budget=4096)
    assert sorted(fake.budgets) == [0, 0, 4096, 4096]  # never sent with thinking off
    score.get_qwen_responses(records[:2], ["claim_only"], ["off", "on"], cache, "qwen27", 1, 5000, fake, budget=2048)
    assert fake.calls == 6  # a new budget requests only the thinking-on responses again
    e = json.loads(cache.read_text().splitlines()[-1])
    assert e["thinking_budget"] == 2048


def test_reasoning_capped_counts_reasoning_tokens_only():
    # vLLM counts its inserted </think> against the budget, so capped reasoning has budget - 1 tokens
    assert score.reasoning_capped({"reasoning_tokens": 16383, "completion_tokens": 16394}, 16384)
    # row 189 of the 16k run: reasoning ended on its own just short of the cap, with completion tokens over it
    assert not score.reasoning_capped({"reasoning_tokens": 16379, "completion_tokens": 16390}, 16384)
    assert not score.reasoning_capped({"reasoning_tokens": 0, "completion_tokens": 10}, 0)


def test_budget_must_leave_room_for_the_answer():
    with pytest.raises(ScoreError):
        score.validate({**score.CONFIG, "qwen_thinking_budget": 4096, "qwen_max_tokens": 4100})
    score.validate({**score.CONFIG, "qwen_thinking_budget": 0, "qwen_max_tokens": 100})
