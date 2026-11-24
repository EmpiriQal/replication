"""Individual human forecasts for the 274 claims, written as a tidy file: data/individual_estimates.csv, one row per claim, source and rater.

Sources (unpacked into data/raw/; see README.md):
- repliCATS_R1: the repliCATS R data package (https://github.com/metamelb-repliCATS/repliCATS, commit 8698fcd; Zenodo https://doi.org/10.5281/zenodo.18870926), data/p1_ratings.rda, rows with round "round_1", question "direct_replication" and element "three_point_best". Round 1 is each rater's private estimate before group discussion.
- RM_survey: the repliCATS-SCORE OSF dataset (https://osf.io/dax9j, downloaded 31 October 2026), data/Markets_demographics_accuracy_data.csv, column q1_replication_estimate_imputed, missing values dropped. The file repeats a rater's single estimate for a paper once per replication of that paper; one row per rater and paper is kept.

Join: the part of Claim ID before the first underscore equals paper_id in both files, matched with case. Estimates are kept as given, 0 to 100.

Usage: uv run python individual_humans.py
"""

import pandas as pd
import pyreadr

import score as S

RAW = S.ROOT / "data/raw"
OUT = S.ROOT / "data/individual_estimates.csv"


def replicats_r1() -> pd.DataFrame:
    d = pyreadr.read_r(RAW / "metamelb-repliCATS-repliCATS-8698fcd/data/p1_ratings.rda")["p1_ratings"]
    d = d[(d["round"] == "round_1") & (d["question"] == "direct_replication") & (d["element"] == "three_point_best")]
    assert not d.duplicated(["paper_id", "user_name"]).any()
    return pd.DataFrame({"paper_id": d["paper_id"], "rater": d["user_name"], "estimate": d["value"]})


def markets_survey() -> pd.DataFrame:
    d = pd.read_csv(RAW / "dax9j-osfstorage-archive/data/Markets_demographics_accuracy_data.csv", low_memory=False)
    d = d.dropna(subset=["q1_replication_estimate_imputed"])
    # the repeats of a rater and paper all carry the same estimate
    assert (d.groupby(["user_id", "paper_id"])["q1_replication_estimate_imputed"].nunique() == 1).all()
    d = d.drop_duplicates(["user_id", "paper_id"])
    return pd.DataFrame({"paper_id": d["paper_id"], "rater": d["user_id"].astype(str), "estimate": d["q1_replication_estimate_imputed"]})


def main() -> None:
    recs = S.load_data(S.ROOT / "data/SCORE_replicated_claims.csv")
    claims = pd.DataFrame({"row": [r.row for r in recs], "claim_id": [r.claim_id for r in recs], "paper_id": [r.paper_id for r in recs]})
    parts = []
    for source, d in (("repliCATS_R1", replicats_r1()), ("RM_survey", markets_survey())):
        parts.append(claims.merge(d, on="paper_id").assign(source=source))
    out = pd.concat(parts)[["row", "claim_id", "source", "rater", "estimate"]].sort_values(["row", "source", "rater"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(S.ROOT)}: {len(out)} estimates")
    for source, g in out.groupby("source"):
        n = g.groupby("row").size()
        for subset in ("human", "highpower"):
            rows = [r.row for r in S.select(recs, subset, "all")]
            m = n.reindex(rows).dropna()
            print(f"  {source:13} {subset:9} claims {len(m)} of {len(rows)}, raters per claim {int(m.min())} to {int(m.max())} (median {m.median():g}), distinct raters {g[g['row'].isin(rows)]['rater'].nunique()}")


if __name__ == "__main__":
    main()
