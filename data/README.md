# Data

| File | Contents | Source |
|---|---|---|
| `SCORE_replicated_claims.csv` | One row per claim that a SCORE replication team tested (274 claims from 164 papers): citation, abstract, claim tested, original finding, replication outcome, the human sets and the two forecasting teams' aggregate estimates. Described below. | Built from the SCORE and repliCATS materials listed in Section 7 |
| `individual_estimates.csv` | Each individual forecaster's estimate for each claim: repliCATS Round 1 and the Replication Markets survey. One row per claim, source and rater. | Written by `individual_humans.py` from the repliCATS R data package and the repliCATS-SCORE OSF dataset |
| `leaked_claims.csv` | The 17 claims whose replication was published in a journal article before the SCORE outcome data were released (8 January 2026), with the article. | Found with `leak_search.py` and by reading its candidates |

## 1. The SCORE data file

The data come from the SCORE program (Systematizing Confidence in Open Research and Evidence) and its papers "Investigating the replicability of the social and behavioral sciences" (Tyner et al., Nature, 2026) and "Large-scale human predictions of the replicability of published social and behavioural science papers" (Mody et al., 2026). Sources are listed in Section 7.

## 2. Background in plain terms

### 2.1 Claims and replications

A **claim** is one specific result reported in a published paper. A **replication** is a new study that tests the same claim with independent data, either by collecting new data ("new data" replication) or by analysing an existing dataset that the original study did not use ("secondary data" replication). A replication counts as **successful** if it finds a statistically significant result (p ≤ .05) in the same direction as the original.

### 2.2 What a row is

Each row of the data file is one claim that a replication team tested: 274 rows from 164 papers. A row is not necessarily a separate replication study. 160 replication studies tested a single claim each; 22 studies tested several claims each (2 to 18) using the same sample, which accounts for the other 114 rows. Example: Vermeulen et al. (2010) has 18 rows, all from one replication study of 1,096 participants, each testing a different cell of the paper's Table 2. 18 papers had more than one replication study.

### 2.3 The two rounds of SCORE

SCORE ran in two rounds. In the first round (Phase 1), one key claim was extracted from each of 3,000 papers; human forecasters predicted whether each would replicate; a sample of these claims was then replicated. In the second round (Phase 2), more claims were extracted and replicated, including studies that tested many claims from one paper. The 274 rows mix both rounds.

### 2.4 The human forecasters

Two independent teams, recruited mainly from graduate students and early-career researchers, estimated for each Phase 1 claim the probability that a high-powered, new data replication would find a statistically significant effect in the same direction as the original.

- **repliCATS (expert groups).** 497 participants from 66 countries. Each claim was rated by a small group (typically 4 or 5 people) who gave an estimate, discussed it, and gave a final estimate. The group estimate used here is the average of the members' final estimates (repliCATS method "ArMean").
- **Replication Markets (prediction market).** 331 active participants trading in a prediction market with cash prizes. The estimate used here is the final market price (Replication Markets method "cs_m1"), typically reflecting about 7 or 8 traders. Each market opened at a starting price set from the original p-value (0.8 for p ≤ .001, 0.4 for .001 < p ≤ .01, 0.3 for p > .01, 0.5 if no p-value), so prices are partly anchored on it; claims with no traders keep the starting price.

Every one of the 3,000 Phase 1 claims received a group estimate from both teams; individual people rated only some claims. These two methods are the ones the paper's authors used to compare the teams head to head.

### 2.5 The human sets

Human accuracy can only be measured on rated claims that were later replicated. Mody et al. scored the humans on two nested sets, always reporting new data and secondary data separately:

| Set | Rows | New data | Secondary data | Replicated |
|---|---|---|---|---|
| Human set (no power filter) | 135 | 84 | 51 | 66 (49%) |
| Human high-power set (replications with at least 80% power to detect half the original effect at α = .05) | 57 | 37 | 20 | 25 (44%) |

The paper's headline result is for the 37 high-powered new data replications. Because the file contains both teams' estimates and the outcomes for every row in these sets, human performance can be computed on any selection of rows, the same rows on which Olmo is scored. The paper's numbers serve as checks (Section 8.1).

### 2.6 The machine models

SCORE also funded three teams to build automated systems (described in Rajtmajer et al., 2026, "Innovations in machine assessment of replicability"): **Synthetic Markets** (Penn State; bot-populated prediction markets), **MACROSCORE** (University of Southern California; knowledge-graph and feature-based modelling) and **A+** (TwoSix Technologies; semantic parsing of argument structure). The systems were trained to predict human expert judgments of replicability, not replication outcomes: they were given about 2,400 Phase 1 claims with the repliCATS judgments for them. They were then tested in three sprints: Sprint 1 scored 600 Phase 1 claims, and Sprints 2 and 3 scored 450 Phase 2 claims each. In each sprint each team submitted several variants of its system (3 to 10), each giving a score from 0 to 1 for every claim. The scores are the machine's prediction of the human judgment, which is itself a probability of replication, so they are treated as the machine's estimated probability of replication. The paper evaluated the machines only against the human judgments, never against replication outcomes.

**Not used by score.py.** The machine-model columns (21 to 23) are kept in the data file for possible future use only. `score.py` does not read them, and no options, outputs or tests refer to them.

**Which scores are in this file.** For each team, the variant the SCORE program judged best in Sprint 1: the variant with the lowest root mean squared error (RMSE) against the repliCATS judgments (method ReasonWAgg) across the sprint's claims. These are Synthetic Markets `cs_m8`, MACROSCORE `cs_m3` and A+ `cs_m1`, with RMSE 0.151, 0.136 and 0.141. They reproduce the best Sprint 1 RMSE values reported in the paper's Table 1 (0.15, 0.14, 0.14).

**Coverage.** Machine scores are per paper, for the paper's Phase 1 claim. 27 rows have them: the rows with human estimates whose paper was in Sprint 1. 26 of these are in the human set (11 in the human high-power set); the 27th is 1574_single-trace (Goeree and Yariv), a rated claim outside the paper's validation set. Sprints 2 and 3 contribute no rows. Multi-claim rows from papers in Sprint 1 are not given the paper's score, since it refers to a different claim.

**Leakage.** The machine-model columns contain each team's preferred variant, as identified by the SCORE program: the variant whose scores best matched repliCATS human judgments (lowest RMSE) across all claims in that test sprint, including the items in this file. The models were trained on human judgments for about 2,400 other claims. The choice of variant therefore used human judgments of these items, so these columns are not clean out-of-sample predictions.


## 3. The data file

- File: `SCORE_replicated_claims.csv`, read-only; `score.py` only reads it.
- Encoding UTF-8 (no byte-order mark). Every field is enclosed in double quotes, with embedded double quotes doubled (Python `csv` module, `QUOTE_ALL`). Some text fields contain commas and line breaks, so the file must be read with a CSV parser (Python `csv` or `pandas.read_csv`), not split by lines.
- One header record, then 274 data records. 23 columns. No field is empty.
- All fields are text in the file; numeric columns must be converted when read (see Type in Section 4).
- A viewing copy, `SCORE_replicated_claims.xlsx` (worksheet `Replicated claims`), has the same content for browsing. The CSV is the source. Excel cannot store more than about 16 significant digits, so a few values in the .xlsx differ from the CSV in the 17th digit (one p-value and nine repliCATS estimates). The CSV should not be edited or resaved from Excel.

Outcome counts across all 274 rows: Replicated "Yes" 151, "No (n.s.)" 98, "No (opposite)" 24, "No (p n/a)" 1. New data 128 rows (70 replicated), secondary data 146 rows (81 replicated).


## 4. Column dictionary

SCORE source objects: `repli_outcomes`, `orig_outcomes` and `paper_metadata` are data frames in `analyst data.RData`; `extracted_claims.csv` is in the parent SCORE project. Human data come from the repliCATS team's evaluation files (Section 7).

| # | Column | Type | Source | Notes |
|---|---|---|---|---|
| 1 | Paper | text | `paper_metadata.citation` | Short citation. Verbatim. |
| 2 | Journal | text | `paper_metadata.publication_standard` | Verbatim. |
| 3 | Year | integer | `paper_metadata.pub_year` | Verbatim. May differ by one from the citation year (online versus print). |
| 4 | Field | text | `paper_metadata.COS_pub_category` | Discipline. Verbatim. |
| 5 | DOI | text | `paper_metadata.DOI` | Verbatim. |
| 6 | Data | text | `repli_outcomes.repli_type` | "New" (SCORE "new data") or "Secondary" (SCORE "secondary data"). |
| 7 | Replicated | text | derived | Outcome label, the ground truth (Section 4.1). |
| 8 | Replication p | number, one text cell | `repli_outcomes.repli_p_value` | Exact SCORE values. One cell (Wetzel et al.) is "test not run". Caveats in Section 4.2. Not used by `score.py`. |
| 9 | Original N | number, some text cells | `orig_outcomes.orig_sample_size_value` | Units vary and are not in the file. 9 text cells: "n/a: meta-analysis" (8, Vadillo et al.), "not reported" (1, Bhattacharjee et al.). |
| 10 | Replication N | number, some text cells | `repli_outcomes.repli_sample_size_value` | Units vary. 8 cells "n/a: meta-analysis" (Vadillo et al.). |
| 11 | Claim ID | text | `repli_outcomes.claim_id` | Unique per row. The part before the first underscore is the paper ID (e.g. `7RR2_bd3w94` belongs to paper `7RR2`). Single-claim Phase 1 and Phase 2 rows end in `_single-trace`. |
| 12 | Claim tested | text | `extracted_claims.coded_claim3`, else `claim3_hyp` | Section 4.3. |
| 13 | Abstract | text | `paper_metadata.abstract` | Original paper's abstract. Verbatim. Repeats on every row of a multi-claim paper. |
| 14 | Original finding | text | `extracted_claims.coded_claim4` | Coder's statement of the specific original result tested, usually with its statistics. Verbatim except 23 invisible PDF control characters deleted from 11 cells. Unique on every row. |
| 15 | Replication finding | text | `repli_outcomes.repli_pattern_description` | Replication team's description of its result. Verbatim. Reveals the outcome, so it must never be used in a prompt. |
| 16 | Paper weight | number | derived | 1 / number of rows from the same paper among all 274. Sums to 164. `score.py` recomputes weights within each selection and does not use this column directly. |
| 17 | Human set | "yes" / "no" | derived (Section 4.4) | "yes" for the 135 rows of the human set. |
| 18 | Human high-power set | "yes" / "no" | derived (Section 4.4) | "yes" for the 57 rows of the human high-power set. |
| 19 | repliCATS estimate | number, or "not rated" | repliCATS `ArMean` | The expert group's estimated probability that a high-powered, new data replication of the claim would find a statistically significant effect in the same direction as the original (average of group members' final estimates). Verbatim from the source file. |
| 20 | Replication Markets estimate | number, or "not rated" | Replication Markets `cs_m1` | The prediction market's estimated probability of the same event (final market price). Verbatim from the source file, including storage noise such as 0.84000003. |
| 21 | Synthetic Markets estimate | number, or "not rated" | Penn State `corr_psu_cs_1.csv`, `cs_m8` | Synthetic Markets' estimated probability of replication for the claim (preferred Sprint 1 variant; Section 2.6). Verbatim. Values on 27 rows. Contains leakage (Section 2.6). Not used by `score.py`. |
| 22 | MACROSCORE estimate | number, or "not rated" | USC `corr_usc_cs_1.csv`, `cs_m3` | MACROSCORE's estimated probability of replication for the claim (preferred Sprint 1 variant). Verbatim. Values on 27 rows. Contains leakage. Not used by `score.py`. |
| 23 | A+ estimate | number, or "not rated" | TwoSix `corr_twosix_cs_1.csv`, `cs_m1` | A+'s estimated probability of replication for the claim (preferred Sprint 1 variant). Verbatim. Values on 27 rows. Contains leakage. Not used by `score.py`. |

### 4.1 Replicated

| Label | Meaning | SCORE fields |
|---|---|---|
| Yes | Significant, same pattern | `repli_p_value` ≤ .05 and `repli_pattern_criteria_met` TRUE |
| No (n.s.) | Not significant | `repli_p_value` > .05 |
| No (opposite) | Significant, opposite pattern | `repli_p_value` ≤ .05 and `repli_pattern_criteria_met` FALSE |
| No (p n/a) | No p-value | Wetzel et al.: the focal test was never run because its prerequisite latent classes could not be recovered; SCORE scores it as a failure |

"Yes" agrees with SCORE's success flag `repli_score_criteria_met` on all 274 rows and with the human-forecasting paper's outcome data on all 135 human-set rows. For scoring, "Yes" means the correct answer is "succeeded"; every "No" label means "failed".

### 4.2 Caveats on Replication p

SCORE's data dictionary describes this field as an exact two-tailed value, which is often not the case. 55 rows have p = 0 (teams reported "p = 0.000" or a bound such as "p < 2e-16", "p < .001" or "p < 0.01"). 23 rows have p = 0.001, of which 17 are write-ups saying "p < .001". Of the 218 non-zero values, teams confirmed 185 as exact, said "no" for 19 and "not sure" for 14. These caveats do not affect `score.py`.

### 4.3 Caveats on Claim tested

133 rows use the coder's cleaned claim (`coded_claim3`); 141 rows have none in SCORE, so the raw hypothesis text copied from the PDF (`claim3_hyp`) is used, with PDF artifacts such as line-break hyphens. For multi-claim papers SCORE often stored the same paper-level hypothesis for every claim, so 83 rows share claim text with another row of the same paper (in 8 papers, 36 rows, identical text has different outcomes). This is acceptable for a probabilistic test. Whitespace was collapsed in 8 cells; no words were changed.

### 4.4 How the human columns were derived

The sets were rebuilt from the repliCATS team's evaluation code and data, reproducing the paper's counts and success rates exactly.

- **Human set.** Replications of rated claims that are SCORE's official result for the claim (`repli_version_of_record`) and carry the paper's inclusion flag `TA2_include_no_power_filter`. 130 of the 135 rows are single-claim rows; 5 are rows from multi-claim studies that SCORE marks as equivalent to the rated claim (`single_trace_equivalent`): 7RR2_g4xnzl, 9OK1_ppwpjr, BrGp_blv93q, D2LY_g81lpo, RYKv_g41jx8.
- **Human high-power set.** The paper's flag `TA2_include_singletrace_50%_alpha_0.05`. For one claim (AgO1, Rodríguez-Lara and Moreno-Garrido) the paper used a duplicate replication record (AgO1_gdvqwq) that is not SCORE's official result and is not in this file; the same claim's official record, AgO1_single-trace, is marked instead. Both records report success, and the human estimates are per claim, so the paper's figures are reproduced exactly. As in the paper, AgO1 is in the high-power set but not in the human set.
- **Estimate columns.** Filled for the 138 rows whose claim was rated (137 claims; RYKv's claim has two rows): the 135 human-set rows plus three rated claims not in the paper's validation set: RYKv_single-trace (the paper scored the same claim's other replication), AgO1_single-trace (see above) and 1574_single-trace (Goeree and Yariv; the paper's inclusion flag is blank, with no reason recorded). All other rows read "not rated". Values are matched by paper ID, since each team rated one claim per paper.
- The question both teams answered concerns a hypothetical high-powered new data replication, not the specific replication SCORE later ran; some actual replications used secondary data or had low power.


## 5. Quality checks performed on the data

- **ID consistency.** Every Claim ID belongs to its paper; each paper has one citation, DOI and abstract.
- **Metadata against Crossref.** All 164 DOIs: journal, first author and year matched (year within one). One DOI (Goeree and Yariv) lists no authors in Crossref; its title and journal match.
- **Abstracts against Crossref.** 75 papers have a Crossref abstract; all 75 match.
- **Claims against abstracts.** By word overlap (TF-IDF), 264 of 274 claim texts matched their own paper's abstract best or within the top three; the rest were inspected manually and are correct.
- **Column-by-column comparison with SCORE.** Only the differences in Section 6 exist.
- **Human data.** The human sets reproduce the paper's counts and success rates; outcomes agree with the paper's data on all 135 rows; the AUC and accuracy of the aggregates match the authors' output exactly.


## 6. Modifications relative to the sources

1. Rows filtered to SCORE's official results (version of record) for non-COVID papers, as in the replication paper's code (`analysis and knit.R`, lines 103 to 104). The full dataset has 427 rows; 35 COVID-paper claims are excluded.
2. Column headers named for readability.
3. Data: "new data" and "secondary data" relabeled "New" and "Secondary".
4. Replicated: labels derived from SCORE fields.
5. Missing values replaced by explanatory text: 9 Original N cells, 8 Replication N cells, 1 Replication p cell; "not rated" in 136 cells of each estimate column.
6. Claim tested: whitespace collapsed in 8 cells; `coded_claim3` or `claim3_hyp` chosen per row.
7. Original finding: 23 invisible control characters deleted from 11 cells.
8. Added columns: Paper weight, Human set, Human high-power set, repliCATS estimate, Replication Markets estimate, Synthetic Markets estimate, MACROSCORE estimate, A+ estimate.


## 7. Archival information

### 7.1 Sources

- SCORE replication paper materials (OSF `g5sny`): https://osf.io/g5sny/wiki?wiki=pa6zy. Analysis package `analysis data and code package 15 Dec 2025 1708.zip` (3,460,701 bytes; https://osf.io/download/5xfdy/): `analyst data.RData`, `analysis and knit.R`, `data-dictionary_replications.xlsx`, `template manuscript.docx`.
- SCORE parent project (OSF `dtzx4`), https://osf.io/dtzx4/: `Analysis-ready Data/Replication and Reproduction Papers/extracted_claims.csv`.
- Mody et al. (2026), "Large-scale human predictions of the replicability of published social and behavioural science papers", preprint, version 2, https://doi.org/10.31222/osf.io/vgyed_v2.
- repliCATS evaluation code and data for that paper (OSF `tcfb9`), https://osf.io/tcfb9/: `data/replication_outcomes_2025-05-08_plus_repliCATS_inclusion_columns.csv` (inclusion flags), `data/anon_p1_singletrace_confidence_scores.csv` (repliCATS estimates), `data/KeyW_CS_2020-10-19.csv` (Replication Markets estimates), `scripts/SCORE_special_issue_performance_evaluation_checks.R`, `scripts/SCORE_special_issue_comparison_between_teams.R`, `scripts/SCORE_performance_analysis_functions.R`, `outputs/performance_table_aggregation_method.csv`.
- repliCATS data package, https://github.com/metamelb-repliCATS/repliCATS, used to check which claims repliCATS rated across both rounds.
- Crossref REST API (https://api.crossref.org) for metadata and abstract checks.
- Rajtmajer et al. (2026), "Innovations in machine assessment of replicability" (manuscript), source of the machine-model descriptions, training and test design, and Table 1 (best variants by sprint).
- SCORE parent project (OSF `dtzx4`), `Analysis-ready Data/Correlations Paper/Confidence Scores/`: `corr_psu_cs_1.csv` (Synthetic Markets), `corr_usc_cs_1.csv` (MACROSCORE), `corr_twosix_cs_1.csv` (A+), Sprint 1 machine scores by paper. The repliCATS ReasonWAgg judgments used to identify each team's preferred variant came from the repliCATS data package (`p1_cs`).

### 7.2 How the data file was constructed

1. The SCORE analysis package was obtained from OSF; `analyst data.RData` was parsed and its data frames extracted.
2. `repli_outcomes` was filtered (Section 6, item 1) and joined to `orig_outcomes` and `extracted_claims.csv` by claim ID and to `paper_metadata` by paper ID. All IDs matched.
3. Columns 1 to 16 were assembled and checked (Section 5), first as `SCORE_replicated_claims.xlsx`, then converted to the CSV and checked field by field.
4. The human sets were rebuilt from the repliCATS evaluation files using the selection rules in the authors' script, and matched to the data file by claim ID (Section 4.4). The estimate columns were matched by paper ID.
5. The machine-model columns were taken from the Sprint 1 score files of each team, using the variant with the lowest RMSE against repliCATS ReasonWAgg across Sprint 1 (Section 2.6), matched by paper ID to the rows with human estimates.

### 7.3 Version history of the data

1. Initial build. In this version the replication p-values were rounded to two significant figures and stored as text, an unrequested change later reverted.
2. Explanatory labels added in 18 formerly blank cells.
3. Abstract, Original finding and Replication finding columns added.
4. Replication p-values restored to SCORE's exact values.
5. Paper weight column added.
6. Converted to the read-only CSV, which became the program input.
7. Human set and Human high-power set columns added.
8. repliCATS estimate and Replication Markets estimate columns added.
9. Synthetic Markets, MACROSCORE and A+ estimate columns added.
