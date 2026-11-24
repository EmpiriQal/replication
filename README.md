# Large language models predict which studies replicate by pattern recognition

Code, model outputs and analyses for the paper "Large language models predict which studies replicate by pattern recognition" by Bradley C. Love (EmpiriQal, Inc.), under review.

Language models were asked whether each of the 274 SCORE replications succeeded, given the original paper's abstract, the claim tested and the original finding, and their predictions were compared with the outcomes and with human forecasters. This repository contains the code that sent the requests, every model output used in the paper, and the code that computes every number and figure in the paper and its Supplementary Information from those outputs.

## Contents

| Path | What it is |
|---|---|
| `score.py` | Asks chat models (token score and stated score) and base models (log odds of " yes" against " no") about each claim; caches the outputs; computes AUC and accuracy for models and human aggregates |
| `thinking32k.py` | Qwen3.8 27B with reasoning capped at 32,000 tokens for the high-power claims whose reasoning did not finish within 16,384 tokens, with forced answers |
| `olmo_top_tokens.py` | Olmo 3 32B's unconstrained top 20 next tokens after "Answer:" (answer-word variants) |
| `recall_probe.py` | Citation-only prompt and the direct question about the SCORE paper |
| `memorization.py` | Completion of numbers from the replication teams' reports |
| `leak_search.py` | OpenAlex search for replications published before 8 January 2026 |
| `plausibility_demo.py` | The eight fictitious papers at p = .001 and p = .030 |
| `gemini_stated.py` | Gemini 3.1 Pro and Gemini 3 Flash on Vertex AI, stated probability |
| `individual_humans.py` | Builds `data/individual_estimates.csv` from the repliCATS and Replication Markets releases |
| `analysis.py` | Every number in the paper and Supplementary Information, from the cached outputs (no model requests) |
| `figures.py` | Figures 1 to 5 (`fig1`, `fig2`, `fig3_v2`, `fig4`, `fig5`) and the three supplementary figures, as PDF in `figures/`; each plotted value is checked against the paper |
| `render_templates.py` | Writes `chat_templates/`: each chat model's chat template applied to one claim |
| `serving.py`, `models.toml` | The local models: model ID, URL, and the `vllm serve` command each was served with |
| `gemini.py`, `openalex.py` | Clients for Vertex AI and OpenAlex |
| `data/` | The SCORE claims and outcomes, individual human forecasts, and the 17 claims published early; see `data/README.md` |
| `results/` | Cached model outputs (JSON lines) and run logs |
| `chat_templates/` | Rendered chat templates |
| `tests/` | Unit tests (no model servers or network needed) |

## System requirements

- Analyses, figures and tests: any operating system with Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU. Package versions are pinned in `uv.lock` (numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pandas 3.1.0, httpx 0.28.1, openai 3.31.0, pyreadr 0.5.9, google-genai 3.3.0).
- Tested on Ubuntu 24.04.5 LTS (aarch64) on an NVIDIA DGX Spark, with uv 0.12.18.
- Running the models (not needed for the analyses): each local model was served by vLLM 0.27.1 in NVIDIA's container `nvcr.io/nvidia/vllm:26.08-py3` on one NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory). The Gemini runs need a Google Cloud project with Vertex AI. The leak search uses the OpenAlex API.

## Installation

```
git clone https://github.com/EmpiriQal/replication.git
cd replication
uv sync
```

Installation takes under a minute on a desktop computer with an internet connection (uv downloads Python 3.12 if it is not installed).

## Demo: reproduce every number and figure

```
uv run python analysis.py      # every number, printed by section (about 3 minutes)
uv run python figures.py       # the figures, written to figures/ (about 1 minute)
uv run pytest                  # unit tests (a few seconds)
```

Times were measured on the DGX Spark's CPU (20 Arm cores); the demo uses no GPU and should take a few minutes on a typical desktop computer.

Expected output: `analysis.py` prints about 1,170 lines, section by section, containing every number reported in the paper and its Supplementary Information; `figures.py` writes `fig1.pdf`, `fig2.pdf`, `fig3_v2.pdf`, `fig4.pdf`, `fig5.pdf`, `figS_humanset.pdf`, `figS_forecasters30.pdf` and `figS_ablation30.pdf` to `figures/`, printing each checked value; `pytest` reports 96 passed.

`analysis.py` reads only the files in `data/` and `results/` and sends no requests. A single section runs with `uv run python analysis.py <section>`; `--list` lists them. `figures.py` stops if any plotted value differs from the paper.

Where each section's numbers appear in the paper (S: Supplementary Information):

| Section of `analysis.py` | In the paper |
|---|---|
| `qwen_off`, `baseline` | Results (AUC of Qwen3.8 without reasoning); S7 |
| `individuals` | Results (typical individual, aggregates); Fig. 2a; S9, S11; Supplementary Fig. S1a |
| `individual_tests` | Results (typical individual, each forecaster); Fig. 2b; S9, S10; Supplementary Figs. S1b and S2 |
| `qwen_agreement` | Results (agreement with the humans); S8 |
| `recall`, `memorization` | Results and Methods (tests of contamination); S14 |
| `leaked` | Results (claims published before the outcomes were public); Discussion (Gemini) |
| `ablation` | Results (prompt ablation); Fig. 3; S12; Supplementary Fig. S3 |
| `thinking_4k`, `thinking_complete`, `traces`, `cost` | Results and Methods (reasoning); Fig. 4; S6 |
| `consistency` | Results (stated probability); S20 |
| `model_set`, `base_chat`, `size` | Results (model size); Fig. 5a; S15, S16, S18 |
| `olmo_think` | Results (reasoning traces); S17 |
| `confidence` | Results (confidence); Fig. 5b; S13 |
| `olmo_yesno`, `variants`, `precision` | Methods (readouts); S5 |
| `plausibility` | Results and Methods (fictitious papers); S21 |
| `pvalue_baseline` | Results and Methods (p-value baseline) |
| `ceiling` | Discussion; S19 |

The Gemini AUCs are also printed by `uv run python gemini_stated.py --report`.

## Running the models

Every script requests only the outputs missing from its cache in `results/`. With the caches as distributed, each command below contacts no server and recomputes its summaries. To request outputs again, remove or rename the cache file.

Each local model must be served at the URL given for it in `models.toml`, with the `serve` command given there (in the container above, with the model's Hugging Face weights). The settings sent with each request (temperature 0, top 20 log-probabilities, answer format enforced by structured output, token caps) are in `score.py` and described in Supplementary Sections S3 to S6.

| Run | Command | Run time on the DGX Spark |
|---|---|---|
| Qwen3.8 27B without reasoning and with reasoning capped at 4,096 tokens, full and no-finding prompts, all claims; Olmo 3 32B base | `uv run python score.py` | About 9 hours for the reasoning runs |
| Prompt ablation, Qwen3.8 27B and Qwen3 30B-A3B Instruct | `uv run python score.py --predictor_base false --chat_servers qwen27 qwen30i --qwen_thinking_on false --template_no_abstract true --template_claim_alone true` | 2.0 and 1.0 minutes |
| Extended model set | `uv run python score.py --base_servers olmo7b qwen08b qwen2b qwen4b qwen9b qwen30b olmo32 --chat_servers olmo7i olmo32i qwen08c qwen2c qwen4c qwen9c qwen30i --qwen_thinking_on false --template_claim_only false --qwen_max_tokens 1024 --qwen_thinking_budget 0` | Minutes per model |
| Qwen3.8 27B, reasoning capped at 16,384 tokens, high-power set | `uv run python score.py --predictor_base false --chat_servers qwen27 --subsets highpower --template_claim_only false --qwen_thinking_off false --qwen_thinking_budget 16384 --qwen_max_tokens 20480 --qwen_force_answer true` | 325 minutes |
| Qwen3.8 27B, reasoning capped at 32,000 tokens, unfinished high-power claims | `uv run python thinking32k.py` | 292 minutes |
| Olmo 3.1 32B Think | `uv run python score.py --predictor_base false --chat_servers olmo32t --template_claim_only false --qwen_thinking_off false --qwen_thinking_budget 0 --qwen_max_tokens 60000 --qwen_force_answer true` | |
| Answer-word variants, Olmo 3 32B base | `uv run python olmo_top_tokens.py` | |
| Citation-only probe and direct question | `uv run python recall_probe.py --server qwen27`, and the same with `--server qwen30i` | |
| Memorization of the reports | `uv run python memorization.py --server qwen27`, and the same with `--server qwen30i`, each with `--variant bare` and `--variant header_2digits` | |
| Fictitious papers | `uv run python plausibility_demo.py` (`--prompts` prints the 16 prompts) | |
| Gemini 3.1 Pro and Gemini 3 Flash | `uv run python gemini_stated.py`, after `gcloud auth application-default login` and with `GOOGLE_CLOUD_PROJECT` set to a project with Vertex AI | |
| Search for early publication of outcomes | `uv run python leak_search.py` (optionally with `OPENALEX_API_KEY` and `OPENALEX_MAILTO` set); the candidates it prints were read to make `data/leaked_claims.csv` | |
| Rendered chat templates | `uv run --with transformers --with jinja2 python render_templates.py` (reads the tokenizers from the Hugging Face cache) | |

Generated text from the Gemini models is not exactly reproducible; the cached replies are those used in the paper.

`score.py` writes per-claim CSVs and `summary.csv` to `results/` (options: `uv run python score.py --help`).

To run the models on other claims, give `score.py` a CSV with the same 23 columns as `data/SCORE_replicated_claims.csv` (`--data_file`) and new cache files (`--cache_file`, `--qwen_cache_file`). The prompts use the Abstract, Claim tested and Original finding columns; AUC and accuracy use Replicated. The integrity checks in `score.load_data` are specific to the SCORE file and must be removed for other data.

## Individual human forecasts

`data/individual_estimates.csv` is built by `uv run python individual_humans.py` from two public releases, unpacked into `data/raw/`:

- the repliCATS R data package at commit 8698fcd, https://github.com/metamelb-repliCATS/repliCATS (zip from https://api.github.com/repos/metamelb-repliCATS/repliCATS/zipball/8698fcd, unpacked as `data/raw/metamelb-repliCATS-repliCATS-8698fcd/`);
- the repliCATS-SCORE OSF dataset, https://osf.io/dax9j (all files as a zip, unpacked as `data/raw/dax9j-osfstorage-archive/`).

## License

The code is released under the MIT License (`LICENSE`). The model outputs in `results/`, `data/individual_estimates.csv`, `data/leaked_claims.csv` and the rendered templates in `chat_templates/` are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). `data/SCORE_replicated_claims.csv` is derived from the SCORE, repliCATS and Replication Markets materials listed in `data/README.md`; please cite those sources when using it.
