# Disentangled Evaluation

This directory contains the code for running the **Section 4 disentangled evaluation** and generating the summarization plots:

- **Performance without CoT** (Figure 3–style summary)
- **Performance change after adding CoT** (with-CoT − no-CoT; Table 2–style comparison, with a more detailed breakdown)

---

## Contents
- `instructions.py` — prompt templates for **zs** and **CoT** under four evaluation settings:
  - original
  - numerical abstraction
  - symbolic abstraction
  - arithmetic computation
- `paths.py` — path configuration (cache/output roots)
- `openai_key.txt` — API key for scoring **symbolic abstraction** via `gpt-4o-mini` (used by `run_evaluation.py`)
- `run_inference.py` — runs model inference and saves generations for each dataset × evaluation setting (4 subtasks) × (zs/cot) condition  
- `run_evaluation.py` — scores saved generations and computes accuracies for each subtask
- `evaluation_utils.py` — shared evaluation utilities for parsing and scoring
- `summarize_behaviour_results.py` — aggregates evaluation outputs and produces plots/tables:
  - Figure 3–style plot: **no-CoT** performance across models
  - detailed Table 2–style plot: **with-CoT − no-CoT** deltas per model and per setting
- `scripts/`:
  - `run_inference.sh`
  - `run_evaluation.sh`
  - `plot_results.sh`

---

## Setup

### 1) Set `CACHE_DIR` in `paths.py`:
In `paths.py`,  **replace** the placeholder `CACHE_DIR` with a valid directory on your machine (and adjust any other roots if needed). 
### 2) Provide an OpenAI key (required for symbolic abstraction scoring):
put your key in `openai_key.txt` if you want `run_evaluation.py` to score symbolic abstraction. By default, it is using `gpt-4o-mini`.

---  

## Workflow

### 1) Run inference (save generations)
```bash
bash scripts/run_inference.sh
```
This runs `run_inference.py` and writes model generations to `inference_results/` dir, for the set of models, datasets, prompt formats (defined in `instructions.py`) configured inside the shell script.

### 2) Evaluate generations (compute accuracies)
```bash
bash scripts/run_evaluation.sh
```
This runs `run_evaluation.py` to read the saved generations, and compute accuracy metrics for each evaluation setting (original, numerical abstraction, symbolic abstraction, arithmetic calculation).

Evaluation outputs are written as summary files under `inference_results/` dir used by the plotting step.

### 3) Aggregate and generate figures
```bash
bash scripts/plot_results.sh
```
This step aggregate evaluation summaries across conditions/models and exports:
- Figure 3-style plot: no-CoT performance across model families and sizes
- Table 2–style comparison (detailed, as plots): performance differences (with CoT vs. no CoT) per model per plot under the four evalution settings.

