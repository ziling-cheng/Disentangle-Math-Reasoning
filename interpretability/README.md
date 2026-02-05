# Interpretability

This folder contains the code for the **Section 5 interpretability experiments** (logit attribution + activation patching) and generates the corresponding plots:

- **Figure 5 (i–iv)**: logit attribution results
- **Figure 5 (iii, vi)**: activation patching results for abstraction / computation
- **Figure 6**: cross-prompt (cross-problem) patching results

---

## Contents

- `run_logit_attribution.py`: runs **logit attribution** and optionally produces plots (Figure 5 i–iv)
- `logit_attribution_utils.py`: helper functions for logit attribution
- `run_activation_patching.py`: runs **activation patching** for different patching regimes and optionally produces plots (Figure 5 iii, vi; Figure 6)
- `activation_patching_utils.py`: helper functions for activation patching
- `plotting_utils.py`: shared plotting utilities
- `config.py`: experiment configuration (paths, model/dataset settings, caching/output). Please set `CACHE_DIR` in `config.py`.
- `scripts/`: shell script to run all interpretability experiments.

---

## Data

These experiments assume the interpretability datasets are available under:

- `../interpretability_data/`

(See the `interpretability_data/` README section for dataset naming conventions and how they are generated.)

---

## Running logit attribution (Figure 5 i–iv)

`run_logit_attribution.py` runs logit attribution over the interpretability datasets and can generate the paper-style plots with `--plotting`.

```bash
python run_logit_attribution.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --plotting
python run_logit_attribution.py --model_id "Qwen/Qwen2.5-7B-Instruct" --plotting
python run_logit_attribution.py --model_id "Qwen/Qwen2.5-14B-Instruct" --plotting
```

---

## Running activation patching (Figure 5 iii, vi; Figure 6)
`run_activation_attribution.py` performs activation patching under three regimes controlled by `--patching_type`:
- `logic` (Section 5.1): perturb only the logic in minimally different pairs
- `computation` (Section 5.1): perturb only the numbers in minimally different pairs)
- `cross` (Section 5.2): patch hidden states across different problems (cross-prompt / cross-problem)


#### Logic patching
```bash
python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "logic" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "logic" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "logic" --plotting
```

#### Computation patching
```bash
python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "computation" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "computation" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "computation" --plotting
```

#### Cross-prompt patching
```bash
python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "cross" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "cross" --plotting
python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "cross" --plotting
```
