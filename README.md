<h1 align="center">
  🤔 Can LLMs Reason Abstractly Over Math Word Problems Without CoT?
    Disentangling Abstract Formulation From Arithmetic Computation
</h1>


<h4 align="center">
  <a href="https://ziling-cheng.github.io/">Ziling Cheng</a>, 
  <a href="https://mcao516.github.io/">Meng Cao</a>, 
  <a href="https://linkedin.com/in/leila-pishdad">Leila Pishdad</a>, 
  <a href="https://yanshuaicao.github.io/">Yanshuai Cao</a>, 
  <a href="https://www.cs.mcgill.ca/~jcheung/">Jackie C.K. Cheung</a>
</h4>

<h4 align="center">
  <a href="https://github.com/ziling-cheng/Disentangle-Math-Reasoning">GitHub</a> •
  <a href="https://aclanthology.org/2025.emnlp-main.723/">Paper</a> •
  <a href="Huggingface">Hugging Face</a> •
  <a href="https://x.com/ziling_cheng/status/1986363306366673323">Twitter</a>
</h4>

This repository contains code and data for the EMNLP 2025 paper "Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation". It includes:
- **Disentangled behavioural evaluation (Section 4)** separating *abstract formulation* from *arithmetic computation*.
- **Mechanistic interpretability experiments (Section 5)** via logit attribution and activation patching (including cross-problem patching).

## Data
`data/` holds all datasets, including:
- Behavioural evaluation datasets (Section 4): disentangled test sets for GSM8K and SVAMP
  - `data/test_gsm8k.csv`
  - `data/test_svamp.csv`
- Interpretability datasets (Section 5): CSV files under `data/interpretability_data/`.


## Disentangled Evaluation
`disentangled_evaluation/` holds code for Section 4 behavioural evaluation:
- runs inference to save model generations under different prompt/settings,
- scores outputs for each sub-skill setting,
- aggregates results and exports plots/tables (Figure 3–style and Table 2–style).

See `disentangled_evaluation/README.md` for details and commands.

## Interpretability Analysis
`interpretability/` holds code for Section 5 interpretability experiments to do logit attribution and activation patching.

See `interpretability/README.md` for details and commands.


## Paper
[Link](https://aclanthology.org/2025.emnlp-main.723.pdf) to our paper.

## Citation
```
@inproceedings{cheng-etal-2025-llms,
    title = "Can {LLM}s Reason Abstractly Over Math Word Problems Without {C}o{T}? Disentangling Abstract Formulation From Arithmetic Computation",
    author = "Cheng, Ziling  and
      Cao, Meng  and
      Pishdad, Leila  and
      Cao, Yanshuai  and
      Cheung, Jackie CK",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.723/",
    doi = "10.18653/v1/2025.emnlp-main.723",
    pages = "14306--14333",
    ISBN = "979-8-89176-332-6"
}
