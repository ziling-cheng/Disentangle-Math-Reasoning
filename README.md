# Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation

The project provides:
- **Disentangled behavioural evaluation (Section 4)** separating *abstract formulation* from *arithmetic computation*.
- **Mechanistic interpretability experiments (Section 5)** via logit attribution and activation patching (including cross-problem patching).

### Data
`data/` holds all datasets used by the codebase, including:
- Disentangled GSM8K and SVAMP files for behavirial evaluation, and
- Interpretability data under `data/interpretability_data/`.


### Disentangled Evaluation
`disentangled_evaluation/` holds code for Section 4 behavioural evaluation:
- runs inference to save model generations under different prompt/settings,
- scores outputs for each sub-skill setting,
- aggregates results and exports plots/tables (Figure 3–style and Table 2–style).

See `disentangled_evaluation/README.md` for details and commands.

### `Interpretability Analysis`
`interpretability/` holds code for Section 5 interpretability experiments to do logit attribution and activation patching.

See `interpretability/README.md` for details and commands.

---
### Paper
Link to the paper

### Citation
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