# Data

This directory contains code and data for:
1) **Disentangle evaluation** experiments on GSM8K/SVAMP with controlled variants.
2) **Interpretability** experiments using template-generated arithmetic data for logit attribution and activation patching.

---

## Disentangle Evaluation Experiments

### Datasets
- **GSM8K**: `test_gsm8k.csv`
- **SVAMP**: `test_svamp.csv`

---

### CSV Schema

**Core fields**
- `question`: original problem statement
- `solution`: original chain-of-thought (CoT) trace (dataset-provided) if exists
- `answer`: gold final answer

**Variant fields**
- `symbolic_question`: question rewritten for the **symbolic abstraction** variant
- `symbolic_abstraction_answer`: symbolic expression for the question (in variables)
- `symbolic_binding`: variable assignment / binding rules mapping the symbolic question back to the original instantiation
- `numerical_abstraction_answer`: numerical expression for the question obtained by substituting bindings into `symbolic_abstraction_answer`
- `arithmetic_question`: question rewritten to directly test **arithmetic computation**

---

### Validation
We validate the derived expressions against the gold answer:

1. **Numeric check:** evaluate `numerical_abstraction_answer` and verify it matches `answer`.
2. **Consistency check:** `numerical_abstraction_answer` is obtained by substituting `symbolic_binding` into `symbolic_abstraction_answer`. Therefore, if the symbolic expression and bindings are correct, the numerical expression should evaluate to the correct final answer.

Any examples that failed these checks were **manually re-annotated and corrected**. The released dataset has 100% expression accuracy under this evaluation.

Note that after the paper was finalized, we found and manually fixed a small number of missed/incorrect annotations to ensure 100% expression accuracy in the released dataset. As a result, the exact values in main figures (e.g., Figure 3) may differ slightly from those reported in the paper (within ~±1%). The updated figures are available in `disentangled_evaluation/plots`.

---

## Interpretability Experiments
This directory contains (i) the generated **interpretability datasets** and (ii) the **code to (re)generate** them from templates.

---

### Data
All generated interpretability data are under `interpretability_data/` and follow these naming conventions.

#### (a) Single-operator datasets
- `method ∈ {logit_attribution, logic_patching, computation_patching}`
- `operator ∈ {addition, subtraction, multiplication, division}`

**Filename pattern**
- `{method}_{operator}_data.csv`

Examples:
- `logit_attribution_addition_data.csv`
- `computation_patching_division_data.csv`

#### (b) Cross-patching datasets
For cross-patching from a *clean* condition to a *corrupted* operator:
- `clean_surface_type ∈ {numerical, symbolic}`
- `clean_operator ∈ {addition, subtraction, multiplication, division}`
- `corrupted_operator ∈ {addition, subtraction, multiplication, division}`

**Filename pattern**
- `cross_patching_{clean_surface_type}_{clean_operator}_to_{corrupted_operator}_data.csv`

Example:
- `cross_patching_symbolic_addition_to_division_data.csv`

---

### Code to (re)generate the interpretability data

**Minimally Different Pairwise Templates**
- `addition_subtraction_templates.csv`
- `multiplication_division_templates.csv`

**Instantiations (sampled names + numbers)**
- `addition_subtraction_sampled_names_and_numbers.csv`
- `multiplication_division_sampled_names_and_numbers.csv`

Instantiations are produced by:
- `sample_numbers_and_names.py`

Templates + instantiations are assembled into interpretability datasets via:
- `make_interpretability_data.py`

**Outputs**
- Generated files are written to: `interpretability_data/`

