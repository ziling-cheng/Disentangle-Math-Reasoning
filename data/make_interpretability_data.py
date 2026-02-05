import pandas as pd
from tqdm import tqdm
import argparse
######## Logic patching dataset #########
def build_logic_patching_dataset(
    problem_type="addition_subtraction",
    n_available=4,
    n_times=2,
    clean_is_first_op=True,
):
    templates_csv = f"{problem_type}_templates.csv"
    sampled_csv = f"{problem_type}_sampled_names_and_numbers.csv"

    op1_name, op2_name = problem_type.split("_")
    out_op1_csv = f"interpretability_data/logic_patching_{op1_name}_data.csv"
    out_op2_csv = f"interpretability_data/logic_patching_{op2_name}_data.csv"

    templates = pd.read_csv(templates_csv)
    sampled_data = pd.read_csv(sampled_csv)

    assert len(templates) % 2 == 0, "templates.csv must have an even number of rows (pairs)."
    n_pairs = len(templates) // 2

    assert n_times <= n_available, f"n_times={n_times} cannot exceed n_available={n_available}."
    assert len(sampled_data) >= n_pairs * n_available, (
        f"sampled_data needs >= {n_pairs * n_available} rows, got {len(sampled_data)}."
    )

    rows = []
    for i in tqdm(range(n_pairs)):
        op1_row = templates.iloc[2 * i]
        op2_row = templates.iloc[2 * i + 1]

        op1_template = op1_row["template"]
        op2_template = op2_row["template"]

        base = i * n_available
        for j in range(n_times):
            d = sampled_data.iloc[base + 2*j]
            name = d["name"]
            a, b = int(d["a"]), int(d["b"])

            op1_q = op1_template.format(x=a, y=b).replace("[name]", name)
            op2_q = op2_template.format(x=a, y=b).replace("[name]", name)

            op1_gt = d[f"{op1_name}"]
            op2_gt = d[f"{op2_name}"]

            if clean_is_first_op:
                rows.append({
                    "clean_prompt": op1_q,
                    "clean_gt_answer": op1_gt,
                    "corrupted_prompt": op2_q,
                    "corrupted_gt_answer": op2_gt,
                    "operand1": a,
                    "operand2": b,
                })
            else:
                rows.append({
                    "clean_prompt": op2_q,
                    "clean_gt_answer": op2_gt,
                    "corrupted_prompt": op1_q,
                    "corrupted_gt_answer": op1_gt,
                    "operand1": a,
                    "operand2": b,
                })

    out_path = out_op1_csv if clean_is_first_op else out_op2_csv
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

######## Computation patching dataset #########
def build_computation_patching_dataset(
    problem_type="addition_subtraction",
    n_available=4,
    n_times=2,
    patch_for_op1=True,
):
    templates_csv = f"{problem_type}_templates.csv"
    sampled_csv = f"{problem_type}_sampled_names_and_numbers.csv"

    op1_name, op2_name = problem_type.split("_")
    out_op1_csv = f"interpretability_data/computation_patching_{op1_name}_data.csv"
    out_op2_csv = f"interpretability_data/computation_patching_{op2_name}_data.csv"
    
    templates = pd.read_csv(templates_csv)
    sampled_data = pd.read_csv(sampled_csv)

    if patch_for_op1:
        templates = templates.iloc[[i for i in range(0, len(templates ), 2)]]
    else:
        templates = templates.iloc[[i for i in range(1, len(templates ), 2)]]
    
    rows = []
    for template_idx in tqdm(range(len(templates))):
        op_template = templates.iloc[template_idx]["template"]

        base = template_idx * n_available
        for j in range(n_times):
            clean_row = sampled_data.iloc[base + 2*j]
            corrupted_row = sampled_data.iloc[base + 2*j + 1]

            name = clean_row["name"]

            clean_a, clean_b = int(clean_row["a"]), int(clean_row["b"])
            corrupted_a, corrupted_b = int(corrupted_row["a"]), int(corrupted_row["b"])

            clean_prompt = op_template.format(x=clean_a, y=clean_b).replace("[name]", name)
            corrupted_prompt = op_template.format(x=corrupted_a, y=corrupted_b).replace("[name]", name)
            
            
            if patch_for_op1:
                clean_gt_answer = clean_row[f"{op1_name}"]
                corrupted_gt_answer = corrupted_row[f"{op1_name}"]
            else:
                clean_gt_answer = clean_row[f"{op2_name}"]
                corrupted_gt_answer = corrupted_row[f"{op2_name}"]

            rows.append({
                "clean_prompt": clean_prompt,
                "clean_gt_answer": clean_gt_answer,
                "corrupted_prompt": corrupted_prompt,
                "corrupted_gt_answer": corrupted_gt_answer,
                "clean_operand1": clean_a,
                "clean_operand2": clean_b,
                "corrupted_operand1": corrupted_a,
                "corrupted_operand2": corrupted_b,
            })

    out_path = out_op1_csv if patch_for_op1 else out_op2_csv
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

######## Cross patching dataset #########
addsub_templates = pd.read_csv("addition_subtraction_templates.csv")
addsub_sampled = pd.read_csv("addition_subtraction_sampled_names_and_numbers.csv")

muldiv_templates = pd.read_csv("multiplication_division_templates.csv")
muldiv_sampled = pd.read_csv("multiplication_division_sampled_names_and_numbers.csv")

TEMPLATES_DATA_MAP = {
    "addition": {
        "templates": addsub_templates.iloc[[i for i in range(0, len(addsub_templates), 2)]],
        "sampled_data": addsub_sampled,
    },
    "subtraction": {
        "templates": addsub_templates.iloc[[i for i in range(1, len(addsub_templates), 2)]],
        "sampled_data": addsub_sampled,
    },
    "multiplication": {
        "templates": muldiv_templates.iloc[[i for i in range(0, len(muldiv_templates), 2)]],
        "sampled_data": muldiv_sampled,
    },
    "division": {
        "templates": muldiv_templates.iloc[[i for i in range(1, len(muldiv_templates), 2)]],
        "sampled_data": muldiv_sampled,
    },
}

OPERATOR_FUNCTIONS = {
        "addition": lambda a, b: int(a) + int(b),
        "subtraction": lambda a, b: int(a) - int(b),
        "multiplication": lambda a, b: int(a) * int(b),
        "division": lambda a, b: int(a) // int(b),
    }

CLEAN_CORR_MAP={
    "addition": [
        'paired_subtraction', 'subtraction',
        'multiplication', 'division'
        ],
    "subtraction": [
        'paired_addition', 'addition',
        'multiplication', 'division'
        ],
    "multiplication": [
        'paired_division', 'division',
        'addition', 'subtraction'
        ],
    "division": [
        'paired_multiplication', 'multiplication',
        'addition', 'subtraction'
        ],
}

def build_cross_patching_dataset(
    clean_opt_type, # addition
    corrupted_opt_type, # subtraction, paired_subtraction, multiplication, division
    clean_surface_form,  # numerical, symbolic
    
    n_available=4,
    n_times=2,
):

    clean_templates = TEMPLATES_DATA_MAP[clean_opt_type.replace('paired_', "")]['templates']
    corrupted_templates = TEMPLATES_DATA_MAP[corrupted_opt_type.replace('paired_', "")]['templates']
    if "paired" not in corrupted_opt_type:
        corrupted_templates = corrupted_templates.sample(frac=1, random_state=42).reset_index(drop=True)
    corrupted_sampled_data = TEMPLATES_DATA_MAP[corrupted_opt_type.replace('paired_', "")]['sampled_data']
    clean_sampled_data = TEMPLATES_DATA_MAP[clean_opt_type.replace('paired_', "")]['sampled_data']

    assert len(clean_templates) == len(corrupted_templates)
    rows = []
    
    
    for i in tqdm(range(len(clean_templates))):
        
        clean_template = clean_templates.iloc[i]['template']
        corrupted_template = corrupted_templates.iloc[i]['template']

        base = i * n_available
        for j in range(n_times):
                corrupted_d = corrupted_sampled_data.iloc[base + 2*j]
                clean_d = clean_sampled_data.iloc[base + 2*j+1]
                
                name = clean_d["name"]
                
                if clean_surface_form == "numerical":
                    clean_x, clean_y = int(clean_d["a"]), int(clean_d["b"])
                    clean_prompt = clean_template.format(x=clean_x, y=clean_y).replace("[name]", name)
                else: #"symbolic"
                    clean_x, clean_y = "x", "y"
                    clean_prompt = clean_template.replace("{x}","x").replace("{y}","y").replace("[name]", name)
                
                corrupted_x, corrupted_y = int(corrupted_d["a"]), int(corrupted_d["b"])
                corrupted_prompt = corrupted_template.format(x=corrupted_x, y=corrupted_y).replace("[name]", name)
                
                
                
                if clean_surface_form == "numerical":
                    clean_gt_answer = OPERATOR_FUNCTIONS[clean_opt_type](clean_x, clean_y)
                else:
                    clean_gt_answer = "x"
                
                corrupted_gt_answer = OPERATOR_FUNCTIONS[corrupted_opt_type.replace('paired_', "")](corrupted_x, corrupted_y)
                
                target_gt_answer = OPERATOR_FUNCTIONS[clean_opt_type](corrupted_x, corrupted_y)
                # clean logic injected to corrupted prompt, combined with corrupted operands
                
                rows.append({
                        "clean_prompt": clean_prompt,
                        "clean_gt_answer": clean_gt_answer,
                        "corrupted_prompt": corrupted_prompt,
                        "corrupted_gt_answer": corrupted_gt_answer,
                        "target_gt_answer": target_gt_answer,
                        "clean_operand1": clean_x,
                        "clean_operand2": clean_y,
                        "corrupted_operand1": corrupted_x,
                        "corrupted_operand2": corrupted_y,
                })
                
    out_path = f"interpretability_data/cross_patching_{clean_surface_form}_{clean_opt_type}_to_{corrupted_opt_type}_data.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    
    return out_path


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="logic_patching",
        choices=["logic_patching", "computation_patching", "logit_attribution", "cross_patching"],
        help="Which paired operation templates to use.",
    )
    args = parser.parse_args()
    
    problem_types = ["addition_subtraction","multiplication_division"]
    
    # logic patching
    if args.experiment_type == "logic_patching":
        for problem_type in problem_types:
            for clean_is_first_op in [True, False]:
                build_logic_patching_dataset(problem_type=problem_type, n_available=4, n_times=2, clean_is_first_op=clean_is_first_op)
    
    
    # computation patching
    elif args.experiment_type == "computation_patching":
        for problem_type in problem_types:
            for patch_for_op1 in [True, False]:
                build_computation_patching_dataset(
                    problem_type=problem_type,
                    n_available=4,
                    n_times=2,
                    patch_for_op1=patch_for_op1,
                )
    elif args.experiment_type == "logit_attribution":
        # logit attribution
        for op in ["addition", "subtraction", "multiplication", "division"]:
            df = pd.read_csv(f"logic_patching_{op}_data.csv")
            df = df.drop(columns={
                "corrupted_prompt",
                "corrupted_gt_answer"
            })
            df = df.rename(columns={
                "clean_prompt":  "prompt",
                "clean_gt_answer": "gt_answer"
            })
            df.to_csv(f"interpretability_data/logit_attribution_{op}_data.csv", index=False)

    # cross-patching
    elif args.experiment_type == "cross_patching":
        clean_opt_types = [
            'addition', 
            'subtraction',
            'multiplication', 
            'division'
            ]
        for clean_opt_type in clean_opt_types:
            for corrupted_opt_type in CLEAN_CORR_MAP[clean_opt_type]:
                for clean_surface_form in ['numerical', 'symbolic']:
                    build_cross_patching_dataset(
                        clean_opt_type,
                        corrupted_opt_type,
                        clean_surface_form,
                        n_available=4,
                        n_times=2
                    )