import pandas as pd
import argparse
from openai import OpenAI
import pandas as pd
from sympy import simplify, sympify, SympifyError,  Eq
from typing import Union
import re
from typing import Union
from tqdm import tqdm
tqdm.pandas()

import os
import csv

from paths import RESULT_DIR
from evaluation_utils import (
    normalize, 
    extract_last_number, 
    get_api_response, 
    extract_lhs, 
    clean_expression, 
    solve_for_x, 
    latex_to_python_math
)
from instructions import TASK_CONFIG

def postprocess_generation(generation, verbose=False):
    '''
        Parse the content after '####'.
    '''
    #print(generation)
    parts = generation.split('####')
    #print(len(parts))
    if len(parts) < 2:
        if verbose:
            print("Instruction Following Error, did not write in the format of ####.")
        return generation
    
    if len(parts) == 2: # #### answer
        if parts[-1].strip() == "": # answer ####
            response = parts[0].strip()
        else: # #### answer
            response = parts[-1].strip()
    elif len(parts) ==3: # #### answer ####
        response = parts[1].strip()
    else:
        return("More than 2 #### in the generation...")
    
    return response


def evaluate_final_answer(generation, reference, verbose=False):
    """
        Extracts the final numeric answer from the generation and compares it to the reference after normalization, 
        returning `pd.NA` when extraction/normalization fails.
    """

    generation = str(generation)
    reference = str(reference)
    try:
        
        number = extract_last_number(generation)
    except Exception as e:
        if verbose:
            print(e)
        return False # instruction following error
    else:
        try:
            if normalize(number) == normalize(reference):
                return True
            return False
        except Exception as e: # instruction following error
            if verbose:
                print(e, generation, reference)
            return pd.NA
    

def evaluate_symbolic_abstraction(
    symbolic_answer,
    abstract_answer,
    client, 
    use_gpt_4o
):

    message = f'''Determine whether the following two mathematical expressions are equivalent.
The expressions may be written in simplified or unsimplified symbolic form (e.g., 1/2x + 3), natural language (e.g., "Susan made 1/2x + 3 buttons") or in Latex notation.
Consider expressions equivalent if they represent the same mathematical value, even if written differently (e.g., different notation, simplification, or variable order when valid).
Respond only with: True or False. 

Example:
1. \\( z - (y - x) \\) 
2. z-y+x
Answer: True

Example:
1. Susan made 1/2*x buttons
2. 0.5x
Answer: True

Example:
1. 2(y+x)
2. \\( M = 2(y + x) \\)
Answer: True

Example:
1. xz* ((1-y)/100)
2. x \times z - (y/100) \times (x \times z)
Answer: True

Now evaluate:
1. {symbolic_answer}
2. {abstract_answer}
Answer:
'''
    response = get_api_response(message, client, use_gpt_4o=use_gpt_4o)
    
    # normalize + parse
    s = str(response).strip().lower()

    if s.startswith("true"):
        return True
    if s.startswith("false"):
        return False
    else:
        return pd.NA


def evaluate_numerical_abstraction(expr: str, expected_value: Union[str, int, float], verbose: bool = False) -> Union[bool, pd.NA]:
    """
        Evaluates whether the generated expression matches the expected value; 
        returns `pd.NA` if parsing or symbolic evaluation fails (invalid expressions).
    """
    expr = str(expr)
    try:
        
        # convert latex to string form
        if "\\" in expr:
            try:
                expr=latex_to_python_math(expr)
            except:
                return pd.NA
        
        if 'x' in expr and '=' in expr:
            return solve_for_x(expr, expected_value, verbose)

        expr_lhs = extract_lhs(expr)
        expr_clean = clean_expression(expr_lhs)
        expected_clean = clean_expression(str(expected_value))

        if verbose:
            print(f"Cleaned expression: {expr_clean}")
            print(f"Cleaned expected value: {expected_clean}")
        
        
        try:
            evaluated = simplify(expr_clean)
            expected = simplify(expected_clean)
        except Exception as e:
            if verbose:
                print("Unexpected error...", e, expr_clean, expected_clean)
            return pd.NA

        if verbose:
            print(f"Evaluated expression: {evaluated}, Expected: {expected}")

        return bool(evaluated == expected)

    except (SympifyError, SyntaxError, TypeError, ValueError, AttributeError) as e:
        if verbose:
            print(f"Error evaluating expression '<{expected_value}>': {e}")
        return pd.NA


def save_metrics(acc_dict: dict, model_short_name: str, save_path: str = "evaluation_summary.csv") -> pd.DataFrame:
    """
    Save accuracies into a table:
      - index: model_short_name
      - columns: acc metrics (keys of acc_dict)
    If model exists, overwrite its row; otherwise append.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # new row
    new_row = pd.DataFrame([acc_dict], index=[model_short_name])
    new_row.index.name = "model"

    # load existing or create new
    if os.path.exists(save_path):
        df = pd.read_csv(save_path, index_col=0)
    else:
        df = pd.DataFrame()

    # ensure columns exist (union)
    df = df.reindex(columns=sorted(set(df.columns) | set(new_row.columns)))

    # write/overwrite row
    df.loc[model_short_name, new_row.columns] = new_row.iloc[0]

    # save
    df.to_csv(save_path)
    return df


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run model evaluation with configurable parameters.")
        
        parser.add_argument("--dataset_name", type=str, default="gsm8k", 
                           choices=["gsm8k", "svamp"], help="Dataset to evaluate on")
        parser.add_argument("--split", type=str, default="test", 
                           choices=["dev", "test", "reverse", "random"], help="Data split to use")
        parser.add_argument("--openai_key_path", type=str, default="openai_key.txt", 
                           help="path to openai key")
        parser.add_argument(
            "--model_name",
            type=str,
            choices=[
                "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
            ],
            required=True,
            help="Choose the HuggingFace model id."
        )
        parser.add_argument(
            "--evaluation_summary_fname",
            type=str,
            default="evaluation_summary.csv", 
            help="evaluation summary filename"
        )

        # Parse arguments
        args = parser.parse_args()
        dataset = args.dataset_name
        split = args.split
        model = args.model_name.split("/")[-1]
        evaluation_summary_fname = dataset + "_" + args.evaluation_summary_fname 
        
        with open(args.openai_key_path, "r") as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        out_path = f"{RESULT_DIR}/{split}_{dataset}_results_{model}.csv"

        print(f"Running evaluation with:")
        print(f"Dataset: {dataset}")
        print(f"Split: {split}")
        print(f"\nEvaluating model: {model}")

        # read inference response df
        df = pd.read_csv(out_path)
        
        print(f"Dataset size: {len(df)}")
        
        # ------------------------------------------------------------------------------------------------------
        # Post processing model generation and extract final answers
        answer_column = "answer"

        for task in TASK_CONFIG.keys():
            for instr in ["zs", "cot"]:
                gen_col = f"{task}_{instr}"
                ans_col = f"{task}_{instr}_answer"
                df[ans_col] = df[gen_col].apply(postprocess_generation)
        df.to_csv(out_path, index=False)

        
        # ------------------------------------------------------------------------------------------------------
        # Evaluation
        EVAL_SPECS = {
            "original": {
                "answer_col": "answer",
                "fn": lambda ref, pred: evaluate_final_answer(pred, ref, verbose=False),
            },
            "numerical_abstraction": {
                "answer_col": "answer",
                "fn": lambda ref, pred: evaluate_numerical_abstraction(pred, ref, verbose=False),
            },
            "arithmetic_computation": {
                "answer_col": "answer",
                "fn": lambda ref, pred: evaluate_final_answer(pred, ref, verbose=False),
            },
            "symbolic_abstraction": {
                "answer_col": "symbolic_abstraction_answer",
                "fn": lambda ref, pred: evaluate_symbolic_abstraction(ref, pred, client, use_gpt_4o=False),
            },
        }
        for task, spec in EVAL_SPECS.items():
            ref_col = spec["answer_col"]

            for instr in ["zs", "cot"]:
                pred_col = f"{task}_{instr}_answer"
                out_col = f"{task}_{instr}_correctness"

                if pred_col in df.columns and ref_col in df.columns:
                    df[out_col] = df.progress_apply(lambda r: spec["fn"](r[ref_col], r[pred_col]), axis=1)

        df.to_csv(out_path, index=False)

        acc_dict = {
            "original_zs_acc": df['original_zs_correctness'].mean(),
            "original_cot_acc": df['original_cot_correctness'].mean(),
            
            'numerical_abstraction_zs_acc': df['numerical_abstraction_zs_correctness'].mean(),
            'numerical_abstraction_cot_acc': df['numerical_abstraction_cot_correctness'].mean(),
            
            'symbolic_abstraction_zs_acc': df['symbolic_abstraction_zs_correctness'].mean(),
            'symbolic_abstraction_cot_acc': df['symbolic_abstraction_cot_correctness'].mean(),
            
            'arithmetic_computation_zs_acc': df['arithmetic_computation_zs_correctness'].mean(),
            'arithmetic_computation_cot_acc': df['arithmetic_computation_cot_correctness'].mean()
        }
        
        summary_path = os.path.join(RESULT_DIR, evaluation_summary_fname)
        acc_table = save_metrics(acc_dict, model, summary_path)