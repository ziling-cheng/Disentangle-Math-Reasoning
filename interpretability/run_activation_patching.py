import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import json
import random
from tqdm import tqdm
import argparse
import gc
import csv
import os

from activation_patching_utils import(
    # for patching
    tokenize,
    make_hook,
    remove_all_hooks,
    register_saving_hooks,
    make_patching_hook,
    calculate_logprob_changes,
)
from plotting_utils import(  # for plotting
    compute_patching_effect,
    make_patching_heatmap,
    plot_cross_patched_logprobs,
    compute_and_plot_effects,
    plot_patching_effects_at_once
    
)
from config import (
    CACHE_DIR, DATA_DIR, INSTRUCTION, MODEL_NAME_MAP, SYSTEM_PROMPT, \
    PROBLEM_TYPES, OPERATOR_PAIRS, CROSS_PATCHING_CLEAN_CORR_MAP,\
    ACTIVATION_PATCHING_RESULT_DIR
)

def load_model_from_hub(model_name,bfloat=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # sets pad_token_id too
    
    if bfloat:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,cache_dir = CACHE_DIR, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir = CACHE_DIR, device_map="auto")
    
    # make sure model config matches tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def to_serializable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    elif hasattr(x, "item"):
        return x.item()
    return x

def load_sampled_data(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def load_results(model_name, patching_type, problem_type, patching_scope):
    with open(f"{ACTIVATION_PATCHING_RESULT_DIR}/{model_name}_{patching_type}_{problem_type}_{patching_scope}_patching.json", "r") as f:
        return [json.loads(line) for line in f if line.strip()]
    
def count_existing_results(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)
    

def update_calculation_error_summary(patching_type, patching_scope, problem_type, skip_ctr, calculation_error_fpath):
    # --- upsert skip_ctr into CSV keyed by (patching_type, problem_type) ---
    row_key = {
        "patching_type": patching_type, 
        "patching_scope": patching_scope, 
        "problem_type": problem_type, 
        "skip_ctr": skip_ctr
    }

    if os.path.exists(calculation_error_fpath):
        df_err = pd.read_csv(calculation_error_fpath)

        # ensure required cols exist
        for c in ["patching_type", "patching_scope", "problem_type", "skip_ctr"]:
            if c not in df_err.columns:
                df_err[c] = pd.NA
    else:
        df_err = pd.DataFrame(columns=["patching_type", "patching_scope", "problem_type", "skip_ctr"])

    mask = (
        (df_err["patching_type"] == patching_type) &
        (df_err["patching_scope"] == patching_scope) &
        (df_err["problem_type"] == problem_type)
    )

    if mask.any():
        # overwrite existing row
        df_err.loc[mask, "skip_ctr"] = skip_ctr
    else:
        # append new row
        df_err = pd.concat([df_err, pd.DataFrame([row_key])], ignore_index=True)

    df_err.to_csv(calculation_error_fpath, index=False)
    
def run_activation_patching(
    model, 
    tokenizer,
    patching_type, # logic, computation, cross
    patching_scope,
    model_name,
    problem_type="addition",
    verbose=False
):
    df = pd.read_csv(f"{DATA_DIR}/{patching_type}_patching_{problem_type}_data.csv")
    os.makedirs(ACTIVATION_PATCHING_RESULT_DIR, exist_ok=True)
    
    output_fpath = f"{ACTIVATION_PATCHING_RESULT_DIR}/{model_name}_{patching_type}_{problem_type}_{patching_scope}_patching.json"
    calculation_error_fpath = f"{ACTIVATION_PATCHING_RESULT_DIR}/{model_name}_first_digit_error.csv"
    
    write_header = not os.path.exists(calculation_error_fpath)
    skip_ctr = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        clean_question = row['clean_prompt']
        corrupted_question = row['corrupted_prompt']
        clean_ans = row['clean_gt_answer']
        corrupted_ans = row['corrupted_gt_answer']
        
        clean_inputs, clean_answer = tokenize(clean_question, model, tokenizer)
        corrupted_inputs, corrupted_answer = tokenize(corrupted_question, model, tokenizer)
        
        clean_ans_id = tokenizer.encode(str(clean_ans), add_special_tokens=False)[0]
        corrupted_ans_id = tokenizer.encode(str(corrupted_ans), add_special_tokens=False)[0]
        
        if clean_ans == corrupted_ans or clean_ans_id == corrupted_ans_id:
            skip_ctr += 1
            if verbose: 
                print("Equal answer, skipping....", skip_ctr)
            continue

            
        target_tokens = [[clean_ans_id], [corrupted_ans_id]]
        
        if patching_type == "cross":
            target_ans = row['target_gt_answer']
            target_ans_id = tokenizer.encode(str(target_ans), add_special_tokens=False)[0]
            target_tokens.append([target_ans_id])
            
            if target_ans == corrupted_ans or target_ans_id == corrupted_ans_id or \
            target_ans == clean_ans or target_ans_id == clean_ans_id:
                skip_ctr += 1
                if verbose: 
                    print("Equal answer, skipping....", skip_ctr)
                continue
        
        remove_all_hooks(model)

        clean_token, clean_logprobs, clean_target_logprobs, \
        corrupted_token, corrupted_logprobs, corrupted_target_logprobs, \
        patched_target_logprobs, patched_predictions, patched_topk_predictions = calculate_logprob_changes(
            clean_inputs,
            corrupted_inputs,
            tokenizer,
            model,
            model,
            target_tokens,
            num_layers=model.config.num_hidden_layers+2,
            patching_scope=patching_scope,
        )

        result = {
            "clean_prompt": clean_question,
            "corrupted_prompt": corrupted_question,
            "clean_gt_answer": clean_ans,
            "corrupted_gt_answer": corrupted_ans,
            "clean_generated_answer": clean_answer,
            "corrupted_generated_answer": corrupted_answer,
            "clean_ans_id": clean_ans_id,
            "corrupted_ans_id": corrupted_ans_id,
            "clean_token": to_serializable(clean_token),
            "clean_logprobs": to_serializable(clean_logprobs),
            "clean_target_logprobs": to_serializable(clean_target_logprobs),
            "corrupted_token": to_serializable(corrupted_token),
            "corrupted_logprobs": to_serializable(corrupted_logprobs),
            "corrupted_target_logprobs": to_serializable(corrupted_target_logprobs),
            "patched_target_logprobs": to_serializable(patched_target_logprobs),
            "patched_predictions": to_serializable(patched_predictions),
            "patched_topk_predictions": to_serializable(patched_topk_predictions)
        }
        if patching_type == "logic":
            result.update({
                "x": row["operand1"],
                "y": row["operand2"],
            })

        elif patching_type == "computation":
            result.update({
                "clean_x": row["clean_operand1"],
                "clean_y": row["clean_operand2"],
                "corrupted_x": row["corrupted_operand1"],
                "corrupted_y": row["corrupted_operand2"],
            })
        else: #'cross'
            result.update({
                "taret_gt_answer": target_ans,
                "taret_ans_id": target_ans_id,
                "clean_x": row["clean_operand1"],
                "clean_y": row["clean_operand2"],
                "corrupted_x": row["corrupted_operand1"],
                "corrupted_y": row["corrupted_operand2"],
            })
            
        with open(output_fpath, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    update_calculation_error_summary(patching_type, patching_scope, problem_type, skip_ctr, calculation_error_fpath)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patching with selected patching type.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--patching_type",
        type=str,
        choices=["logic", "computation", "cross"],
        default="logic",
    )
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model_from_hub(args.model_id)
    model_name = MODEL_NAME_MAP[args.model_id]
    num_layers = model.config.num_hidden_layers
    patching_type = args.patching_type
    
    if (patching_type == "logic") or (patching_type == "computation"):
        for problem_type in PROBLEM_TYPES:
            for patching_scope in ["layer", "attn_output", "mlp"]:
                run_activation_patching(
                    model,
                    tokenizer,
                    patching_type,
                    patching_scope,
                    model_name=model_name,
                    problem_type=problem_type,
                    verbose=args.verbose
                )
                if args.plotting: # plot heatmap
                    results = load_results(model_name, patching_type, problem_type, patching_scope)
                    compute_and_plot_effects(results, patching_type, problem_type, patching_scope, model_name, report_logprob_diff=[True])
            
            stacked_results = {
                "layer": load_results(model_name, patching_type, problem_type, "layer"),
                "attn":  load_results(model_name, patching_type, problem_type, "attn_output"),
                "mlp": load_results(model_name, patching_type, problem_type, "mlp")
            }
            plot_patching_effects_at_once(stacked_results, patching_type, problem_type, model_name)
            
            
    elif patching_type == "cross":
        patching_scope="layer"
        for clean_opt_type in PROBLEM_TYPES:
            for corrupted_opt_type in CROSS_PATCHING_CLEAN_CORR_MAP[clean_opt_type]:
                for clean_surface_form in ['numerical', 'symbolic']:
                    problem_type = f"{clean_surface_form}_{clean_opt_type}_to_{corrupted_opt_type}"
                    run_activation_patching(
                        model,
                        tokenizer,
                        patching_type,
                        patching_scope=patching_scope,
                        model_name=model_name,
                        problem_type=problem_type,
                        verbose=args.verbose
                    )
        if args.plotting:
            for clean_opt_type in PROBLEM_TYPES:
                for corrupted_opt_type in CROSS_PATCHING_CLEAN_CORR_MAP[clean_opt_type]:
                    for clean_surface_form in ['numerical', 'symbolic']:
                        problem_type = f"{clean_surface_form}_{clean_opt_type}_to_{corrupted_opt_type}"
                        results = load_results(model_name, patching_type, problem_type, patching_scope)
                        patched_target_logprobs_list = [r["patched_target_logprobs"] for r in results]

                        title = f"Patch {clean_surface_form} {clean_opt_type} {patching_scope} activations to concrete {corrupted_opt_type}"
                        save_path = f"{ACTIVATION_PATCHING_RESULT_DIR}/{model_name}_{patching_type}_{problem_type}_{patching_scope}_logprob_line_graph.png"

                        plot_cross_patched_logprobs(
                            patched_target_logprobs_list,
                            tokenizer,
                            save_path,
                            patched_predictions=None,
                            num_layers=model.config.num_hidden_layers,
                            title=title
                        )

    else:
        print("Unrecognized patching type!")
    
    
    
    