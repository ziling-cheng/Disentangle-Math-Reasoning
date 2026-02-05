import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import json
import argparse
import os

import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from config import CACHE_DIR, DATA_DIR, INSTRUCTION, LOGIT_ATTR_RESULT_DIR, MODEL_NAME_MAP, SYSTEM_PROMPT, PROBLEM_TYPES, OPERATOR_PAIRS
from logit_attribution_utils import register_saving_hooks, remove_all_hooks, generate_with_hidden_logging
from plotting_utils import plot_avg_logit_diff_by_path, plot_avg_logits_by_path

def load_model_from_hub(model_name,bfloat=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = CACHE_DIR)
    torch_dtype = torch.bfloat16 if bfloat else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, cache_dir = CACHE_DIR, device_map="auto")
        
    return model, tokenizer

def generate_message(
    question
):
    user_prompt = f"{question}\n{INSTRUCTION}"

    messages = [
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    
    # Terminators
    terminators = [tokenizer.eos_token_id]
    
    if "qwen" in model.config.model_type.lower():
        terminators.append(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    else:
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    ### getting input_ids
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = tokenizer([text], return_tensors="pt").to(model.device)
    
    return input_ids, terminators


def logit_attribution_at_step(
    model, tokenizer, model_name,
    problem_type="addition", step=0, verbose=False
):

    df = pd.read_csv(f"{DATA_DIR}/logit_attribution_{problem_type}_data.csv")
    os.makedirs(LOGIT_ATTR_RESULT_DIR, exist_ok=True)
    
    lm_head = model.lm_head 
    norm = model.model.norm
    torch.set_grad_enabled(False)
    L = model.config.num_hidden_layers
    
    comp_error_ctr = 0
    role_token_logits = defaultdict(lambda: [[] for _ in range(L)])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        formatted_question = row['prompt']
        a = int(row['operand1'])
        b = int(row['operand2'])
        c = int(row['gt_answer'])
        
        input_ids, terminators = generate_message(formatted_question)
        input_ids = input_ids.to(model.device)
        gen_ids, hidden_logs = generate_with_hidden_logging(model, tokenizer, input_ids, terminators) #max_tokens = 1?
        output_text = tokenizer.decode(gen_ids[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        try:
            if int(output_text) != c:
                if verbose:
                    print(formatted_question)
                    print("Input:", a, b, "Generated Output:", output_text)
                comp_error_ctr += 1
                
        except Exception as e:
            if verbose:
                comp_error_ctr += 1
                print(e)
                print("Input:", formatted_question, "Generated Output:", output_text)
            
                
        roles = {
                'a': a,
                'b': b,
                'c+': a + b,
                'c-': a - b,
                'c*': a * b,
                'c÷': int(a / b),
                'op (+)': [' +', '+', ' plus', ' addition'],
                'op (-)': [' -', '-', ' minus', ' substract'],
                'op (*)': ['*', ' *', 'x', ' x', ' multiply', ' multiplication'],
                'op (÷)': ['÷', ' ÷', ' /', '/', ' divide', ' division'],
        }

        for name_repr in ["attn_out", "mlp_in", "mlp_out", "resid_mid", "resid_final"]:
            reps = hidden_logs[name_repr][step]
            for layer in range(reps.shape[0]):
                hidden = reps[layer].to(lm_head.weight.device)
                logits = lm_head(norm(hidden))

                for role, val in roles.items():
                    if isinstance(val, list):
                        tok_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in val]
                        max_logit = max(logits[tok_id].item() for tok_id in tok_ids)
                        role_token_logits[(name_repr, role)][layer].append(max_logit)
                    else:
                        tok_id = tokenizer.encode(str(val), add_special_tokens=False)[0]
                        role_token_logits[(name_repr, role)][layer].append(logits[tok_id].item())

    np.save(f"{LOGIT_ATTR_RESULT_DIR}/role_token_logits_{problem_type}_{model_name}.npy", dict(role_token_logits))
    if verbose:
        print("Calculation Accuracy: ", 100-(comp_error_ctr/len(df))*100, "%")
    return role_token_logits, comp_error_ctr


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--plotting", action="store_true")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    
    model, tokenizer = load_model_from_hub(args.model_id)
    model_name = MODEL_NAME_MAP[args.model_id]
    num_layers = model.config.num_hidden_layers
    
    for problem_type in PROBLEM_TYPES:
        role_token_logits, comp_error_ctr = logit_attribution_at_step(
            model, tokenizer, model_name,
            problem_type=problem_type, step=args.step, verbose=args.verbose,
        )
    
    if args.plotting:
        # per-task plots
        for problem_type in PROBLEM_TYPES:
            role_token_logits = np.load(f"{LOGIT_ATTR_RESULT_DIR}/role_token_logits_{problem_type}_{model_name}.npy", allow_pickle=True).item()
            plot_avg_logits_by_path(
                role_token_logits, 
                num_layers, 
                problem_type=problem_type, 
                model_name=model_name
            )
        # operator diffs
        for p1, p2, label in OPERATOR_PAIRS:
            logits1 = np.load(f"{LOGIT_ATTR_RESULT_DIR}/role_token_logits_{p1}_{model_name}.npy", allow_pickle=True).item()
            logits2 = np.load(f"{LOGIT_ATTR_RESULT_DIR}/role_token_logits_{p2}_{model_name}.npy", allow_pickle=True).item()
            plot_avg_logit_diff_by_path(
                logits1,
                logits2,
                num_layers, 
                problem_type=label,
                model_name=model_name,
            )