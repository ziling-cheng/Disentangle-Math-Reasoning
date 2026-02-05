from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import argparse
from typing import List, Dict, Tuple
import os
import torch
from tqdm import tqdm

from paths import DATA_DIR, RESULT_DIR, CACHE_DIR
from instructions import TASK_CONFIG, SYSTEM_PROMPT


def merge_lora_and_save(base_model_name: str, lora_model_name: str, merged_dir: str):
    """Merge LoRA adapter with base model and save to disk."""
    print(f"Merging LoRA model {lora_model_name} into base model {base_model_name}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token = os.getenv("HF_TOKEN")
    )
    
    # Load LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name, cache_dir=CACHE_DIR, token = os.getenv("HF_TOKEN"), force_download=True)
    
    # Merge and save
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=CACHE_DIR)
    tokenizer.save_pretrained(merged_dir)
    
    print(f"Merged model saved to {merged_dir}")
    return merged_dir


def load_model_from_hub(model_name: str, lora_model_name: str = None):
    """Load base model or merged LoRA model with vLLM."""
    
    if lora_model_name:
        # Merge LoRA and save to a temporary directory
        merged_dir = f"{CACHE_DIR}/merged_{model_name.replace('/', '_')}_{lora_model_name.replace('/', '_')}"
        if not os.path.exists(merged_dir):
            merge_lora_and_save(model_name, lora_model_name, merged_dir)
        
        # Load merged model with vLLM
        model_name = merged_dir
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    # Load model with vLLM
    llm = LLM(
        model=model_name,
        download_dir=CACHE_DIR,
        dtype="float16",
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.90,
    )
    
    return llm, tokenizer


def prepare_batches(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """Prepare batches of questions for each instruction type."""
    batches = {}
    for task_name, cfg in TASK_CONFIG.items():
        question_col = cfg["question_col"]
        questions = df[question_col].tolist()

        for instr_type in ["zs", "cot"]:
            instruction = cfg[instr_type]
            batches[f"{task_name}_{instr_type}"] = list(zip(questions, [instruction] * len(questions)))

    return batches

def generate_batch_responses(batch: List[tuple], tokenizer, model) -> List[str]:
    """Generate responses for a batch of (question, instruction) pairs."""
    messages_batch = []
    for question, instruction in batch:
        messages = [
            #{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n{instruction}"},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        messages_batch.append(text)
    
    sampling_params = SamplingParams(
        #do_sample=False,
        temperature=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=512,
        stop=[tokenizer.eos_token, "<|eot_id|>", "<|endoftext|>"]
    )
    
    outputs = model.generate(messages_batch, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def save_model_generations(df: pd.DataFrame, model_short_name: str, split: str, dataset: str):
    """Save the results to CSV and JSON."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(f"{RESULT_DIR}/{split}_{dataset}_results_{model_short_name}.csv", index=False)


def process_model_generations(
    df: pd.DataFrame,
    model,
    tokenizer,
    model_short_name: str,
    split: str,
    dataset: str,
    batch_size: int = 32,
) -> pd.DataFrame:

    """Process all questions in batches and add responses to dataframe."""
    df = df.copy()
    batches = prepare_batches(df)
    
    for col in tqdm(batches.keys(), desc="Processing question + instruction types"):
        question_batches = [batches[col][i:i + batch_size] for i in range(0, len(batches[col]), batch_size)]
        responses = []
        
        for batch in tqdm(question_batches, desc=f"Processing {col} batches", leave=False):
            batch_responses = generate_batch_responses(batch, tokenizer, model)
            responses.extend(batch_responses)
        
        df[col] = responses
        
        save_model_generations(df,  model_short_name, split, dataset)
    return df


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run model generation.")
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
        "--lora_model", 
        type=str, 
        default=None, 
        help="HuggingFace path to your LoRA model"
    )
    parser.add_argument(
        "--data_split",
        type=str,
        choices=["dev", "test", "reverse", "random"],
        required=True,
        help="Choose the data split: 'dev' or 'test' or 'reverse'."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["gsm8k", "svamp"],
        required=True,
        help="Choose the dataset_name: 'gsm8k', 'svamp'."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="If set, sample 8 examples from the dataset."
    )
    
    args = parser.parse_args()
    
    
    df = pd.read_csv(f"{DATA_DIR}/{args.data_split}_{args.dataset_name}.csv", encoding="utf-8")

    if args.sample:
        df = df.sample(n=8, random_state=42)

    model, tokenizer = load_model_from_hub(model_name=args.model_name, lora_model_name=args.lora_model)
        
        
    ########
    # Load previously saved results if they exist
    model_short_name = args.model_name.split("/")[-1]
        
    if args.lora_model:
        model_short_name += "_ft"
        
    print(model_short_name)
        
    result_path = f"{RESULT_DIR}/{args.data_split}_{args.dataset_name}_results_{model_short_name}.csv"

    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        print("Reading from existing file...")
    else:
        print(f"Previous result file not found at, continuing with default df: {result_path}")

    ########
    # Process in batches
    result_df = process_model_generations(
            df=df,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            model_short_name=model_short_name,
            split = args.data_split,
            dataset = args.dataset_name
    )
        
    # Save results
    save_model_generations(
            df=result_df,
            model_short_name=model_short_name,
            split=args.data_split,
            dataset= args.dataset_name
    )
        
    # Clean up
    del model
    del tokenizer