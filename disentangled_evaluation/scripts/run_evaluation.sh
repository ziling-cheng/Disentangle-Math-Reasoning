MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)

for m in "${MODELS[@]}"; do
  python run_evaluation.py \
    --dataset_name gsm8k \
    --split test \
    --model_name "$m"
done

for m in "${MODELS[@]}"; do
  python run_evaluation.py \
    --dataset_name svamp \
    --split test \
    --model_name "$m"
done