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
  python run_inference.py \
    --model_name "$m" \
    --data_split test \
    --dataset gsm8k \
    --batch_size 8
done

for m in "${MODELS[@]}"; do
  python run_inference.py \
    --model_name "$m" \
    --data_split test \
    --dataset svamp \
    --batch_size 8
done
