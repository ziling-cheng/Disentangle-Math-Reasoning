python run_logit_attribution.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --plotting
python run_logit_attribution.py --model_id "Qwen/Qwen2.5-7B-Instruct" --plotting
python run_logit_attribution.py --model_id "Qwen/Qwen2.5-14B-Instruct" --plotting

python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "logic" --plotting;
python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "computation" --plotting;
python run_activation_patching.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --patching_type "cross" --plotting

python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "logic" --plotting;
python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "computation" --plotting;
python run_activation_patching.py --model_id "Qwen/Qwen2.5-7B-Instruct" --patching_type "cross" --plotting;

python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "logic" --plotting;
python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "computation" --plotting;
python run_activation_patching.py --model_id "Qwen/Qwen2.5-14B-Instruct" --patching_type "cross" --plotting