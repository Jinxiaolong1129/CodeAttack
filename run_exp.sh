#!/bin/bash

# Models to evaluate
declare -a models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "Qwen/Qwen1.5-7B-Chat"
    "Qwen/Qwen2-7B-Instruct"
)

# Available GPUs
declare -a gpus=(0 1 2 5 6)

NUM_SAMPLES=1
MAX_TOKENS=2048
QUERY_FILES="./data/jailbreakbench.csv ./data/harmbench.csv"

# Create logs directory if it doesn't exist
mkdir -p logs/python_stack_plus

# Run experiments in parallel on different GPUs
for i in "${!models[@]}"; do
    gpu_index=$((i % ${#gpus[@]}))
    gpu=${gpus[$gpu_index]}
    
    # Create log file name
    model_name=$(echo ${models[$i]} | tr '/' '_')
    log_file="logs/python_stack_plus/run_${model_name}_gpu${gpu}.log"
    
    echo "Starting experiment for ${models[$i]} on GPU $gpu"
    
    # Run with nohup
    nohup env CUDA_VISIBLE_DEVICES=$gpu python main.py \
        --target-model "${models[$i]}" \
        --target-max-n-tokens $MAX_TOKENS \
        --num-samples $NUM_SAMPLES \
        --query-files $QUERY_FILES \
        --prompt-type "python_stack_plus" \
        > "$log_file" 2>&1 &
    
    # Optional: add small delay between launches
    sleep 2
done

echo "All experiments launched!"
echo "Check logs directory for outputs."



# nohup python main-API.py \
#     --target-model "gpt-4o-mini" \
#     --target-max-n-tokens 2048 \
#     --num-samples 1 \
#     --max-workers 20 \
#     --query-files "./data/jailbreakbench.csv" "./data/harmbench.csv" \
#     --exp-name "exp_gpt4o_mini" \
#     --temperature 0.0 \
#     > "logs/gpt4o_mini.log" 2>&1 &


