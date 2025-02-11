#!/bin/bash

# Models to evaluate
declare -a models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "Qwen/Qwen1.5-7B-Chat"
    "Qwen/Qwen2-7B-Instruct"
)

# Prompt types to evaluate
declare -a prompt_types=(
    "python_list_plus"
    "python_string_plus"
)

# Available GPUs
declare -a gpus=(0 1 2 5 6)

NUM_SAMPLES=1
MAX_TOKENS=2048
QUERY_FILES="./data/jailbreakbench.csv ./data/harmbench.csv"

# Create logs directory if it doesn't exist
mkdir -p logs/prompt_comparison

# Counter for managing GPU assignments
counter=0

# Run experiments in parallel for each model and prompt type combination
for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        gpu_index=$((counter % ${#gpus[@]}))
        gpu=${gpus[$gpu_index]}
        
        # Create log file name
        model_name=$(echo ${model} | tr '/' '_')
        log_file="logs/prompt_comparison/run_${model_name}_${prompt_type}_gpu${gpu}.log"
        
        echo "Starting experiment for ${model} with ${prompt_type} on GPU $gpu"
        
        # Run with nohup
        nohup env CUDA_VISIBLE_DEVICES=$gpu python main.py \
            --target-model "${model}" \
            --target-max-n-tokens $MAX_TOKENS \
            --num-samples $NUM_SAMPLES \
            --query-files $QUERY_FILES \
            --prompt-type "${prompt_type}" \
            > "$log_file" 2>&1 &
        
        # Increment counter for next GPU assignment
        counter=$((counter + 1))
        
        # Optional: add small delay between launches
        sleep 2
    done
done

echo "All experiments launched!"
echo "Check logs/prompt_comparison directory for outputs."