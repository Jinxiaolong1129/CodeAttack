import argparse
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict
from tqdm import tqdm

from data_preparation import DataPreparer
from judge import GPT4Judge
from post_processing import PostProcessor
from target_llm import TargetLLM, TargetLLM_vllm

# Configure logging
def setup_logging(exp_name: str) -> str:
    """Set up logging configuration and return log file path"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/experiment_{exp_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_filename

def get_target_llm(model_name: str, max_tokens: int, temperature: float) -> Any:
    """Factory function to create appropriate LLM instance based on model name"""
    vllm_models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Qwen/Qwen1.5-7B-Chat",
        "Qwen/Qwen2-7B-Instruct"
    ]
    
    logging.info(f"Initializing model: {model_name}")
    if any(model in model_name for model in vllm_models):
        logging.info("Using vLLM backend")
        return TargetLLM_vllm(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
    elif "gpt" in model_name.lower():
        logging.info("Using API backend")
        return TargetLLM(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def save_results(results: List[Dict], args: Any, data_key: str, dataset_name: str) -> str:
    """Save results to JSON file and return filename"""
    results_dumped = json.dumps(results, indent=2)
    
    # Extract model name from full path
    target_model = args.target_model.split("/")[-1]
    
    # Create nested directory structure
    result_dir = Path("results") / dataset_name / target_model
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    # cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # res_filename = result_dir / f"{args.exp_name}_{data_key}_temp_{args.temperature}_results_{cur_time}.json"
    
    res_filename = result_dir / f"{args.exp_name}_{data_key}_temp_{args.temperature}_results.json"
    
    # Save results
    with open(res_filename, "w+") as f:
        f.write(results_dumped)
    
    logging.info(f"Results saved to: {res_filename}")
    return res_filename



def process_data(args: Any, datas: list, results: list, data_key: str, dataset_name: str) -> None:
    """Process data with appropriate threading strategy based on model type"""
    judgeLLM = GPT4Judge(args.judge_model)
    postprocessor = PostProcessor(args.prompt_type)
    total_jailbreaks = 0
    
    # 预先创建结果文件路径
    target_model = args.target_model.split("/")[-1]
    result_dir = Path("results") / dataset_name / target_model
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.exp_name}_{data_key}_temp_{args.temperature}_results.json"

    def save_current_results():
        """保存当前结果到固定文件"""
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to: {result_file}")

    def func_wrap(idx, data, targetLLM=None):
        nonlocal total_jailbreaks
        if not targetLLM:
            targetLLM = get_target_llm(
                args.target_model,
                args.target_max_n_tokens,
                args.temperature
            )

        question = data[data_key]
        plain_attack = data["plain_attack"]
        
        results[idx]["idx"] = idx
        results[idx]["query"] = plain_attack
        results[idx]["qA_pairs"] = []
        results[idx]["sample_results"] = []

        target_response_list = []
        successful_jailbreaks = 0

        for j in range(args.num_samples):
            sample_result = {}
            
            target_response = targetLLM.generate(question)
            target_response_list.append(target_response)
            
            resp = postprocessor.core(target_response)
            sample_result["response_simplified"] = resp

            is_jailbreak = judgeLLM.infer(plain_attack, resp)
            sample_result["is_jailbreak"] = is_jailbreak
            
            if is_jailbreak:
                successful_jailbreaks += 1
                total_jailbreaks += 1

            results[idx]["sample_results"].append(sample_result)

        results[idx]["qA_pairs"].append(
            {"Q": question, "A": target_response_list}
        )
        results[idx]["success_rate"] = successful_jailbreaks / args.num_samples
        
        logging.info(f"Processed idx {idx}: Success rate = {results[idx]['success_rate']:.2%}")
        
        if len(results[idx]) == 0:
            raise ValueError("Results not updated")
        
        save_current_results()
        
        return

    # 处理线程策略
    is_gpt_model = "gpt" in args.target_model.lower()
    
    if args.multi_thread and is_gpt_model:
        logging.info(f"Using multi-threading for GPT model: {args.target_model}")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(func_wrap,
                                 list(range(len(datas))),
                                 datas),
                     total=len(datas),
                     desc="Processing samples"))
    else:
        use_status = "multi-thread disabled" if is_gpt_model else "vLLM model"
        logging.info(f"Using single thread for {args.target_model} ({use_status})")
        targetLLM = get_target_llm(
            model_name=args.target_model,
            max_tokens=args.target_max_n_tokens,
            temperature=args.temperature
        )
        for idx, data in tqdm(enumerate(datas), total=len(datas), desc="Processing samples"):
            func_wrap(idx, data, targetLLM)

    # 记录最终统计信息
    total_samples = len(datas) * args.num_samples
    overall_success_rate = total_jailbreaks / total_samples
    logging.info(f"Experiment completed:")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Total successful jailbreaks: {total_jailbreaks}")
    logging.info(f"Overall success rate: {overall_success_rate:.2%}")
    
    
    
def run_experiment(args: Any, query_file: str, prompt_type: str) -> None:
    args.prompt_type = prompt_type
    """Run experiment for a single dataset"""
    dataset_name = Path(query_file).stem
    logging.info(f"Starting experiment for dataset: {dataset_name}")
    
    # 1. Generate the prompts based on CodeAttack
    data_preparer = DataPreparer(query_file=query_file,
                                prompt_name=f"code_{args.prompt_type}.txt",
                                prompt_type=args.prompt_type)
    data_preparer.infer()

    if args.no_attack:
        logging.info("Only generate adversarial examples. Don't run attacks.")
        return

    # 2. Attack the victim model and Auto-evaluate the results
    data_key = f"code_wrapped_{args.prompt_type}"
    with open(f"./prompts/data_{dataset_name}_{args.prompt_type}.json") as f:
        datas = json.load(f)
        if args.end_idx == -1:
            args.end_idx = len(datas)
        datas = datas[args.start_idx: args.end_idx]

    results = [{} for _ in range(len(datas))]
    
    # Process data with appropriate threading strategy
    process_data(args, datas, results, data_key, dataset_name)
    
    # Save results
    save_results(results, args, data_key, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ########### Target model parameters ##########
    parser.add_argument("--multi-thread", action="store_true", help="multi-thread generation (only for GPT models)")
    parser.add_argument("--max-workers", type=int, default=10, help="max workers for multi-thread generation")
    parser.add_argument("--target-model", type=str, help="Name of target model.")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="Name of judge model.")
    parser.add_argument("--target-max-n-tokens", type=int, default=2048, help="Maximum number of generated tokens for the target.")
    parser.add_argument("--exp-name", type=str, default="main", help="Experiment file name")
    parser.add_argument("--num-samples", type=int, default=1, help="number of output samples for each prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--prompt-type", type=str, default="python_stack_plus", help="type of adversarial prompt")
    parser.add_argument("--start-idx", type=int, default=0, help="start index of the data")
    parser.add_argument("--end-idx", type=int, default=-1, help="end index of the data")
    parser.add_argument("--query-files", nargs="+", default=["./data/jailbreakbench.csv", "./data/harmbench.csv"],
                      help="List of query files to process")
    
    parser.add_argument("--prompt-types", nargs="+", 
                    default=None,
                    help="List of prompt types to evaluate")
    parser.add_argument("--no-attack", action="store_true", help="set true when only generating adversarial examples")

    args = parser.parse_args()
    
    # Setup logging
    base_exp_name = args.exp_name
    log_file = setup_logging(f"{base_exp_name}_{args.target_model.split('/')[-1]}")
    
    logging.info(f"Starting experiments with configuration:")
    logging.info(f"Arguments: {vars(args)}")
    
    # Nested loop for prompt types and query files
    for prompt_type in args.prompt_types:
        logging.info(f"\nStarting experiments with prompt type: {prompt_type}")
        logging.info(f'='*100)
        logging.info(f'='*100)
        logging.info(f'='*100)
        # Update experiment name to include prompt type
        current_exp_name = f"{base_exp_name}_{prompt_type}"
        args.exp_name = current_exp_name
        
        for query_file in args.query_files:
            logging.info(f"\nProcessing {query_file} with {prompt_type}")
            logging.info(f'='*100)
            logging.info(f'='*100)
            logging.info(f'='*100)
            try:
                run_experiment(args, query_file, prompt_type)
            except Exception as e:
                logging.error(f"Error processing {query_file} with {prompt_type}: {str(e)}", 
                            exc_info=True)
                continue
    
    logging.info("All experiments completed successfully")