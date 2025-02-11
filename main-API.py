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
import threading

from data_preparation import DataPreparer
from judge import GPT4Judge
from post_processing import PostProcessor
from target_llm import TargetLLM, TargetLLM_vllm

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
    logging.info(f"Initializing model: {model_name}")
    return TargetLLM(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature
    )

def save_results(results: List[Dict], args: Any, data_key: str, dataset_name: str) -> str:
    """Save results to JSON file and return filename"""
    results_dumped = json.dumps(results, indent=2)
    
    target_model = args.target_model.split("/")[-1]
    result_dir = Path("results") / dataset_name / target_model
    result_dir.mkdir(parents=True, exist_ok=True)
    
    res_filename = result_dir / f"{args.exp_name}_{data_key}_temp_{args.temperature}_results.json"
    
    with open(res_filename, "w+") as f:
        f.write(results_dumped)
    
    logging.info(f"Results saved to: {res_filename}")
    return res_filename

class ResultManager:
    def __init__(self, result_file: Path, initial_results: List[Dict]):
        self.result_file = result_file
        self.results = initial_results
        self.lock = threading.Lock()
        self.total_jailbreaks = 0
        
    def update_result(self, idx: int, result: Dict, jailbreak_count: int) -> None:
        with self.lock:
            self.results[idx] = result
            self.total_jailbreaks += jailbreak_count
            self._save_current_results()
            
    def _save_current_results(self) -> None:
        with open(self.result_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
    def get_total_jailbreaks(self) -> int:
        with self.lock:
            return self.total_jailbreaks

def process_single_sample(data: Dict, idx: int, args: Any, data_key: str, 
                         result_manager: ResultManager, targetLLM: TargetLLM = None) -> None:
    """Process a single sample with its own LLM instance if not provided"""
    if targetLLM is None:
        targetLLM = get_target_llm(
            model_name=args.target_model,
            max_tokens=args.target_max_n_tokens,
            temperature=args.temperature
        )
        
    judgeLLM = GPT4Judge(args.judge_model)
    postprocessor = PostProcessor(args.prompt_type)
    
    result = {"idx": idx, "query": data["plain_attack"], "qA_pairs": [], "sample_results": []}
    question = data[data_key]
    successful_jailbreaks = 0
    target_response_list = []

    try:
        for _ in range(args.num_samples):
            sample_result = {}
            
            target_response = targetLLM.generate(question)
            target_response_list.append(target_response)
            
            resp = postprocessor.core(target_response)
            sample_result["response_simplified"] = resp

            is_jailbreak = judgeLLM.infer(data["plain_attack"], resp)
            sample_result["is_jailbreak"] = is_jailbreak
            
            if is_jailbreak:
                successful_jailbreaks += 1

            result["sample_results"].append(sample_result)

        result["qA_pairs"].append({"Q": question, "A": target_response_list})
        result["success_rate"] = successful_jailbreaks / args.num_samples
        
        logging.info(f"Processed idx {idx}: Success rate = {result['success_rate']:.2%}")
        
        # Update results with thread safety
        result_manager.update_result(idx, result, successful_jailbreaks)
        
    except Exception as e:
        logging.error(f"Error processing sample {idx}: {str(e)}", exc_info=True)

def process_data(args: Any, datas: list, results: list, data_key: str, dataset_name: str) -> None:
    """Process data with thread pool for gpt-4o-mini"""
    # Create result file path and initialize result manager
    target_model = args.target_model.split("/")[-1]
    result_dir = Path("results") / dataset_name / target_model
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.exp_name}_{data_key}_temp_{args.temperature}_results.json"
    
    result_manager = ResultManager(result_file, results)
    
    logging.info(f"Processing samples using thread pool with {args.max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for idx, data in enumerate(datas):
            future = executor.submit(
                process_single_sample,
                data=data,
                idx=idx,
                args=args,
                data_key=data_key,
                result_manager=result_manager
            )
            futures.append(future)
            
        # Monitor progress
        for _ in tqdm(
            futures,
            total=len(futures),
            desc="Processing samples"
        ):
            _.result()  # This will also raise any exceptions that occurred

    # Log final statistics
    total_samples = len(datas) * args.num_samples
    total_jailbreaks = result_manager.get_total_jailbreaks()
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
    
    # Generate the prompts based on CodeAttack
    data_preparer = DataPreparer(
        query_file=query_file,
        prompt_name=f"code_{args.prompt_type}.txt",
        prompt_type=args.prompt_type
    )
    data_preparer.infer()

    if args.no_attack:
        logging.info("Only generate adversarial examples. Don't run attacks.")
        return

    # Attack the victim model and Auto-evaluate the results
    data_key = f"code_wrapped_{args.prompt_type}"
    with open(f"./prompts/data_{dataset_name}_{args.prompt_type}.json") as f:
        datas = json.load(f)
        if args.end_idx == -1:
            args.end_idx = len(datas)
        datas = datas[args.start_idx: args.end_idx]

    results = [{} for _ in range(len(datas))]
    
    # Process data
    process_data(args, datas, results, data_key, dataset_name)
    
    # Save final results
    save_results(results, args, data_key, dataset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=str, help="Name of target model.")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="Name of judge model.")
    parser.add_argument("--target-max-n-tokens", type=int, default=2048, help="Maximum number of generated tokens for the target.")
    parser.add_argument("--exp-name", type=str, default="main", help="Experiment file name")
    parser.add_argument("--num-samples", type=int, default=1, help="number of output samples for each prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--start-idx", type=int, default=0, help="start index of the data")
    parser.add_argument("--end-idx", type=int, default=-1, help="end index of the data")
    parser.add_argument("--max-workers", type=int, default=10, help="maximum number of worker threads")
    parser.add_argument("--no-attack", action="store_true", help="set true when only generating adversarial examples")
    
    parser.add_argument("--prompt-types", nargs="+", 
                    default=None,
                    help="List of prompt types to evaluate")
    parser.add_argument("--query-files", nargs="+", default=["./data/jailbreakbench.csv", "./data/harmbench.csv"],
                      help="List of query files to process")

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