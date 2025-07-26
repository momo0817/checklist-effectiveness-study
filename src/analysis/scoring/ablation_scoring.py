import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import sys
import argparse
import logging
import hashlib
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.data import  load_json, save_json, load_prompt, make_output_dir
from utils.model import load_model
from abs_evaluate_response import  request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__) 

def load_args():
    args = argparse.ArgumentParser()
    
    args.add_argument(
        "--question_path",
        type = str,
        help = "Path to the question",
        #  default = "./analysis/data/classification/scoring/{subset}/generated_{checklist_model}/preprocessed_questions.json"
        default="./Dataset/InFoBench/dataset.json"
    )
    
    args.add_argument(
        "--variation_type",
        type = str,
        help = "Type of variation",
        default = "all"
    )
    args.add_argument(
        "--checklist_model",
        type = str,
        help = "checklist generate model",
        default = "gpt-4o-2024-08-06"
    )
    args.add_argument(
         "--eval_model",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
    )
    args.add_argument(
        "--base_prompt_path",
        type=str,
        help="Path to the base prompt file. If the prompt not contains {checklist}, the checklist will be ignored.",
        default="./data/prompt/abs_evaluate_response/checklist.txt",
    )
    args.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to evaluate responses",
        default="You are a helpful assistant.",
    )
    args.add_argument(
        "-n",
        "--num-evaluation-trials",
        type=int,
        help="Number of evaluation trials. To account for position bias, half of the evaluation trials will reverse the order of the responses provided to the evaluation model.",
        default=2,
    )
    args.add_argument(
        "--temperature",
        type=float,
        help="Temperature for the generation",
        default=1.0,
    )
    args.add_argument(
        "--top-p",
        type=float,
        help="Top probability for the generation",
        default=0.95,
    )

    args.add_argument(
        "--max-retries",
        type=int,
        help="Max number of retries to generate the checklist",
        default=3,
    )

    args.add_argument(
        "--debug-mode",
        action="store_true",
        help="Run in debug mode",
    )
    args.add_argument(
        "--positive_checklist_path",
        type=str,
        default = "./analysis/classification/scoring/{policy}:{checklist_model}/{eval_model}/checklist/positive_checklist.json"
    )
    args.add_argument(
        "--negative_checklist_path",
        type=str,
        default = "./analysis/classification/scoring/{policy}:{checklist_model}/{eval_model}/checklist/negative_checklist.json"
    )
    args.add_argument(
        "--ablation_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/scoring/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/scoring/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--miss_ablation_negative_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/scoring/{policy}:{checklist_model}/{eval_model}/negative/miss_result.json"
    )
    args.add_argument(
        "--miss_ablation_positive_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/scoring/{policy}:{checklist_model}/{eval_model}/positive/miss_result.json"
    )
    args.add_argument(
        "--checklist_ablation_stats_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/scoring/InFoBench/checklist/{checklist_model}/ablation/{eval_model}_{policy}.json"
    )
    return args.parse_args()

def save_missing_entry(file_path, missing_entry):
    
    if os.path.exists(file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            existing_data = load_json(file_path)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.append(missing_entry)
        except Exception as e:
            logger.error(f"Failed to save missing entry: {e}")
            existing_data = [missing_entry]
    else:  
        existing_data = [missing_entry]
    save_json(file_path, existing_data)
   
def get_item_key(question, bad_checklist):

    key_string = f"{question}_{bad_checklist}"
    return hashlib.md5(key_string.encode()).hexdigest() 

def run_scoring_ablation(args, base_prompt, dataset, checklist_model, eval_model, checklist_items, save_path, miss_path,threshold=1.5, is_positive=True):
    matched_ablation_results = []
    missed_ablation_results = []
    for idx, d in enumerate(tqdm(dataset)):
        question = d['input']
        response = d['output'] 
        
        if question not in checklist_items:
            continue
        checklists= checklist_items[question]["checklist"]
        gold_label = checklist_items[question]["gold_label"]
        ave_checklist= checklist_items[question]["checklist_avg"]
        
        if not checklists:
            logger.warning(f"No checklist found for question at index {idx}")
            continue
        
        if len(checklists) <= 1:
            logger.warning(f"Too few checklist items to ablate at index {idx}")
            continue
            
        for remove_idx in range(len(checklists)):
            ablated_checklist = checklists[:remove_idx] + checklists[remove_idx+1:]
            text_checklists = '\n'.join(
                [f"{i+1}. {c}" for i,c in enumerate(ablated_checklist)]
            )
            prompt = base_prompt.format(
                question=question,
                response=response, 
                checklist=text_checklists
            )
            
            logger.debug(f"Prompting model for question {idx}, ablation {remove_idx}")
            scores, responses, dict_responses = request(
                args.system_prompt,
                prompt,
                eval_model,
                max_retries=args.max_retries,
                n=args.num_evaluation_trials,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            if not scores or len(scores) == 0:
                logger.warning("No scores found.")
                missing_entry = {
                    "question": question,
                    "response": response,
                    "removed_item": checklists[remove_idx],
                    "scores": scores,
                    "note": "no scores",
                }
                missed_ablation_results.append(missing_entry)
                continue

            ave_ablation_checklist = scores
            if isinstance(scores, list):
                if len(scores) == 1:
                    ave_ablation_checklist = scores[0]  # リストの場合は最初の要素を使用
                else:
                    logger.warning(f"Multiple scores found for question {idx}, using average")
                    ave_ablation_checklist = sum(scores) / len(scores)
            ablation_improvement_score = abs(gold_label - ave_checklist) - abs(gold_label - ave_ablation_checklist)
            
            if ablation_improvement_score:
                ablation_success = {
                    "question": question,
                    "response": response,
                    "removed_item": checklists[remove_idx],
                    "dict_responses": dict_responses,
                    "no_checklist_avg": checklist_items[question]["no_checklist_avg"],
                    "checklist_avg": checklist_items[question]["checklist_avg"],
                    "improvement_score": checklist_items[question]["improvement_score"],
                    'ablation_improvement_score': ablation_improvement_score 
                }
                matched_ablation_results.append(ablation_success)
                logger.info(f"Finished ablation {remove_idx+1}/{len(checklists)} for question {idx}")
            else:
                missing_entry  = {
                    "question": question,
                    "response": response,
                    "removed_item": checklists[remove_idx],
                    "dict_responses": dict_responses,
                    "no_checklist_avg": checklist_items[question]["no_checklist_avg"],
                    "checklist_avg": checklist_items[question]["checklist_avg"],
                    "improvement_score": checklist_items[question]["improvement_score"],
                    'ablation_improvement_score': ablation_improvement_score 
                }
                missed_ablation_results.append(missing_entry)
                logger.info(f"Finished ablation {remove_idx+1}/{len(checklists)} for question {idx}")
                
    if matched_ablation_results:
        logger.info(f"[{checklists}] Saving results to {save_path}")
        make_output_dir(save_path)
        save_json(save_path, matched_ablation_results)
    if missed_ablation_results:
        make_output_dir(os.path.dirname(miss_path))
        save_json(miss_path, missed_ablation_results)
        logger.info(f"Saved {len(missed_ablation_results)} missed ablation results to {miss_path}")
    
    return matched_ablation_results

def scoring_evaluation(checklist_model, eval_model, policy):
    args = load_args()
    logger.info(f'Loading model from {args.eval_model}')
    model = load_model(args.eval_model)
    
    logger.info(f'Loading base prompt from: {args.base_prompt_path}')
    base_prompt = load_prompt(args.base_prompt_path)
    stats = {}
     
    
        
    positive_checklist_path = args.positive_checklist_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    negative_checklist_path = args.negative_checklist_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    ablation_positive_checklist_path = args.ablation_positive_checklist_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    ablation_negative_checklist_path = args.ablation_negative_checklist_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    ) 
    miss_ablation_negative_path = args.miss_ablation_negative_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    )   
    miss_ablation_positive_path = args.miss_ablation_positive_path.format(
        policy=policy, 
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    make_output_dir(positive_checklist_path)
    make_output_dir(negative_checklist_path)
    make_output_dir(ablation_positive_checklist_path)
    make_output_dir(ablation_negative_checklist_path)
    make_output_dir(miss_ablation_positive_path)
    make_output_dir(miss_ablation_negative_path)
        
    if "{checklist}" in base_prompt:
        positive_results = None
        negative_results = None
        if os.path.exists(positive_checklist_path):
            logger.info(f'Loading positive checklist from: {positive_checklist_path}')
            positive_checklist = load_json(positive_checklist_path)
            logger.info(f'Found {len(positive_checklist)} positive checklists')
            logger.info('Running ablation for positive checklists')
            positive_results = run_scoring_ablation(
                args, 
                base_prompt, 
                args.dataset, 
                checklist_model, 
                model, 
                positive_checklist, 
                ablation_positive_checklist_path, 
                miss_ablation_positive_path,
                threshold=1.5, 
                is_positive=True
            )
            logger.info(f'Completed positive checklist ablation with {len(positive_results)} results')
        else:
            logger.warning(f'Positive checklist file not found: {positive_checklist_path}')
            positive_results = []
        
        if os.path.exists(negative_checklist_path):
            logger.info(f'Loading negative checklist from: {negative_checklist_path}')
            negative_checklist = load_json(negative_checklist_path)
            logger.info(f'Found {len(negative_checklist)} negative checklists')
            
            # Negative checklistに対するablation実行
            logger.info('Running ablation for negative checklists')
            negative_results = run_scoring_ablation(
                args, 
                base_prompt, 
                args.dataset, 
                checklist_model, 
                model, 
                negative_checklist, 
                ablation_negative_checklist_path, 
                miss_ablation_negative_path,
                threshold=1.5, 
                is_positive=False
            )
            logger.info(f'Completed negative checklist ablation with {len(negative_results)} results')
        else:
            logger.warning(f'Negative checklist file not found: {negative_checklist_path}')
            negative_results = []
        
        # 両方の結果を統計に格納
        stats = {
            "positive_total": len(positive_checklist) if 'positive_checklist' in locals() and positive_checklist else 0,
            "positive_ablation_success": len(positive_results) if positive_results else 0,
            "negative_total": len(negative_checklist) if 'negative_checklist' in locals() and negative_checklist else 0,
            "negative_ablation_success": len(negative_results) if negative_results else 0
        }
        
    return stats
        
def main():
    args = load_args()
    checklist_model = args.checklist_model.replace("/", "_")
    eval_model = args.eval_model.replace("/", "_")
    
    checklist_generation_policies = [
        "baseline",
        "adjust_0.5_baseline",
        "adjust_1.5_baseline", 
        "ticking", 
        "refine_baseline", 
        "specify"
    ]
    
    if args.variation_type == "all":
        target_policies = checklist_generation_policies
    elif args.variation_type in checklist_generation_policies:
        target_policies = [args.variation_type]
    else:
        print(f"Invalid variation type: {args.variation_type}")
        return
    
    all_stats = {}

    for policy in target_policies:
        print(f"Processing policy: {policy}")
        checklist_ablation_stats_path = args.checklist_ablation_stats_path.format(
        checklist_model=checklist_model, 
        eval_model=eval_model,
        policy=policy
    )
        stats = scoring_evaluation(checklist_model, eval_model, policy)
        all_stats[policy] = stats

    make_output_dir(checklist_ablation_stats_path)
    save_json(checklist_ablation_stats_path, all_stats)
    print(f"Saved results to: {checklist_ablation_stats_path}")
     
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()