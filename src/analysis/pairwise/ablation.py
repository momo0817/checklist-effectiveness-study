import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
import sys
import argparse
import logging
import math
from tqdm import tqdm
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.data import load_json, load_prompt, save_json, make_output_dir
from utils.model import load_model
from evaluate_response import request

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
        # default = "./analysis/classification/pairwise/generated_{checklist_model}/preprocessed_questions.json"
        default="./Dataset/LLMBar/dataset.json"
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
        help = "generate checklist model",
        default = "gpt-4o-2024-08-06"
    )
    args.add_argument(
        "--eval_model ",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
    )
    args.add_argument(
        "--base-prompt-path",
        type=str,
        help="Path to the base prompt file. If the prompt not contains {checklist}, the checklist will be ignored.",
        default="./data/prompt/pairwise_evaluate_response/checklist.txt",
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
        default = "./analysis/classification/pairwise/{policy}:{checklist_model}/{eval_model}/checklist/positive_checklist.json"
    )
    args.add_argument(
        "--negative_checklist_path",
        type=str,
        default = "./analysis/classification/pairwise/{policy}:{checklist_model}/{eval_model}/checklist/negative_checklist.json"
    )
    args.add_argument(
        "--ablation_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--miss_ablation_negative_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/negative/miss_result.json"
    )
    args.add_argument(
        "--miss_ablation_positive_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/positive/miss_result.json"
    )
    args.add_argument(
        "--checklist_ablation_stats_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/pairwise/LLMBar/checklist/{checklist_model}/ablation/{eval_model}_{policy}.json"
    )
    return args.parse_args()
        
def process_ablation_item(args_dict, base_prompt, question_data, eval_model, checklist_items, remove_idx):
    args = argparse.Namespace(**args_dict)
    question = question_data['input']
    response_1 = question_data['output_1']
    response_2 = question_data['output_2']
    label = question_data.get("label", None)
    
    ablated_checklist = checklist_items[:remove_idx] + checklist_items[remove_idx+1:]
    text_checklist_items = '\n'.join(
        [f"{i+1}. {c}" for i,c in enumerate(ablated_checklist)]
    )
    
    try:
        prompt = base_prompt.format(
            question=question,
            response_1=response_1,
            response_2=response_2,
            checklist=text_checklist_items
        )
        
        judges, responses, dict_responses = request(
            args.system_prompt,
            prompt,
            eval_model,
            max_retries=args.max_retries,
            n=math.ceil(args.num_evaluation_trials / 2),
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        rev_prompt = base_prompt.format(
            question=question,
            response_1=response_2,
            response_2=response_1,
            checklist=text_checklist_items,
        )
        
        rev_judges, rev_responses, rev_dict_responses = request(
            args.system_prompt,
            rev_prompt,
            eval_model,
            max_retries=args.max_retries,
            n=math.floor(args.num_evaluation_trials / 2),
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        rev_judges = [1 if j == 2 else 2 for j in rev_judges]
        
        all_judges = judges + rev_judges
        all_dict_responses = dict_responses + rev_dict_responses
        
        counter = Counter(all_judges)
        if len(counter) == 0:
            return {
                "status": "no_votes",
                "question": question,
                "removed_idx": remove_idx, 
                "removed_item": checklist_items[remove_idx],
                "judges": all_judges
            }
            
        final_judge = counter.most_common(1)[0][0]
        result = {
            "status": "success",
            "question": question,
            "removed_idx": remove_idx,
            "removed_item": checklist_items[remove_idx],
            "remaining_checklist": ablated_checklist,
            "judges": all_judges,
            "final_judge": final_judge,
            "gold_label": label,
            "match_gold": final_judge,
            "dict_responses": all_dict_responses
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error", 
            "question": question,
            "removed_idx": remove_idx,
            "removed_item": checklist_items[remove_idx],
            "error": str(e)
        }

def run_ablation(args, base_prompt, dataset, checklist_model, eval_model_name,  bad_checklist, save_path, miss_path):
    matched_ablation_results = []
    all_ablation_results = []
    
    make_output_dir(save_path)
    make_output_dir(miss_path)
    
    num_workers = min(mp.cpu_count(), 4)  # CPUコア数か4の小さい方
    logger.info(f"Using {num_workers} parallel workers for ablation")
    
    args_dict = vars(args)
    
    
    # 処理タスクのリストを準備
    ablation_tasks = []
    
    for idx, d in enumerate(dataset):
        question = d['input']
        
        # 質問がbad_checklistに存在するか確認
        if question not in bad_checklist:
            continue
            
        checklist_items = bad_checklist[question]["checklist"]
        # gold_label = bad_checklist[question]["gold_label"]
        # ave_all_chacklist = bad_checklist[question]["checklist_ave"]
        
        if not checklist_items or len(checklist_items) <= 1:
            continue
            
        # この質問のすべてのablationタスクを追加
        for remove_idx in range(len(checklist_items)):
            ablation_tasks.append((idx, d, question, checklist_items, remove_idx))
    for idx, d, question, checklist_items, remove_idx in tqdm(ablation_tasks, desc="Running ablations"):
        try:
            result = process_ablation_item(
                args_dict, base_prompt, d, eval_model_name, checklist_items.copy(), remove_idx
            )
            all_ablation_results.append(result)

            if result["status"] == "success":
                if result["match_gold"]:
                    gold_label = int(dataset[idx].get('label', 0))  # defaultを0にしておくのも安全
                    no_checklist_avg=bad_checklist[question]["no_checklist_avg"]
                    checklist_avg=bad_checklist[question]["checklist_avg"]
                    improvement_score=bad_checklist[question]["improvement_score"]
                    ave_ablation_label =calculate_judge_average(result)
                    print("gold_label", gold_label, type(gold_label))
                    print("checklist_avg", checklist_avg, type(checklist_avg))
                    print("ave_ablation_label", ave_ablation_label, type(ave_ablation_label))

                    ablation_improvement_score = abs(gold_label - checklist_avg) - abs(gold_label - ave_ablation_label)
                    print("ablation_improvement_score",ablation_improvement_score)
                    ablation_success = {
                        "question": question,
                        "response_1": dataset[idx]['output_1'],
                        "response_2": dataset[idx]['output_2'],
                        "removed_checklist_item": result["removed_item"],
                        "remaining_checklist": result["remaining_checklist"],
                        "dict_responses": result,
                        "no_checklist_avg":no_checklist_avg,
                        "checklist_avg": checklist_avg,
                        "improvement_score": improvement_score,
                        "ablation_improvement_score": ablation_improvement_score
                    }
                    matched_ablation_results.append(ablation_success)

            elif result["status"] == "no_votes":
                missing_entry = {
                    "question": question,
                    "response_1": dataset[idx]['output_1'],
                    "response_2": dataset[idx]['output_2'],
                    "removed_checklist_item": result["removed_item"],
                    "judges": result["judges"],
                    "note": "no votes"
                }
                miss_file = miss_path.replace('.json', f'_{idx}_{remove_idx}.json')
                save_json(miss_file, missing_entry)

            elif result["status"] == "error":
                logger.error(f"Error in ablation for Q{idx}, item {remove_idx}: {result['error']}")

        except Exception as e:
            logger.error(f"Exception processing result for Q{idx}, item {remove_idx}: {str(e)}")

    if matched_ablation_results:
        logger.info(f"Saving {len(matched_ablation_results)} successful ablation results to {save_path}")
        save_json(save_path, matched_ablation_results)
    else:
        logger.warning("No matched ablation results found.")

    all_results_path = save_path.replace('.json', '_all_results.json')
    save_json(all_results_path, all_ablation_results)

    return matched_ablation_results

def calculate_judge_average(data):

    if isinstance(data, dict):
        dict_responses = data.get("dict_responses", [])
    else:
        dict_responses = data  # dataがリストならそのまま使用
    
    # Check if dict_responses exists and is a list
    if not dict_responses or not isinstance(dict_responses, list) or len(dict_responses) == 0:
        print("dict_responses is empty or not a list")
        return None
    
    # Calculate the sum of judge values
    judge_sum = 0
    for response in dict_responses:
        if isinstance(response, dict) and "judge" in response:
            judge_sum += response["judge"]
    
    # Calculate the average
    average = judge_sum / len(dict_responses)
    
    return average

def evaluation(checklist_model, eval_model, policy):
    args = load_args()
    logger.info(f'Loading model from {args.eval_model}')
    model = args.eval_model.replace("/", "_")
    eval_model_name = load_model(args.eval_model)

    logger.info(f'Loading base prompt from: {args.base_prompt_path}')
    base_prompt = load_prompt(args.base_prompt_path)
    stats = {}

    question_path = args.question_path
    if not os.path.exists(question_path):
        logger.warning(f"Question file not found: {question_path}")
        return stats

    dataset = load_json(question_path)

    negative_checklist_path = args.negative_checklist_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )
    positive_checklist_path = args.positive_checklist_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )
    ablation_negative_checklist_path = args.ablation_negative_checklist_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )
    ablation_positive_checklist_path = args.ablation_positive_checklist_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )
    miss_ablation_negative_path = args.miss_ablation_negative_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )
    miss_ablation_positive_path = args.miss_ablation_positive_path.format(
        policy=policy, checklist_model=checklist_model, eval_model=model
    )

    negative_checklist = []
    positive_checklist = []
    ablation_negative_checklist = []
    ablation_positive_checklist = []

    if "{checklist}" in base_prompt:
        logger.info('Checklist detected in prompt.')

        # --- 悪化 checklist ablation ---
        if os.path.exists(negative_checklist_path):
            logger.info(f'Loading negative checklists from: {negative_checklist_path}')
            negative_checklist = load_json(negative_checklist_path)
            logger.info(f'Found {len(negative_checklist)} negative checklists')

            ablation_negative_checklist = run_ablation(
                args, base_prompt, dataset, checklist_model, eval_model_name,
                negative_checklist, ablation_negative_checklist_path,
                miss_ablation_negative_path
            ) or []

            if not os.path.exists(ablation_negative_checklist_path):
                logger.warning(f"Failed to save ablation negative checklist: {ablation_negative_checklist_path}")
                make_output_dir(os.path.dirname(ablation_negative_checklist_path))
                save_json(ablation_negative_checklist_path, ablation_negative_checklist)
        else:
            logger.warning(f'Worsened checklist file not found: {negative_checklist_path}')
            ablation_negative_checklist = []

        # --- 改善 checklist ablation ---
        if os.path.exists(positive_checklist_path):
            logger.info(f'Loading positive checklists from: {positive_checklist_path}')
            positive_checklist = load_json(positive_checklist_path)
            logger.info(f'Found {len(positive_checklist)} positive checklists')

            ablation_positive_checklist = run_ablation(
                args, base_prompt, dataset, checklist_model, eval_model_name,
                positive_checklist, ablation_positive_checklist_path,
                miss_ablation_positive_path
            ) or []

            if not os.path.exists(ablation_positive_checklist_path):
                logger.warning(f"Failed to save ablation positive checklist: {ablation_positive_checklist_path}")
                make_output_dir(os.path.dirname(ablation_positive_checklist_path))
                save_json(ablation_positive_checklist_path, ablation_positive_checklist)
        else:
            logger.warning(f'Improvement checklist file not found: {positive_checklist_path}')
            ablation_positive_checklist = []

        stats = {
            "negative_total": len(negative_checklist),
            "negative_ablation_success": len(ablation_negative_checklist),
            "positive_total": len(positive_checklist),
            "positive_ablation_success": len(ablation_positive_checklist)
        }

    return stats

def main():
    args = load_args()
    checklist_model = args.checklist_model.replace("/", "_")
    eval_model = args.eval_model .replace("/", "_")
    checklist_generation_policies = [
        "baseline", "adjust_0.5_baseline","adjust_1.5_baseline", 
        "ticking", "refine_baseline", "specify"
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
        checklist_ablation_stats_path = args.checklist_ablation_stats_path.format(checklist_model=checklist_model, eval_model=eval_model,policy=policy)
        print(f"Processing policy: {policy}")
        stats = evaluation(checklist_model, eval_model, policy)
        print("stats:", stats)
        all_stats[policy] = stats

    make_output_dir(checklist_ablation_stats_path)
    save_json(checklist_ablation_stats_path, all_stats)
    print(f"Saved results to: {checklist_ablation_stats_path}")
    try:
        save_json(checklist_ablation_stats_path, all_stats)
        print(f"Saved results to: {checklist_ablation_stats_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        # 代替ファイル名で保存を試みる
        alt_path = f"{checklist_ablation_stats_path}.backup"
        try:
            save_json(alt_path, all_stats)
            print(f"Saved backup results to: {alt_path}")
        except Exception as e2:
            print(f"Failed to save backup as well: {str(e2)}")
     
if __name__ == '__main__':
    main()