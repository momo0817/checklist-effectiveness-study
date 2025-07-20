import os
import sys
import glob
import json
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from calc_pairwise_acc import load_ignore_questions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.data import load_jsonl, load_json, save_json,make_output_dir

def load_args():
    args =  argparse.ArgumentParser()

    args.add_argument(
        "--variation_type",
        type = str,
        help = "Type of variation",
        default = "all"
    )
    args.add_argument(
        "--question_path",
        type = str,
        help = "Path to the question",
        default = "./LLMBar/Dataset/Sample/Test/dataset.json"
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
        "--checklist_path",
        type = str,
        help = "Path to the question",
        default = "./outputs/checklist/{policy}/Sample_Test/{checklist_model}.jsonl"
    )
    args.add_argument(
        "--eval_path",
        type = str,
        help = "Path to the evaluation",
        default = "./outputs/evaluation/checklist/{policy}:{checklist_model}/Sample_Test/{eval_model}.jsonl"
    )
    args.add_argument(
        "--no_checklist_eval_path",
        type = str,
        help = "Path to the evaluation",
        default = "./outputs/evaluation/baseline/Sample_Test/{eval_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/preprocessed/pairwise/Sample_Test/checklist/{policy}/{checklist_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_eval_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/preprocessed/pairwise/Sample_Test/evaluation/checklist/{policy}:{checklist_model}/{eval_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_no_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/preprocessed/pairwise/Sample_Test/evaluation/baseline/{policy}/{eval_model}.jsonl"
    )
    args.add_argument(
        "--checklist_stats_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/stats/pairwise/Sample_Test/checklist/{checklist_model}/summary/{eval_model}.jsonl"
    )
    return args.parse_args()

# def preprocess_checklist(checklist_data):
    
#     checklist_dict = {}
#     for data in checklist_data:
#         # question = data["question"]
#         # checklist = data["checklist"]
#         # checklist_dict[question] = checklist
#         question_text = data.get("input", "")
#         eval_item = {
#                     "question": question_text,
#                     "eval_responses": data.get(question_text, {})
#                 }
#         checklist_dict[question_text] = eval_item
#     return checklist_dict
def preprocess_checklist(checklist_data):
    
    checklist_dict = {}
    for data in checklist_data:
        question = data["question"]
        checklist = data["checklist"]
        checklist_dict[question] = checklist
    return checklist_dict

def preprocess_eval(no_checklist_eval_data,eval_data):
    
    no_checklist_eval_dict = {}
    eval_dict = {}
    for no_checklist_eval, eval in zip(no_checklist_eval_data, eval_data):
        question = no_checklist_eval.get("question", "")
        no_checklist_eval_dict_responses = no_checklist_eval.get("dict_responses", [])
        eval_dict_responses = eval.get("dict_responses", [])
        if question and eval_dict_responses:
            eval_dict[question] = eval_dict_responses
        if question and no_checklist_eval_dict_responses:
            no_checklist_eval_dict[question] = no_checklist_eval_dict_responses
    return no_checklist_eval_dict, eval_dict


def verified_checklist(no_checklist_eval_data,eval_data, checklist_dict):

    preprocessed_no_checklist_eval, preprocessed_eval = preprocess_eval(no_checklist_eval_data, eval_data)
    verified_eval_list = []
    verified_no_checklist_eval_list = []
    for question, eval_responses in preprocessed_eval.items():
        valid = True
        if question not in checklist_dict:
            valid = False
            continue
        
        for eval_response in eval_responses:
            checklist = eval_response.get("checklist", {})
            response_1_keys = checklist.get("response_1", {})
            response_2_keys = checklist.get("response_2", {})
            if len(response_1_keys) != len(response_2_keys):
                valid = False
                break
        if valid:
            for checklist_question, checklist_items in checklist_dict.items():
                if checklist_question == question:
                    if checklist_items is not None and response_1_keys is not None:
                        if len(checklist_items) != len(response_1_keys):
                            valid = False
        if valid:
            # verified_eval_dict[question] = eval_responses
            verified_eval_dict = {
                "question": question,
                "eval_responses": eval_responses,
            }
            verified_eval_list.append(verified_eval_dict)
            if question in preprocessed_no_checklist_eval:
                # verified_eval_dict[question] = preprocessed_no_checklist_eval[question]
                # verified_no_checklist_eval_dict[question] = preprocessed_no_checklist_eval[question]
                verified_no_checklist_eval_dict = {
                    "question": question,
                    "eval_responses": preprocessed_no_checklist_eval[question],
                }
                verified_no_checklist_eval_list.append(verified_no_checklist_eval_dict)
            else:
                valid = False

    original_checklist_count = sum(len(items) for q, items in checklist_dict.items() if items is not None)
    verified_checklist_count = 0
    for verified_item in verified_eval_list:
        question = verified_item["question"]
        if question in checklist_dict and checklist_dict[question] is not None:
            verified_checklist_count += len(checklist_dict[question])

    reduced_count = original_checklist_count - verified_checklist_count
    varified_ratio = verified_checklist_count / original_checklist_count if original_checklist_count else 0.0
    print("======Result:======")
    print(f"Original checklist count: {original_checklist_count} items")
    print(f"Verified checklist count: {verified_checklist_count} items")
    print(f"Reduced Count: {reduced_count}")
    print(f"Varified Rate: {varified_ratio:.2%}")
    checklist_stats = {
    "original_checklist_count": original_checklist_count,
    "verified_checklist_count": verified_checklist_count,
    "reduced_count": reduced_count,
    "reduced_ratio": varified_ratio
}
    # return verified_eval_dict, verified_no_checklist_eval_dict, checklist_stats
    return verified_eval_list, verified_no_checklist_eval_list, checklist_stats
   
def process_policy(args, policy,no_checklist_eval_data):
    print(f"\n======Processing policy: {policy}=====") 
    checklist_model_name = args.checklist_model.replace("/", "_")
    eval_model_name = args.eval_model.replace("/", "_")
    checklist_path = args.checklist_path.format(policy=policy, checklist_model=checklist_model_name)
    eval_path = args.eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
    
    preprocessed_checklist_path = args.preprocessed_checklist_path.format(policy=policy, checklist_model=checklist_model_name)
    preprocessed_eval_path = args.preprocessed_eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
    preprocessed_no_checklist_path = args.preprocessed_no_checklist_path.format(policy=policy, eval_model=eval_model_name)
    
    make_output_dir(preprocessed_checklist_path)
    make_output_dir(preprocessed_eval_path)
    make_output_dir(preprocessed_no_checklist_path)
    
    if not os.path.exists(checklist_path):
        print(f"File not found: {checklist_path}")
        return
    if not os.path.exists(eval_path):  
        print(f"File not found: {eval_path}")
        return
    try:
        checklist_data = load_jsonl(checklist_path)
        eval_path = args.eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
        eval_data = load_jsonl(eval_path)
        preprocessed_checklist_dict = preprocess_checklist(checklist_data)
        print(f"Preprocessed checklist data loaded from {preprocessed_checklist_dict}")
        verified_eval_list, verified_no_checklist_eval_list, checklist_stats = verified_checklist(no_checklist_eval_data,eval_data, preprocessed_checklist_dict)
        save_json(preprocessed_checklist_path,preprocessed_checklist_dict) 
        save_json(preprocessed_no_checklist_path,verified_no_checklist_eval_list)
        save_json(preprocessed_eval_path,verified_eval_list)       
        # save_json(checklist_stats_path,checklist_stats)
        print(f"Finished processing policy: {policy}")
        return checklist_stats
    except Exception as e:
        print(f"Error processing policy {policy}: {e}")
        import traceback
        traceback.print_exc()  
        return None
    
def main():
    args = load_args()
    checklist_generation_policies = ["baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
                                    "ticking", "refine_baseline", "detail"]
    all_stats = {}
    # eval_model_name = args.eval_model
    eval_model_name = args.eval_model.replace("/", "_")
    checklist_model_name = args.checklist_model.replace("/", "_")
    no_checklist_eval_path = args.no_checklist_eval_path.format(eval_model=eval_model_name)
    checklist_stats_path = args.checklist_stats_path.format(checklist_model=checklist_model_name, eval_model=eval_model_name)
    checklist_generation_policies_path = "./analysis/data/info/checklist_generation_policies.json"
    make_output_dir(checklist_generation_policies_path)
    save_json(checklist_generation_policies_path, checklist_generation_policies)
    
    if not os.path.exists(no_checklist_eval_path):
        print(f"File not found: {no_checklist_eval_path}")
        return
    no_checklist_eval_data = load_jsonl(no_checklist_eval_path)
    if args.variation_type == "all":
        for policy in checklist_generation_policies:
            stats = process_policy(args, policy,no_checklist_eval_data)
            if stats:
                all_stats[policy] = stats
    else:
        if args.variation_type in checklist_generation_policies:
            stats = process_policy(args, args.variation_type, no_checklist_eval_data)
            if stats:
                all_stats[args.variation_type] = stats
        else:
            print(f"Invalid variation type: {args.variation_type}")
            return
    print("======Final Result:======")
    for policy, stats in all_stats.items():
        print(f"Policy: {policy}")
        print(f"Original checklist count: {stats['original_checklist_count']} items")
        print(f"Verified checklist count: {stats['verified_checklist_count']} items")
        print(f"Reduced Count: {stats['reduced_count']}")
        print(f"Varified Rate: {stats['reduced_ratio']:.2%}")
    
    make_output_dir(checklist_stats_path)
    save_json(checklist_stats_path, all_stats)
    print(f"Save to : {checklist_stats_path}")  
    
if __name__ == "__main__":
    main()
    print("All done!")


            
            
        
        
