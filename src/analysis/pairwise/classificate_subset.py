import os
import sys
import glob
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from calc_pairwise_acc import load_ignore_questions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.data import load_json, load_json, save_json, make_output_dir

def load_args():
    args = argparse.ArgumentParser()
    
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
        "--subset_questions_path",
        type = str,
        help = "Path to the output file",
        default = "./analysis/data/classification/pairwise/{subset}/generated_{checklist_model}/preprocessed_questions.json"
    )
    args.add_argument(
        "--subset_checklists_path",
        type = str,
        help = "Path to the output file",
        default = "./analysis/data/classification/pairwise/{subset}/generated_{checklist_model}/preprocessed_checklists.json"
    )
    args.add_argument(
        "--subset_eval_results_path",
        type = str,
        help = "Path to the output file",
        default = "./analysis/data/classification/pairwise/{subset}/{policy}:{checklist_model}/{eval_model}/eval_results.json"
    )
    args.add_argument(
        "--subset_no_checklist_eval_results_path",
        type = str,
        help = "Path to the output file",
        default = "./analysis/data/classification/pairwise/{subset}/{policy}:{checklist_model}/{eval_model}/no_checklist_eval_results.json"
    )
    args.add_argument(
        "--checklist_subset_stats_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/stats/pairwise/Sample_Test/checklist/{checklist_model}/subset_classification/{eval_model}.jsonl"
    )
    return args.parse_args()


def classificate_subset(args,question_data, checklist_model_name):
    # subset_questions_path = args.subset_questions_path.format(subset=subset)
    classified_question_data = {}
    
    for item in question_data:
        full_subset = item.get("subset", "")
        if "/" in full_subset:
            parts = full_subset.split("/")
            last_part = parts[-1]
            
            if "_" in last_part:
                subset_name = last_part.split("_")[0]
            else:
                subset_name = last_part
        else:
            subset_name = full_subset
        
        if subset_name not in classified_question_data:
            classified_question_data[subset_name] = []
        
        classified_question_data[subset_name].append(item)
    
    for subset_name, items in classified_question_data.items():
        subset_questions_path = args.subset_questions_path.format(subset=subset_name, checklist_model=checklist_model_name)
        make_output_dir(subset_questions_path)
        save_json(subset_questions_path, items)
        print(f"Saved {len(items)} question items for subset: {subset_name}")
    
    return classified_question_data

def classificate_checklist(args, classified_question_data, preprocess_checklist, checklist_model_name):
    for subset_name, questions_data in classified_question_data.items():
        subset_checklists = []
        
        for item in questions_data:
            question_text = item.get("input", "")
            
            if question_text in preprocess_checklist:
                checklist_item = {
                    "question": question_text,
                    "checklist": preprocess_checklist[question_text]
                }
                subset_checklists.append(checklist_item)
        subset_checklists_path = args.subset_checklists_path.format(subset=subset_name,checklist_model=checklist_model_name)
        make_output_dir(subset_checklists_path)
        save_json(subset_checklists_path, subset_checklists)
        
        print(f"Saved {len(subset_checklists)} checklist items for subset: {subset_name}")
    
    
def classificate_eval_data(args, classified_question_data, preprocessed_no_checklist_data,preprocessed_eval_data, checklist_model_name, eval_model, policy,subset_list):
    all_stats = {}
    
    no_checklist_eval_dict = {item['question']: item for item in preprocessed_no_checklist_data}
    checklist_eval_dict = {item['question']: item for item in preprocessed_eval_data}
    
    for subset_name in subset_list:
        subset_eval_results_path = args.subset_eval_results_path.format(subset=subset_name, policy=policy, eval_model=eval_model, checklist_model=checklist_model_name)
        subset_no_checklist_eval_results_path = args.subset_no_checklist_eval_results_path.format(subset=subset_name, policy=policy, eval_model=eval_model, checklist_model=checklist_model_name)
        
        subset_evals = []
        subset_no_checklist_evals = []
        questions_data = classified_question_data.get(subset_name, [])
        for item in questions_data:
            question_text = item.get("input", "")
            if question_text in checklist_eval_dict:
                eval_item = {
                    "question": question_text,
                    "eval_responses": checklist_eval_dict[question_text]["eval_responses"] 
                }
                subset_evals.append(eval_item)
            if question_text in no_checklist_eval_dict:
                eval_item = {
                    "question": question_text,
                    "eval_responses": no_checklist_eval_dict[question_text]["eval_responses"] 
                }
                subset_no_checklist_evals.append(eval_item)
        make_output_dir(subset_eval_results_path)
        make_output_dir(subset_no_checklist_eval_results_path)
        save_json(subset_eval_results_path, subset_evals)
        save_json(subset_no_checklist_eval_results_path, subset_no_checklist_evals)
        all_stats[subset_name] = {
            "questions_count": len(questions_data),
            "checklist_eval_count": len(subset_evals),
            "no_checklist_eval_count": len(subset_no_checklist_evals)
        }

    return all_stats
        
def main():
    args = load_args()
    eval_model = args.eval_model.replace("/", "_")
    checklist_model_name = args.checklist_model
    checklist_subset_stats_path = args.checklist_subset_stats_path.format(checklist_model=checklist_model_name, eval_model=eval_model)

    checklist_generation_policies = [
        "baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
        "ticking", "refine_baseline", "detail"
    ]
    subset_list = [
        "MT-Bench", "LLMEval^2", "FairEval", "Natural",
        "GPTInst", "GPTOut", "Manual", "Neighbor"
    ]
    # eval_model = args.eval_model

    question_data = load_json(args.question_path)
    classified_question_data = classificate_subset(args, question_data, checklist_model_name)

    all_stats = {}
    checklist_generation_policies_path = "./analysis/data/info/checklist_generation_policies.json"
    subset_list_path = "./analysis/data/info/subset_list.json"
    make_output_dir(checklist_generation_policies_path)
    make_output_dir(subset_list_path)
    save_json(checklist_generation_policies_path, checklist_generation_policies)
    save_json(subset_list_path, subset_list)
    

    if args.variation_type == "all":
        target_policies = checklist_generation_policies
    elif args.variation_type in checklist_generation_policies:
        target_policies = [args.variation_type]
    else:
        print(f"Invalid variation type: {args.variation_type}")
        return

    for policy in target_policies:
        preprocessed_checklist_path = args.preprocessed_checklist_path.format(
            checklist_model=checklist_model_name, eval_model=eval_model, policy=policy
        )
        preprocess_checklist = load_json(preprocessed_checklist_path)

        preprocessed_eval_path = args.preprocessed_eval_path.format(
            checklist_model=checklist_model_name, eval_model=eval_model, policy=policy
        )
        preprocessed_eval_data = load_json(preprocessed_eval_path)

        preprocessed_no_checklist_path = args.preprocessed_no_checklist_path.format(
            policy=policy, eval_model=eval_model
        )
        preprocessed_no_checklist_data = load_json(preprocessed_no_checklist_path)

        classificate_checklist(args, classified_question_data, preprocess_checklist, checklist_model_name)

        stats = classificate_eval_data(
            args, classified_question_data, preprocessed_no_checklist_data, preprocessed_eval_data, 
            checklist_model_name, eval_model, policy, subset_list
        )
        if stats:
            all_stats[policy] = stats
        

    print("\n=== Summary ===")
    for policy, policy_stats in all_stats.items():
        print(f"\nPolicy: {policy}")
        for subset_name, subset_stats in policy_stats.items():
            print(f"- {subset_name}: questions={subset_stats['questions_count']}, "
                  f"checklist_eval={subset_stats['checklist_eval_count']}, "
                  f"no_checklist_eval={subset_stats['no_checklist_eval_count']}")
    make_output_dir(checklist_subset_stats_path)
    save_json(checklist_subset_stats_path, all_stats)
    
if __name__ == "__main__":
    main()
    print("finish")

    
    
    
    
    
    
    


