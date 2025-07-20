import os
import csv
import glob
import json
import numpy as np
import argparse

from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


def load_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--evaluation-dir",
        type=str,
        help="Path to the checklist directory",
        default="./outputs",
    )
    args.add_argument(
        "--output-path",
        type=str,
        help="Path to the output file",
        default="analysis/data/pairwise_accuracy.xlsx",
    )

    return args.parse_args()


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def make_final_judge(judges):
    if len(judges) == 0:
        return -1

    judge_counter = Counter(judges)
    most_common_judge = judge_counter.most_common(1)[0][0]
    if len(judges) % 2 == 0 and len(judges) // 2 == judge_counter[most_common_judge]:
        return -2
    return most_common_judge


def calc_acc(data):
    corrects = []
    for d in data:
        if d["label"] == make_final_judge(d["judges"]):
            corrects.append(1)
        elif d["label"] == -2:
            corrects.append(0.5)
        else:
            corrects.append(0)
    accuracy = sum(corrects) / len(corrects)
    return accuracy


def merge(check_eval_data, baseline_eval_data, thres=1):
    idx_to_check_judges = {d["idx"]: d["judges"] for d in check_eval_data}

    merge_data = []
    count = {"checklist": 0, "baseline": 0}
    for d in baseline_eval_data:
        merge_d = {"idx": d["idx"], "label": d["label"]}
        cnt = Counter(d["judges"])
        if len(cnt) <= 1:
            merge_d["judges"] = d["judges"]
            count["baseline"] += 1
        elif len(cnt) == 2:
            inconsistency = cnt.most_common(2)[-1][-1]
            if inconsistency >= thres:
                merge_d["judges"] = idx_to_check_judges[d["idx"]]
                count["checklist"] += 1
            else:
                merge_d["judges"] = d["judges"]
                count["baseline"] += 1
        else:
            print(d["judges"])
            raise ValueError()
        merge_data.append(merge_d)
    return merge_data, count


def load_ignore_questions(evaluation_dir):
    checklist_paths = evaluation_dir.glob("checklist/**/*.jsonl")
    checklists = defaultdict(dict)
    for file_path in checklist_paths:
        relative_file_path = file_path.relative_to(evaluation_dir)
        _, checklist_type, dataset_name, file_name = relative_file_path.parts

        *checklist_model_name, _ = file_name.split(".")
        checklist_model_name = ".".join(checklist_model_name)
        checklist_model_name = checklist_model_name.replace("_", "/")

        checklists[dataset_name][checklist_model_name] = load_jsonl(file_path)

    ignore_questions = {}
    for dataset_name, checklists_data in checklists.items():
        if dataset_name not in ignore_questions:
            ignore_questions[dataset_name] = set()
        for checklist_model_name, data in checklists_data.items():
            for d in data:
                if d.get("checklist") is None:
                    ignore_questions[dataset_name].add(d["question"])

    return ignore_questions


def filter_data(data, ignore_questions):
    return [d for d in data if d["question"] not in ignore_questions]


def load_baseline_experiments(evaluation_dir, ignore_questions):
    baseline_paths = evaluation_dir.glob("evaluation/baseline/**/*.jsonl")

    experiments = {}
    for file_path in baseline_paths:
        relative_file_path = file_path.relative_to(evaluation_dir)
        _, _, *dataset_name, file_name = relative_file_path.parts
        dataset_name = "/".join(dataset_name)

        *model_name, _ = file_name.split(".")
        model_name = ".".join(model_name)

        dataset_name = dataset_name.replace("/", "_")
        model_name = model_name.replace("_", "/")

        data = load_jsonl(file_path)
        data = filter_data(data, ignore_questions[dataset_name])
        print(f"Baseline: {dataset_name}, {model_name}, len: {len(data)}")
        experiments[(dataset_name, model_name)] = data

    return experiments


def load_checklist_experiments(evaluation_dir, ignore_questions):
    experiments = {}
    checklist_paths = evaluation_dir.glob("evaluation/checklist/**/*.jsonl")
    for file_path in checklist_paths:
        relative_file_path = file_path.relative_to(evaluation_dir)
        _, _, checklist_info, *dataset_name, file_name = relative_file_path.parts
        checklist_type, *checklist_model_name = checklist_info.split(":")

        checklist_model_name = ":".join(checklist_model_name)

        dataset_name = "/".join(dataset_name)

        *model_name, _ = file_name.split(".")
        model_name = ".".join(model_name)

        dataset_name = dataset_name.replace("/", "_")
        model_name = model_name.replace("_", "/")

        check_eval_data = load_jsonl(file_path)
        check_eval_data = filter_data(check_eval_data, ignore_questions[dataset_name])
        print(
            f"Checklist: {dataset_name}, {model_name}, {checklist_type}, {checklist_model_name}, len: {len(check_eval_data)}"
        )
        experiments[
            (dataset_name, model_name, checklist_type, checklist_model_name)
        ] = check_eval_data

    return experiments

def bootstrap_accuracy(data,n_iter=1000,seed=42):
    np.random.seed(seed)
    accuracies = []
    
    n = len(data)
    for _ in range(n_iter):
        sample = [data[i] for i in np.random.randint(0,n,n)]
        acc = calc_acc(sample)
        accuracies.append(acc)
    return np.array(accuracies)

def run_bootstrap_analysis(all_merge_data_dict):
    
    results = []
    keys = list(all_merge_data_dict)
    accuracy_distributions = {
        key: bootstrap_accuracy(data) for key, data in all_merge_data_dict.items()
    }
    
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            k1, k2 =keys[i], keys[j]
            dist1, dist2 = accuracy_distributions[k1], accuracy_distributions[k2]
            diff = dist2 - dist1
            lower, upper = np.percentile(diff, [2.5, 97.5])
            is_sig = not (lower <= 0 <= upper)
            results.append({
                "comp": f"{k2} - {k1}",
                "mean_diff": diff.mean(),
                "95CI_lower": lower,
                "95CI_upper": upper,
                "significant": is_sig,
                "significant_positive": lower > 0  # ←追加
            })

    return results


def main():
    args = load_args()

    evaluation_dir = Path(args.evaluation_dir)

    ignore_questions = load_ignore_questions(evaluation_dir)

    for dataset_name, questions in ignore_questions.items():
        print(f"Ignored questions in {dataset_name}: {len(questions)}")

    experiments = load_baseline_experiments(evaluation_dir, ignore_questions)
    checklist_experiments = load_checklist_experiments(evaluation_dir, ignore_questions)

    table, dist_table = [], []
    bootstrap_summary = []
    for (
        dataset_name,
        model_name,
        checklist_type,
        checklist_model_name,
    ), check_eval_data in checklist_experiments.items():
        acc_checklist = calc_acc(check_eval_data)

        if (dataset_name, model_name) not in experiments:
            continue
        baseline_eval_data = experiments[(dataset_name, model_name)]

        acc_baseline = calc_acc(baseline_eval_data)

        row = [
            dataset_name,
            checklist_model_name,
            model_name,
            checklist_type,
            acc_baseline,
        ]
        dist_row = [
            dataset_name,
            model_name,
            checklist_type,
            checklist_model_name,
            0,
        ]
        
        all_merge_data = {
            "none":baseline_eval_data,
            "All":check_eval_data
        }

        for i in reversed(range(1, 6)):
            merge_data, count = merge(check_eval_data, baseline_eval_data, thres=i)
            all_merge_data[f"threshold_{i}"] = merge_data
            acc_marge = calc_acc(merge_data)
            row.append(acc_marge)
            dist_row.append(count["checklist"])        

        row.append(acc_checklist)
        table.append(row)

        dist_row.append(len(check_eval_data))
        dist_row.append(len(baseline_eval_data))
        dist_table.append(dist_row)
        
        bootstrap_results = run_bootstrap_analysis(all_merge_data)
        for res in bootstrap_results:
            res.update({
                "evaluation_model": model_name,
                "checklist_type": checklist_type,
            })
            bootstrap_summary.append(res)
        
    header = [
        "dataset",
        "checklist_model",
        "evaluation_model",
        "checklist_type",
        "w/o checklist",
    ]
    dist_header = [
        "dataset",
        "evaluation_model",
        "checklist_type",
        "checklist_model",
        "w/o checklist",
    ]
    for i in reversed(range(1, 6)):
        header.append(f"threshold_{i}")
        dist_header.append(f"threshold_{i}")
    header.append("w/ checklist")
    dist_header.append("w/ checklist")

    dist_header.append("total")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    score_df = pd.DataFrame(table, columns=header)
    distribution_df = pd.DataFrame(dist_table, columns=dist_header)
    bootstrap_df = pd.DataFrame(bootstrap_summary)
    bootstrap_df.to_csv("analysis/data/bootstrap_results.csv", index=False)


    with pd.ExcelWriter(args.output_path) as writer:
        score_df.to_excel(writer, sheet_name="score", index=False)
        distribution_df.to_excel(writer, sheet_name="distribution", index=False)
        bootstrap_df.to_excel(writer, sheet_name="bootstrap", index=False)  

if __name__ == "__main__":
    main()
