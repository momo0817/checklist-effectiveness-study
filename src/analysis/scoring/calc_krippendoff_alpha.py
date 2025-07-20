import os
import sys
import json
import argparse
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from collections import defaultdict, Counter
import krippendorff
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.data import load_jsonl


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
        default="analysis/data/abs_krippendorff_alpha.xlsx",
    )

    return args.parse_args()

def make_final_judge(judges):
    if len(judges) == 0:
        return -1

    judge_counter = Counter(judges)
    most_common_judge = judge_counter.most_common(1)[0][0]
    if len(judges) % 2 == 0 and len(judges) // 2 == judge_counter[most_common_judge]:
        return -1
    return most_common_judge


def round_half_up(value, decimals="0"):
    round_value = Decimal(value).quantize(Decimal(decimals), rounding=ROUND_HALF_UP)
    return int(round_value)


def calc_ave_score(scores):
    if len(scores) == 0:
        return 3

    return round_half_up(sum(scores) / len(scores))


def calc_krip_alpha(data):
    abs_ratings = [calc_ave_score(d["scores"]) for d in data]
    abs_labels = [calc_ave_score(list(d["label"].values())) for d in data]
    reliability_data = [abs_ratings, abs_labels]
    return krippendorff.alpha(reliability_data, level_of_measurement="ordinal")


def merge(check_eval_data, baseline_eval_data, thres=1):
    idx_to_check_judges = {d["idx"]: d["scores"] for d in check_eval_data}

    merge_data = []
    count = {"checklist": 0, "baseline": 0}
    for d in baseline_eval_data:
        merge_d = {"idx": d["idx"], "label": d["label"]}
        if len(d["scores"]) == 0:
            merge_d["scores"] = d["scores"]
            count["baseline"] += 1
        else:
            judge_std = np.std(d["scores"])
            judge_std = np.round(judge_std, 10)  # Avoid floating point error
            if judge_std >= thres:
                merge_d["scores"] = idx_to_check_judges[d["idx"]]
                count["checklist"] += 1
            else:
                merge_d["scores"] = d["scores"]
                count["baseline"] += 1
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
    baseline_paths = evaluation_dir.glob("abs_evaluation/baseline/**/*.jsonl")

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
    checklist_paths = evaluation_dir.glob("abs_evaluation/checklist/**/*.jsonl")
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


def bootstrap_krip_alpha(data,num_sumples=1000):
    n = len(data)
    scores = []
    for _ in range(num_sumples):
        sample = [data[i] for i in np.random.choice(n, size=n, replace=True)]
        alpha = calc_krip_alpha(sample)
        scores.append(alpha)
    return np.array(scores)

def run_bootstrap_analysis(all_merge_data_dict):
    results = []
    keys = list(all_merge_data_dict.keys())
    alpha_distributions = {
        key: bootstrap_krip_alpha(data) for key, data in all_merge_data_dict.items()
    }

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            dist1, dist2 = alpha_distributions[k1], alpha_distributions[k2]
            diff = dist2 - dist1
            lower, upper = np.percentile(diff, [2.5, 97.5])
            is_sig = not (lower <= 0 <= upper)
            results.append({
                "comp": f"{k2} - {k1}",
                "mean_diff": diff.mean(),
                "95CI_lower": lower,
                "95CI_upper": upper,
                "significant": is_sig,
                "significant_positive": lower > 0 
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

    table = []
    dist_table = []
    bootstrap_summary = []
    for (
        dataset_name,
        model_name,
        checklist_type,
        checklist_model_name,
    ), check_eval_data in checklist_experiments.items():
        krip_alpha_checklist = calc_krip_alpha(check_eval_data)

        if (dataset_name, model_name) not in experiments:
            continue
        baseline_eval_data = experiments[(dataset_name, model_name)]

        krip_alpha_baseline = calc_krip_alpha(baseline_eval_data)

        row = [
            dataset_name,
            checklist_model_name,
            model_name,
            checklist_type,
            krip_alpha_baseline,
        ]
        dist_row = [
            dataset_name,
            checklist_model_name,
            model_name,
            checklist_type,
            0,
        ]
        all_merge_data = {
            "none": baseline_eval_data,
            "All": check_eval_data,
        }

        for i in reversed(range(1, 210)):
            thres = i / 100
            merge_data, count = merge(check_eval_data, baseline_eval_data, thres=thres)
            krip_alpha_marge = calc_krip_alpha(merge_data)
            row.append(krip_alpha_marge)
            dist_row.append(count["checklist"])
            all_merge_data[f"threshold_{thres:.2f}"] = merge_data

        row.append(krip_alpha_checklist)
        table.append(row)

        dist_row.append(len(check_eval_data))
        dist_row.append(len(baseline_eval_data))
        dist_table.append(dist_row)
        
        threshold_range = [f"threshold_{t:.2f}" for t in np.linspace(0.70, 0.30, 9)]  # 0.70, 0.65, ..., 0.30
        bootstrap_targets = ["none"] + threshold_range + ["All"]


        target_merge_data = {
            key: all_merge_data[key] for key in bootstrap_targets if key in all_merge_data
        }

        bootstrap_results = run_bootstrap_analysis(target_merge_data)
        
        for res in bootstrap_results:
            res.update({
                "evaluation_model": model_name,
                "checklist_type": checklist_type,
            })
            print(
                f"[{dataset_name} / {model_name} / {checklist_type} / {checklist_model_name}] "
                f"{res['comp']} → diff={res['mean_diff']:.3f}, "
                f"95%CI=({res['95CI_lower']:.3f}, {res['95CI_upper']:.3f}), "
                f"significant={res['significant']}"
            )
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
        "checklist_model",
        "evaluation_model",
        "checklist_type",
        "w/o checklist",
    ]
    for i in reversed(range(1, 210)):
        thres = i / 100
        header.append(f"threshold_{thres}")
        dist_header.append(f"threshold_{thres}")
    header.append("w/ checklist")
    dist_header.append("w/ checklist")
    dist_header.append("total")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Convert tables to DataFrames
    score_df = pd.DataFrame(table, columns=header)
    distribution_df = pd.DataFrame(dist_table, columns=dist_header)
    bootstrap_df = pd.DataFrame(bootstrap_summary)
    bootstrap_df.to_csv("analysis/data/bootstrap_result_scoring.csv", index=False)

    # Write DataFrames to Excel file with separate sheets
    with pd.ExcelWriter(args.output_path) as writer:
        score_df.to_excel(writer, sheet_name="score", index=False)
        distribution_df.to_excel(writer, sheet_name="distribution", index=False)
        bootstrap_df.to_excel("analysis/data/abs_krippendorff_alpha_bootstrap.xlsx", index=False)


if __name__ == "__main__":
    main()
