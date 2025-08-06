
import os
import sys
import glob
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.data import load_json, save_json, make_output_dir, load_jsonl, save_jsonl
from collections import Counter
import random


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question_path",
        type = str,
        help = "Path to the question",
        # default = "./analysis/classification/pairwise/generated_{checklist_model}/preprocessed_questions.json"
        default="./Dataset/LLMBar/dataset.json"
    )
    parser.add_argument(
        "--variation_type",
        type = str,
        help = "Type of variation",
        default = "all"
    )
    parser.add_argument(
        "--checklist_model",
        type = str,
        help = "checklist generate model",
        default = "gpt-4o-2024-08-06"
    )
    parser.add_argument(
        "--eval_model",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--inconsistency_threshold",
        type = int,
        help = "inconsistency_threshold",
        default = 9
    )
    
    args = parser.parse_args()

    args.checklist_path =f"./outputs/checklist/{{policy}}/LLMBar/{{checklist_model}}.jsonl"
    args.checklist_eval_output_path = f"./outputs/evaluation/checklist/{{policy}}:{{checklist_model}}/LLMBar/{{eval_model}}.jsonl"
    args.baseline_eval_output_path = f"./outputs/evaluation/no_checklist/LLMBar/{{eval_model}}.jsonl"
    args.negative_checklist_path =f"./analysis/classification/pairwise/{{policy}}:{{checklist_model}}/{{eval_model}}/checklist/negative_checklist.json"
    args.positive_checklist_path = f"./analysis/classification/pairwise/{{policy}}:{{checklist_model}}/{{eval_model}}/checklist/positive_checklist.json"
    args.checklist_improvement_stats_path = f"./analysis/stats/pairwise/LLMBar/checklist/{{checklist_model}}/classification/improvement_stats/{{eval_model}}.json"
    args.histogram_path = f"./analysis/histgram/pairwise/LLMBar/checklist/{{checklist_model}}/classification/{{eval_model}}_{{policy}}.pdf"
    args.stats_json_path = f"./analysis/stats/pairwise/LLMBar/checklist/{{checklist_model}}/classification/{{eval_model}}_{{policy}}_stats.json"

    return args

def calculate_average_label(labels):
    if not labels:
        return None
    scores = [score for score in labels.values() if isinstance(score, (int, float))]
    return sum(scores) / len(scores) if scores else None

def plot_and_save_improvement_statistics(improvement_scores, policy, output_path, threshold=None, stats_path=None):
    """改善スコアのヒストグラムを作成し統計情報を保存する関数"""
    mean_score = np.mean(improvement_scores) if improvement_scores else 0
    median_score = np.median(improvement_scores) if improvement_scores else 0
    std_dev = np.std(improvement_scores) if improvement_scores else 0
    min_score = min(improvement_scores) if improvement_scores else 0
    max_score = max(improvement_scores) if improvement_scores else 0
    
    # 改善/悪化/中立の分類
    positive = sum(1 for score in improvement_scores if score >= threshold) if threshold is not None else sum(1 for score in improvement_scores if score > 0)
    negative = sum(1 for score in improvement_scores if score <= -threshold) if threshold is not None else sum(1 for score in improvement_scores if score < 0)
    neutral = len(improvement_scores) - positive - negative
    
    # 統計情報をJSON形式で保存
    statistics = {
        "policy": policy,
        "threshold": threshold,
        "total_samples": len(improvement_scores),
        "positive": {
            "count": positive,
            "percentage": round(positive / len(improvement_scores) * 100, 2) if improvement_scores else 0
        },
        "negative": {
            "count": negative,
            "percentage": round(negative / len(improvement_scores) * 100, 2) if improvement_scores else 0
        },
        "neutral": {
            "count": neutral,
            "percentage": round(neutral / len(improvement_scores) * 100, 2) if improvement_scores else 0
        },
        "statistics": {
            "mean": round(mean_score, 4),
            "median": round(median_score, 4),
            "std_dev": round(std_dev, 4),
            "min": round(min_score, 4),
            "max": round(max_score, 4),
            "quartiles": {
                "q1": round(np.percentile(improvement_scores, 25), 4) if improvement_scores else 0,
                "q3": round(np.percentile(improvement_scores, 75), 4) if improvement_scores else 0
            },
            "percentiles": {
                "p10": round(np.percentile(improvement_scores, 10), 4) if improvement_scores else 0,
                "p90": round(np.percentile(improvement_scores, 90), 4) if improvement_scores else 0
            }
        },
        "histogram_bins": {
            "count": min(max(5, len(improvement_scores) // 2), 30),
            "values": improvement_scores
        },
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if stats_path:
        make_output_dir(os.path.dirname(stats_path))
        save_json(stats_path, statistics)
        print(f"Statistics saved to: {stats_path}")
    
    plt.figure(figsize=(12, 7))
    
    max_abs_value = max(abs(min_score), abs(max_score)) if improvement_scores else 1
    x_limit = 2.0 
    # データポイントが少ない場合の対応
    if len(improvement_scores) < 10:
        # データポイントが少ない場合はスウォームプロットを追加
        plt.subplot(2, 1, 1)
        sns.swarmplot(x=improvement_scores, color='blue', size=16)
        plt.axvline(x=0, color='r', linestyle='--', label='No Change')
        if threshold is not None:
            plt.axvline(x=threshold, color='g', linestyle='--', label=f'Threshold (+{threshold})')
            plt.axvline(x=-threshold, color='orange', linestyle='--', label=f'Threshold (-{threshold})')
        plt.xlabel('Improvement Score', fontsize=16)
        plt.title(f'Distribution of Improvement Scores - {policy}')
        plt.legend()
        plt.xlim(-x_limit, x_limit)
        x_ticks = np.linspace(-x_limit, x_limit, 20) 
        plt.xticks(x_ticks)
        plt.subplot(2, 1, 2)
    
    n_bins = min(max(5, len(improvement_scores) // 2), 30)
    
    # ヒストグラムとKDEプロット
    sns.histplot(improvement_scores, kde=True, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 平均値と中央値を表示
    plt.axvline(x=mean_score, color='blue', linestyle='-', label=f'Mean: {mean_score:.3f}')
    plt.axvline(x=median_score, color='green', linestyle='-', label=f'Median: {median_score:.3f}')
    
    # 0点と閾値のライン
    plt.axvline(x=0, color='r', linestyle='--', label='No Change')
    if threshold is not None:
        plt.axvline(x=threshold, color='g', linestyle='--', label=f'Threshold (+{threshold})')
        plt.axvline(x=-threshold, color='orange', linestyle='--', label=f'Threshold (-{threshold})')
    
    # 改善と悪化の領域に色をつける
    if threshold is not None and improvement_scores:
        plt.axvspan(threshold, x_limit, alpha=0.2, color='green', label='Positive')
        plt.axvspan(-x_limit, -threshold, alpha=0.2, color='red', label='Negative')
    
    # 分布情報を表示
    stats_text = f"Total: {len(improvement_scores)}\n"
    stats_text += f"Positive: {positive} ({positive/len(improvement_scores)*100:.1f}%)\n"
    stats_text += f"Negative: {negative} ({negative/len(improvement_scores)*100:.1f}%)\n"
    if threshold is not None:
        stats_text += f"Neutral: {neutral} ({neutral/len(improvement_scores)*100:.1f}%)\n"
    stats_text += f"Mean: {mean_score:.3f}\n"
    stats_text += f"Median: {median_score:.3f}\n"
    stats_text += f"Std: {std_dev:.3f}"
    
    # テキスト情報を追加
    plt.figtext(0.70, 0.65, stats_text, fontsize=16, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ('Improvement Score')
    plt.ylabel('Frequency' ,fontsize=16)
    plt.title(f'Distribution of Improvement Scores - {policy}',  fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    
    plt.xlim(-x_limit, x_limit)
    x_ticks = np.linspace(-x_limit, x_limit, 11)  # 11分割
    plt.xticks(x_ticks)
    
    # 出力ディレクトリが存在しない場合は作成
    make_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)  # 高解像度で保存
    plt.close()
    
    print(f"Histogram saved to: {output_path}")
    
    return statistics

def classificate_checklist_improvement(checklist_model, eval_model, policy,threshold=None):

    args = load_args()
    
    if threshold is None:
        threshold = 0.3
    
    
    all_stats = {}
    all_improvement_scores = []
    all_statistics = {} 
    
    improvement_results = {
        "positive_checklists": {},
        "negative_checklists": {}
    }
    
    checklist_path = args.checklist_path.format(policy=policy,checklist_model=checklist_model)
    eval_path = args.checklist_eval_output_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
    baseline_eval_outout_path = args.baseline_eval_output_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

    positive_checklist_path = args.positive_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
    negative_checklist_path = args.negative_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
    histogram_path = args.histogram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
    stats_json_path = args.stats_json_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

    question_data = load_json(args.question_path)
    checklist_data = load_jsonl(checklist_path)
    baseline_eval_outout_data = load_jsonl(baseline_eval_outout_path)
    eval_data = load_jsonl(eval_path)

    make_output_dir(positive_checklist_path)
    make_output_dir(negative_checklist_path)
    make_output_dir(histogram_path)
    make_output_dir(stats_json_path)
    
    # 統計情報を計算するための配列
    improvement_scores = []


    for question, checklist_item, no_checklist, eval_data_item in tqdm(zip(question_data, checklist_data, baseline_eval_outout_data, eval_data)):
        question_key = question["input"]
        checklist = checklist_item.get("checklist", [])
        gold_label = question['label']

        print("Gold label:", gold_label)
        # チェックリストなし評価の平均を計算
        no_checklist_ratings = []
        for key in ["eval_responses"]:
            no_checklist_ratings.extend(
                response.get("rating") or response.get("judge") 
                for response in no_checklist.get(key, []) 
                if response.get("rating") is not None or response.get("judge") is not None
            )
        ave_no_checklist = sum(no_checklist_ratings) / len(no_checklist_ratings) if no_checklist_ratings else None
        
        # チェックリストあり評価の平均を計算
        checklist_ratings = []
        for key in ["eval_responses"]:
            checklist_ratings.extend(
                response.get("rating") or response.get("judge") 
                for response in eval_data_item.get(key, []) 
                if response.get("rating") is not None or response.get("judge") is not None
            )
        ave_checklist = sum(checklist_ratings) / len(checklist_ratings) if checklist_ratings else None
        
        if gold_label is not None and ave_no_checklist is not None and ave_checklist is not None:
            # 改善スコアを計算: abs(gold - no_checklist) - abs(gold - checklist)
            # 正の値なら改善、負の値なら悪化
            improvement_score = abs(gold_label - ave_no_checklist) - abs(gold_label - ave_checklist)
            improvement_scores.append(improvement_score)
            all_improvement_scores.append(improvement_score)
            
            # 閾値に基づいて分類
            if improvement_score >= threshold:
                statistics = plot_and_save_improvement_statistics(
                    improvement_scores, 
                    policy, 
                    histogram_path, 
                    threshold,
                    stats_json_path
                )
                # 大幅な改善
                improvement_results["positive_checklists"][question_key] = {
                    "checklist": checklist,
                    "gold_label": gold_label,
                    "no_checklist_avg": ave_no_checklist,
                    "checklist_avg": ave_checklist,
                    "improvement_score": improvement_score
                }
                
            else:
                # 大幅な悪化
                improvement_results["negative_checklists"][question_key] = {
                    "checklist": checklist,
                    "gold_label": gold_label,
                    "no_checklist_avg": ave_no_checklist,
                    "checklist_avg": ave_checklist,
                    "improvement_score": improvement_score
                }
        
        # 改善スコアの統計情報
        avg_improvement = sum(improvement_scores) / len(improvement_scores) if improvement_scores else 0
        all_stats[policy] = {
            "avg_improvement_score": round(avg_improvement,  3),
            "positive_count": len(improvement_results["positive_checklists"]),
            "negative_count": len(improvement_results["negative_checklists"]),
            "total_count": len(improvement_scores),
            "positive_ratio": round(len(improvement_results["positive_checklists"]) / len(improvement_scores) if improvement_scores else 0, 3),
            "negative_ratio": round(len(improvement_results["negative_checklists"]) / len(improvement_scores) if improvement_scores else 0, 3),
            "threshold_used": threshold,
        }
        
        
        save_json(positive_checklist_path, improvement_results["positive_checklists"])
        save_json(negative_checklist_path, improvement_results["negative_checklists"])
    
    # 全サブセットの合計ヒストグラムを作成
    if all_improvement_scores:
        
        policy_title_map = {
            "baseline": "Baseline",
            "ticking": "Ticking",
            "specify": "Specify",
            "adjust_0.5_baseline": "Length*0.5",
            "adjust_1.5_baseline": "Length*1.5",
            "refine_baseline": "Self-refine"
        }

        # 表示用タイトルだけ変換
        display_policy = policy_title_map.get(policy, policy)
        histogram_path = args.histogram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        stats_json_path = args.stats_json_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

        # タイトルとして渡す
        combined_statistics = plot_and_save_improvement_statistics(
            all_improvement_scores, 
            "All Subsets", 
            display_policy, 
            histogram_path, 
            threshold,
            stats_json_path
        )
        all_stats["combined"] = {
        "avg_improvement_score": round(np.mean(all_improvement_scores), 3),
        "positive_count": combined_statistics["positive"]["count"],
        "negative_count": combined_statistics["negative"]["count"],
        "neutral_count": combined_statistics["neutral"]["count"],
        "total_count": combined_statistics["total_samples"],
        "positive_ratio": round(combined_statistics["positive"]["percentage"] / 100, 3),
        "negative_ratio": round(combined_statistics["negative"]["percentage"] / 100, 3),
        "neutral_ratio": round(combined_statistics["neutral"]["percentage"] / 100, 3),
        "threshold_used": threshold,
    }
    # 全体の統計情報を保存
    stats_path = args.checklist_improvement_stats_path.format(
        policy=policy, 
       checklist_model=checklist_model, 
        eval_model=eval_model
    )
    make_output_dir(stats_path)
    save_json(stats_path, all_stats)
    
    return all_stats
    
def main():
    args = load_args()
    checklist_model = args.checklist_model.replace("/", "_")
    eval_model_name = args.eval_model.replace("/", "_")
    checklist_improvement_stats_path = args.checklist_improvement_stats_path.format(checklist_model=checklist_model, eval_model=eval_model_name)
    checklist_generation_policies = [
            "baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
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
        print(f"Processing policy: {policy}")
        stats = classificate_checklist_improvement(checklist_model, eval_model_name, policy,threshold=None)
        if stats:
            all_stats[policy] = stats


    make_output_dir(checklist_improvement_stats_path)
    save_json(checklist_improvement_stats_path, all_stats)
    print(f"Results saved to {checklist_improvement_stats_path}")
    
if __name__ == "__main__":
    main()
