import os
import sys
import json
import argparse
from collections import defaultdict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import pandas as pd
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.data import load_json, save_json, make_output_dir


def load_args():
    args = argparse.ArgumentParser()
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
        "--eval_model",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
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
        "--ablation_final_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_final/scoring/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_final_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_final/scoring/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--ablation_filtered_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_filtered/scoring/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_filtered_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_filtered/scoring/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--negative_histgram_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/scoring/InFoBench/histgrams/negative/{policy}:{checklist_model}/{eval_model}/final_result.pdf"
    )
    args.add_argument(
        "--positive_histgram_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/scoring/InFoBench/histgrams/positive/{policy}:{checklist_model}/{eval_model}/final_result.pdf"
    )
    
    return args.parse_args()


def plot_histgram_statistics(ablation_improvement_scores, policy, output_path, pos_or_neg, threshold=None, stats_path=None):
    """改善スコアのヒストグラムを作成し統計情報を保存する関数"""
    # 統計情報を計算
    mean_score = np.mean(ablation_improvement_scores) if ablation_improvement_scores else 0
    median_score = np.median(ablation_improvement_scores) if ablation_improvement_scores else 0
    std_dev = np.std(ablation_improvement_scores) if ablation_improvement_scores else 0
    min_score = min(ablation_improvement_scores) if ablation_improvement_scores else 0
    max_score = max(ablation_improvement_scores) if ablation_improvement_scores else 0
    
    pos_count = sum(1 for score in ablation_improvement_scores if score > 0)
    neg_count = sum(1 for score in ablation_improvement_scores if score < 0)
    zero_count = sum(1 for score in ablation_improvement_scores if score == 0)
    
    
    # 統計情報をJSON形式で保存
    statistics = {

        "policy": policy,
        "threshold": threshold,
        "total_samples": len(ablation_improvement_scores),
        "statistics": {
            "mean": round(mean_score, 4),
            "median": round(median_score, 4),
            "std_dev": round(std_dev, 4),
            "min": round(min_score, 4),
            "max": round(max_score, 4),
            "quartiles": {
                "q1": round(np.percentile(ablation_improvement_scores, 25), 4) if ablation_improvement_scores else 0,
                "q3": round(np.percentile(ablation_improvement_scores, 75), 4) if ablation_improvement_scores else 0
            },
            "percentiles": {
                "p10": round(np.percentile(ablation_improvement_scores, 10), 4) if ablation_improvement_scores else 0,
                "p90": round(np.percentile(ablation_improvement_scores, 90), 4) if ablation_improvement_scores else 0
            }
        },
        "histogram_bins": {
            "count": min(max(5, len(ablation_improvement_scores) // 2), 30),
            "values": ablation_improvement_scores
        },
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 統計情報をJSONとして保存
    if stats_path:
        make_output_dir(stats_path)
        save_json(stats_path, statistics)
        print(f"Statistics saved to: {stats_path}")
    
    # ヒストグラムを描画して保存（前のコードと同じ）
    plt.figure(figsize=(12, 5))
    if pos_or_neg=="negative":
        x_min_limit = -2.55
        x_max_limit = 2.65
    else:
        x_min_limit = -2.55
        x_max_limit = 2.65
    
    max_abs_value = max(abs(min_score), abs(max_score)) if ablation_improvement_scores else 1

    
    major_ticks = np.arange(-2.5, 3.0, 0.1)
    sns.histplot(ablation_improvement_scores, kde=True, bins=major_ticks, alpha=0.7, color='skyblue', edgecolor='black')
        # pos_or_neg に応じた塗り分け
    if pos_or_neg == "positive":
        plt.axvspan(x_min_limit, 0, alpha=0.2, color='green', label='Positive Checklist Items\n (ΔS_all < 0)')
    elif pos_or_neg == "negative":
        plt.axvspan(0, x_max_limit, alpha=0.2, color='red', label='Negative Checklist Items\n(ΔS_all > 0)')
        

    
    # 平均値と中央値を表示
    plt.axvline(x=mean_score, color='blue', linestyle='-', label=f'Mean: {mean_score:.3f}')
    plt.axvline(x=median_score, color='green', linestyle='-', label=f'Median: {median_score:.3f}')
    
    
    if len(ablation_improvement_scores) > 0:
        pos_rate = round(100 * pos_count / len(ablation_improvement_scores), 2)
        neg_rate = round(100 * neg_count / len(ablation_improvement_scores), 2)
    else:
        pos_rate = neg_rate = 0.0

    stats_text = f"Total: {len(ablation_improvement_scores)}\n"
    # if threshold is not None:
    stats_text += f"Mean: {mean_score:.3f}\n"
    stats_text += f"Median: {median_score:.3f}\n"
    stats_text += f"Std: {std_dev:.3f}\n"
    
    if pos_or_neg == "negative":
        stats_text  +=  f"Negative Checklist Items: {pos_count}\n"
        stats_text += f"Negative Rate: {pos_rate}%"

    else:
        stats_text += f"Positive Checklist Items:{neg_count}\n"
        stats_text += f"Positive Rate: {neg_rate}%"
    
    if pos_or_neg == "negative":
        plt.figtext(0.68, 0.65, stats_text, fontsize=16, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    else:
        plt.figtext(0.70, 0.65, stats_text, fontsize=16, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.xlabel('Contribution Score', fontsize=19)
    plt.ylabel('Frequency', fontsize=19, labelpad=5)  
    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # labelpadの値を大きくするほどラベルが軸から離れる
    policy_title_map = {
            "baseline": "Baseline",
            "ticking": "Ticking",
            "specify": "Specify",
            "adjust_0.5_baseline": "Length*0.5",
            "adjust_1.5_baseline": "Length*1.5",
            "refine_baseline": "Self-refine"
        }

    display_policy = policy_title_map.get(policy, policy)
    display_pos_neg = "Negative" if pos_or_neg == "negative" else "Positive"
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    
    plt.xlim(x_min_limit, x_max_limit)
    x_ticks = np.arange(x_min_limit, x_max_limit, 0.1) # 11分割
    plt.axvline(x=0, color='r', linestyle='--', label='No Change')
    major_ticks = np.arange(-2.5, 3.0, 0.1)
    label_ticks = major_ticks[::5]  # ← 0.1刻みなので、5個ごとで0.5間隔になる
    plt.xticks(label_ticks, fontsize=15) 
    plt.yticks(fontsize=15)
    
    # 出力ディレクトリが存在しない場合は作成
    make_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, format="pdf") 
    plt.close()
    
    print(f"Histogram saved to: {output_path}")
    
    return statistics

def safe_load_json(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return load_json(path)
        except json.JSONDecodeError:
            print(f"Error: The file {path} contains invalid JSON.")
            return []
    else:
        print(f"Error: The file {path} does not exist or is empty.")
        return []

def classificate_ablation_improvement_scores(
    checklist: list,
    policy: str,
    score_type: str,  # "positive" or "negative"
    hist_path: str,
) -> Tuple[list, list, list, list]:
    """ablation_improvement_score を分類し、スコアと元データを返す（チェックリストごとに平均）"""

    # removed_item ごとに entry を集約
    checklist_by_removed_item = defaultdict(list)
    
    for entry in checklist:
        q = entry["removed_item"]
        checklist_by_removed_item[q].append(entry)

    checklists = []
    scores = []

    for removed_item, entries in checklist_by_removed_item.items():
        all_scores = [e.get("ablation_improvement_score", 0) for e in entries]
        avg_score = np.mean(all_scores)
        
        # 最初のentryを代表として使い、平均スコアに置き換えた形で保存
        representative_entry = entries[0].copy()
        representative_entry["ablation_improvement_score"] = avg_score
        
        checklists.append(representative_entry)
        scores.append(avg_score)

    # filtered / final の抽出
    filtered_checklist = []
    filtered_scores = []
    final_checklist = []
    final_scores = []

    for entry in checklists:
        score = entry["ablation_improvement_score"]
        q = entry["removed_item"]

        if (score_type == "positive" and score < -1.0) or (score_type == "negative" and score > 0.9):
            filtered_scores.append(score)
            filtered_checklist.append(entry)

        if (score_type == "positive" and score < 0) or (score_type == "negative" and score > 0):
            final_scores.append(score)
            final_checklist.append(entry)

    # ヒストグラムと円グラフ出力
    plot_histgram_statistics(scores, policy, hist_path, score_type)

    return scores, checklists, final_checklist, filtered_checklist


def main():
    args = load_args()
    checklist_model = args.checklist_model.replace("/", "_")
    eval_model = args.eval_model.replace("/", "_")
    
    checklist_generation_policies = [
            "baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
            "ticking", "refine_baseline", "specify"
        ]
    grand_total_pos = 0
    grand_filtered_pos = 0
    grand_total_neg = 0
    grand_filtered_neg = 0
    grand_final_pos = 0
    grand_final_neg = 0

    # 全ポリシーのフィルタリングされたチェックリストを統合するための変数
    all_policies_filtered_pos_checklist = []
    all_policies_filtered_neg_checklist = []
    all_policies_final_pos_checklist = []
    all_policies_final_neg_checklist = []
    
    # 使用するポリシーを決定
    if args.variation_type == "all":
        target_policies = checklist_generation_policies
    elif args.variation_type in checklist_generation_policies:
        target_policies = [args.variation_type]
    else:
        print(f"Invalid variation type: {args.variation_type}")
        return

    for policy in target_policies:
        print(f"Processing policy: {policy}")
        all_positive_scores = []
        all_negative_scores = []
        policy_filtered_pos_count = 0
        policy_final_pos_count = 0
        policy_filtered_neg_count = 0
        policy_final_neg_count = 0

            
        ablation_final_positive_checklist_path = args.ablation_final_positive_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        ablation_final_negative_checklist_path = args.ablation_final_negative_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        ablation_filtered_positive_checklist_path = args.ablation_filtered_positive_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        ablation_filtered_negative_checklist_path = args.ablation_filtered_negative_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        
        ablation_positive_checklist_path = args.ablation_positive_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        ablation_negative_checklist_path = args.ablation_negative_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        
        ablation_positive_checklist = safe_load_json(ablation_positive_checklist_path) 
        ablation_negative_checklist = safe_load_json(ablation_negative_checklist_path) 
        
        miss_ablation_positive_path = args.miss_ablation_positive_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        miss_ablation_negative_path = args.miss_ablation_negative_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        
        miss_ablation_positive_checklist = safe_load_json(miss_ablation_positive_path) 
        miss_ablation_negative_checklist = safe_load_json(miss_ablation_negative_path) 
        
        pos_checklist = ablation_positive_checklist + miss_ablation_positive_checklist 
        neg_checklist = ablation_negative_checklist + miss_ablation_negative_checklist

        pos_hist_path = args.positive_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        neg_hist_path = args.negative_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

        make_output_dir(ablation_final_positive_checklist_path)
        make_output_dir(ablation_final_negative_checklist_path)
        make_output_dir(ablation_filtered_positive_checklist_path)
        make_output_dir(ablation_filtered_negative_checklist_path)
        make_output_dir(pos_hist_path)
        make_output_dir(neg_hist_path)
        
        pos_scores, pos_checklists, final_pos_checklist, filtered_pos_checklist = classificate_ablation_improvement_scores(
            pos_checklist, policy, "positive", pos_hist_path
        )
        neg_scores, neg_checklists, final_neg_checklist, filtered_neg_checklist = classificate_ablation_improvement_scores(
            neg_checklist, policy, "negative", neg_hist_path
        )
        
        # 各サブセットのカウント
        policy_filtered_pos_count += len(filtered_pos_checklist)
        policy_final_pos_count += len(final_pos_checklist)
        policy_filtered_neg_count += len(filtered_neg_checklist)
        policy_final_neg_count += len(final_neg_checklist)

        for item in filtered_pos_checklist:
            item_with_metadata = item.copy() if isinstance(item, dict) else {'original_item': item}
            item_with_metadata['policy'] = policy

            all_policies_filtered_pos_checklist.append(item_with_metadata)
            
        for item in filtered_neg_checklist:
            item_with_metadata = item.copy() if isinstance(item, dict) else {'original_item': item}
            item_with_metadata['policy'] = policy
            all_policies_filtered_neg_checklist.append(item_with_metadata)
            
        for item in final_pos_checklist:
            item_with_metadata = item.copy() if isinstance(item, dict) else {'original_item': item}
            item_with_metadata['policy'] = policy
            all_policies_final_pos_checklist.append(item_with_metadata)
            
        for item in final_neg_checklist:
            item_with_metadata = item.copy() if isinstance(item, dict) else {'original_item': item}
            item_with_metadata['policy'] = policy
            all_policies_final_neg_checklist.append(item_with_metadata)

        save_json(ablation_final_positive_checklist_path, final_pos_checklist)
        save_json(ablation_final_negative_checklist_path, final_neg_checklist)
        save_json(ablation_filtered_positive_checklist_path, filtered_pos_checklist)
        save_json(ablation_filtered_negative_checklist_path, filtered_neg_checklist)

        # 全体集計用に追加
        all_positive_scores.extend(pos_scores)
        all_negative_scores.extend(neg_scores)

        # ポリシーごとの全体数を更新
        grand_total_pos += len(all_positive_scores)
        grand_filtered_pos += policy_filtered_pos_count
        grand_final_pos += policy_final_pos_count
        grand_total_neg += len(all_negative_scores)
        grand_filtered_neg += policy_filtered_neg_count
        grand_final_neg += policy_final_neg_count

        # === policy単位の統計処理（"all"用） ===
        pos_hist_path_all = args.positive_combined_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        neg_hist_path_all = args.negative_combined_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

        pos_dir = os.path.dirname(pos_hist_path_all)
        neg_dir = os.path.dirname(neg_hist_path_all)
        make_output_dir(pos_dir)
        make_output_dir(neg_dir)


        plot_histgram_statistics(all_positive_scores, policy, pos_hist_path_all, "positive")
        plot_histgram_statistics(all_negative_scores, policy, neg_hist_path_all, "negative")

    print("===== Overall Summary (All Policies) =====")
    if grand_total_pos > 0:
        grand_pos_filtered_ratio = grand_filtered_pos / grand_total_pos
        grand_pos_final_ratio = grand_final_pos / grand_total_pos
        print(f"Total Positive filtered: {grand_filtered_pos}/{grand_total_pos} ({grand_pos_filtered_ratio:.1%})")
        print(f"Total Positive final: {grand_final_pos}/{grand_total_pos} ({grand_pos_final_ratio:.1%})")
    else:
        print("No positive scores found.")

    if grand_total_neg > 0:
        grand_neg_filtered_ratio = grand_filtered_neg / grand_total_neg
        grand_neg_final_ratio = grand_final_neg / grand_total_neg
        print(f"Total Negative filtered: {grand_filtered_neg}/{grand_total_neg} ({grand_neg_filtered_ratio:.1%})")
        print(f"Total Negative final: {grand_final_neg}/{grand_total_neg} ({grand_neg_final_ratio:.1%})")
    else:
        print("No negative scores found.")

if __name__ == '__main__':
    main()






