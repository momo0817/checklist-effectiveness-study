import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.data import load_json, save_json, make_output_dir
from collections import defaultdict

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
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablation/pairwise/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--ablation_final_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_final/pairwise/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_final_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_final/pairwise/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--ablation_filtered_positive_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_filtered/pairwise/{policy}:{checklist_model}/{eval_model}/positive/ablation_result.json"
    )
    args.add_argument(
        "--ablation_filtered_negative_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/ablated_filtered/pairwise/{policy}:{checklist_model}/{eval_model}/negative/ablation_result.json"
    )
    args.add_argument(
        "--negative_histgram_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/pairwise/LLMBar/histgrams/negative/{policy}:{checklist_model}/{eval_model}/final_result.pdf"
    )
    args.add_argument(
        "--positive_histgram_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/pairwise/LLMBar/histgrams/positive/{policy}:{checklist_model}/{eval_model}/final_result.pdf"
    )
    
    return args.parse_args()

def filter_stable_judgements(ablation_path, filtered_output_path):
    stable_items = []
    unstable_count = 0

    items = load_json(ablation_path)
    for item in tqdm(items, desc="Filtering unstable items"):
        judges = [res["judge"] for res in item.get("dict_responses", [])]
        if len(set(judges)) > 1:
            unstable_count += 1
            continue
        stable_items.append(item)

    print(f"{unstable_count} / {len(items)} items were excluded due to unstable judgement (order bias)")
    print(f"Bad checklist items: {len(items) - unstable_count}")
    save_json(filtered_output_path, stable_items)
    return len(items), unstable_count, len(stable_items)


def plot_histgram_statistics(ablation_improvement_scores,  policy, output_path, pos_or_neg, threshold=None, stats_path=None):
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
        x_min_limit = -1.05
        x_max_limit = 1.15
    else:
        x_min_limit = -1.05
        x_max_limit = 1.15
    
    max_abs_value = max(abs(min_score), abs(max_score)) if ablation_improvement_scores else 1
    # データポイントが少ない場合の対応
    if len(ablation_improvement_scores) < 10:
        # データポイントが少ない場合はスウォームプロットを追加
        plt.subplot(2, 1, 1)
        sns.swarmplot(x=ablation_improvement_scores, color='blue', size=16)
        plt.axvline(x=0, color='r', linestyle='--', label='No Change')

        plt.xlabel('Improvement Score', fontsize=16)
        policy_title_map = {
            "baseline": "Baseline",
            "ticking": "Ticking",
            "detail": "Specify",
            "adjust_0.5_baseline": "Length*0.5",
            "adjust_1.5_baseline": "Length*1.5",
            "refine_baseline": "Self-refine"
        }


        # 表示用タイトルだけ変換
        display_policy = policy_title_map.get(policy, policy)
    
        display_pos_neg = "Negative" if pos_or_neg == "negative" else "Positive"
        plt.title(f'Distribution of {display_pos_neg} Improvement Scores - {display_policy}')
        plt.legend()
        plt.xlim(x_min_limit, x_max_limit)
        x_ticks = np.arange(x_min_limit, x_max_limit, 0.1)  
        major_ticks = np.arange(-1.0, 1.1, 0.1)
        plt.xticks(major_ticks)# 11分割（-x_limit, -0.8*x_limit, ..., 0, ..., 0.8*x_limit, x_limit）
        plt.axvline(x=0, color='r', linestyle='--', label='No Change')
        plt.subplot(2, 1, 2)
    
    # ビンの数を調整（データポイントが少ない場合は少なく、多い場合は多く）
    n_bins = min(max(5, len(ablation_improvement_scores) // 2), 30)
    
    # ヒストグラムとKDEプロット
    sns.histplot(ablation_improvement_scores, kde=True, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
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
        plt.figtext(0.68, 0.65, stats_text, fontsize=16, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.xlabel('Contribution Score', fontsize=19)
    plt.ylabel('Frequency', fontsize=19, labelpad=5)  
    policy_title_map = {
            "baseline": "Baseline",
            "ticking": "Ticking",
            "detail": "Specify",
            "adjust_0.5_baseline": "Length*0.5",
            "adjust_1.5_baseline": "Length*1.5",
            "refine_baseline": "Self-refine"
        }

        # 表示用タイトルだけ変換
    display_policy = policy_title_map.get(policy, policy)
    display_pos_neg = "Negative" if pos_or_neg == "negative" else "Positive"
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    
    plt.xlim(x_min_limit, x_max_limit)
    x_ticks = np.arange(x_min_limit, x_max_limit, 0.1) # 11分割
    plt.axvline(x=0, color='r', linestyle='--', label='No Change')
    major_ticks = np.arange(-1.0, 1.1, 0.1)
    label_ticks = major_ticks[::2] 
    plt.xticks(label_ticks, fontsize=16)  # x軸の目盛りを設定
    plt.yticks(fontsize=16)
    
    # 出力ディレクトリが存在しない場合は作成
    make_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)  # 高解像度で保存
    plt.close()
    
    print(f"Histogram saved to: {output_path}")
    
    return statistics

def classificate_ablation_improvement_scores(
    checklist: list,
    policy: str,
    score_type: str,  # "positive" or "negative"
    hist_path: str
) -> Tuple[list, list, list, list]:
    """ablation_improvement_score を分類し、スコアと元データを返す（チェックリストごとに平均）"""

    # removed_item ごとに entry を集約
    checklist_by_removed_item = defaultdict(list)
    
    for entry in checklist:
        q = entry["removed_checklist_item"]
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
        q = entry["removed_checklist_item"]

        if (score_type == "positive" and score < -0.9) or (score_type == "negative" and score > 0.9):
            filtered_scores.append(score)
            filtered_checklist.append(entry)

        if (score_type == "positive" and score < 0) or (score_type == "negative" and score > 0):
            final_scores.append(score)
            final_checklist.append(entry)

    # ヒストグラムと円グラフ出力
    plot_histgram_statistics(scores,  policy, hist_path, score_type)

    return scores, checklists, final_checklist, filtered_checklist

def main():
    args = load_args()
    checklist_model = args.checklist_model.replace("/", "_")
    eval_model = args.eval_model.replace("/", "_")
    
    checklist_generation_policies = [
            "baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
            "ticking", "refine_baseline", "detail"
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
        
        raw_pos_path = args.ablation_positive_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        raw_neg_path = args.ablation_negative_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

        # フィルタ済みの一時ファイルを用意
        tmp_filtered_pos_path = f"./tmp_filtered_pos_{policy}.json"
        tmp_filtered_neg_path = f"./tmp_filtered_neg_{policy}.json"

        _, _, _ = filter_stable_judgements(raw_pos_path, tmp_filtered_pos_path)
        _, _, _ = filter_stable_judgements(raw_neg_path, tmp_filtered_neg_path)

        # その出力を使って処理を続行
        pos_checklist = load_json(tmp_filtered_pos_path)
        neg_checklist = load_json(tmp_filtered_neg_path)

        pos_hist_path = args.positive_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)
        neg_hist_path = args.negative_histgram_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model)

        make_output_dir(ablation_final_positive_checklist_path)
        make_output_dir(ablation_final_negative_checklist_path)
        make_output_dir(ablation_filtered_positive_checklist_path)
        make_output_dir(ablation_filtered_negative_checklist_path)
        make_output_dir(pos_hist_path)
        make_output_dir(neg_hist_path)
        
        pos_scores, pos_checklists, final_pos_checklist, filtered_pos_checklist = classificate_ablation_improvement_scores(
            pos_checklist,  policy, "positive", pos_hist_path
        )
        neg_scores, neg_checklists, final_neg_checklist, filtered_neg_checklist = classificate_ablation_improvement_scores(
            neg_checklist,  policy, "negative", neg_hist_path
        )
        
        # 各サブセットのカウント
        policy_filtered_pos_count += len(filtered_pos_checklist)
        policy_final_pos_count += len(final_pos_checklist)
        policy_filtered_neg_count += len(filtered_neg_checklist)
        policy_final_neg_count += len(final_neg_checklist)
        print(f"Policy: {policy}")
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
        
        make_output_dir(pos_hist_path_all)
        make_output_dir(neg_hist_path_all)


        plot_histgram_statistics(all_positive_scores, "all", policy, pos_hist_path_all, "positive")
        plot_histgram_statistics(all_negative_scores, "all", policy, neg_hist_path_all, "negative")
    
    
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


