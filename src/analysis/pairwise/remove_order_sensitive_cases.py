import os
import sys
import argparse
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from utils.data import load_json, save_json, make_output_dir

logger = logging.getLogger(__name__)


def load_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--variation_type",
        type = str,
        help = "Type of variation",
        default = "all"
    )
    args.add_argument(
        "--checklist_model_name_or_path",
        type = str,
        help = "generate checklist model",
        default = "gpt-4o-2024-08-06"
    )
    args.add_argument(
        "--eval_model_name_or_path",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
    )
    args.add_argument(
        "--ablation_change_bad_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/bad_checklists/pairwise/{policy}:{checklist_model}/{eval_model}/ablation/change_bad_checklist/{inconsistency_threshold}_result.json"
    )
    args.add_argument(
        "--ablation_stay_bad_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/bad_checklists/pairwise/{policy}:{checklist_model}/{eval_model}/ablation/stay_bad_checklist/{inconsistency_threshold}_result.json"
    )
    args.add_argument(
        "--debug-mode",
        action="store_true",
        help="Run in debug mode",
    )

    args.add_argument(
        "--final_change_bad_checklist_path",
        type = str,
        help = "Path to the output",
        default =  "./analysis/data/bad_checklists/pairwise/{policy}:{checklist_model}/{eval_model}/final/change_bad_checklist/{inconsistency_threshold}_result.json"
    )
    args.add_argument(
        "--final_stay_bad_checklist_path",
        type = str,
        help = "Path to the output",
        default =  "./analysis/data/bad_checklists/pairwise/{policy}:{checklist_model}/{eval_model}/final/stay_bad_checklist/{inconsistency_threshold}_result.json"
    )
    args.add_argument(
        "--figure_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/figures/pairwise/LLMBar/checklist/{checklist_model}/final/{change_or_stay}_threshold_comparison.png"
    )
    args.add_argument(
        "--average_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/data/figures/pairwise/LLMBar/checklist/{checklist_model}/final/average_threshold_comparison.png"
    )
    args.add_argument(
        "--checklist_evaluation_final_stats_path",
        type = str,
        help = "Path to the evaluation stats",
        default = "./analysis/data/stats/pairwise/LLMBar/checklist/{checklist_model}/evaluation/final_{eval_model}.json"
    )
    args.add_argument(
        "--checklist_ablation_final_stats_path",
        type = str,
        help = "Path to the ablation stats",
        default = "./analysis/data/stats/pairwise/LLMBar/checklist/{checklist_model}/ablation/final_{eval_model}.json"
    )
    args.add_argument(
    "--change_or_stay",
    type = str,
    help = "Type of change_or_stay (change_bad_checklist or stay_bad_checklist)",
    default = "change_bad_checklist"
)
    return args.parse_args()
    



def filter_stable_judgements(ablation_path, filtered_output_path):
    args = load_args()
    stable_items = []
    unstable_count = 0
    
    items = load_json(ablation_path)
    for item in tqdm(items, desc="Filtering unstable items"):
        print(item)
        judges = [res["judge"] for res in item.get("dict_responses", [])]
        if len(set(judges)) > 1:
            unstable_count += 1
            continue
        stable_items.append(item)

    print(f"{unstable_count} / {len(items)} items were excluded due to unstable judgement (order bias)")
    print(f"Bad checklist items: {len(items) - unstable_count}")
    save_json(filtered_output_path, stable_items)
    return len(items), unstable_count, len(items) - unstable_count

def plot_results(all_stats, policy_list, thresholds, checklist_model, eval_model, change_or_stay):
    args = load_args()
    
    for change_or_stay in ["change_bad_checklist", "stay_bad_checklist"]:
        plt.figure(figsize=(15,10))
        plt.style.use('tableau-colorblind10')
        markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']
        linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
        
        legend_labels = []
        plot_count = 0
        
        for policy in policy_list:
                
                ratio_values = []
                for threshold in thresholds:
                    t_str = str(threshold)
                    if ("stable_ratio" in all_stats[change_or_stay][policy][t_str]):
                        ratio_values.append(all_stats[change_or_stay][policy][t_str]["stable_ratio"])
                    else:
                        ratio_values.append(0)
                if all(v==0 for v in ratio_values):
                    continue
                label = f"{policy}"
                plt.plot(
                    thresholds,
                    ratio_values,
                    marker=markers[plot_count % len(markers)],
                    linestyle=linestyles[plot_count % len(linestyles)],
                    label=label
                )
                plot_count += 1
        plt.xlabel('Inconsistency Threshold')
        plt.ylabel('Stable Ratio')
        plt.title(f'Stable Judgement Ratio by Threshold ({change_or_stay})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Y軸の範囲を0-1に設定
        plt.ylim(0, 1)
        figure_path = args.figure_path.format(
            change_or_stay=change_or_stay, 
            checklist_model=checklist_model
        )
        make_output_dir(figure_path)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 全体のサマリーグラフ（平均値）も作成
    plt.figure(figsize=(12, 8))
    
    for change_or_stay in ["change_bad_checklist", "stay_bad_checklist"]:
        avg_values = []
        for threshold in thresholds:
            t_str = str(threshold)
            total_ratio = 0
            count = 0
            
            for policy in policy_list:
                if (t_str in all_stats[change_or_stay].get(policy, {}) and 
                    "stable_ratio" in all_stats[change_or_stay][policy][t_str]):
                    total_ratio += all_stats[change_or_stay][policy][t_str]["stable_ratio"]
                    count += 1

            avg_values.append(total_ratio / count if count > 0 else 0)
        
        plt.plot(
            thresholds, 
            avg_values, 
            marker='o' if change_or_stay == "change_bad_checklist" else 's',
            linestyle='-' if change_or_stay == "change_bad_checklist" else '--',
            label=f'Average {change_or_stay}'
        )
    
    plt.xlabel('Inconsistency Threshold')
    plt.ylabel('Average Stable Ratio')
    plt.title('Average Stable Judgement Ratio by Threshold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim(0, 1)
    average_path = args.average_path.format(
        checklist_model=checklist_model
    )
    make_output_dir(average_path)
    plt.savefig(average_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphs have been saved as 'change_bad_checklist_threshold_comparison.png', 'stay_bad_checklist_threshold_comparison.png' and 'average_threshold_comparison.png'")
        

def main():
    args = load_args()
    change_or_stay = args.change_or_stay
    checklist_model = args.checklist_model_name_or_path.replace("/", "_")
    eval_model = args.eval_model_name_or_path.replace("/", "_")
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
    all_stats = {
        "change_bad_checklist":{},
        "stay_bad_checklist":{}
        
    }
    results = {
        "change_bad_checklist":{},
        "stay_bad_checklist":{}
    }

    for policy in target_policies:
        print(f"Processing policy: {policy}")
        results["change_bad_checklist"][policy] = {}
        results["stay_bad_checklist"][policy] = {}
        
        results["change_bad_checklist"][policy] = {}
        results["stay_bad_checklist"][policy] = {}

        if "change_bad_checklist" not in all_stats:
            all_stats["change_bad_checklist"] = {}
        if policy not in all_stats["change_bad_checklist"]:
            all_stats["change_bad_checklist"][policy] = {}

        if "stay_bad_checklist" not in all_stats:
            all_stats["stay_bad_checklist"] = {}
        if policy not in all_stats["stay_bad_checklist"]:
            all_stats["stay_bad_checklist"][policy] = {}

        
        for threshold in range(4,11):
            print(f"\nProcessing: policy={policy}, threshold={threshold}")
            ablation_change_path = args.ablation_change_bad_checklist_path.format(
                policy=policy, 
                checklist_model=checklist_model,
                eval_model=eval_model,
                inconsistency_threshold=threshold
            )
            
            ablation_stay_path = args.ablation_stay_bad_checklist_path.format(
                policy=policy, 
                checklist_model=checklist_model,
                eval_model=eval_model,
                inconsistency_threshold=threshold
            )

            final_change_bad_checklist_path = args.final_change_bad_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model, inconsistency_threshold=threshold)
            final_stay_bad_checklist_path = args.final_stay_bad_checklist_path.format(policy=policy, checklist_model=checklist_model, eval_model=eval_model, inconsistency_threshold=threshold)
            make_output_dir(final_change_bad_checklist_path)
            make_output_dir(final_stay_bad_checklist_path)
            total_change, unstable_change, stable_change = filter_stable_judgements(
                ablation_change_path, final_change_bad_checklist_path
            )
            total_stay, unstable_stay, stable_stay = filter_stable_judgements(
                ablation_stay_path, final_stay_bad_checklist_path
            )
            results["change_bad_checklist"][policy][threshold] = {
                "total": total_change,
                "unstable": unstable_change,
                "stable": stable_change,
                "stable_ratio": stable_change / total_change if total_change > 0 else 0
            }
            
            results["stay_bad_checklist"][policy][threshold] = {
                "total": total_stay,
                "unstable": unstable_stay,
                "stable": stable_stay,
                "stable_ratio": stable_stay / total_stay if total_stay > 0 else 0
            }
            all_stats["change_bad_checklist"][policy][str(threshold)] = {
                "total": total_change,
                "unstable": unstable_change,
                "stable": stable_change,
                "stable_ratio": stable_change / total_change if total_change > 0 else 0
            }
            
            all_stats["stay_bad_checklist"][policy][str(threshold)] = {
                "total": total_stay,
                "unstable": unstable_stay,
                "stable": stable_stay,
                "stable_ratio": stable_stay / total_stay if total_stay > 0 else 0
            }
    
    checklist_evaluation_final_stats_path = args.checklist_evaluation_final_stats_path.format(
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    
    make_output_dir(checklist_evaluation_final_stats_path)
    save_json(checklist_evaluation_final_stats_path, all_stats)
    print(f"Results saved to {checklist_evaluation_final_stats_path}")
    
    # 統計情報も保存
    checklist_ablation_final_stats_path = args.checklist_ablation_final_stats_path.format(
        checklist_model=checklist_model, 
        eval_model=eval_model
    )
    
    make_output_dir(checklist_ablation_final_stats_path)
    save_json(checklist_ablation_final_stats_path, all_stats)
    
    # グラフの作成
    plot_results(all_stats, target_policies,  [i  for i in range(4, 11)], checklist_model, eval_model, change_or_stay)

if __name__ == "__main__":
    main()
    