import argparse

from collections import Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils.data import load_jsonl
from calc_score import make_final_judge


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pairwise-baseline-judge-path",
        type=str,
        help="Path to the dataset",
        required=True,
    )
    parser.add_argument(
        "--pairwise-checklist-judge-path",
        type=str,
        help="Path to the dataset",
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        help="Threshold for determining whether to use the checklist. If the baseline evaluation fluctuates beyond this threshold, a re-evaluation is performed.",
        default=1,
    )

    return parser.parse_args()


def main():
    args = load_args()
    pairwise_baseline_judge = load_jsonl(args.pairwise_baseline_judge_path)
    pairwise_checklist_judge = load_jsonl(args.pairwise_checklist_judge_path)

    pairwise_checklist_judge = {d["idx"]: d for d in pairwise_checklist_judge}

    corrects = []
    for d in pairwise_baseline_judge:
        judge_count = Counter(d["judges"])

        if (
            len(judge_count) >= 2
            and judge_count.most_common()[-1][-1] >= args.threshold
        ):
            if d["idx"] not in pairwise_checklist_judge:
                print(f"Checklist evaluation not found: {d['idx']}")
                final_judge = make_final_judge(d["judges"])
            else:
                final_judge = make_final_judge(
                    pairwise_checklist_judge[d["idx"]]["judges"]
                )
        else:
            final_judge = make_final_judge(d["judges"])
        corrects.append(d["label"] == final_judge)

    accuracy = sum(corrects) / len(corrects)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
