import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from collections import Counter

from utils.data import load_jsonl


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pairwise-judge-path",
        type=str,
        help="Path to the dataset",
        default="./outputs/evaluation/no_checklist/LLMBar/gpt-4o-08-06.jsonl",
    )

    return parser.parse_args()


def make_final_judge(judges):
    if len(judges) == 0:
        return -1

    judge_counter = Counter(judges)
    most_common_judge = judge_counter.most_common(1)[0][0]
    if len(judges) % 2 == 0 and len(judges) // 2 == judge_counter[most_common_judge]:
        return -1
    return most_common_judge


def main():
    args = load_args()
    pairwise_judge = load_jsonl(args.pairwise_judge_path)

    corrects = [d["label"] == make_final_judge(d["judges"]) for d in pairwise_judge]
    accuracy = sum(corrects) / len(corrects)

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
