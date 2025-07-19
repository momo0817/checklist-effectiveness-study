import argparse

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

from utils.data import load_jsonl

import krippendorff


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--absolute-rating-path",
        type=str,
        help="Path to the dataset",
        default="./outputs/abs_evaluation/baseline/InFoBench_expert_annotation/gpt-4o-2024-08-06.jsonl",
    )

    return parser.parse_args()


def round_half_up(value, decimals="0"):
    round_value = Decimal(value).quantize(Decimal(decimals), rounding=ROUND_HALF_UP)
    return int(round_value)


def calc_ave_score(scores):
    if len(scores) == 0:
        return 3

    return round_half_up(sum(scores) / len(scores))


def main():
    args = load_args()
    abs_evaluation = load_jsonl(args.absolute_rating_path)
    abs_ratings = [calc_ave_score(d["scores"]) for d in abs_evaluation]
    abs_labels = [calc_ave_score(list(d["label"].values())) for d in abs_evaluation]

    reliability_data = [abs_ratings, abs_labels]
    reliability = krippendorff.alpha(reliability_data, level_of_measurement="ordinal")
    print(f"Krippendorff's Alpha: {reliability}")


if __name__ == "__main__":
    main()
