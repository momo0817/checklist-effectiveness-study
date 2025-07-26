import argparse
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils.data import load_json, save_json


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llmbar-dir",
        type=str,
        help="Path to the dataset",
        default="./LLMBar/Dataset",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        help="List of subsets to use",
        default=[
            "LLMBar/Adversarial/GPTInst",
            "LLMBar/Adversarial/GPTOut",
            "LLMBar/Adversarial/Manual",
            "LLMBar/Adversarial/Neighbor",
            "LLMBar/Natural",
            "Processed/FairEval",
            "Processed/LLMEval^2",
            "Processed/MT-Bench",
        ],
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset",
        default="dataset.json",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for random sampling",
        default=1234,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output file",
        default="./Dataset/LLMBar/dataset.json",
    )
     

    return parser.parse_args()


def main():
    args = load_args()

    # 親ディレクトリだけ作る
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
        print(f"{args.output_path} is exists.")
    else:
        print(f"{args.output_path} is not exists.")

    all_samples = []
    for subset in args.subsets:
        dataset_path = os.path.join(args.llmbar_dir, subset, args.dataset_name)
        dataset = load_json(dataset_path)

        print(f"Number of samples in {subset}: {len(dataset)}")

        for idx, d in enumerate(dataset):
            d["subset"] = f"{subset}_{idx}"

        random.seed(args.seed)
        random.shuffle(dataset)

        all_samples.extend(dataset)

    print(f"Number of all samples: {len(all_samples)}")
    save_json(args.output_path, all_samples)



if __name__ == "__main__":
    main()
