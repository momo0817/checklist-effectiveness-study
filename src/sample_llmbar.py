import os
import random
import argparse

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
        "--output-dir",
        type=str,
        help="Path to the output file",
        default="./LLMBar/Dataset/Sample",
    )

    return parser.parse_args()


def main():
    args = load_args()

    assert (
        args.dev_sampling_rate + args.test_sampling_rate <= 1
    ), "Dev and test sampling rates should sum to 1"

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
    os.makedirs(os.path.join(args.output_dir, "All"), exist_ok=True)
    save_json(os.path.join(args.output_dir, "All", "dataset.json"), all_samples)


if __name__ == "__main__":
    main()
