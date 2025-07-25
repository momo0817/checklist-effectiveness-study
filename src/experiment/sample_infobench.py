import os
import glob
import json
import argparse

import pandas as pd

PROMPT_TEMPLATE = """\
{instruction}

{input}"""


def load_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--infobench-dir",
        type=str,
        help="Path to the dataset directory",
        default="./InFoBench/expert_annotation",
    )
    args.add_argument(
        "--output-path",
        type=str,
        help="Path to the output path",
        default="./Dataset/InFoBench/dataset.json",
    )

    args.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=1234,
    )


    return args.parse_args()


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def decompose(df):
    models = ["gpt-3.5-turbo", "gpt-4", "claude-v1", "alpaca-7b", "vicuna-13b"]
    decomposed_data = []

    for _, row in df.iterrows():
        for model in models:
            input_text = row["input"]
            instruct = row["instruction"]

            if input_text is not None:
                instruct = PROMPT_TEMPLATE.format(
                    instruction=instruct, input=input_text
                )

            decomposed_data.append(
                {
                    "id": f"{row['id']}:{model}",
                    "model": model,
                    "input": instruct,
                    "output": row[model],
                    "labels": {
                        "annotator1": row[f"{model}-annotation-overall-annotator1"],
                        "annotator2": row[f"{model}-annotation-overall-annotator2"],
                        "annotator3": row[f"{model}-annotation-overall"],
                    },
                }
            )

    return decomposed_data


def main():
    args = load_args()

    file_paths = glob.glob(os.path.join(args.infobench_dir, "*.csv"))
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
        print(f"{args.output_path} は存在しています。")
    else:
        print(f"{args.output_path} は存在していません。")

        all_data = []
        for file_path in file_paths:
            if "Easy" in os.path.basename(file_path):
                difficulty = "easy"
            elif "Hard" in os.path.basename(file_path):
                difficulty = "hard"
            else:
                raise ValueError("Unknown difficulty")

            df = pd.read_csv(file_path)
            df = df.head(25)
            df = df.where(pd.notnull(df), None)

            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

            # Split the dataframe into dev and test sets
            for d in decompose(df):
                d["subset"] = difficulty  # 難易度情報は保持
                all_data.append(d)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        save_json(args.output_path, all_data)

if __name__ == "__main__":
    main()
