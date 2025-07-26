import re
import json
import math
import argparse

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from utils.data import load_prompt, load_jsonl, load_json, save_jsonl
from utils.model import load_model


def load_args():
    parser = argparse.ArgumentParser(
        description="Generate a checklist for a given questions"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset",
        default="./InFoBench/expert_annotation/dataset.json",
    )
    parser.add_argument(
        "--checklist-path",
        type=str,
        help="Path to the checklist",
        default="./outputs/checklist/baseline/InFoBench/gpt-4o-2024-08-06.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output file",
        default="./outputs/abs_evaluation/baseline/InFoBench/gpt-4o-2024-08-06.jsonl",
    )

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="Model name or path to generate the checklist",
        default="gpt-4o-2024-08-06",  # or "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    parser.add_argument(
        "--base-prompt-path",
        type=str,
        help="Path to the base prompt file. If the prompt not contains {checklist}, the checklist will be ignored.",
        default="./data/prompt/abs_evaluate_response/checklist.txt",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to generate the checklist",
        default="You are a helpful assistant.",
    )

    parser.add_argument(
        "-n",
        "--num-evaluation-trials",
        type=int,
        help="Number of evaluation trials.",
        default=10,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for the generation",
        default=1.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top probability for the generation",
        default=0.95,
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        help="Max number of retries to generate the checklist",
        default=3,
    )

    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Run in debug mode",
    )

    return parser.parse_args()


def request(
    system_prompt, prompt, model, n, max_retries=3, temperature=1.0, top_p=0.95
):
    num_retries = 0
    scores = []
    responses = []
    dict_responses = []
    while num_retries <= max_retries and len(scores) < n:
        try:
            raw_responses = model(
                system_prompt,
                prompt,
                n=n - len(scores),
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            print(f"Error: {e}")
            num_retries += 1
            continue
        for response in raw_responses:
            try:
                dict_response = parse_response(response)
                rating = int(dict_response["rating"])
            except Exception as e:
                continue
            if rating not in [1, 2, 3, 4, 5]:
                continue
            scores.append(rating)
            responses.append(response)
            dict_responses.append(dict_response)

        if len(scores) < n:
            print(f"Response parse error: {n-len(scores)}/{n}")
            num_retries += 1

    return scores, responses, dict_responses


def parse_response(response):
    if "```json" in response:
        response = re.match(r"```json((.|\s)*?)```", response).group(1)
    return json.loads(response)


def main():
    args = load_args()

    if os.path.exists(args.output_path):
        print(f"Output file already exists: {args.output_path}")
        return

    model = load_model(args.model_name_or_path)
    dataset = load_json(args.dataset_path)
    base_prompt = load_prompt(args.base_prompt_path)

    if "{checklist}" in base_prompt:
        checklist = load_jsonl(args.checklist_path)
        qustion2checklist = {d["question"]: d["checklist"] for d in checklist}

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    results = []
    for idx, d in enumerate(tqdm(dataset)):
        question = d["input"]
        response = d["output"]

        if "{checklist}" in base_prompt:
            checklist_items = qustion2checklist[question]
            if checklist_items is None:
                continue
            text_checklist_items = "\n".join(
                [f"  {i+1}. {c}" for i, c in enumerate(checklist_items)]
            )
            prompt = base_prompt.format(
                question=question,
                response=response,
                checklist=text_checklist_items,
            )
        else:
            prompt = base_prompt.format(
                question=question,
                response=response,
            )

        scores, responses, dict_responses = request(
            args.system_prompt,
            prompt,
            model,
            n=args.num_evaluation_trials,
            max_retries=args.max_retries,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        if args.debug_mode:
            print(
                f"------------------System Prompt------------------\n{args.system_prompt}\n"
            )
            print(f"------------------Prompt------------------\n{prompt}\n")
            print(f"------------------Responses------------------\n{responses}\n")
            print(f"------------------Scores------------------\n{scores}\n")
            return

        result = {
            "idx": idx,
            "question": question,
            "response": response,
            "label": dataset[idx]["labels"],
            "prompt": prompt,
            "responses": responses,
            "dict_responses": dict_responses,
            "scores": scores,
        }
        results.append(result)

    save_jsonl(args.output_path, results)


if __name__ == "__main__":
    main()
