import os
import re
import argparse

from utils.data import load_prompt, load_quesions, load_jsonl, save_jsonl
from utils.model import load_model

from tqdm import tqdm


def load_args():
    parser = argparse.ArgumentParser(
        description="Generate a checklist for a given questions"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset",
        default="./LLMBar/Dataset/Sample/dataset.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output file",
        default="./outputs/checklist/baseline/gpt-4o.jsonl",
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
        help="Path to the base prompt file",
        default="data/prompt/generate_checklist/baseline.txt",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to generate the checklist",
        default="You are a helpful assistant.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling",
        default=1.0,
    )
    parser.add_argument(
        "--top-p",
        type=int,
        help="Top k for sampling",
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


def parse_response(response):
    return re.findall(r"\[\[(.*?)\]\]", response)


def main():
    args = load_args()

    if os.path.exists(args.output_path):
        print(f"Output file already exists: {args.output_path}")
        return

    model = load_model(args.model_name_or_path)
    questions = load_quesions(args.dataset_path)
    base_prompt = load_prompt(args.base_prompt_path)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    results = []
    for question in tqdm(questions, desc="Generating checklist"):
        prompt = base_prompt.format(question=question)

        num_retries = 0
        checklist = None
        response = None
        while num_retries <= args.max_retries:
            try:
                response = model(
                    args.system_prompt,
                    prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                print(f"Error: {e}")
                num_retries += 1
                continue
            checklist = parse_response(response)

            if len(checklist) > 0:
                break

            num_retries += 1

        if args.debug_mode:
            print(f"------------------Question------------------\n{question}\n")
            print(
                f"------------------System Prompt------------------\n{args.system_prompt}\n"
            )
            print(f"------------------Prompt------------------\n{prompt}\n")
            print(f"------------------Checklist------------------\n{checklist}")
            return

        result = {
            "question": question,
            "prompt": prompt,
            "response": response,
            "checklist": checklist,
        }
        results.append(result)

    save_jsonl(args.output_path, results)


if __name__ == "__main__":
    main()
