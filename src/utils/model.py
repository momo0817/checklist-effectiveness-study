import os
import time

import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from openai import AzureOpenAI

from anthropic import AnthropicBedrock
from vllm import LLM, SamplingParams

load_dotenv(override=True)
SEED = 1234


def load_model(model_name_or_path):
    if model_name_or_path.startswith("gpt"):
        return GPT(model=model_name_or_path)
    elif model_name_or_path.startswith("anthropic"):
        return Anthropic(model=model_name_or_path)
    else:
        return VLLM(model_name_or_path)


class GPT:
    def __init__(self, model="gpt-4o-2024-08-06"):
        self.model = model
        self.client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ.get("OPENAI_API_VERSION", "2023-05-15"),
            azure_endpoint=os.environ["OPENAI_API_BASE"],
        )

    def __call__(self, system_content, user_content, n=1, temperature=1.0, top_p=0.95):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            top_p=top_p,
            n=n,
        )
        time.sleep(1)
        if n == 1:
            return response.choices[0].message.content
        return [c.message.content for c in response.choices]

class Anthropic:
    def __init__(self, model="anthropic.claude-3-5-sonnet-20240620-v1:0"):
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        if aws_access_key is None:
            aws_access_key = input("Enter AWS Bedrock access key: ")

        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        if aws_secret_key is None:
            aws_secret_key = input("Enter AWS Bedrock secret key: ")

        aws_region = os.getenv("AWS_REGION")
        if aws_region is None:
            aws_region = input("Enter AWS Bedrock region: ")

        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )
        self.model_name = model

    def __call__(self, system_content, user_content, n=1, temperature=1.0, top_p=0.95):
        if n != 1:
            raise ValueError("Anthropic only supports n=1")

        messages = [
            {"role": "user", "content": user_content},
        ]

        completions = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system_content,
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
        )
        time.sleep(1)
        return completions.content[0].text


class VLLM:
    def __init__(self, model_name_or_path, num_gpus=None):
        download_dir = os.getenv("HF_CACHE_DIR", "./data/cache")

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        try:
            self.model = LLM(
                model=model_name_or_path,
                download_dir=download_dir,
                tensor_parallel_size=num_gpus,
                seed=SEED,
                max_model_len=2048, 
                enforce_eager=True,  
                trust_remote_code=True,
            )
            self.model_name = model_name_or_path
        except Exception as e:
            print(f"VLLM初期化エラー: {e}")
            raise

    def __call__(self, system_content, user_content, n=1, temperature=1.0, top_p=0.95):
        params = SamplingParams(
            max_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )
        
        if "gemma" in self.model_name.lower():
            messages = [
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            
        outputs = self.model.chat(
            messages=[messages], sampling_params=params, use_tqdm=False
        )
        return [output.text for output in outputs[0].outputs]


if __name__ == "__main__":
    model = load_model("gpt-4o-2024-08-06")
    print("gpt-4o-2024-08-06")
    print(model("You are a helpful assistant.", "What is the capital of Tokyo?"))

    model = load_model("anthropic.claude-3-5-sonnet-20240620-v1:0")
    print("anthropic.claude-3-5-sonnet-20240620-v1:0")
    print(model("You are a helpful assistant.", "What is the capital of Tokyo?"))

    model = load_model("meta-llama/Llama-3.1-8B-Instruct")
    print("meta-llama/Llama-3.1-8B-Instruct")
    print(model("You are a helpful assistant.", "What is the capital of Tokyo?"))

    model = load_model("google/gemma-2-27b-it")
    print("google/gemma-2-27b-it")
    print(model("You are a helpful assistant.", "What is the capital of Tokyo?"))
