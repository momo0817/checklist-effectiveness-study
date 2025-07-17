import json
import os


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_quesions(file_path):
    data = load_json(file_path)
    questions = [d["input"] for d in data]
    return sorted(set(questions))

def make_output_dir(filepath: str):
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)
