#!/bin/bash
set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <checklist_model_name> <eval_model_name> <checklist_type>"
    exit 1
fi

CHECKLIST_MODEL_NAME=$1 # e.g. gpt-4o-2024-08-06
EVAL_MODEL_NAME=$2 # e.g. gpt-4o-2024-08-06 or meta-llama/Llama-3.1-8B-Instruct
CHECKLIST_TYPE=$3 # e.g. detailpp

echo "Checklist model name: ${CHECKLIST_MODEL_NAME}"
echo "Evaluation model name: ${EVAL_MODEL_NAME}"
echo "Checklist type: ${CHECKLIST_TYPE}"

SIMPLE_CHECKLIST_MODEL_NAME=$(echo "$CHECKLIST_MODEL_NAME" | sed 's|/|_|g')
SIMPLE_EVAL_MODEL_NAME=$(echo "$EVAL_MODEL_NAME" | sed 's|/|_|g')

echo "Simple checklist model name: ${SIMPLE_CHECKLIST_MODEL_NAME}"
echo "Simple evaluation model name: ${SIMPLE_EVAL_MODEL_NAME}"

DATASET_PATH="./Dataset/LLMBar/dataset.json"

echo "Evaluating without checklist..."
BASELINE_EVAL_OUTPUT_PATH="./outputs/evaluation/baseline/LLMBar/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
echo "Baseline eval output path: ${BASELINE_EVAL_OUTPUT_PATH}"
python3 src/experiment/evaluate_response.py \
    --model-name-or-path ${EVAL_MODEL_NAME} \
    --base-prompt-path ./data/prompt/pairwise_evaluate_response/no_checklist.txt \
    --dataset-path ${DATASET_PATH} \
    --output-path ${BASELINE_EVAL_OUTPUT_PATH} \
    -n 10

echo "Generating checklist..."
CHECKLIST_PROMPT_PATH="data/prompt/generate_checklist/${CHECKLIST_TYPE}.txt"
CHECKLIST_PATH="./outputs/checklist/${CHECKLIST_TYPE}/LLMBar/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"
echo "Checklist path: ${CHECKLIST_PATH}"
python3 src/experiment/generate_checklist.py \
    --model-name-or-path ${CHECKLIST_MODEL_NAME} \
    --base-prompt-path ${CHECKLIST_PROMPT_PATH} \
    --dataset-path ${DATASET_PATH} \
    --output-path ${CHECKLIST_PATH}

echo "Adjust checklist..."
ADJUST_CHECKLIST_PROMPT_PATH="data/prompt/generate_checklist/adjust_${CHECKLIST_TYPE}.txt"
ADJUST_CHECKLIST_PATH="./outputs/checklist/adjust_${FACTOR}_${CHECKLIST_TYPE}/LLMBar/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"
echo "Adjust checklist path: ${ADJUST_CHECKLIST_PATH}"
python3 src/experiment/adjust_checklist.py \
    --model-name-or-path ${CHECKLIST_MODEL_NAME} \
    --base-prompt-path ${ADJUST_CHECKLIST_PROMPT_PATH} \
    --dataset-path ${DATASET_PATH} \
    --base-checklist-path ${CHECKLIST_PATH} \
    --output-path ${ADJUST_CHECKLIST_PATH} \
    --factor ${FACTOR}

echo "Evaluating with checklist..."
CHECKLIST_EVAL_OUTPUT_PATH="./outputs/evaluation/checklist/adjust_${FACTOR}_${CHECKLIST_TYPE}:${SIMPLE_CHECKLIST_MODEL_NAME}/LLMBar/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
echo "Checklist eval output path: ${CHECKLIST_EVAL_OUTPUT_PATH}"
python3 src/experiment/evaluate_response.py \
    --model-name-or-path ${EVAL_MODEL_NAME} \
    --base-prompt-path ./data/prompt/pairwise_evaluate_response/checklist.txt \
    --dataset-path ${DATASET_PATH} \
    --checklist-path ${ADJUST_CHECKLIST_PATH} \
    --output-path ${CHECKLIST_EVAL_OUTPUT_PATH} \
    -n 10

echo "Calculating score of evaluation without checklist..."
python3 src/experiment/calc_score.py \
    --pairwise-judge-path ${BASELINE_EVAL_OUTPUT_PATH}

echo "Calculating score of evaluation with checklist..."
python3 src/experiment/calc_score.py \
    --pairwise-judge-path ${CHECKLIST_EVAL_OUTPUT_PATH}

for threshold in 1 2 3 4 5
do
    echo "Calculating merge score of evaluation with and without checklist... (threshold: ${threshold})"
    python3 src/experiment/calc_merge_score.py \
        --pairwise-baseline-judge-path ${BASELINE_EVAL_OUTPUT_PATH} \
        --pairwise-checklist-judge-path ${CHECKLIST_EVAL_OUTPUT_PATH} \
        --threshold ${threshold}
done