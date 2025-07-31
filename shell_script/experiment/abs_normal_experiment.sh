set -e

if [ $# -ne 3 ] && [ $# -ne 4 ]; then
    echo "Usage: $0 <checklist_model_name> <eval_model_name> <checklist_type> [factor]"
    exit 1
fi

CHECKLIST_MODEL_NAME=$1 # e.g. gpt-4o-2024-08-06
EVAL_MODEL_NAME=$2 # e.g. gpt-4o-2024-08-06 or meta-llama/Llama-3.1-8B-Instruct
CHECKLIST_TYPE=$3 # e.g. specify
FACTOR=${5:-1.5}  

echo "Checklist model name: ${CHECKLIST_MODEL_NAME}"
echo "Evaluation model name: ${EVAL_MODEL_NAME}"
echo "Checklist type: ${CHECKLIST_TYPE}"

SIMPLE_CHECKLIST_MODEL_NAME=$(echo "$CHECKLIST_MODEL_NAME" | sed 's|/|_|g')
SIMPLE_EVAL_MODEL_NAME=$(echo "$EVAL_MODEL_NAME" | sed 's|/|_|g')

echo "Simple checklist model name: ${SIMPLE_CHECKLIST_MODEL_NAME}"
echo "Simple evaluation model name: ${SIMPLE_EVAL_MODEL_NAME}"

DATASET_PATH="./Dataset/InFoBench/dataset.json"

echo "Evaluating without checklist..."
BASELINE_EVAL_OUTPUT_PATH="./outputs/abs_evaluation/no_checklist/InFoBench/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
echo "Baseline eval output path: ${BASELINE_EVAL_OUTPUT_PATH}"
python3 src/experiment/abs_evaluate_response.py \
    --model-name-or-path ${EVAL_MODEL_NAME} \
    --base-prompt-path ./data/prompt/abs_evaluate_response/no_checklist.txt \
    --dataset-path ${DATASET_PATH} \
    --output-path ${BASELINE_EVAL_OUTPUT_PATH} \
    -n 5

echo "Generating checklist..."
CHECKLIST_PROMPT_PATH="./data/prompt/generate_checklist/${CHECKLIST_TYPE}.txt"
CHECKLIST_PATH="./outputs/checklist/${CHECKLIST_TYPE}/InFoBench/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"
echo "Checklist path: ${CHECKLIST_PATH}"
python3 src/experiment/generate_checklist.py \
    --model-name-or-path ${CHECKLIST_MODEL_NAME} \
    --base-prompt-path ${CHECKLIST_PROMPT_PATH} \
    --dataset-path ${DATASET_PATH} \
    --output-path ${CHECKLIST_PATH}

echo "Evaluating with checklist..."
CHECKLIST_EVAL_OUTPUT_PATH="./outputs/abs_evaluation/checklist/${CHECKLIST_TYPE}:${SIMPLE_CHECKLIST_MODEL_NAME}/InFoBench/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
echo "Checklist eval output path: ${CHECKLIST_EVAL_OUTPUT_PATH}"
python3 src/experiment/abs_evaluate_response.py \
    --model-name-or-path ${EVAL_MODEL_NAME} \
    --base-prompt-path ./data/prompt/abs_evaluate_response/checklist.txt \
    --dataset-path ${DATASET_PATH} \
    --checklist-path ${CHECKLIST_PATH} \
    --output-path ${CHECKLIST_EVAL_OUTPUT_PATH} \
    -n 5

echo "Calculating score of evaluation without checklist..."
python3 src/experiment/abs_calc_krippendorff_alpha.py \
    --absolute-rating-path ${BASELINE_EVAL_OUTPUT_PATH}

echo "Calculating score of evaluation with checklist..."
python3 src/experiment/abs_calc_krippendorff_alpha.py \
    --absolute-rating-path ${CHECKLIST_EVAL_OUTPUT_PATH}
