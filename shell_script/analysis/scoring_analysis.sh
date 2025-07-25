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

echo "Calulating scoring krippendoff alpha..."
KRIPPENDORFF_OUTPUT_PATH="./outputs/analysis/scoring/krippendorff_alpha.xlsx"
echo "Krippendorff output path: ${KRIPPENDORFF_OUTPUT_PATH}"
BOOTSTRAP_OUTPUT_PATH="./outputs/analysis/bootstrap/scoring_results.csv"
echo "Bootstrap output path: ${BOOTSTRAP_OUTPUT_PATH}"

python3 src/analysis/scoring/calc_krippendoff_alpha.py \
    --evaluation-dir "./outputs" \
    --krippendorff-output-path ${KRIPPENDORFF_OUTPUT_PATH} \
    --bootstrap-output-path ${BOOTSTRAP_OUTPUT_PATH}


CHECKLIST_STATS_PATH="./analysis/stats/scoring/InFoBench/checklist/${CHECKLIST_MODEL_NAME}/summary/${EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_EVAL_PATH="./analysis/preprocessed/scoring/InFoBench/evaluation/checklist/${CHECKLIST_TYPE}/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_NO_EVAL_PATH="./analysis/preprocessed/scoring/InFoBench/evaluation/no_checklist/${CHECKLIST_TYPE}/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_CHECKLIST_PATH="./analysis/preprocessed/scoring/InFoBench/checklist/${CHECKLIST_TYPE}/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"


python3 src/analysis/scoring/verify_checklist_scoring.py \
    --variation_type "all" \
    --question_path ${DATASET_PATH} \
    --checklist_model ${CHECKLIST_MODEL_NAME} \
    --eval_model ${EVAL_MODEL_NAME} \
    --checklist_path ${CHECKLIST_PATH} \
    --eval_path ${CHECKLIST_EVAL_OUTPUT_PATH} \
    --no_checklist_eval_path ${BASELINE_EVAL_OUTPUT_PATH} \
    --preprocessed_no_checklist_eval_path ${PREPROCESSED_NO_EVAL_PATH} \
    --preprocessed_checklist_path ${PREPROCESSED_CHECKLIST_PATH} \
    --preprocessed_eval_path ${PREPROCESSED_EVAL_PATH} \
    --checklist_stats_path ${CHECKLIST_STATS_PATH}

# python3 src/analysis/scoring/classificate_subset_scoring.py \

echo "Running ablation study..."
POSITIVE_CHECKLIST_PATH=./analysis/classification/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/checklist/positive_checklist.json
NEGATIVE_CHECKLIST_PATH=./analysis/classification/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/checklist/negative_checklist.json
ABLATION_POSITIVE_CHECKLIST_PATH=./analysis/ablation/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/ablation_result.json
ABLATION_NEGATIVE_CHECKLIST_PATH=./analysis/ablation/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/ablation_result.json
MISS_ABLATION_POSITIVE_PATH=./analysis/ablation/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/miss_result.json
MISS_ABLATION_NEGATIVE_PATH=./analysis/ablation/scoring/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/miss_result.json
CHECKLIST_ABLATION_STATS_PATH=./analysis/stats/scoring/InFoBench/checklist/${CHECKLIST_MODEL_NAME}/ablation/${EVAL_MODEL_NAME}_${CHECKLIST_TYPE}.json

python3 src/analysis/scoring/ablation_scoring.py \
    --question_path ${DATASET_PATH} \
    --variation_type "all" \
    --checklist_model ${CHECKLIST_MODEL_NAME} \
    --eval_model ${EVAL_MODEL_NAME} \
    --base_prompt_path ./data/prompt/abs_evaluate_response/checklist.txt \
    --positive_checklist_path  ${POSITIVE_CHECKLIST_PATH} \
    --negative_checklist_path  ${NEGATIVE_CHECKLIST_PATH} \
    --ablation_positive_checklist_path  ${ABLATION_POSITIVE_CHECKLIST_PATH} \
    --ablation_negative_checklist_path  ${ABLATION_NEGATIVE_CHECKLIST_PATH} \
    --miss_ablation_positive_path  ${MISS_ABLATION_POSITIVE_PATH} \
    --miss_ablation_negative_path  ${MISS_ABLATION_NEGATIVE_PATH} \
    --checklist_ablation_stats_path  ${CHECKLIST_ABLATION_STATS_PATH} \


python3 src/analysis/scoring/analyze_ablation_checklist_scoring.py \
    --variation_type "all" \
    --checklist_model ${CHECKLIST_MODEL_NAME} \
    --eval_model ${EVAL_MODEL_NAME} \
    --ablation_positive_checklist_path  ${ABLATION_POSITIVE_CHECKLIST_PATH} \
    --ablation_negative_checklist_path  ${ABLATION_NEGATIVE_CHECKLIST_PATH} \
    --miss_ablation_positive_path  ${MISS_ABLATION_POSITIVE_PATH} \
    --miss_ablation_negative_path  ${MISS_ABLATION_NEGATIVE_PATH} \
    --ablation_final_positive_checklist_path  ${ABLATION_FINAL_POSITIVE_PATH} \
    --ablation_final_negative_checklist_path  ${ABLATION_FINAL_NEGATIVE_PATH} \
    --ablation_filtered_positive_checklist_path  ${ABLATION_FILTERED_POSITIVE_PATH} \
    --ablation_filtered_negative_checklist_path  ${ABLATION_FILTERED_NEGATIVE_PATH} \
    --negative_histgram_path  ${NEGATIVE_HISTGRAM_PATH} \
    --positive_histgram_path  ${POSITIVE_HISTGRAM_PATH} \

