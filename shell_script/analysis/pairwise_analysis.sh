
echo "Checklist model name: ${CHECKLIST_MODEL_NAME}"
echo "Evaluation model name: ${EVAL_MODEL_NAME}"
echo "Checklist type: ${CHECKLIST_TYPE}"

SIMPLE_CHECKLIST_MODEL_NAME=$(echo "$CHECKLIST_MODEL_NAME" | sed 's|/|_|g')
SIMPLE_EVAL_MODEL_NAME=$(echo "$EVAL_MODEL_NAME" | sed 's|/|_|g')
DATASET_PATH="./Dataset/LLMBar/dataset.json"

BASELINE_EVAL_OUTPUT_PATH="./outputs/evaluation/no_checklist/LLMBar/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
CHECKLIST_PATH="./outputs/checklist/${CHECKLIST_TYPE}/LLMBar/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"
CHECKLIST_EVAL_OUTPUT_PATH="./outputs/evaluation/checklist/${CHECKLIST_TYPE}:${SIMPLE_CHECKLIST_MODEL_NAME}/LLMBar/${SIMPLE_EVAL_MODEL_NAME}.jsonl"

echo "Simple checklist model name: ${SIMPLE_CHECKLIST_MODEL_NAME}"
echo "Simple evaluation model name: ${SIMPLE_EVAL_MODEL_NAME}"

echo "Calulating pairwise accuracy..."
ACCURACY_OUTPUT_PATH="./outputs/analysis/pairwise/accuracy.xlsx"
echo "Accuracy output path: ${ACCURACY_OUTPUT_PATH}"
BOOTSTRAP_OUTPUT_PATH="./outputs/analysis/bootstrap/pairwise_results.csv"
echo "Bootstrap output path: ${BOOTSTRAP_OUTPUT_PATH}"

python3 src/analysis/pairwise/calc_pairwise_acc.py \
    --evaluation-dir "./outputs" \
    --accuracy-output-path ${ACCURACY_OUTPUT_PATH} \
    --bootstrap-output-path ${BOOTSTRAP_OUTPUT_PATH}

CHECKLIST_STATS_PATH="./analysis/stats/pairwise/LLMBar/checklist/${CHECKLIST_MODEL_NAME}/summary/${EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_EVAL_PATH="./analysis/preprocessed/pairwise/LLMBar/evaluation/checklist/${CHECKLIST_TYPE}/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_NO_EVAL_PATH="./analysis/preprocessed/pairwise/LLMBar/evaluation/no_checklist/${CHECKLIST_TYPE}/${SIMPLE_EVAL_MODEL_NAME}.jsonl"
PREPROCESSED_CHECKLIST_PATH="./analysis/preprocessed/pairwise/LLMBar/checklist/${CHECKLIST_TYPE}/${SIMPLE_CHECKLIST_MODEL_NAME}.jsonl"

echo "Verifying pairwise results..."
python3 src/analysis/pairwise/verify_checklist_pairwise.py \
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


# echo "Classifying subset of pairwise results..."
# python3 src/analysis/pairwise/classificate_subset.py \
#     --variation_type "all" \
#     --question_path ${DATASET_PATH} \
#     --checklist_model ${CHECKLIST_MODEL_NAME} \
#     --checklist_path ${CHECKLIST_PATH} \
#     --eval_model ${EVAL_MODEL_NAME} \
#     --no_checklist_eval_path ${BASELINE_EVAL_OUTPUT_PATH} \
#     --preprocessed_eval_path ${PREPROCESSED_EVAL_PATH} \
#     --preprocessed_no_checklist_eval_path ${PREPROCESSED_NO_EVAL_PATH} \
#     --checklist_stats_path ${CHECKLIST_STATS_PATH}

echo "Running ablation study..."
POSITIVE_CHECKLIST_PATH=./analysis/classification/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/checklist/positive_checklist.json
NEGATIVE_CHECKLIST_PATH=./analysis/classification/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/checklist/negative_checklist.json
ABLATION_POSITIVE_CHECKLIST_PATH=./analysis/ablation/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/ablation_result.json
ABLATION_NEGATIVE_CHECKLIST_PATH=./analysis/ablation/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/ablation_result.json
MISS_ABLATION_POSITIVE_PATH=./analysis/ablation/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/miss_result.json
MISS_ABLATION_NEGATIVE_PATH=./analysis/ablation/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/miss_result.json
CHECKLIST_ABLATION_STATS_PATH=./analysis/stats/pairwise/LLMBar/checklist/${CHECKLIST_MODEL_NAME}/ablation/${EVAL_MODEL_NAME}_${CHECKLIST_TYPE}.json


python3 src/analysis/pairwise/ablation.py \
    --question_path ${DATASET_PATH} \
    --variation_type "all" \
    --checklist_model ${CHECKLIST_MODEL_NAME} \
    --eval_model ${EVAL_MODEL_NAME} \
    --base_prompt_path ./data/prompt/pairwise_evaluate_response/checklist.txt \
    --positive_checklist_path  ${POSITIVE_CHECKLIST_PATH} \
    --negative_checklist_path  ${NEGATIVE_CHECKLIST_PATH} \
    --ablation_positive_checklist_path  ${ABLATION_POSITIVE_CHECKLIST_PATH} \
    --ablation_negative_checklist_path  ${ABLATION_NEGATIVE_CHECKLIST_PATH} \
    --miss_ablation_positive_path  ${MISS_ABLATION_POSITIVE_PATH} \
    --miss_ablation_negative_path  ${MISS_ABLATION_NEGATIVE_PATH} \
    --checklist_ablation_stats_path  ${CHECKLIST_ABLATION_STATS_PATH} \


# python3 src/analysis/remove_order_sensitive_cases.py \

echo "Analyzing final ablation study..."
ABLATION_FINAL_POSITIVE_PATH=./analysis/ablated_final/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/ablation_result.json
ABLATION_FINAL_NEGATIVE_PATH=./analysis/ablated_final/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/ablation_result.json
ABLATION_FILTERED_POSITIVE_PATH=./analysis/ablated_filtered/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/positive/ablation_result.json
ABLATION_FILTERED_NEGATIVE_PATH=./analysis/ablated_filtered/pairwise/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/negative/ablation_result.json
NEGATIVE_HISTGRAM_PATH=./analysis/stats/pairwise/LLMBar/histgrams/negative/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/final_result.pdf
POSITIVE_HISTGRAM_PATH=./analysis/stats/pairwise/LLMBar/histgrams/positive/${CHECKLIST_TYPE}:${CHECKLIST_MODEL_NAME}/${EVAL_MODEL_NAME}/final_result.pdf


python3 src/analysis/pairwise/analyze_ablation_checklist.py \
    --variation_type "all" \
    --checklist_model ${CHECKLIST_MODEL_NAME} \
    --eval_model ${EVAL_MODEL_NAME} \
    --ablation_positive_checklist_path  ${ABLATION_POSITIVE_CHECKLIST_PATH} \
    --ablation_negative_checklist_path  ${ABLATION_NEGATIVE_CHECKLIST_PATH} \
    --ablation_final_positive_checklist_path  ${ABLATION_FINAL_POSITIVE_PATH} \
    --ablation_final_negative_checklist_path  ${ABLATION_FINAL_NEGATIVE_PATH} \
    --ablation_filtered_positive_checklist_path  ${ABLATION_FILTERED_POSITIVE_PATH} \
    --ablation_filtered_negative_checklist_path  ${ABLATION_FILTERED_NEGATIVE_PATH} \
    --negative_histgram_path  ${NEGATIVE_HISTGRAM_PATH} \
    --positive_histgram_path  ${POSITIVE_HISTGRAM_PATH} \