import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.data import load_jsonl,  save_json,make_output_dir

def load_args():
    args =  argparse.ArgumentParser()

    args.add_argument(
        "--variation_type",
        type = str,
        help = "Type of variation",
        default = "all"
    )
    args.add_argument(
        "--question_path",
        type = str,
        help = "Path to the question",
        default = "./Dataset/InFoBench/dataset.json"
    )
    args.add_argument(
        "--checklist_model",
        type = str,
        help = "checklist generate model",
        default = "gpt-4o-2024-08-06"
    )
    args.add_argument(
        "--eval_model",
        type = str,
        help = "eval model",
        default = "Qwen/Qwen2.5-7B-Instruct"
    )
    args.add_argument(
        "--checklist_path",
        type = str,
        help = "Path to the question",
        default = "./outputs/checklist/{policy}/InFoBench/{checklist_model}.jsonl"
    )
    args.add_argument(
        "--eval_path",
        type = str,
        help = "Path to the evaluation",
        default = "./outputs/abs_evaluation/checklist/{policy}:{checklist_model}/InFoBench/{eval_model}.jsonl"
    )
    args.add_argument(
        "--no_checklist_eval_path",
        type = str,
        help = "Path to the evaluation",
        default = "./outputs/abs_evaluation/no_checklist/InFoBench/{eval_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/preprocessed/scoring/InFoBench/checklist/{policy}/{checklist_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_eval_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/preprocessed/scoring/InFoBench/checklist/{policy}:{checklist_model}/{eval_model}.jsonl"
    )
    args.add_argument(
        "--preprocessed_no_checklist_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/preprocessed/scoring/InFoBench/no_checklist/{policy}/{eval_model}.jsonl"
    )
    args.add_argument(
        "--checklist_stats_path",
        type = str,
        help = "Path to the output",
        default = "./analysis/stats/scoring/InFoBench/{checklist_model}/summary/{eval_model}.jsonl"
    )
    return args.parse_args()


def preprocess_checklist(checklist_data):
    
    checklist_dict = {}
    for data in checklist_data:
        question = data["question"]
        checklist = data["checklist"]
        checklist_dict[question] = checklist
    return checklist_dict

def preprocess_eval(no_checklist_eval_data,eval_data):
    
    no_checklist_eval_dict = {}
    eval_dict = {}
    for no_checklist_eval, eval in zip(no_checklist_eval_data, eval_data):
        question = no_checklist_eval.get("question", "")
        no_checklist_eval_dict_responses = no_checklist_eval.get("dict_responses", [])
        eval_dict_responses = eval.get("dict_responses", [])
        if question and eval_dict_responses:
            eval_dict[question] = eval_dict_responses
        if question and no_checklist_eval_dict_responses:
            no_checklist_eval_dict[question] = no_checklist_eval_dict_responses
    return no_checklist_eval_dict, eval_dict


def verified_checklist(no_checklist_eval_data, eval_data, checklist_dict):
    preprocessed_no_checklist_eval, preprocessed_eval = preprocess_eval(no_checklist_eval_data, eval_data)
    verified_eval_list = []
    verified_no_checklist_eval_list = []
    
    # デバッグのためのカウンター
    total_questions = 0
    rejected_reasons = {
        "missing_checklist": 0,
        "missing_response": 0,
        "length_mismatch": 0,
        "missing_no_checklist": 0,
        "valid": 0
    }
    
    for question, eval_responses in preprocessed_eval.items():
        total_questions += 1
        valid = True
        
        # チェックリストが存在するか確認
        if question not in checklist_dict:
            print(f"Question not found in checklist_dict: {question}")
            rejected_reasons["missing_checklist"] += 1
            valid = False
            continue
        
        # レスポンスデータの検査と修正
        for i, eval_response in enumerate(eval_responses):
            checklist = eval_response.get("checklist", {})
            
            # レスポンスデータを取得
            response_data = checklist.get("response", {})
            
            # レスポンスデータの構造を調査
            print(f"Debug - Question: {question}, Response data type: {type(response_data)}")
            print(f"Debug - Response data: {response_data}")
            
            # レスポンスが空の辞書なら、チェックリスト項目に対応するキーを作成
            if isinstance(response_data, dict) and len(response_data) == 0:
                checklist_items = checklist_dict.get(question, [])
                if checklist_items:
                    # チェックリスト項目に対応する空の応答を作成
                    new_response = {}
                    for idx, item in enumerate(checklist_items):
                        # キーとして項目のインデックスを使用
                        new_response[str(idx)] = False
                    
                    # 応答を更新
                    checklist["response"] = new_response
                    eval_response["checklist"] = checklist
                    eval_responses[i] = eval_response
                    
                    print(f"Debug - Created new response structure: {new_response}")
            
            # 他のレスポンス形式も確認（例: リスト形式）
            elif isinstance(response_data, list):
                print(f"Debug - Response is a list with {len(response_data)} items")
                checklist_items = checklist_dict.get(question, [])
                if len(response_data) != len(checklist_items):
                    print(f"List length mismatch: checklist={len(checklist_items)}, response={len(response_data)}")
                    valid = False
        
        # チェックリスト長の最終確認
        if valid and checklist_dict.get(question) is not None:
            checklist_items = checklist_dict.get(question)
            
            # 少なくとも1つのレスポンスを確認
            if len(eval_responses) > 0:
                eval_response = eval_responses[0]
                checklist = eval_response.get("checklist", {})
                response_data = checklist.get("response", {})
                
                # 構造が一致するか確認
                if isinstance(response_data, dict):
                    if len(checklist_items) != len(response_data):
                        print(f"Final check - Length mismatch: checklist={len(checklist_items)}, response={len(response_data)}")
                        print(f"Checklist: {checklist_items}")
                        print(f"Response: {response_data}")
                        rejected_reasons["length_mismatch"] += 1
                        valid = False
        
        # 有効なデータを追加
        if valid:
            verified_eval_dict = {
                "question": question,
                "eval_responses": eval_responses,
            }
            verified_eval_list.append(verified_eval_dict)
            
            # no_checklist データが存在するか確認
            if question in preprocessed_no_checklist_eval:
                verified_no_checklist_eval_dict = {
                    "question": question,
                    "eval_responses": preprocessed_no_checklist_eval[question],
                }
                verified_no_checklist_eval_list.append(verified_no_checklist_eval_dict)
                rejected_reasons["valid"] += 1
            else:
                print(f"No checklist eval data missing for question: {question}")
                rejected_reasons["missing_no_checklist"] += 1
                valid = False

    # 統計情報の計算
    original_checklist_count = sum(len(items) for q, items in checklist_dict.items() if items is not None)
    verified_checklist_count = 0
    for verified_item in verified_eval_list:
        question = verified_item["question"]
        if question in checklist_dict and checklist_dict[question] is not None:
            verified_checklist_count += len(checklist_dict[question])

    reduced_count = original_checklist_count - verified_checklist_count
    varified_ratio = verified_checklist_count / original_checklist_count if original_checklist_count else 0.0
    
    # 拒否理由の表示
    print("======Rejection Reasons:======")
    print(f"Total questions: {total_questions}")
    for reason, count in rejected_reasons.items():
        print(f"{reason}: {count}")
    
    print("======Result:======")
    print(f"Original checklist count: {original_checklist_count} items")
    print(f"Verified checklist count: {verified_checklist_count} items")
    print(f"Reduced Count: {reduced_count}")
    print(f"Varified Rate: {varified_ratio:.2%}")
    
    checklist_stats = {
        "original_checklist_count": original_checklist_count,
        "verified_checklist_count": verified_checklist_count,
        "reduced_count": reduced_count,
        "reduced_ratio": varified_ratio
    }
    
    return verified_eval_list, verified_no_checklist_eval_list, checklist_stats
   
def process_policy(args, policy,no_checklist_eval_data):
    print(f"\n======Processing policy: {policy}=====") 
    checklist_model_name = args.checklist_model.replace("/", "_")
    eval_model_name = args.eval_model.replace("/", "_")
    checklist_path = args.checklist_path.format(policy=policy, checklist_model=checklist_model_name)
    eval_path = args.eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
    
    preprocessed_checklist_path = args.preprocessed_checklist_path.format(policy=policy, checklist_model=checklist_model_name)
    preprocessed_eval_path = args.preprocessed_eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
    preprocessed_no_checklist_path = args.preprocessed_no_checklist_path.format(policy=policy, eval_model=eval_model_name)
    
    make_output_dir(preprocessed_checklist_path)
    make_output_dir(preprocessed_eval_path)
    make_output_dir(preprocessed_no_checklist_path)
    
    if not os.path.exists(checklist_path):
        print(f"File not found: {checklist_path}")
        return
    if not os.path.exists(eval_path):  
        print(f"File not found: {eval_path}")
        return
    try:
        checklist_data = load_jsonl(checklist_path)
        eval_path = args.eval_path.format(policy=policy, checklist_model=checklist_model_name, eval_model=eval_model_name)
        eval_data = load_jsonl(eval_path)
        preprocessed_checklist_dict = preprocess_checklist(checklist_data)
        print(f"Preprocessed checklist data loaded from {preprocessed_checklist_dict}")
        verified_eval_list, verified_no_checklist_eval_list, checklist_stats = verified_checklist(no_checklist_eval_data,eval_data, preprocessed_checklist_dict)
        save_json(preprocessed_checklist_path,preprocessed_checklist_dict) 
        save_json(preprocessed_no_checklist_path,verified_no_checklist_eval_list)
        save_json(preprocessed_eval_path,verified_eval_list)       
        # save_json(checklist_stats_path,checklist_stats)
        print(f"Finished processing policy: {policy}")
        return checklist_stats
    except Exception as e:
        print(f"Error processing policy {policy}: {e}")
        import traceback
        traceback.print_exc()  
        return None
    
def main():
    args = load_args()
    checklist_generation_policies = ["baseline", "adjust_0.5_baseline", "adjust_1.5_baseline", 
                                    "ticking", "refine_baseline", "specify"]
    all_stats = {}
    # eval_model_name = args.eval_model
    eval_model_name = args.eval_model.replace("/", "_")
    checklist_model_name = args.checklist_model.replace("/", "_")
    no_checklist_eval_path = args.no_checklist_eval_path.format(eval_model=eval_model_name)
    checklist_stats_path = args.checklist_stats_path.format(checklist_model=checklist_model_name, eval_model=eval_model_name)
    checklist_generation_policies_path = "./analysis/info/checklist_generation_policies_scoring.json"
    make_output_dir(checklist_generation_policies_path)
    save_json(checklist_generation_policies_path, checklist_generation_policies)
    
    if not os.path.exists(no_checklist_eval_path):
        print(f"File not found: {no_checklist_eval_path}")
        return
    no_checklist_eval_data = load_jsonl(no_checklist_eval_path)
    if args.variation_type == "all":
        for policy in checklist_generation_policies:
            stats = process_policy(args, policy,no_checklist_eval_data)
            if stats:
                all_stats[policy] = stats
    else:
        if args.variation_type in checklist_generation_policies:
            stats = process_policy(args, args.variation_type, no_checklist_eval_data)
            if stats:
                all_stats[args.variation_type] = stats
        else:
            print(f"Invalid variation type: {args.variation_type}")
            return
    print("======Final Result:======")
    for policy, stats in all_stats.items():
        print(f"Policy: {policy}")
        print(f"Original checklist count: {stats['original_checklist_count']} items")
        print(f"Verified checklist count: {stats['verified_checklist_count']} items")
        print(f"Reduced Count: {stats['reduced_count']}")
        print(f"Varified Rate: {stats['reduced_ratio']:.2%}")
    
    make_output_dir(checklist_stats_path)
    save_json(checklist_stats_path, all_stats)
    print(f"Save to : {checklist_stats_path}")  
    
if __name__ == "__main__":
    main()
    print("All done!")


            
            
        
        
