# Created by MacBook Pro at 30.08.25
import config
import json
from src.utils import file_utils, data_utils


def merge_two_gpt_results(file1, file2, output_file):
    import json

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged_data = {**data1, **data2}  # Merge dictionaries

    with open(output_file, 'w') as out_f:
        json.dump(merged_data, out_f, indent=4)

    print(f"Merged results saved to {output_file}")


def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)


def get_all_subfolder_names(directory):
    from pathlib import Path
    directory = Path(directory)
    return [f.name for f in directory.iterdir() if f.is_dir()]


def save_json(data, file_path):
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def merge_proximity(res_path):
    principle = "proximity"
    cont_names = sorted(get_all_subfolder_names(config.get_raw_patterns_path(False) / principle / "train"))
    file_lists = [
        [0, 233]
    ]
    merged_cont_json = {}
    for start, end in file_lists:
        acc_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_acc.csv")[f"{start}_{end} - {principle}/test_accuracy"].values
        f1_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_f1_score.csv")[f"{start}_{end} - {principle}/f1_score"].values
        precision_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_precision.csv")[f"{start}_{end} - {principle}/precision"].values
        recall_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_recall.csv")[f"{start}_{end} - {principle}/recall"].values

        for i in range(len(acc_1)):
            merged_cont_json[cont_names[i + start]] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            # print(f"Pattern: {cont_names[i+start]}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
    print("Final length of merged results:", len(merged_cont_json))
    save_json(merged_cont_json, res_path / principle / f"internVL3_78B_{principle}_merged_results.json")


def merge_similarity(res_path):
    principle = "similarity"
    cont_names = sorted(get_all_subfolder_names(config.get_raw_patterns_path(False) / principle / "train"))
    file_lists = [
        [0, 182]
    ]
    merged_cont_json = {}
    for start, end in file_lists:
        acc_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_acc.csv")[f"{start}_{end} - {principle}/test_accuracy"].values
        f1_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_f1_score.csv")[f"{start}_{end} - {principle}/f1_score"].values
        precision_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_precision.csv")[f"{start}_{end} - {principle}/precision"].values
        recall_1 = load_csv(res_path / principle / f"{principle}_{start}_{end}_recall.csv")[f"{start}_{end} - {principle}/recall"].values

        for i in range(len(acc_1)):
            merged_cont_json[cont_names[i + start]] = {
                "accuracy": float(acc_1[i]),
                "f1_score": float(f1_1[i]),
                "precision": float(precision_1[i]),
                "recall": float(recall_1[i])
            }
            # print(f"Pattern: {cont_names[i+start]}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
    print("Final length of merged results:", len(merged_cont_json))
    save_json(merged_cont_json, res_path / principle / f"internVL3_78B_{principle}_merged_results.json")


def merge_principle(res_path, principle,model_name, file_lists):
    cont_names = sorted(get_all_subfolder_names(config.get_raw_patterns_path(False) / principle / "train"))
    merged_cont_json = {}
    for start, end in file_lists:
        acc_1 = load_csv(res_path / principle / f"{principle}_{model_name}_{start}_{end}_acc.csv")[f"{start}_{end} - {principle}/test_accuracy"].values
        f1_1 = load_csv(res_path / principle / f"{principle}_{model_name}_{start}_{end}_f1_score.csv")[f"{start}_{end} - {principle}/f1_score"].values
        precision_1 = load_csv(res_path / principle / f"{principle}_{model_name}_{start}_{end}_precision.csv")[f"{start}_{end} - {principle}/precision"].values
        recall_1 = load_csv(res_path / principle / f"{principle}_{model_name}_{start}_{end}_recall.csv")[f"{start}_{end} - {principle}/recall"].values

        for i in range(end - start + 1):
            try:
                merged_cont_json[cont_names[i + start]] = {
                    "accuracy": float(acc_1[i]),
                    "f1_score": float(f1_1[i]),
                    "precision": float(precision_1[i]),
                    "recall": float(recall_1[i])
                }
            except Exception as e:
                print(f"Error processing index {i} (pattern: {cont_names[i + start]}): {e}")
            # print(f"Pattern: {cont_names[i+start]}, Accuracy: {acc_1[i]:.4f}, F1 Score: {f1_1[i]:.4f}, Precision: {precision_1[i]:.4f}, Recall: {recall_1[i]:.4f}")
    print("Final length of merged results:", len(merged_cont_json))
    save_json(merged_cont_json, res_path / principle / f"{model_name}_{principle}_merged_results.json")


def process_grm_o_results(res_path, principle, json_file):
    json_data = file_utils.load_json(res_path / principle / json_file)
    all_task_names = [t['task_name'[0]] for t in json_data['per_task_results']['hard_obj']]

    result_o_dict = {}
    for t_i, name in enumerate(all_task_names):
        all_preds = [s > 0.5 for s in json_data['per_task_results']['hard_obj']["analysis"]["calibrated_scores"]]
        all_labels = json_data['per_task_results']['hard_obj']["analysis"]["groundtruth_labels"]
        TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_preds, all_labels)
        precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

        result_o_dict[name] = {
            "accuracy": json_data['per_task_results']['hard_obj'][t_i]['acc'],
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall
        }
    print("Final length of merged results:", len(result_o_dict))
    save_json(result_o_dict, res_path / principle / f"grm_{principle}_obj_results.json")


def process_grm_ogc_results(res_path, principle, json_file):
    json_data = file_utils.load_json(res_path / principle / json_file)
    all_task_names = [t['task_name'][0] for t in json_data['per_task_results']['hard_ogc']]

    result_ogc_dict = {}
    for t_i, name in enumerate(all_task_names):
        all_preds = [s > 0.5 for s in json_data['per_task_results']['hard_ogc'][t_i]["analysis"]["calibrated_scores"]]
        all_labels = json_data['per_task_results']['hard_ogc'][t_i]["analysis"]["groundtruth_labels"]
        TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_preds, all_labels)
        precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

        result_ogc_dict[name] = {
            "accuracy": json_data['per_task_results']['hard_ogc'][t_i]['acc'],
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall
        }
    print("Final length of merged results:", len(result_ogc_dict))
    save_json(result_ogc_dict, res_path / principle / f"grm_{principle}_ogc_results.json")

def process_vit_json(res_path, principle, json_file):
    json_data = file_utils.load_json(res_path / principle / json_file)
    data = json_data[principle]
    save_json(data, res_path / principle / f"vit_{principle}_results.json")
if __name__ == "__main__":
    res_path = config.get_proj_output_path(False)

    # ViT
    # merge_principle(res_path, "symmetry","vit", [[0, 125], [126,215]])
    # process_vit_json(res_path, "proximity", "vit_base_patch16_224_eval_principle_proximity_20250903_134724_img_num_3_start_0_task_num_full.json")
    # process_vit_json(res_path, "continuity", "vit_base_patch16_224_eval_principle_continuity_20250903_140831_img_num_3_start_0_task_num_full.json")
    # process_vit_json(res_path, "similarity", "vit_base_patch16_224_eval_principle_similarity_20250903_140015_img_num_3_start_0_task_num_full.json")
    process_vit_json(res_path, "closure", "vit_base_patch16_224_eval_principle_closure_20250903_140516_img_num_3_start_0_task_num_full.json")



    # GRM
    # process_grm_ogc_results(res_path, "proximity", "ablation_summary_proximity_grm_20250720_070246.json")
    # process_grm_ogc_results(res_path, "similarity", "ablation_summary_similarity_grm_20250721_092307.json")
    # process_grm_ogc_results(res_path, "closure", "ablation_summary_closure_grm_20250718_110518.json")
    # process_grm_ogc_results(res_path, "continuity", "ablation_summary_continuity_grm_20250718_110525.json")
    # process_grm_ogc_results(res_path, "symmetry", "ablation_summary_symmetry_grm_20250722_095458.json")


    # internVL3_78B
    # merge_proximity(res_path)
    # merge_similarity(res_path)
    # merge_principle(res_path, "closure", [[0, 170]])
    # merge_principle(res_path, "symmetry", [[0, 119]])
    # merge_principle(res_path, "continuity", [[0, 224]])
    print("program finished.")
