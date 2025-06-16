# Created by MacBook Pro at 12.06.25
# ablation_main.py
import os, json
import torch
import wandb
from collections import defaultdict
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
from mbg.training import training
from mbg.evaluation import evaluation
import config

ABLATED_CONFIGS = {
    # "soft_og": {"use_hard": False, "use_soft": True, "use_obj": True, "use_group": True},
    # "soft_group": {"use_hard": False, "use_soft": True, "use_obj": False, "use_group": True},
    # "hs_og": {"use_hard": True, "use_soft": True, "use_obj": True, "use_group": True},
    "hard_obj": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False},
    "hard_group": {"use_hard": True, "use_soft": False, "use_obj": False, "use_group": True},
    "hard_og": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True},
    # "soft_obj": {"use_hard": False, "use_soft": True, "use_obj": True, "use_group": False},
    # "hs_obj": {"use_hard": True, "use_soft": True, "use_obj": True, "use_group": False},
    # "hs_group": {"use_hard": True, "use_soft": True, "use_obj": False, "use_group": True},

}


def run_ablation(train_data, val_data, test_data, obj_model, train_principle, args, mode_name, ablation_flags):
    task_name = train_data["task"]
    train_val_data = {
        "task": task_name,
        "positive": train_data["positive"] + val_data["positive"],
        "negative": train_data["negative"] + val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])

    # train rule + calibrator
    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, hyp_params,
                                                                         train_principle, args.device, ablation_flags)
    base_rules = training.train_rules(hard, soft,obj_list, group_list, group_nums, train_img_labels, hyp_params, ablation_flags)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)
    calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params)

    return evaluation.eval_rules(test_data, obj_model, final_rules, hyp_params, train_principle, args.device,
                                 calibrator)


def main_ablation():
    args = args_utils.get_args()
    train_principle = args.principle
    principle_path = getattr(config, f"grb_{train_principle}")
    combined_loader = dataset.load_combined_dataset(principle_path)
    obj_model = eval_patch_classifier.load_model(args.device)

    wandb.init(project=f"grb_ablation_{train_principle}", config=args.__dict__, name=args.exp_name)
    results_summary = defaultdict(lambda: defaultdict(list))  # setting -> metric -> list
    all_f1 = {conf:[] for conf in ABLATED_CONFIGS}
    all_auc = {conf:[] for conf in ABLATED_CONFIGS}
    all_acc = {conf:[] for conf in ABLATED_CONFIGS}
    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        if task_idx < 125:
            continue
        task_name = train_data["task"]
        print(f"\nTask {task_idx + 1}/{len(combined_loader)}: {task_name}")

        log_dicts = {}
        for mode_name, ablation_flags in ABLATED_CONFIGS.items():
            print(f"  Running ablation: {mode_name}")
            test_metrics = run_ablation(train_data, val_data, test_data, obj_model,
                                        train_principle, args, mode_name, ablation_flags)
            for k in ["acc", "f1", "auc"]:
                results_summary[mode_name][k].append(test_metrics.get(k, 0))
                print(f"task: {task_idx + 1}/{len(combined_loader)}: {k} {test_metrics.get(k, 0)}")
            test_acc = test_metrics.get("acc", 0)
            test_auc = test_metrics.get("auc", 0)
            test_f1 = test_metrics.get("f1", 0)

            all_f1[mode_name].append(test_f1)
            all_auc[mode_name].append(test_auc)
            all_acc[mode_name].append(test_acc)

            log_dicts.update({f"{mode_name}_{k}": test_metrics.get(k, 0) for k in test_metrics})
            log_dicts.update({f"{mode_name}_acc_avg": torch.tensor(all_acc[mode_name]).mean(),
                f"{mode_name}_auc_avg": torch.tensor(all_auc[mode_name]).mean(),
                f"{mode_name}_f1_avg": torch.tensor(all_f1[mode_name]).mean()})

        wandb.log(log_dicts)
        # wandb.log({
        #     f"{mode_name}_{k}": test_metrics.get(k, 0) for k in test_metrics
        # })
        #

    # save and summarize
    final_summary = {
        mode: {f"avg_{k}": float(torch.tensor(v).mean()) for k, v in metric_dict.items()}
        for mode, metric_dict in results_summary.items()
    }
    with open(f"ablation_summary_{args.exp_name}.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print("\n=== Final Summary ===")
    for mode, metrics in final_summary.items():
        print(f"{mode}: {metrics}")

    wandb.finish()


if __name__ == "__main__":
    main_ablation()
