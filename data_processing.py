# Created by MacBook Pro at 18.06.25
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

import config


def legacy_fun():
    prox_path = config.wandb_path / "prox.csv"
    simi_path = config.wandb_path / "simi.csv"
    closure_path = config.wandb_path / "closure.csv"
    cont_path = config.wandb_path / "cont.csv"
    symm_path = config.wandb_path / "symm.csv"

    prox = pd.read_csv(prox_path)
    simi = pd.read_csv(simi_path)
    closure = pd.read_csv(closure_path)
    cont = pd.read_csv(cont_path)
    symm = pd.read_csv(symm_path)

    prox_mean = prox["prox_01 - hard_ogc_acc_avg"].mean()
    prox_std = prox["prox_01 - hard_ogc_acc_avg"].std()
    simi_mean = simi["simi_01 - hard_ogc_acc_avg"].mean()
    simi_std = simi["simi_01 - hard_ogc_acc_avg"].std()
    closure_mean \
        = closure["closure_01 - hard_ogc_acc_avg"].mean()
    closure_std = closure["closure_01 - hard_ogc_acc_avg"].std()

    cont_mean = cont["cont_01 - hard_ogc_acc_avg"].mean()
    cont_std = cont["cont_01 - hard_ogc_acc_avg"].std()

    prin = "prox"
    prox_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
    prox_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

    prox_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
    prox_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

    prox_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
    prox_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()

    prin = "simi"
    simi_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
    simi_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

    simi_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
    simi_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

    simi_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
    simi_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()

    prin = "cont"
    cont_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
    cont_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

    cont_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
    cont_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

    cont_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
    cont_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()

    prin = "closure"
    closure_o_mean = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].mean()
    closure_o_std = pd.read_csv(config.wandb_path / f"{prin}_o.csv")[f"{prin}_01 - hard_obj_acc_avg"].std()

    closure_oc_mean = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].mean()
    closure_oc_std = pd.read_csv(config.wandb_path / f"{prin}_oc.csv")[f"{prin}_01 - hard_obj_calib_acc_avg"].std()

    closure_og_mean = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].mean()
    closure_og_std = pd.read_csv(config.wandb_path / f"{prin}_og.csv")[f"{prin}_01 - hard_og_acc_avg"].std()

    symm_mean = symm["symm_01 - hard_ogc_acc_avg"].mean()
    symm_std = symm["symm_01 - hard_ogc_acc_avg"].std()

    symm_o_mean = pd.read_csv(config.wandb_path / "symm_o.csv")["symm_01 - hard_obj_acc_avg"].mean()
    symm_o_std = pd.read_csv(config.wandb_path / "symm_o.csv")["symm_01 - hard_obj_acc_avg"].std()

    symm_oc_mean = pd.read_csv(config.wandb_path / "symm_oc.csv")["symm_01 - hard_obj_calib_acc_avg"].mean()
    symm_oc_std = pd.read_csv(config.wandb_path / "symm_oc.csv")["symm_01 - hard_obj_calib_acc_avg"].std()

    symm_og_mean = pd.read_csv(config.wandb_path / "symm_og.csv")["symm_01 - hard_og_acc_avg"].mean()
    symm_og_std = pd.read_csv(config.wandb_path / "symm_og.csv")["symm_01 - hard_og_acc_avg"].std()

    print("")


def print_ablation_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    per_task = data.get("per_task_results", {})

    # Get hard_obj average accuracy for comparison
    hard_obj_acc = summary.get("hard_obj", {}).get("avg_acc", None)

    for mode, tasks in per_task.items():
        print(f"\n=== Mode: {mode} ===")
        accs = []
        f1s = []
        print("Per-task results:")
        for task in tasks:
            acc = task.get("acc", None)
            f1 = task.get("f1", None)
            accs.append(acc)
            f1s.append(f1)
            # print(f"  Task {task.get('task_idx', '?')}: {task.get('task_name', '?')}, acc={acc:.4f}, f1={f1:.4f}")
        accs = np.array(accs)
        f1s = np.array(f1s)
        print(f"\nAverage acc: {accs.mean():.2f} ± {accs.std():.2f}")
        print(f"Average f1:  {f1s.mean():.2f} ± {f1s.std():.2f}")

        avg_acc = summary.get(mode, {}).get("avg_acc", None)
        avg_f1 = summary.get(mode, {}).get("avg_f1", None)
        if avg_acc is not None and avg_f1 is not None:
            print(f"(From summary) avg_acc: {avg_acc:.2f}, avg_f1: {avg_f1:.2f}")

            # Print percentage change compared to hard_obj
            if hard_obj_acc is not None and mode != "hard_obj":
                pct_change = ((avg_acc - hard_obj_acc) / hard_obj_acc) * 100
                print(f"Change vs hard_obj: {pct_change:+.0f}%")


def draw_final_calibrator_gain_figure(json_path, output_path=config.output / "calibrator_gain_vs_clause_quality.pdf"):
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22
    })

    with open(json_path, "r") as f:
        data = json.load(f)

    ogc_results = data["per_task_results"]["hard_ogc"]
    og_results = data["per_task_results"]["hard_og"]

    def get_counts(results, key_prefix="calibrated"):
        helps, no_help, worse = [], [], []
        for res in results:
            analysis = res.get("analysis", {})
            cal_scores = analysis.get(f"{key_prefix}_scores", [])
            van_scores = analysis.get("vanilla_scores", [])
            groundtruth_labels = analysis.get("groundtruth_labels", [])
            for c, v, gt in zip(cal_scores, van_scores, groundtruth_labels):
                if key_prefix == "calibrated":
                    if (gt == 1 and v < 0.5 < c) or (gt == 0 and v > 0.5 > c):
                        helps.append(1)
                    elif (gt == 1 and v > 0.5 and c > 0.5) or (gt == 0 and v < 0.5 and c < 0.5):
                        no_help.append(1)
                    else:
                        worse.append(1)
                else:
                    if (gt == 1 and v < 0.5) or (gt == 0 and v > 0.5):
                        worse.append(1)
                    else:
                        no_help.append(1)
        return [sum(helps), sum(no_help), sum(worse)]

    ogc_counts = get_counts(ogc_results, key_prefix="calibrated")
    og_counts = get_counts(og_results, key_prefix="vanilla")

    categories = ["Helps", "No Help", "Worse"]
    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 7))
    plt.bar(x - width/2, ogc_counts, width, label="Calibrated (ogc)", color="#4C72B0", edgecolor="black")
    plt.bar(x + width/2, og_counts, width, label="Vanilla (og)", color="#DD8452", edgecolor="black")
    plt.ylabel("Count")
    plt.title("Calibrator Effect Summary (Grouped)")
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    print("Ablation study result analysis")

    # print_ablation_results(config.output/"ablation_summary_closure_20250718_110518.json")
    # draw_final_calibrator_gain_figure(config.output / "ablation_summary_closure_20250718_110518.json")

    print_ablation_results(config.output/"grm_ablation_summary_closure_transformer_20251121_104235.json")

    # continuity
    # print_ablation_results(config.output/"ablation_summary_continuity_20250718_110525.json")
    # draw_final_calibrator_gain_figure(config.output / "ablation_summary_continuity_20250718_110525.json")
    # print_ablation_results(config.output/"grm_ablation_summary_continuity_transformer_20251120_104518.json")
    
    # symmetry
    # print_ablation_results(config.output/"ablation_summary_symmetry_20250722_095458.json")
    # draw_final_calibrator_gain_figure(config.get_proj_output_path() / "symmetry"/ "ablation_summary_symmetry_20250904_202041.json")
    # print_ablation_results(config.output/"grm_ablation_summary_symmetry_transformer_20251120_125143.json")
    

    # proximity
    # print_ablation_results(config.output/"ablation_summary_proximity_20250720_070246.json")
    # print_ablation_results(config.output/"grm_ablation_summary_proximity_transformer_20251120_193841.json")
    
    
    # draw_final_calibrator_gain_figure(config.output / "ablation_summary_proximity_20250720_070246.json")

    # similarity
    # print_ablation_results(config.output/"ablation_summary_similarity_20250721_092307.json")
    # print_ablation_results(config.output/"grm_ablation_summary_similarity_transformer_20251121_104125.json")
    # draw_final_calibrator_gain_figure(config.output / "ablation_summary_similarity_20250721_092307.json")


    # draw_final_calibrator_gain_figure(config.output / "ablation_summary_proximity_20250720_070246.json")
    # main_ablation()
    # run_ablation()
