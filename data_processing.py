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
        print(f"\nAverage acc: {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"Average f1:  {f1s.mean():.4f} ± {f1s.std():.4f}")

        avg_acc = summary.get(mode, {}).get("avg_acc", None)
        avg_f1 = summary.get(mode, {}).get("avg_f1", None)
        if avg_acc is not None and avg_f1 is not None:
            print(f"(From summary) avg_acc: {avg_acc:.4f}, avg_f1: {avg_f1:.4f}")


def draw_final_calibrator_gain_figure(json_path, output_path=config.output / "calibrator_gain_vs_clause_quality.pdf"):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract analysis from hard_obj_calib mode
    results = data["per_task_results"]["hard_obj_calib"]
    gain_with_clause = []
    gain_without_clause = []

    for task in results:
        analysis = task.get("analysis", {})
        cal_scores = analysis.get("calibrated_scores", [])
        van_scores = analysis.get("vanilla_scores", [])
        clause_flags = analysis.get("rule_pool_has_good_clause", [])

        for c, v, has_good in zip(cal_scores, van_scores, clause_flags):
            gain = c - v
            if has_good:
                gain_with_clause.append(gain)
            else:
                gain_without_clause.append(gain)

    # Compute stats
    def describe(gains):
        return np.mean(gains), np.std(gains), len(gains)

    mean_with, std_with, n_with = describe(gain_with_clause)
    mean_without, std_without, n_without = describe(gain_without_clause)

    # Run t-test
    t_stat, p_value = ttest_ind(gain_with_clause, gain_without_clause, equal_var=False)

    # Plot setup
    plt.figure(figsize=(7, 5))
    categories = ["Good Clause Exists", "No Good Clause"]
    means = [mean_with, mean_without]
    stds = [std_with, std_without]

    # Bar plot
    bars = plt.bar(categories, means, yerr=stds, capsize=8, color=["#4C72B0", "#DD8452"], edgecolor="black")

    # Overlay jittered dots
    jitter = 0.08
    for i, data in enumerate([gain_with_clause, gain_without_clause]):
        x = np.random.normal(i, jitter, size=len(data))
        plt.scatter(x, data, color="black", alpha=0.4, s=10)

    # Labels and formatting
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Average Score Gain (Calibrated − Vanilla)", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Calibrator Score Gain Relative to Clause Pool Quality", fontsize=13)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    # Annotate p-value
    p_label = f"t-test p = {p_value:.1e}"
    plt.text(0.5, max(means) + max(stds) * 1.2, p_label,
             ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Figure saved to {output_path}")
    print(f"Good Clause Exists: n={n_with}, mean={mean_with:.4f}, std={std_with:.4f}")
    print(f"No Good Clause:     n={n_without}, mean={mean_without:.4f}, std={std_without:.4f}")
    print(f"T-test p-value: {p_value:.4e}")


if __name__ == "__main__":
    print_ablation_results(config.output/"ablation_summary_continuity_20250717_141123.json")
    draw_final_calibrator_gain_figure(config.output / "ablation_summary_continuity_20250717_084756.json")

    # main_ablation()
    # run_ablation()
