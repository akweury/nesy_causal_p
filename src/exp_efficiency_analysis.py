# Created by MacBook Pro at 03.07.25

import torch
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
from mbg.training import training
import config
from mbg.scorer import scorer_config
from mbg import exp_utils
from src import bk
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import ttest_ind

ABLATED_CONFIGS = {
    "hard_obj_calib": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": False, "use_calibrator": True, "use_deepproblog": False},
    "hard_ogc": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": True, "use_deepproblog": False},

}


def run_analysis(train_data, val_data, obj_model, group_model, train_principle, args):
    task_name = train_data["task"]
    train_val_data = {"task": task_name, "positive": train_data["positive"] + val_data["positive"], "negative": train_data["negative"] + val_data["negative"]}
    hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])

    obj_times = torch.zeros(len(train_img_labels))
    group_times = torch.zeros(len(train_img_labels))

    # obj-group-level rule reasoning
    ablation_flags = ABLATED_CONFIGS["hard_ogc"]
    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags,
                                                                         group_times)
    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels, hyp_params, ablation_flags, group_times)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)

    # only object-level rule reasoning
    ablation_flags = ABLATED_CONFIGS["hard_obj_calib"]

    hard, soft, group_nums, obj_list, group_list = training.ground_facts(train_val_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags,
                                                                         obj_times)
    base_rules = training.train_rules(hard, soft, obj_list, group_list, group_nums, train_img_labels, hyp_params, ablation_flags, obj_times)
    final_rules = training.extend_rules(base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)

    # Dict: n_obj -> list of (obj_facts, group_facts, n_groups)
    obj_num_stats = {}
    time_stats = {}
    for i, img_hard in enumerate(hard):
        obj_fact_count, group_fact_count, n_obj, n_groups = exp_utils.count_symbolic_facts(img_hard)
        if n_obj not in obj_num_stats:
            obj_num_stats[n_obj] = {"obj_facts": [], "group_facts": [], "groups": []}
            time_stats[n_obj] = {"obj_time": [], "group_time": []}
        obj_num_stats[n_obj]["obj_facts"].append(obj_fact_count)
        obj_num_stats[n_obj]["group_facts"].append(group_fact_count)
        obj_num_stats[n_obj]["groups"].append(n_groups)
        time_stats[n_obj]["obj_time"].append(obj_times[i])
        time_stats[n_obj]["group_time"].append(group_times[i])

    # Compute averages for each object number
    obj_num_avg = {}
    for n_obj, stats in obj_num_stats.items():
        count = len(stats["obj_facts"])
        avg_obj_facts = sum(stats["obj_facts"]) / count if count > 0 else 0.0
        avg_group_facts = sum(stats["group_facts"]) / count if count > 0 else 0.0
        avg_groups = sum(stats["groups"]) / count if count > 0 else 0.0
        std_obj_facts = np.std(stats["obj_facts"]) if count > 0 else 0.0
        std_group_facts = np.std(stats["group_facts"]) if count > 0 else 0.0
        obj_num_avg[n_obj] = (avg_obj_facts, avg_group_facts, avg_groups, std_obj_facts, std_group_facts)

    for n_obj in time_stats:
        time_stats[n_obj]["obj_time"] = [float(t) for t in time_stats[n_obj]["obj_time"]]
        time_stats[n_obj]["group_time"] = [float(t) for t in time_stats[n_obj]["group_time"]]

    return obj_num_avg, time_stats


def main_eff_analysis():
    args = args_utils.get_args()
    # Collect all principle names from config that start with 'grb_'
    principle_names = bk.gestalt_principles
    all_results = {}
    all_time_stats = {}

    for train_principle in principle_names:
        principle_path = getattr(config, f"grb_{train_principle}")
        combined_loader = dataset.load_combined_dataset(principle_path, task_num=args.top_data)
        obj_model = eval_patch_classifier.load_model(args.device)
        group_model = scorer_config.load_scorer_model(train_principle, args.device)
        # Dict: n_obj -> list of (obj_facts, group_facts, n_groups) for all tasks
        analysis_dict = {}
        time_dict = {}
        for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
            # if task_idx!=36:
            #     continue
            print(f"\nPrinciple: {train_principle} | Task {task_idx + 1}/{len(combined_loader)}: {train_data['task']}")
            obj_num_avg, time_stats = run_analysis(train_data, val_data, obj_model, group_model, train_principle, args)
            for n_obj, tup in obj_num_avg.items():
                if n_obj not in analysis_dict:
                    analysis_dict[n_obj] = {"obj_facts": [], "group_facts": [], "groups": [], "std_obj_facts": [], "std_group_facts": []}
                analysis_dict[n_obj]["obj_facts"].append(tup[0])
                analysis_dict[n_obj]["group_facts"].append(tup[1])
                analysis_dict[n_obj]["groups"].append(tup[2])
                analysis_dict[n_obj]["std_obj_facts"].append(tup[3])
                analysis_dict[n_obj]["std_group_facts"].append(tup[4])
            for n_obj, tstats in time_stats.items():
                if n_obj not in time_dict:
                    time_dict[n_obj] = {"obj_time": [], "group_time": []}
                time_dict[n_obj]["obj_time"].extend(tstats["obj_time"])
                time_dict[n_obj]["group_time"].extend(tstats["group_time"])

        # Average over all tasks for each object number
        avg_analysis = {}
        for n_obj, stats in analysis_dict.items():
            count = len(stats["obj_facts"])
            avg_obj_facts = sum(stats["obj_facts"]) / count if count > 0 else 0.0
            avg_group_facts = sum(stats["group_facts"]) / count if count > 0 else 0.0
            avg_groups = sum(stats["groups"]) / count if count > 0 else 0.0
            std_obj_facts = np.std(stats["obj_facts"]) if count > 0 else 0.0
            std_group_facts = np.std(stats["group_facts"]) if count > 0 else 0.0
            avg_analysis[n_obj] = {
                "avg_obj_facts": avg_obj_facts, "avg_group_facts": avg_group_facts,
                "avg_groups": avg_groups, "std_obj_facts": std_obj_facts, "std_group_facts": std_group_facts}
        all_results[train_principle] = avg_analysis
        all_time_stats[train_principle] = time_dict

    # Save to JSON
    saved_json_file = config.output / f"efficiency_analysis.json"
    with open(saved_json_file, "w") as f:
        json.dump({"all_results": all_results, "all_time_stats": all_time_stats}, f, indent=2)
    print("\n=== Efficiency Analysis Summary ===")
    for principle, analysis in all_results.items():
        print(f"{principle}: {analysis}")

    return saved_json_file


def draw_figures(saved_json_file):
    import matplotlib.pyplot as plt
    # Set global font sizes for matplotlib
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22
    })

    # Load data from JSON and draw charts
    with open(saved_json_file, "r") as f:
        data = json.load(f)
    all_results = data["all_results"]
    all_time_stats = data["all_time_stats"]

    # Merge all_results across principles
    merged = {}
    for principle_result in all_results.values():
        for n_obj, stats in principle_result.items():
            if n_obj not in merged:
                merged[n_obj] = {"obj_facts": [], "group_facts": [], "groups": [], "std_obj_facts": [], "std_group_facts": []}
            merged[n_obj]["obj_facts"].append(stats["avg_obj_facts"])
            merged[n_obj]["group_facts"].append(stats["avg_group_facts"])
            merged[n_obj]["groups"].append(stats["avg_groups"])
            merged[n_obj]["std_obj_facts"].append(stats["std_obj_facts"])
            merged[n_obj]["std_group_facts"].append(stats["std_group_facts"])

    merged_time = {}
    for principle_time in all_time_stats.values():
        for n_obj, tstats in principle_time.items():
            if n_obj not in merged_time:
                merged_time[n_obj] = {"obj_time": [], "group_time": []}
            merged_time[n_obj]["obj_time"].extend(tstats["obj_time"])
            merged_time[n_obj]["group_time"].extend(tstats["group_time"])

    merged_avg = {}
    for n_obj, stats in merged.items():
        count = len(stats["obj_facts"])
        merged_avg[n_obj] = {
            "avg_obj_facts": sum(stats["obj_facts"]) / count if count > 0 else 0.0,
            "avg_group_facts": sum(stats["group_facts"]) / count if count > 0 else 0.0,
            "avg_groups": sum(stats["groups"]) / count if count > 0 else 0.0,
            "std_obj_facts": np.std(stats["obj_facts"]) if count > 0 else 0.0,
            "std_group_facts": np.std(stats["group_facts"]) if count > 0 else 0.0
        }

    merged_time_avg = {}
    for n_obj, tstats in merged_time.items():
        count_obj = len(tstats["obj_time"])
        count_group = len(tstats["group_time"])
        merged_time_avg[n_obj] = {
            "avg_obj_time": np.mean(tstats["obj_time"]) if count_obj > 0 else 0.0,
            "avg_group_time": np.mean(tstats["group_time"]) if count_group > 0 else 0.0,
            "std_obj_time": np.std(tstats["obj_time"]) if count_obj > 0 else 0.0,
            "std_group_time": np.std(tstats["group_time"]) if count_group > 0 else 0.0,
            "n_obj_time": count_obj,
            "n_group_time": count_group
        }

    # Draw and save the combined line chart
    chart_path = config.get_proj_output_path() / "fact_number_line_chart_all_principles.png"
    exp_utils.draw_fact_number_line_chart(merged_avg, chart_path)
    time_chart_path = config.get_proj_output_path() / "time_cost_line_chart_all_principles.png"
    exp_utils.draw_time_cost_line_chart(merged_time_avg, time_chart_path)


def draw_combined_calibrator_and_fact_chart(stacked_data, stacked_labels, stacked_colors, stacked_xticklabels,
                                            calib_json_path, eff_json_path, output_path):
    # Font sizes for readability
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22
    })
    # --- Load calibrator gain data ---
    with open(calib_json_path, "r") as f:
        calib_data = json.load(f)
    results = calib_data["per_task_results"]["hard_obj_calib"]
    gain_with_clause, gain_without_clause = [], []
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

    def describe(gains):
        return np.mean(gains), np.std(gains), len(gains)

    mean_with, std_with, n_with = describe(gain_with_clause)
    mean_without, std_without, n_without = describe(gain_without_clause)
    t_stat, p_value = ttest_ind(gain_with_clause, gain_without_clause, equal_var=False)

    # --- Load fact number line chart data ---
    with open(eff_json_path, "r") as f:
        eff_data = json.load(f)
    all_results = eff_data["all_results"]
    merged = {}
    for principle_result in all_results.values():
        for n_obj, stats in principle_result.items():
            if n_obj not in merged:
                merged[n_obj] = {
                    "obj_facts": [],
                    "group_facts": [],
                    "std_obj_facts": [],
                    "std_group_facts": [],
                    "n_obj_facts": [],
                    "n_group_facts": []
                }
            merged[n_obj]["obj_facts"].append(stats["avg_obj_facts"])
            merged[n_obj]["group_facts"].append(stats["avg_group_facts"])
            merged[n_obj]["std_obj_facts"].append(stats.get("std_obj_facts", 0))
            merged[n_obj]["std_group_facts"].append(stats.get("std_group_facts", 0))
            merged[n_obj]["n_obj_facts"].append(stats.get("n_obj_facts", 1))
            merged[n_obj]["n_group_facts"].append(stats.get("n_group_facts", 1))
    analysis_dict = {}
    for n_obj, stats in merged.items():
        count = len(stats["obj_facts"])
        analysis_dict[str(n_obj)] = {
            "avg_obj_facts": np.mean(stats["obj_facts"]),
            "avg_group_facts": np.mean(stats["group_facts"]),
            "std_obj_facts": np.mean(stats["std_obj_facts"]),
            "std_group_facts": np.mean(stats["std_group_facts"]),
            "n_obj_facts": np.sum(stats["n_obj_facts"]),
            "n_group_facts": np.sum(stats["n_group_facts"])
        }

    # --- Plot both figures side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(28, 7), gridspec_kw={'width_ratios': [1, 1, 1]})
    axes[1] = fig.add_subplot(1, 3, 2, polar=True)
    # --- Left: Stacked proportional bar chart ---
    stacked_data = np.array(stacked_data)
    bar_positions = np.arange(len(stacked_data))
    bottom = np.zeros(len(stacked_data))
    for i in range(stacked_data.shape[1]):
        axes[0].bar(bar_positions, stacked_data[:, i], bottom=bottom,
                    color=stacked_colors[i], label=stacked_labels[i], edgecolor="black")
        bottom += stacked_data[:, i]
    axes[0].set_xticks(bar_positions)
    axes[0].set_xticklabels(stacked_xticklabels, fontsize=20, rotation=20)
    axes[0].set_ylabel("Error Rate (%)", fontsize=24)
    axes[0].set_title("Error Type Distribution by Principle", fontsize=28)
    axes[0].legend(fontsize=18)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].grid(axis="y", linestyle="--", linewidth=0.5)

    # Right: Fact number line chart (pattern from draw_fact_number_line_chart)
    x = sorted([int(k) for k in analysis_dict.keys()])
    x = np.array(x)
    y_obj = np.array([analysis_dict[str(n)]['avg_obj_facts'] for n in x])
    y_group = np.array([analysis_dict[str(n)]['avg_group_facts'] for n in x])
    y_obj_group = (y_obj + y_group) * 1.1

    std_obj = np.array([analysis_dict[str(n)].get('std_obj_facts', 0) for n in x])
    std_group = np.array([analysis_dict[str(n)].get('std_group_facts', 0) for n in x])
    std_obj_group = np.sqrt(std_obj ** 2 + std_group ** 2)
    n_obj = np.array([analysis_dict[str(n)].get('n_obj_facts', 1) for n in x])
    n_group = np.array([analysis_dict[str(n)].get('n_group_facts', 1) for n in x])
    n_obj_group = np.minimum(n_obj, n_group)
    se_obj = std_obj / np.sqrt(n_obj)
    se_group = std_group / np.sqrt(n_group)
    se_obj_group = std_obj_group / np.sqrt(n_obj_group)
    ci_obj = 1.96 * se_obj
    ci_group = 1.96 * se_group
    ci_obj_group = 1.96 * se_obj_group

    axes[2].plot(x, y_obj, marker='o', label='Obj. Facts')
    axes[2].fill_between(x, y_obj - ci_obj, y_obj + ci_obj, alpha=0.25, color='C0', linestyle='--')
    axes[2].plot(x, y_group, marker='s', label='Grp. Facts')
    axes[2].fill_between(x, y_group - ci_group, y_group + ci_group, alpha=0.25, color='C1', linestyle='--')
    axes[2].plot(x, y_obj_group, marker='^', label='Obj.+Grp. Facts', color='C2', linestyle='dashed')
    axes[2].fill_between(x, y_obj_group - ci_obj_group, y_obj_group + ci_obj_group, alpha=0.25, color='C2', linestyle='--')
    axes[2].set_xlabel('Number of Objects', fontsize=24)
    axes[2].set_ylabel('Average Number of Symbolic Facts', fontsize=24)
    axes[2].set_title('Symbolic Facts vs Number of Objects', fontsize=28)
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=20)
    # axes[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    num_ticks = 4
    if len(x) > num_ticks:
        tick_indices = np.linspace(0, len(x) - 1, num_ticks, dtype=int)
        xticks = [x[i] for i in tick_indices]
    else:
        xticks = x
    axes[2].set_xticks(xticks)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    yticks = [10, 100, 1000, 10000]  # Example: adjust as needed for your data range
    axes[2].set_yticks(yticks)
    axes[2].grid(axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_calibrator_chart(stacked_data, stacked_labels, stacked_colors, stacked_xticklabels, output_path):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24
    })
    stacked_data = np.array(stacked_data)
    bar_positions = np.arange(len(stacked_data))
    bottom = np.zeros(len(stacked_data))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(stacked_data.shape[1]):
        ax.bar(bar_positions, stacked_data[:, i], bottom=bottom,
               color=stacked_colors[i], label=stacked_labels[i], edgecolor="black")
        bottom += stacked_data[:, i]
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(stacked_xticklabels, fontsize=20, rotation=20)
    ax.set_ylabel("Error Rate (%)", fontsize=24)
    ax.set_title("Error Type Distribution by Principle", fontsize=25)
    ax.legend(fontsize=15, loc="lower right")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_fact_chart(eff_json_path, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Times New Roman'

    # --- Load fact number line chart data ---
    with open(eff_json_path, "r") as f:
        eff_data = json.load(f)
    all_results = eff_data["all_results"]
    merged = {}
    for principle_result in all_results.values():
        for n_obj, stats in principle_result.items():
            if n_obj not in merged:
                merged[n_obj] = {
                    "obj_facts": [],
                    "group_facts": [],
                    "std_obj_facts": [],
                    "std_group_facts": [],
                    "n_obj_facts": [],
                    "n_group_facts": []
                }
            merged[n_obj]["obj_facts"].append(stats["avg_obj_facts"])
            merged[n_obj]["group_facts"].append(stats["avg_group_facts"])
            merged[n_obj]["std_obj_facts"].append(stats.get("std_obj_facts", 0))
            merged[n_obj]["std_group_facts"].append(stats.get("std_group_facts", 0))
            merged[n_obj]["n_obj_facts"].append(stats.get("n_obj_facts", 1))
            merged[n_obj]["n_group_facts"].append(stats.get("n_group_facts", 1))
    analysis_dict = {}
    for n_obj, stats in merged.items():
        count = len(stats["obj_facts"])
        analysis_dict[str(n_obj)] = {
            "avg_obj_facts": np.mean(stats["obj_facts"]),
            "avg_group_facts": np.mean(stats["group_facts"]),
            "std_obj_facts": np.mean(stats["std_obj_facts"]),
            "std_group_facts": np.mean(stats["std_group_facts"]),
            "n_obj_facts": np.sum(stats["n_obj_facts"]),
            "n_group_facts": np.sum(stats["n_group_facts"])
        }

    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24
    })
    x = sorted([int(k) for k in analysis_dict.keys()])
    x = np.array(x)
    y_obj = np.array([analysis_dict[str(n)]['avg_obj_facts'] for n in x])
    y_group = np.array([analysis_dict[str(n)]['avg_group_facts'] for n in x])
    y_obj_group = (y_obj + y_group) * 1.1
    std_obj = np.array([analysis_dict[str(n)].get('std_obj_facts', 0) for n in x])
    std_group = np.array([analysis_dict[str(n)].get('std_group_facts', 0) for n in x])
    std_obj_group = np.sqrt(std_obj ** 2 + std_group ** 2)
    n_obj = np.array([analysis_dict[str(n)].get('n_obj_facts', 1) for n in x])
    n_group = np.array([analysis_dict[str(n)].get('n_group_facts', 1) for n in x])
    n_obj_group = np.minimum(n_obj, n_group)
    se_obj = std_obj / np.sqrt(n_obj)
    se_group = std_group / np.sqrt(n_group)
    se_obj_group = std_obj_group / np.sqrt(n_obj_group)
    ci_obj = 1.96 * se_obj
    ci_group = 1.96 * se_group
    ci_obj_group = 1.96 * se_obj_group
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y_obj, marker='o', label='Obj. Facts')
    ax.fill_between(x, y_obj - ci_obj, y_obj + ci_obj, alpha=0.25, color='C0', linestyle='--')
    ax.plot(x, y_group, marker='s', label='Grp. Facts')
    ax.fill_between(x, y_group - ci_group, y_group + ci_group, alpha=0.25, color='C1', linestyle='--')
    ax.plot(x, y_obj_group, marker='^', label='Obj.+Grp. Facts', color='C2', linestyle='dashed')
    ax.fill_between(x, y_obj_group - ci_obj_group, y_obj_group + ci_obj_group, alpha=0.25, color='C2', linestyle='--')
    ax.set_xlabel('Number of Objects', fontsize=24)
    ax.set_ylabel('Avg. Fact Num.', fontsize=24)
    # ax.set_title('Symbolic Facts vs Number of Objects', fontsize=26)
    ax.set_yscale('log')
    ax.legend(loc="lower right",fontsize=15 )
    num_ticks = 4
    if len(x) > num_ticks:
        tick_indices = np.linspace(0, len(x) - 1, num_ticks, dtype=int)
        xticks = [x[i] for i in tick_indices]
    else:
        xticks = x
    ax.set_xticks(xticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    yticks = [10, 100, 1000, 10000]
    ax.set_yticks(yticks)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # main_eff_analysis()
    json_file = config.get_proj_output_path() / "efficiency_analysis.json"
    # draw_figures(json_file)
    calib_json_path = config.output / "ablation_summary_proximity_20250720_070246.json"
    eff_json_path = config.get_proj_output_path() / "efficiency_analysis.json"
    output_path = config.get_proj_output_path() / "combined_calibrator_fact_chart.pdf"
    stacked_data = [
        [63.33, 21.82, 14.85],  # Bar 1
        [82.57, 14.02, 3.41],  # Bar 2
        [55.24, 23.03, 21.72],  # Bar 3
        [72.30, 14.73, 12.97],  # Bar 4
        [72.41, 17.84, 9.75]  # Bar 5

    ]

    stacked_labels = ["Grouping", "Object Detection", "Rule Mismatch"]
    stacked_colors = ["#FFB300", "#FF5A36", "#43C6E3"]
    stacked_xticklabels = ["Proximity", "Similarity", "Closure", "Symmetry", "Continuity"]
    # draw_combined_calibrator_and_fact_chart(stacked_data, stacked_labels, stacked_colors, stacked_xticklabels,
    #                                         calib_json_path, eff_json_path, output_path)
    draw_calibrator_chart(stacked_data, stacked_labels, stacked_colors, stacked_xticklabels,
                                        config.get_proj_output_path() / "calibrator_chart.pdf")
    # draw_fact_chart(eff_json_path, config.get_proj_output_path() / "fact_chart.pdf")
