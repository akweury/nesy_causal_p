# Created by jing at 03.03.25
import argparse
import os
import ace_tools_open as tools
import json
import pandas as pd
import scipy.stats as stats
import torch
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

import config
from src.utils import analysis_utils

model_dict = {
    "ViT": {"model": "vit", "img_num": 3},
    "Llava": {"model": "llava", "img_num": 3},
    "InternVL3": {"model": "internVL3_78B", "img_num": 3},
    "GPT-5": {"model": "gpt5", "img_num": 3},
    "GRM": {"model": "grm", "img_num": 3},
}

principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]


def json_to_csv(json_data, csv_file_path):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data).T
    df["accuracy"] /= 100  # Convert accuracy to percentage
    # Calculate performance statistics
    # mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-20)
    # Save F1-score to CSV with row names (index)
    f1_score_df = pd.DataFrame({"F1 Score": f1_score}, index=df.index)
    f1_score_df.to_csv(csv_file_path, index=True)
    print(f"F1-score data saved to {csv_file_path}")
    return df, f1_score


def json_to_csv_llava(json_data, csv_file_path):
    f1_score = torch.tensor([v["f1_score"] for k, v in json_data.items()])
    # Convert JSON to DataFrame
    # Remove the "logic_rules" field from each entry
    for key in json_data.keys():
        if "logic_rules" in json_data[key]:
            del json_data[key]["logic_rules"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(json_data, orient="index")

    df["accuracy"] /= 100  # Convert accuracy to percentage
    # Calculate performance statistics
    # mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall) + 1e-20)
    # Save F1-score to CSV with row names (index)
    f1_score_df = pd.DataFrame({"F1 Score": f1_score}, index=df.index)
    f1_score_df.to_csv(csv_file_path, index=True)
    print(f"F1-score data saved to {csv_file_path}")
    return df, f1_score


def get_per_task_data(json_path, principle):
    # load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    if principle in data:
        return data[principle]
    else:
        if "average" in data:
            data.pop("average", None)
        return data


def analysis_average_performance(args, json_path):
    principle, model_name, img_num = args.principle, args.model, args.img_num
    # load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    per_task_data = data
    per_task_data.pop("average", None)  # Remove 'average' if it exists

    avg_acc = np.mean([v['accuracy'] for v in per_task_data.values()])
    avg_f1 = np.mean([v['f1_score'] for v in per_task_data.values()])
    avg_precision = np.mean([v['precision'] for v in per_task_data.values()])
    avg_recall = np.mean([v['recall'] for v in per_task_data.values()])

    std_acc = np.std([v['accuracy'] for v in per_task_data.values()])
    std_f1 = np.std([v['f1_score'] for v in per_task_data.values()])
    std_precision = np.std([v['precision'] for v in per_task_data.values()])
    std_recall = np.std([v['recall'] for v in per_task_data.values()])

    msg = (f"{principle}\n"
           f"#task: {len(per_task_data)}\n"
           f"#Img: {img_num}\n"
           f"Acc: {avg_acc:.2f} ± {std_acc:.2f}\n"
           f"F1: {avg_f1:.2f} ± {std_f1:.2f}\n"
           f"Prec: {avg_precision:.2f} ± {std_precision:.2f}\n"
           f"Recall: {avg_recall:.2f} ± {std_recall:.2f}")
    # draw the performance as a line chart
    x = list(range(1, len(per_task_data) + 1))
    y = [v['accuracy'] for v in per_task_data.values()]
    x_label = "Task Index"
    y_label = "Accuracy"
    title = f"{model_name}_{img_num} Accuracy for each task in {principle}"

    figure_path = config.get_figure_path(args.remote)
    analysis_utils.draw_line_chart(x, y, x_label, y_label, title, figure_path / f"{principle}_acc_{model_name}_{img_num}.pdf", msg)


def analysis_models(args):
    csv_files = {}
    save_path = config.get_figure_path(args.remote) / f"all_category_all_principles_heat_map.pdf"
    for principle in principles:
        csv_files[principle] = []
        # path = config.results / principle
        for model_name, model_info in model_dict.items():
            json_path = analysis_utils.get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
            per_task_data = get_per_task_data(json_path, principle)
            # replace the soloar with solar if exists in the keys of the per_task_data
            per_task_data = {re.sub(r"soloar", "solar", k): v for k, v in per_task_data.items()}
            new_per_task_data = {}
            for k, v in per_task_data.items():
                if "non_intersected_n_splines" in k:
                    new_per_task_data[k] = v
                elif "intersected_n_splines" in k:
                    new_key = k.replace("intersected_n_splines", "with_intersected_n_splines")
                    new_per_task_data[new_key] = v
                else:
                    new_per_task_data[k] = v
            per_task_data = new_per_task_data
            csv_file_name = config.get_proj_output_path(args.remote) / principle / f"{model_info['model']}_{model_info['img_num']}.csv"
            if model_name == "llava":
                df, f1_score = json_to_csv_llava(per_task_data, csv_file_name)
            else:
                df, f1_score = json_to_csv(per_task_data, csv_file_name)

            csv_files[principle].append(csv_file_name)
    analysis_utils.draw_f1_heat_map(args, csv_files, config.categories)

def analysis_ablation_performance(args):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "ytick.labelsize": 30
    })

    props = ["group_num", "group_size"]
    all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
    model_names = list(model_dict.keys())
    n_props = len(props)
    n_principles = len(all_principles)
    n_models = len(model_names)

    fig, axes = plt.subplots(n_props, n_principles, figsize=(5 * n_principles, 5 * n_props), sharey='row')
    palette = sns.color_palette("Set2", n_colors=n_models)
    bar_width = 0.10
    group_gap = 0.05

    # Collect results for each principle and property value
    results = {prop: {principle: {model: [] for model in model_names} for principle in all_principles} for prop in props}
    for row_idx, prop in enumerate(props):
        for principle in all_principles:
            for model_name, model_info in model_dict.items():
                json_path = analysis_utils.get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)
                for task_name, task_res in per_task_data.items():
                    task_info = analysis_utils.parse_task_name(task_name)
                    if prop in task_info:
                        value = task_info[prop]
                        acc = task_res["accuracy"] * 100 if model_name == "GRM" else task_res["accuracy"]
                        results[prop][principle][model_name].append(acc)

    # Main matrix plot
    for row_idx, prop in enumerate(props):
        for col_idx, principle in enumerate(all_principles):
            ax = axes[row_idx, col_idx] if n_props > 1 else axes[col_idx]
            has_data = any(len(results[prop][principle][model]) > 0 for model in model_names)
            if not has_data:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                ax.text(0.5, 0.5, f"{principle}\nNo data", fontsize=24, color="gray",
                        ha='center', va='center', transform=ax.transAxes)
                continue
            x = np.arange(len(model_names))
            n = 1
            group_width = n * bar_width + group_gap
            for j, model_name in enumerate(model_names):
                value = np.mean(results[prop][principle][model_name]) if results[prop][principle][model_name] else np.nan
                ax.bar(x[j] * group_width, value, width=bar_width, color=palette[j], label=model_name if row_idx == 0 and col_idx == 0 else "")
                if not np.isnan(value):
                    ax.text(x[j] * group_width, value + 2, f"{value:.1f}", rotation=30, ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x * group_width + (bar_width) / 2)
            ax.set_xticklabels(model_names, fontsize=30, ha='right', rotation=30)
            ax.set_xlabel(prop, fontsize=25)
            if row_idx == 0:
                ax.set_title(principle, fontsize=35, fontweight='bold')
            ax.set_ylim(0, 100)
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=25)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=len(model_names),
               fontsize=30, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = config.get_figure_path(args.remote) / f"ablation_all_props_all_principles_matrix.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Matrix of grouped ablation bar charts saved to: {save_path}")

    # Extra figure: average over all principles for each prop
    avg_results = {prop: [] for prop in props}
    for prop in props:
        for model_name in model_names:
            accs = []
            for principle in all_principles:
                accs.extend(results[prop][principle][model_name])
            avg_results[prop].append(np.mean(accs) if accs else np.nan)

    fig, axes = plt.subplots(1, len(props), figsize=(6 * len(props), 6), sharey=True)
    if len(props) == 1:
        axes = [axes]
    bar_width = 0.60
    x = np.arange(len(model_names))
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    legend_handles, legend_labels = None, None
    for idx, prop in enumerate(props):
        ax = axes[idx]
        vals = avg_results[prop]
        for i, v in enumerate(vals):
            bar = ax.bar(x[i], v, width=bar_width, color=palette[i], label=model_names[i])
            if not np.isnan(v):
                ax.text(x[i], v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=0)
        ax.set_title(f"Avg Acc: {prop}", fontsize=30, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=25)
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.legend_.remove() if ax.get_legend() else None
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=25, rotation=30)
        ax.set_ylim(0, 100)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # if legend_handles and legend_labels:
    #     fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(model_names), fontsize=20)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = config.get_figure_path(args.remote) / "ablation_avg_accuracy_per_prop_overall.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
    for idx, prop in enumerate(props):
        fig_single, ax_single = plt.subplots(figsize=(6, 6), sharey=True)
        vals = avg_results[prop]
        x = np.arange(len(model_names))
        palette = sns.color_palette("Set2", n_colors=len(model_names))
        for i, v in enumerate(vals):
            ax_single.bar(x[i], v, width=bar_width, color=palette[i], label=model_names[i])
            if not np.isnan(v):
                ax_single.text(x[i], v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=0)
        ax_single.set_title(f"Avg Acc: {prop}", fontsize=30, fontweight='bold')
        ax_single.set_ylabel("Accuracy", fontsize=25)
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(model_names, fontsize=25, rotation=25)
        ax_single.set_ylim(0, 100)
        ax_single.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax_single.spines['top'].set_visible(False)
        ax_single.spines['right'].set_visible(False)
        # ax_single.legend(fontsize=20, loc='upper left')
        plt.tight_layout()
        save_path = config.get_figure_path(args.remote) / f"obj_ablation_avg_accuracy_{prop}_overall.pdf"
        fig_single.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig_single)



def analysis_obj_ablation_performance(args):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "ytick.labelsize": 30})
    props = ["color", "shape", "size"]
    all_principles = ["proximity", "similarity", "closure", "symmetry", "continuity"]
    model_names = list(model_dict.keys())
    official_model_names = list(model_dict.keys())
    n_props = len(props)
    n_principles = len(all_principles)
    fig, axes = plt.subplots(n_props, n_principles, figsize=(5 * n_principles, 5 * n_props), sharey='row')
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    bar_width = 0.10
    group_gap = 0.05

    # Collect results for each (prop, principle, model)
    results = {prop: {principle: {model: [] for model in official_model_names} for principle in all_principles} for prop in props}
    for row_idx, prop in enumerate(props):
        for principle in all_principles:
            for model_name, model_info in model_dict.items():
                json_path = analysis_utils.get_results_path(args.remote, principle, model_info["model"], model_info["img_num"])
                per_task_data = get_per_task_data(json_path, principle)
                for task_name, task_res in per_task_data.items():
                    if task_name[-1] != "_":
                        task_name += "_"
                    task_info = analysis_utils.parse_task_name(task_name)
                    if prop in task_info["related_concepts"]:
                        acc = task_res["accuracy"] * 100 if model_name == "GRM" else task_res["accuracy"]
                        results[prop][principle][model_name].append(acc)

    # Main matrix plot
    for row_idx, prop in enumerate(props):
        for col_idx, principle in enumerate(all_principles):
            ax = axes[row_idx, col_idx] if n_props > 1 else axes[col_idx]
            has_data = any(len(results[prop][principle][model]) > 0 for model in official_model_names)
            if not has_data:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
                ax.text(0.5, 0.5, f"{principle}\nNo data", fontsize=24, color="gray",
                        ha='center', va='center', transform=ax.transAxes)
                continue
            x = np.arange(len(model_names))
            n = 1
            group_width = n * bar_width + group_gap
            for j, model_name in enumerate(official_model_names):
                value = np.mean(results[prop][principle][model_name]) if results[prop][principle][model_name] else np.nan
                ax.bar(x[j] * group_width, value, width=bar_width, color=palette[j], label=model_name if row_idx == 0 and col_idx == 0 else "")
                if not np.isnan(value):
                    ax.text(x[j] * group_width, value + 2, f"{value:.1f}", rotation=30, ha='center', va='bottom', fontsize=10)
            ax.set_xticks(x * group_width + (bar_width) / 2)
            ax.set_xticklabels(model_names, fontsize=30, ha='right', rotation=30)
            ax.set_xlabel(prop, fontsize=25)
            if row_idx == 0:
                ax.set_title(principle, fontsize=35, fontweight='bold')
            ax.set_ylim(0, 100)
            if col_idx == 0:
                ax.set_ylabel('Accuracy', fontsize=25)
            ax.axhline(50, color='gray', linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:len(model_names)], model_names, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(model_names), fontsize=30)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = config.get_figure_path(args.remote) / f"ablation_obj_all_props_all_principles_matrix.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    # Extra figure: average over all principles for each prop
    avg_results = {prop: [] for prop in props}
    for prop in props:
        for model_name in official_model_names:
            accs = []
            for principle in all_principles:
                accs.extend(results[prop][principle][model_name])
            avg_results[prop].append(np.mean(accs) if accs else np.nan)

    fig, axes = plt.subplots(1, len(props), figsize=(6 * len(props), 6), sharey=True)
    if len(props) == 1:
        axes = [axes]
    bar_width = 0.6
    x = np.arange(len(official_model_names))
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    legend_handles, legend_labels = None, None
    for idx, prop in enumerate(props):
        ax = axes[idx]
        vals = avg_results[prop]
        bars = []
        for i, v in enumerate(vals):
            bar = ax.bar(x[i], v, width=bar_width, color=palette[i], label=model_names[i])
            bars.append(bar)
            if not np.isnan(v):
                ax.text(x[i], v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=0)
        ax.set_title(f"Avg Acc: {prop}", fontsize=30, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=25)
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.legend_.remove() if ax.get_legend() else None  # Remove legend for other subplots
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=25, rotation=25)
        ax.set_ylim(0, 100)
        ax.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(model_names), fontsize=20)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = config.get_figure_path(args.remote) / "obj_ablation_avg_accuracy_per_prop_overall.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

    # After plotting and saving the overall figure (avg accuracy per prop)
    for idx, prop in enumerate(props):
        fig_single, ax_single = plt.subplots(figsize=(6, 6), sharey=True)
        vals = avg_results[prop]
        x = np.arange(len(official_model_names))
        palette = sns.color_palette("Set2", n_colors=len(model_names))
        for i, v in enumerate(vals):
            ax_single.bar(x[i], v, width=bar_width, color=palette[i], label=model_names[i])
            if not np.isnan(v):
                ax_single.text(x[i], v + 2, f"{v:.1f}", ha='center', va='bottom', fontsize=15, rotation=0)
        ax_single.set_title(f"Avg Acc: {prop}", fontsize=30, fontweight='bold')
        ax_single.set_ylabel("Accuracy", fontsize=25)
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(model_names, fontsize=25, rotation=25)
        ax_single.set_ylim(0, 100)
        ax_single.axhline(50, color='gray', linestyle='--', linewidth=1)
        ax_single.spines['top'].set_visible(False)
        ax_single.spines['right'].set_visible(False)
        # ax_single.legend(fontsize=20, loc='upper left')  # Optional: add legend if needed
        plt.tight_layout()
        save_path = config.get_figure_path(args.remote) / f"obj_ablation_avg_accuracy_{prop}_overall.pdf"
        fig_single.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig_single)
def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--model", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--principle", type=str, required=True)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--mode", type=str, default="avg_principle")
    parser.add_argument("--img_num", type=int)
    args = parser.parse_args()
    json_path = analysis_utils.get_results_path(args.remote, args.principle, args.model, args.img_num)

    if args.mode == "principle":
        analysis_average_performance(args, json_path)
    elif args.mode == "category":
        analysis_models(args)
    elif args.mode == "ablation":
        analysis_ablation_performance(args)
    elif args.mode == "obj_ablation":
        analysis_obj_ablation_performance(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")



if __name__ == "__main__":
    main()
