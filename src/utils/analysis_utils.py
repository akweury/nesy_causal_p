# Created by MacBook Pro at 02.09.25

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


def draw_line_chart(x, y, xlabel, ylabel, title, save_path=None, msg=""):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    if msg:
        plt.text(0.05, 0.8, msg, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def get_results_path(remote=False, principle=None, model_name=None, img_num=None):
    results_path = config.get_proj_output_path(remote)
    prin_path = results_path / principle

    # get all the json file start with vit_
    if model_name == "vit":
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
        all_json_files = [f for f in all_json_files if f"img_num_{img_num}" in f.name]

    else:
        all_json_files = list(prin_path.glob(f"{model_name}_*.json"))
    if all_json_files:
        latest_json_file = max(all_json_files, key=os.path.getmtime)
        json_path = latest_json_file
    else:
        raise FileNotFoundError(f"No JSON files found for model {model_name} in principle {principle}")
    return json_path


def parse_task_name(task_name):
    info = {}
    # Related concepts after 'rel_'
    rel_match = re.search(r'rel_([a-zA-Z_]+)', task_name)
    if rel_match:
        info['related_concepts'] = rel_match.group(1).split('_')[:-1]

    # Group number (number after related concepts)
    group_num_match = re.search(r'rel_[a-zA-Z_]+_(\d+)', task_name)
    if group_num_match:
        info['group_num'] = int(group_num_match.group(1))
    else:
        if "grid" in task_name:
            info['group_num'] = 4
        else:
            raise ValueError(f"Task name '{task_name}' does not contain a valid group number.")
    # Group size (s, m, l, xl after group number)
    size_match = re.search(r'rel_[a-zA-Z_]+_\d+_(s|m|l|xl)', task_name)
    if size_match:
        info['group_size'] = size_match.group(1)

    # Irrelated concepts after 'irrel_'
    irrel_match = re.search(r'irrel_([a-zA-Z_]+)', task_name)
    if irrel_match:
        info['irrelated_concepts'] = irrel_match.group(1).split('_')[:-1]

    # Rule type at the end ('all' or 'exist')
    rule_match = re.search(r'(all|exist)$', task_name)
    if rule_match:
        info['rule_type'] = rule_match.group(1)

    return info


def draw_f1_heat_map(args, csv_files, gestalt_principles):
    # Function to categorize tasks
    category_acc_scores = {
        "vit_base_patch16_224": pd.Series(dtype=float),
        "llava-onevision-qwen2-7b": pd.Series(dtype=float),
        "InternVL3-78B": pd.Series(dtype=float),
        "GRM": pd.Series(dtype=float)
    }

    for principle, principle_csv_files in csv_files.items():
        tmp_df = pd.read_csv(principle_csv_files[0], index_col=0)  # Load CSV and set first column as index (model names)
        for file in principle_csv_files:
            df = pd.read_csv(file, index_col=0)  # Load CSV and set first column as index (model names)
            def categorize_task(task_name):
                for category in categories:
                    if category in task_name:
                        return category
            if "vit_3" in file.name:
                model_name = "vit_base_patch16_224"
            elif "llava" in file.name:
                model_name = "llava-onevision-qwen2-7b"
            elif "internVL3_78B" in file.name:
                model_name = "InternVL3-78B"
            else:
                raise ValueError("Unknown model in file name:", file.name)
            # df = df.reindex(tmp_df.index, fill_value=0)
            categories = config.categories[principle]
            df["Category"] = df.index.map(categorize_task)
            category_avg_f1 = df.groupby("Category")["F1 Score"].mean()

            # iterative the category_avg_f1, check the name if it is in the config.name_map, then replace it with the value
            if "symmetry" in str(file):
                for cat in category_avg_f1.index:
                    if cat in config.name_map:
                        new_name = config.name_map[cat]
                        category_avg_f1 = category_avg_f1.rename(index={cat: new_name})

            category_acc_scores[model_name] = pd.concat([category_acc_scores[model_name], category_avg_f1])  # Store results
    # Convert dictionary to DataFrame for heatmap
    heatmap_data = pd.DataFrame(category_acc_scores)

    # Adjust figure size dynamically based on the number of columns
    plt.figure(figsize=(max(15, len(heatmap_data.columns) * 1.5), 4))  # Auto-scale width
    ax = sns.heatmap(heatmap_data.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.8,
                     cbar_kws={'label': 'F1 Score'}
                     )
    ax2 = ax.twiny()  # Create a secondary x-axis
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)  # Increase y-axis label font size

    counts = 0
    principle_pos = []
    principle_names = []
    # Compute column splits based on Gestalt principles
    for principle, categories in gestalt_principles.items():
        principle_names.append(principle)
        principle_pos.append(counts + len(categories) / 2)
        if principle == "continuity": continue
        # Draw vertical lines between sections
        pos = int(counts + len(categories))
        ax.axvline(pos, color='black', linestyle='dashed', linewidth=1.5)
        counts += len(categories)
    # Increase the font size of x ticks below the chart
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=30, ha="right")

    ax2.set_xticks(principle_pos)
    ax2.set_xticklabels(principle_names, rotation=0, fontsize=12, fontweight="bold")
    ax2.set_xlim(ax.get_xlim())  # Align with main x-axis
    # ax2.set_xlabel("Gestalt Principles", fontsize=12, fontweight="bold")

    # plt.xlabel("Category", fontsize=12)
    plt.ylabel("Models", fontsize=12)

    # Adjust the layout to remove extra space
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)  # Reduce right margin

    # Save the heatmap
    heat_map_filename = config.get_figure_path(args.remote) / f"f1_heat_map.pdf"
    plt.savefig(heat_map_filename, format="pdf", bbox_inches="tight")  # Ensures no extra space

    print(f"Heatmap saved to: {heat_map_filename}")
