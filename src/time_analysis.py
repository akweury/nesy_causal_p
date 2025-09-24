# Created by MacBook Pro at 24.09.25

import config


def average_and_max_hard_og_time(log_path):
    """
    Calculates the average and maximum time for 'hard_og' ablation across all tasks in the log file.
    Args:
        log_path (str): Path to the log file.
    Returns:
        tuple: (average_time, max_time) in seconds for 'hard_og'.
    """
    import re

    hard_og_times = []
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"Running ablation: hard_obj in ([\d\.]+) seconds", line)
            if match:
                hard_og_times.append(float(match.group(1)))
    if not hard_og_times:
        raise ValueError("No 'hard_obj' entries found in the log.")
    avg_time = sum(hard_og_times) / len(hard_og_times)
    max_time = max(hard_og_times)
    return avg_time, max_time


def draw_grouped_time_comparison():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data from the table
    principles = ["Proximity", "Similarity", "Closure", "Symmetry", "Continuity"]
    grm_avg = [6.28, 88.76, 10.29, 34.42, 9.32]
    internvl_avg = [14.11, 13.34, 15.58, 12.30, 14.89]
    gpt5_avg = [109.45, 94.34, 105.46, 157.56, 82.59]

    bar_width = 0.2
    x = np.arange(len(principles))

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.bar(x - 1 * bar_width, grm_avg, width=bar_width, label="GRM", color="lightblue")
    # ax.bar(x - 0.5*bar_width, grm_max, width=bar_width, label="GRM (Maximum)")
    ax.bar(x + 0 * bar_width, internvl_avg, width=bar_width, label="InternVL3-78B", color="pink")
    ax.bar(x + 1 * bar_width, gpt5_avg, width=bar_width, label="GPT-5", color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(principles, fontsize=24)
    ax.set_ylabel("Time (seconds)", fontsize=24)
    ax.tick_params(axis='y', labelsize=24)  # Set y-tick font size

    ax.set_title("Rule Induction Time Across Gestalt Principles", fontsize=24)
    # ax.set_yscale("log")

    ax.legend(fontsize=24, loc='upper left')
    plt.tight_layout()

    save_path = config.get_figure_path() / f"grouped_time_comparison.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # log_file_path = config.get_proj_output_path()/"grm_proximity_log.log"
    # log_file_path = config.get_proj_output_path()/"grm_similarity_log.log"
    # log_file_path = config.get_proj_output_path()/"grm_closure_log.log"
    # log_file_path = config.get_proj_output_path()/"grm_continuity_log.log"
    log_file_path = config.get_proj_output_path() / "grm_symmetry_log.log"
    avg_time, max_time = average_and_max_hard_og_time(log_file_path)
    print(f"Average time for 'hard_og': {avg_time:.2f} seconds")
    print(f"Maximum time for 'hard_og': {max_time:.2f} seconds")
    draw_grouped_time_comparison()
