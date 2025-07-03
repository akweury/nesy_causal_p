# Created by MacBook Pro at 03.07.25

import torch
import numpy as np
import matplotlib.pyplot as plt

def count_symbolic_facts(fact_dict):
    obj_fact_count = 0
    group_fact_count = 0
    n_obj = fact_dict["has_shape"].shape[0]
    n_groups = fact_dict["in_group"].shape[1] if "in_group" in fact_dict else 0

    # 1. Unary object-level facts
    unary_keys = ["has_shape", "has_color", "x", "y", "w", "h"]
    for key in unary_keys:
        obj_fact_count += fact_dict[key].shape[0]

    # 2. Binary object-object relations
    binary_keys = ["same_shape", "same_color", "same_size", "mirror_x", "same_y"]
    for key in binary_keys:
        mat = fact_dict[key]
        if isinstance(mat, torch.Tensor) and mat.ndim == 2:
            obj_fact_count += int(mat.sum().item())

    # 3. Negative shape assertions (not_has_shape_*)
    for key in fact_dict:
        if key.startswith("not_has_shape"):
            obj_fact_count += int(fact_dict[key].sum().item())

    # 4. Group membership (in_group: [n_obj, n_groups])
    if "in_group" in fact_dict:
        group_fact_count += int(fact_dict["in_group"].sum().item())

    # 5. Group-level unary predicates
    group_level_keys = [
        "group_size", "no_member_triangle", "no_member_rectangle", "no_member_circle",
        "diverse_shapes", "unique_shapes",
        "diverse_colors", "unique_colors",
        "diverse_sizes", "unique_sizes"
    ]
    for key in group_level_keys:
        val = fact_dict[key]
        if isinstance(val, torch.Tensor):
            group_fact_count += int(val.sum().item())

    return obj_fact_count, group_fact_count, n_obj, n_groups


def draw_time_cost_line_chart(time_cost_dict, save_path):
    """
    Draws and saves a line chart:
    - x-axis: number of objects
    - y-axis: average time cost (log scale)
    - Lines: object extraction, group extraction (group only), obj+group extraction (cumulative)
    - Shaded error regions: std and 95% confidence interval
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x = sorted(time_cost_dict.keys())
    y_obj = np.array([time_cost_dict[n]['avg_obj_time'] for n in x])
    y_obj_group = np.array([time_cost_dict[n]['avg_group_time'] for n in x])
    y_group_only = y_obj_group - y_obj

    std_obj = np.array([time_cost_dict[n].get('std_obj_time', 0) for n in x])
    std_obj_group = np.array([time_cost_dict[n].get('std_group_time', 0) for n in x])
    # For group_only, estimate std by sqrt(std_group^2 + std_obj^2) (assuming independence)
    std_group_only = np.sqrt(std_obj_group**2 + std_obj**2)

    n_obj = np.array([time_cost_dict[n].get('n_obj_time', 1) for n in x])
    n_group = np.array([time_cost_dict[n].get('n_group_time', 1) for n in x])
    n_group_only = np.minimum(n_obj, n_group)  # conservative

    # Standard error and 95% CI
    se_obj = std_obj / np.sqrt(n_obj)
    se_obj_group = std_obj_group / np.sqrt(n_group)
    se_group_only = std_group_only / np.sqrt(n_group_only)
    ci_obj = 1.96 * se_obj
    ci_obj_group = 1.96 * se_obj_group
    ci_group_only = 1.96 * se_group_only

    plt.figure()

    # Object extraction time
    plt.plot(x, y_obj, marker='o', label='Object Extraction Time')
    plt.fill_between(x, y_obj - ci_obj, y_obj + ci_obj, alpha=0.25, color='C0', linestyle='--', label='Obj 95% CI')

    # Obj+Group extraction time (cumulative)
    plt.plot(x, y_obj_group, marker='s', label='Obj+Group Extraction Time')
    plt.fill_between(x, y_obj_group - ci_obj_group, y_obj_group + ci_obj_group, alpha=0.25, color='C1', linestyle='--', label='Obj+Group 95% CI')

    # Group extraction time (group only)
    plt.plot(x, y_group_only, marker='^', label='Group Extraction Time (Group Only)')
    plt.fill_between(x, y_group_only - ci_group_only, y_group_only + ci_group_only, alpha=0.25, color='C2', linestyle='--', label='Group Only 95% CI')

    plt.xlabel('Number of Objects')
    plt.ylabel('Average Time Cost (s)')
    plt.title('Time Cost vs Number of Objects')
    plt.legend()
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def draw_fact_number_line_chart(analysis_dict, save_path):
    """
    Draws and saves a line chart: x-axis is number of objects,
    y-axis is avg object facts and avg group facts (log scale),
    with shaded error regions (std).
    """
    x = sorted(analysis_dict.keys())
    y_obj = np.array([analysis_dict[n]['avg_obj_facts'] for n in x])
    y_group = np.array([analysis_dict[n]['avg_group_facts'] for n in x])
    y_obj_group = y_obj + y_group


    std_obj = np.array([analysis_dict[n].get('std_obj_facts', 0) for n in x])
    std_group = np.array([analysis_dict[n].get('std_group_facts', 0) for n in x])
    # For obj+group, assuming independence
    std_obj_group = np.sqrt(std_obj**2 + std_group**2)


    n_obj = np.array([analysis_dict[n].get('n_obj_facts', 1) for n in x])
    n_group = np.array([analysis_dict[n].get('n_group_facts', 1) for n in x])
    n_obj_group = np.minimum(n_obj, n_group)  # conservative

    se_obj = std_obj / np.sqrt(n_obj)
    se_group = std_group / np.sqrt(n_group)
    se_obj_group = std_obj_group / np.sqrt(n_obj_group)

    ci_obj = 1.96 * se_obj
    ci_group = 1.96 * se_group
    ci_obj_group = 1.96 * se_obj_group

    plt.figure()
    plt.plot(x, y_obj, marker='o', label='Avg Object Facts')
    plt.fill_between(x, y_obj - ci_obj, y_obj + ci_obj, alpha=0.25, color='C0', linestyle='--', label='Obj 95% CI')

    plt.plot(x, y_group, marker='s', label='Avg Group Facts')
    plt.fill_between(x, y_group - ci_group, y_group + ci_group, alpha=0.25, color='C1', linestyle='--', label='Group 95% CI')

    plt.plot(x, y_obj_group, marker='^', label='Avg Obj+Group Facts')
    plt.fill_between(x, y_obj_group - ci_obj_group, y_obj_group + ci_obj_group, alpha=0.25, color='C2', linestyle='--', label='Obj+Group 95% CI')


    plt.xlabel('Number of Objects')
    plt.ylabel('Average Number of Symbolic Facts')
    plt.title('Symbolic Facts vs Number of Objects')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
