# Created by jing at 17.06.24

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config


def part_of_patch(input_patch, output_patch):
    """ the small_array can be x% identical to large_array
    return: top n identical patches in large array, its position, width, and different tiles if any
    """
    input_patch = np.array(input_patch)
    output_patch = np.array(output_patch)
    if input_patch.shape[0] == output_patch.shape[0] and input_patch.shape[1] == output_patch.shape[1]:
        part_type = config.code_group_relation["a_eq_b"]
        large_patch = output_patch
        small_patch = input_patch
    elif input_patch.shape[0] <= output_patch.shape[0] and input_patch.shape[1] <= output_patch.shape[1]:
        part_type = config.code_group_relation["b_inc_a"]
        large_patch = output_patch
        small_patch = input_patch
    elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
        part_type = config.code_group_relation["a_inc_b"]
        large_patch = input_patch
        small_patch = output_patch
    else:
        part_type = config.code_group_relation["else"]
        return part_type
    # Get sliding windows of shape (3, 3) from the large array
    windows = sliding_window_view(large_patch, small_patch.shape)
    match_counts = np.sum(windows == small_patch, axis=(2, 3))
    similarity = match_counts / small_patch.size * 100
    if similarity.max() < 100:
        part_type = config.code_group_relation["else"]
    return part_type


def check_patch_exists(space_patch, target_patch):
    """ the small_array can be x% identical to large_array
    return: top n identical patches in large array, its position, width, and different tiles if any
    """
    space_patch = np.array(space_patch)
    target_patch = np.array(target_patch)
    # Get sliding windows of shape (3, 3) from the large array
    windows = sliding_window_view(space_patch, target_patch.shape)
    # Calculate the similarity percentage for each window
    match_counts = np.sum(windows == target_patch, axis=(2, 3))
    similarity = match_counts / target_patch.size * 100
    # Generate the positions and differences
    positions = np.argwhere(similarity >= 0)

    results = []
    for pos in positions:
        i, j = pos
        differences = list(zip(*np.where(windows[i, j] != target_patch)))
        differences = [(i + x, j + y) for x, y in differences]
        results.append({
            'position': (i, j),
            'similarity': similarity[i, j],
            'differences': differences
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results


def find_identical_groups(space_patch, target_patch, is_visual=True):
    identical_groups = check_patch_exists(space_patch, target_patch)
    return identical_groups
