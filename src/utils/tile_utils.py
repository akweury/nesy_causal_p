# Created by jing at 17.06.24

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.utils import visual_utils


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


def individual_grouping(param):
    pass


def find_identical_groups(space_patch, target_patch, is_visual=True):
    identical_groups = check_patch_exists(space_patch, target_patch)
    return identical_groups
