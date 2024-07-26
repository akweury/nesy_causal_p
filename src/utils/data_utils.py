# Created by shaji at 21/06/2024

import numpy as np
import torch

def data2patch(data):
    input_patches = []
    output_patches = []
    for example in data:
        input_patch = example['input']
        output_patch = example['output']
        input_patches.append(input_patch)
        output_patches.append(output_patch)
    return input_patches, output_patches


def patch2line_patches(patch):
    rows, cols = patch.shape
    row_patches = []
    col_patches = []
    for col in range(cols):
        if col == 0:
            data = np.concatenate((np.zeros_like(patch[:, col:col + 1])+10, patch[:, col:col + 2]), axis=1)
        elif col == cols - 1:
            data = np.concatenate((patch[:, col - 1:], np.zeros_like(patch[:, col:col + 1])+10), axis=1)
        else:
            data = patch[:, col - 1:col + 2]
        data = data.T
        col_patches.append(data)

    for row in range(rows):
        if row == 0:
            data = np.concatenate((np.zeros_like(patch[row:row + 1, :])+10,patch[row:row + 2, :]), axis=0)
        elif row == rows - 1:
            data = np.concatenate((patch[row - 1:, :],np.zeros_like(patch[row:(row + 1), :])+10), axis=0)
        else:
            data = patch[(row - 1):(row + 2), :]
        row_patches.append(data)
    return row_patches, col_patches


def patch2tensor(patch):
    patch = torch.tensor(patch).float()
    patch[patch != 10] = 1
    patch[patch == 10] = 0
    return patch


def group2patch(whole_patch, group):
    data = np.array(whole_patch)
    group_patch = np.zeros_like(data) + 10
    for pos in group:
        group_patch[pos] = data[pos]

    return group_patch
