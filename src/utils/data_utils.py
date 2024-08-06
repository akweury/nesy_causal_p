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
            data = np.concatenate((np.zeros_like(patch[:, col:col + 1]) + 10, patch[:, col:col + 2]), axis=1)
        elif col == cols - 1:
            data = np.concatenate((patch[:, col - 1:], np.zeros_like(patch[:, col:col + 1]) + 10), axis=1)
        else:
            data = patch[:, col - 1:col + 2]
        data = data.T
        col_patches.append(data)

    for row in range(rows):
        if row == 0:
            data = np.concatenate((np.zeros_like(patch[row:row + 1, :]) + 10, patch[row:row + 2, :]), axis=0)
        elif row == rows - 1:
            data = np.concatenate((patch[row - 1:, :], np.zeros_like(patch[row:(row + 1), :]) + 10), axis=0)
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


def closest_distance(points):
    """
    Finds the closest distance between two points in a set of 2D points
    along the x-axis and y-axis.

    Parameters:
    - points (list of tuples): A list where each tuple represents a point (x, y).

    Returns:
    - min_x_dist (float): The smallest distance between any two points along the x-axis.
    - min_y_dist (float): The smallest distance between any two points along the y-axis.
    """
    if len(points) < 2:
        return float('inf'), float('inf')  # Not enough points to compare

    # Sort points by x-coordinate and calculate minimum distance along x-axis
    points_x_sorted = points[np.argsort(points[:, 0])]
    min_x_dist = float('inf')
    for i in range(len(points_x_sorted) - 1):
        x_diff = abs(points_x_sorted[i + 1][0] - points_x_sorted[i][0])
        if x_diff < min_x_dist:
            min_x_dist = x_diff

    # Sort points by y-coordinate and calculate minimum distance along y-axis
    points_y_sorted = points[np.argsort(points[:, 1])]
    min_y_dist = float('inf')
    for i in range(len(points_y_sorted) - 1):
        y_diff = abs(points_y_sorted[i + 1][1] - points_y_sorted[i][1])
        if y_diff < min_y_dist:
            min_y_dist = y_diff

    return min_x_dist, min_y_dist


def oco2patch(data):
    positions = torch.cat([torch.tensor([[d["x"], d["y"]]]) for d in data], dim=0)
    patch = None
    max_counts = 100
    scale = 64
    grid_position = torch.ceil(positions * scale).to(dtype=torch.int)

    # while max_counts > 1:
    patch = torch.zeros((scale, scale))
    grid_position = torch.ceil(positions * (scale - 1)).to(dtype=torch.int)
    unique_pos, pos_counts = torch.unique(grid_position, dim=0, return_counts=True)
    max_counts = torch.max(pos_counts)
    scale += 1
    for p_i in range(len(grid_position)):
        patch[grid_position[p_i, 1], grid_position[p_i, 0]] = 1
    return patch


def patch2info_patch(matrix):
    matrix = matrix.squeeze().numpy()
    n = len(matrix)
    m = 3 * n
    expanded_matrix = np.zeros((m, m))

    for i in range(n):
        for j in range(n):
            patch = np.zeros((3, 3))
            value = matrix[i, j]

            # Center of the patch is the value itself
            patch[1, 1] = value

            # Fill in neighbor information
            if i > 0:
                patch[0, 1] = matrix[:i - 1, j].sum()  # Top neighbor
            if i < n - 1:
                patch[2, 1] = matrix[i + 1:, j].sum()  # Bottom neighbor
            if j > 0:
                patch[1, 0] = matrix[i, :j - 1].sum()  # Left neighbor
            if j < n - 1:
                patch[1, 2] = matrix[i, j + 1:].sum()  # Right neighbor
            if i > 0 and j > 0:
                patch[0, 0] = matrix[:i - 1, :j - 1].sum()  # Top-left neighbor
            if i > 0 and j < n - 1:
                patch[0, 2] = matrix[:i - 1, j + 1:].sum()  # Top-right neighbor
            if i < n - 1 and j > 0:
                patch[2, 0] = matrix[i + 1:, :j - 1].sum()  # Bottom-left neighbor
            if i < n - 1 and j < n - 1:
                patch[2, 2] = matrix[i + 1:, j + 1:].sum()  # Bottom-right neighbor

            # Place the patch into the expanded matrix
            expanded_matrix[3 * i:3 * i + 3, 3 * j:3 * j + 3] = patch

    return torch.from_numpy(expanded_matrix).unsqueeze(0).to(dtype=torch.float)
