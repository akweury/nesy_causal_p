# Created by shaji at 21/06/2024

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


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


def create_identity_kernel(n):
    # Create an nxn kernel with all zeros
    identity_kernel = torch.zeros((1, 1, n, n), dtype=torch.float32)

    # Set the center value to 1
    identity_kernel[0, 0, n // 2, n // 2] = 1

    return identity_kernel


def find_submatrix(matrix_64x64):
    # Unfold the 64x64 matrix into non-overlapping 3x3 patches
    patches = F.unfold(matrix_64x64.unsqueeze(0).unsqueeze(0), kernel_size=(3, 3), stride=3)

    # Reshape the patches to (number_of_patches, 3, 3)
    patches = patches.transpose(1, 2).reshape(-1, 3, 3)

    # Calculate positions of patches
    num_patches_x = matrix_64x64.size(0) // 3
    num_patches_y = matrix_64x64.size(1) // 3
    positions = torch.stack(torch.meshgrid(torch.arange(num_patches_x), torch.arange(num_patches_y)), dim=-1).reshape(-1, 2)

    # Filter out zero patches
    non_zero_mask = patches.sum(dim=(1, 2)) != 0
    non_zero_patches = patches[non_zero_mask]
    non_zero_positions = positions[non_zero_mask]

    return non_zero_patches, non_zero_positions

# Method 1: Cosine Similarity
def cosine_similarity_mapping(A, B):
    A_flat = A.view(-1)
    B_flat = B.view(-1)
    cosine_sim = F.cosine_similarity(A_flat, B_flat, dim=0)
    # Scale cosine similarity from [-1, 1] to [0, 1]
    return (cosine_sim + 1) / 2


# Method 2: Element-wise Dot Product followed by Sigmoid
def dot_product_sigmoid_mapping(A, B):
    dot_product = torch.sum(A * B)
    sigmoid_value = torch.sigmoid(dot_product)
    return sigmoid_value


# Method 3: Neural Network with Sigmoid Output
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 2, 1)

    def forward(self, A, B):
        x = torch.cat((A.view(-1), B.view(-1)), dim=0)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


def neural_network_mapping(A, B, model):
    with torch.no_grad():
        return model(A, B)


# Method to map the matrix to a unique value
def matrix_to_value(A):
    width, height = A.shape[-2], A.shape[-1]
    # Flatten the matrix and convert to a binary string
    A_flat = A.view(-1)
    binary_str = ''.join(A_flat.int().cpu().numpy().astype(str))

    # Convert the binary string to an integer
    value_int = int(binary_str, 2)

    # Calculate the maximum possible value for a 64x64 binary matrix
    max_value = 2 ** (width * height) - 1

    # Normalize the integer to a value in the range [0, 1]
    normalized_value = value_int / max_value

    # Flatten the matrices to 1D

    # Get the indices of 1s for each matrix in the batch
    compressed = [torch.nonzero(A_flat).squeeze(1)]

    return normalized_value


# Method to recover the matrix from the unique value
def value_to_matrix(value):
    # Calculate the maximum possible value for a 64x64 binary matrix
    max_value = 2 ** (64 * 64) - 1

    # Convert the floating-point value back to the corresponding integer
    value_int = int(value * max_value)

    # Convert the integer back to a binary string
    binary_str = bin(value_int)[2:].zfill(64 * 64)

    # Convert the binary string to a 1D tensor and reshape it into a 64x64 matrix
    matrix = torch.tensor([int(b) for b in binary_str], dtype=torch.float32).view(64, 64)

    return matrix
