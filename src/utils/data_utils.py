# Created by shaji at 21/06/2024

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import config
import cv2
from tqdm import tqdm


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


def data2positions(data):
    positions = torch.cat([torch.tensor([[d["x"], d["y"]]]) for d in data], dim=0)
    scale = 64
    grid_position = torch.ceil(positions * (scale - 1)).to(dtype=torch.int)
    return grid_position


def oco2patch(data):
    positions = torch.cat([torch.tensor([[d["x"], d["y"]]]) for d in data], dim=0)
    scale = config.pixel_size
    patch = torch.zeros((scale, scale))
    grid_position = torch.ceil(positions * (scale - 1)).to(dtype=torch.int)
    unique_pos, pos_counts = torch.unique(grid_position, dim=0, return_counts=True)
    max_counts = torch.max(pos_counts)
    if max_counts != 1:
        raise ValueError(f"pixel size is too small, overlay {max_counts}")
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


def find_submatrix(matrix_64x64, kernel_size):
    # Unfold the 64x64 matrix into non-overlapping 3x3 patches
    patches = F.unfold(matrix_64x64.unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, kernel_size),
                       stride=kernel_size)

    # Reshape the patches to (number_of_patches, 3, 3)
    patches = patches.transpose(1, 2).reshape(-1, kernel_size, kernel_size)

    # Calculate positions of patches
    num_patches_x = matrix_64x64.size(0) // kernel_size
    num_patches_y = matrix_64x64.size(1) // kernel_size
    positions = torch.stack(torch.meshgrid(torch.arange(num_patches_x),
                                           torch.arange(num_patches_y)), dim=-1).reshape(-1, 2)
    # Filter out zero patches
    non_zero_mask = patches.sum(dim=(1, 2)) != 0
    non_zero_patches = patches[non_zero_mask]
    non_zero_positions = positions[non_zero_mask]
    non_zero_patches_shifted = shift_content_to_top_left(non_zero_patches)
    return non_zero_patches, non_zero_patches_shifted, non_zero_positions


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


def shift_up(matrix):
    shift_count = 0
    while matrix.sum() > 0 and torch.all(matrix[:, 0] == 0):  # Check if the top row is full of zeros
        matrix = torch.roll(matrix, shifts=-1, dims=1)  # Shift all rows up
        matrix[:, -1] = 0  # Fill the last row with zeros after the shift
        shift_count += 1
    return matrix, shift_count


# Function to shift the matrix left if the leftmost column is all zeros
def shift_left(matrix):
    shift_count = 0
    while matrix.sum() > 0 and torch.all(matrix[:, :, 0] == 0):  # Check if the leftmost column is full of zeros
        matrix = torch.roll(matrix, shifts=-1, dims=2)  # Shift all columns to the left
        matrix[:, :, -1] = 0  # Fill the last column with zeros after the shift
        shift_count += 1
    return matrix, shift_count


def shift_content_to_top_left(batch_matrices, given_rs=None, given_cs=None):
    # Function to shift the matrix up if the top row is all zeros
    rs = torch.zeros(batch_matrices.shape[0])
    cs = torch.zeros(batch_matrices.shape[0])
    shifted_matrices = []
    for i in range(batch_matrices.shape[0]):
        matrix = batch_matrices[i]
        if given_rs is not None:
            matrix = torch.roll(matrix, shifts=int(-given_rs[i].item()), dims=1)  # Shift all rows up
        else:
            matrix, shift_row = shift_up(matrix)  # Apply upward shift
            rs[i] = shift_row

        if given_cs is not None:
            matrix = torch.roll(matrix, shifts=int(-given_cs[i].item()), dims=2)  # Shift all rows up
        else:
            matrix, shift_col = shift_left(matrix)  # Apply leftward shift
            cs[i] = shift_col
        shifted_matrices.append(matrix.unsqueeze(0))
    shifted_matrices = torch.cat(shifted_matrices, dim=0)
    return shifted_matrices, rs, cs


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    else:
        return None


def load_bw_img(img_path, size=None):
    # Load an image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image[image != 211] = 1
    image[image == 211] = 0
    img_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    img_resized = torch.from_numpy(img_resized).unsqueeze(0)

    return img_resized

def rgb2bw(rgb, resize=None):

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.int32)
    gray[gray != 0] = 1
    if resize is not None:
        gray = cv2.resize(gray.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
        gray = torch.from_numpy(gray).unsqueeze(0)
    return gray


def find_valid_radius(matrix):
    # Find indices of valid elements (nonzero items)
    valid_indices = torch.nonzero(matrix, as_tuple=False).float()  # Convert to float for calculations

    if valid_indices.numel() == 0:
        # No valid elements in the tensor.
        return 0
    else:
        # Compute the centroid of valid elements
        centroid = valid_indices.mean(dim=0)  # Mean along rows gives (y, x)

        # Calculate distances from the centroid to each valid point
        distances = torch.sqrt((valid_indices[:, 0] - centroid[0]) ** 2 + (valid_indices[:, 1] - centroid[1]) ** 2)

        # Measure the radius: Maximum distance from the centroid
        radius = distances.max().item()
        return radius


def matrix_equality(matrix1, matrix2):
    """
    Calculate the normalized equal item count between two matrices.

    Parameters:
    - matrix1: np.ndarray, first matrix.
    - matrix2: np.ndarray, second matrix.

    Returns:
    - float: Normalized equal item count in the range [0, 1].
    """
    # Ensure input matrices have the same number of columns
    matrix1_flatten = matrix1.sum(dim=1).view(matrix1.size(0), -1)
    matrix2_flatten = matrix2.sum(dim=1).view(matrix2.size(0), -1)
    num_features = matrix2.sum(dim=[1, 2, 3])

    batch_size = 128
    similarity_matrix = torch.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in tqdm(range(0, matrix1.shape[0], batch_size),
                  desc="Calculating Equality"):
        end_i = min(i + batch_size, matrix1.shape[0])
        batch1 = matrix1_flatten[i:end_i].unsqueeze(1).bool()
        batch2 = matrix2_flatten.unsqueeze(0).bool()

        # Sum over the feature dimension to count matches
        equal_counts = (batch1 * batch2).sum(dim=2)  # Shape: (4096, 197)

        # Normalize by the number of features to get similarity in range [0, 1]
        similarity_matrix[i:end_i] = equal_counts / (num_features + 1e-20)

    return similarity_matrix

