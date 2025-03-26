# Created by X at 21/06/2024

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import config
import cv2
from src import bk
from src.utils.chart_utils import van
from src.utils import chart_utils


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
            data = np.concatenate(
                (np.zeros_like(patch[:, col:col + 1]) + 10, patch[:, col:col + 2]),
                axis=1)
        elif col == cols - 1:
            data = np.concatenate(
                (patch[:, col - 1:], np.zeros_like(patch[:, col:col + 1]) + 10),
                axis=1)
        else:
            data = patch[:, col - 1:col + 2]
        data = data.T
        col_patches.append(data)

    for row in range(rows):
        if row == 0:
            data = np.concatenate(
                (np.zeros_like(patch[row:row + 1, :]) + 10, patch[row:row + 2, :]),
                axis=0)
        elif row == rows - 1:
            data = np.concatenate(
                (patch[row - 1:, :], np.zeros_like(patch[row:(row + 1), :]) + 10),
                axis=0)
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
    patches = F.unfold(matrix_64x64.unsqueeze(0).unsqueeze(0),
                       kernel_size=(kernel_size, kernel_size),
                       stride=kernel_size)

    # Reshape the patches to (number_of_patches, 3, 3)
    patches = patches.transpose(1, 2).reshape(-1, kernel_size, kernel_size)

    # Calculate positions of patches
    num_patches_x = matrix_64x64.size(0) // kernel_size
    num_patches_y = matrix_64x64.size(1) // kernel_size
    positions = torch.stack(torch.meshgrid(torch.arange(num_patches_x),
                                           torch.arange(num_patches_y)),
                            dim=-1).reshape(-1, 2)
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
    matrix = torch.tensor([int(b) for b in binary_str], dtype=torch.float32).view(64,
                                                                                  64)

    return matrix


def shift_up(matrix):
    shift_count = 0
    while matrix.sum() > 0 and torch.all(
            matrix[:, 0] == 0):  # Check if the top row is full of zeros
        matrix = torch.roll(matrix, shifts=-1, dims=1)  # Shift all rows up
        matrix[:, -1] = 0  # Fill the last row with zeros after the shift
        shift_count += 1
    return matrix, shift_count


# Function to shift the matrix left if the leftmost column is all zeros
def shift_left(matrix):
    shift_count = 0
    while matrix.sum() > 0 and torch.all(
            matrix[:, :, 0] == 0):  # Check if the leftmost column is full of zeros
        matrix = torch.roll(matrix, shifts=-1,
                            dims=2)  # Shift all columns to the left
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
            matrix = torch.roll(matrix, shifts=int(-given_rs[i].item()),
                                dims=1)  # Shift all rows up
        else:
            matrix, shift_row = shift_up(matrix)  # Apply upward shift
            rs[i] = shift_row

        if given_cs is not None:
            matrix = torch.roll(matrix, shifts=int(-given_cs[i].item()),
                                dims=2)  # Shift all rows up
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


def to_bw_img(image):
    # Load an image
    image[image > 0] = 1
    return image


def resize_img(img, resize):
    # rgb = rgb_np.numpy().astype(np.uint8)
    # bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # bw_img[bw_img != 211] = 1
    # bw_img[bw_img == 211] = 0
    # if crop:
    #     # bw image to cropped bw image
    #     bw_img, _ = crop_img(torch.from_numpy(bw_img).squeeze(), resize=resize)
    # else:
    #     if resize:

    resized_image = cv2.resize(img.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST)

    # else:
    #     bw_img = torch.from_numpy(bw_img).unsqueeze(0)
    return resized_image


def find_valid_radius(matrix):
    # Find indices of valid elements (nonzero items)
    valid_indices = torch.nonzero(matrix,
                                  as_tuple=False).float()  # Convert to float for calculations

    if valid_indices.numel() == 0:
        # No valid elements in the tensor.
        return 0
    else:
        # Compute the centroid of valid elements
        centroid = valid_indices.mean(dim=0)  # Mean along rows gives (y, x)

        # Calculate distances from the centroid to each valid point
        distances = torch.sqrt((valid_indices[:, 0] - centroid[0]) ** 2 + (
                valid_indices[:, 1] - centroid[1]) ** 2)

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
    matrix1_flatten = matrix1.view(matrix1.size(0), -1)
    matrix2_flatten = matrix2.view(matrix2.size(0), -1)
    # num_features = matrix2.sum(dim=[1, 2, 3])

    batch_size = 16
    similarity_matrix = torch.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in range(0, matrix1.shape[0], batch_size):
        end_i = min(i + batch_size, matrix1.shape[0])
        batch1 = matrix1_flatten[i:end_i].unsqueeze(1)
        batch2 = matrix2_flatten.unsqueeze(0)
        mask = (batch2 == 0).squeeze()
        # Sum over the feature dimension to count matches
        equal_counts = (batch1 == batch2)  # Shape: (4096, 197)
        equal_counts[:, :, mask] = 0
        # Normalize by the number of features to get similarity in range [0, 1]
        similarity_matrix[i:end_i] = equal_counts.sum(dim=-1) / equal_counts.shape[
            -1]

    return similarity_matrix


def crop_img(img, crop_data=None):
    rgb = img.numpy().astype(np.uint8)
    bg_mask = np.all(rgb == bk.color_matplotlib["lightgray"], axis=-1)
    rgb[bg_mask] = [0, 0, 0]
    bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bw_img = torch.from_numpy(bw_img).squeeze()
    if crop_data is None:
        height, width = bw_img.shape[-2], bw_img.shape[-1]
        # Find the bounding box of the nonzero values
        nonzero_coords = torch.nonzero(bw_img)
        if nonzero_coords.numel() == 0:  # Handle completely empty images
            return bw_img, [0, 0, 0, 0]

        min_y, min_x = nonzero_coords.min(dim=0).values
        max_y, max_x = nonzero_coords.max(dim=0).values

        # Compute the side length of the square
        side_length = max(max_y - min_y + 1, max_x - min_x + 1)

        # Adjust the bounding box to make it square
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        half_side = side_length // 2 + 5

        # Compute the new square bounding box
        new_min_y = max(center_y - half_side, 0)
        new_max_y = min(center_y + half_side + 1, height)
        new_min_x = max(center_x - half_side, 0)
        new_max_x = min(center_x + half_side + 1, width)
    else:
        new_min_y, new_max_y, new_min_x, new_max_x = crop_data
    # Crop the image
    cropped_image = bw_img[new_min_y:new_max_y, new_min_x:new_max_x]

    # if resize is not None:
    #     cropped_image = cv2.resize(cropped_image.numpy(), (resize, resize),
    #                                interpolation=cv2.INTER_AREA)
    #     cropped_image = torch.from_numpy(cropped_image)
    cropped_image = cropped_image.unsqueeze(0)
    return cropped_image, [new_min_y, new_max_y, new_min_x, new_max_x]


def merge_segments(segments):
    merged_img = segments[0].clone()
    for segment in segments:
        mask = (segment != torch.tensor(bk.color_matplotlib["lightgray"])).any(dim=-1)
        merged_img[mask] = segment[mask]
    return merged_img


def load_json(file):
    import json
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def get_contours(input_tensor):
    """
    Find contours of isolated areas in a binary image, draw them, and crop the image
    to include only the bounding box containing the contours.

    Args:
        input_tensor (torch.Tensor): A 2D binary tensor of shape (H, W) with values 0 or 1.

    Returns:
        torch.Tensor: Cropped tensor containing only the area with contours.
    """
    # Ensure the image is binary
    binary_image = cv2.threshold(input_tensor, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours with all points along the edges
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract the first (and only) contour
    contour = contours[0]

    # Convert the contour points to a list of (x, y) tuples
    contour_points = [tuple(point[0]) for point in contour]

    # Sort contour points to align with the original shape
    # Sort by y first, then by x for each row
    contour_points_sorted = sorted(contour_points, key=lambda p: (p[1], p[0]))

    return contour
    #
    # # Convert PyTorch tensor to numpy array
    # input_array = input_tensor.astype(np.uint8)
    #
    # # Find contours with all points along the edges
    # contours, _ = cv2.findContours(input_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # # Extract the first (and only) contour
    # contour = contours[0]
    #
    # # Convert the contour points to a list of (x, y) tuples
    # contour_points = torch.tensor([tuple(point[0]) for point in contour])
    # return contours

    # # Create an empty array to draw contours
    # contour_array = np.zeros_like(input_array)
    #
    # # Draw contours on the empty array
    # cv2.drawContours(contour_array, contours, -1, color=1, thickness=1)
    #
    # # Find bounding box for all contours
    # x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Combine all contours into one set
    #
    # # Crop the contour area
    # cropped_array = contour_array[y:y + h, x:x + w]
    # pad_size = 32
    # padded_image = np.pad(cropped_array, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant',
    #                       constant_values=0)
    # resized_array = resize_img(padded_image, 128)
    # return resized_array, contours[0]


def crop_to_valid_area(image, threshold=10):
    """
    Crop the image to the smallest bounding box that contains valid information.

    Args:
        image (np.ndarray): Grayscale test image as a NumPy array.
        threshold (int): Pixel intensity threshold to identify non-empty regions.

    Returns:
        np.ndarray: Cropped image.
        tuple: (x_min, y_min, x_max, y_max) coordinates of the cropped region.
    """
    # Find non-zero areas
    valid_rows = np.any(image > threshold, axis=1)
    valid_cols = np.any(image > threshold, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        # No valid area
        return image, (0, 0, image.shape[1], image.shape[0])

    y_min, y_max = np.where(valid_rows)[0][[0, -1]]
    x_min, x_max = np.where(valid_cols)[0][[0, -1]]

    # Include bordering cases by padding
    padding = 0  # Add padding to account for partial patches
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, (x_min, y_min, x_max, y_max)


def direction_vectors_to_angles(direction_vectors):
    """
    Convert each direction vector to its corresponding angle in degrees.

    Args:
        direction_vectors (np.ndarray): An Nx2 array where each row is a direction vector [dx, dy].

    Returns:
        np.ndarray: A 1D array of angles in degrees corresponding to each direction vector.
    """
    # Extract dx and dy components
    dx = direction_vectors[:, 0]
    dy = direction_vectors[:, 1]

    # Compute angles in radians using arctan2
    angles_radians = np.arctan2(dy, dx)

    # Convert radians to degrees
    angles_degrees = np.degrees(angles_radians)

    # Ensure all angles are in the range [0, 360)
    angles_degrees = np.mod(angles_degrees, 360)

    return angles_degrees


def smooth_directions_degrees(directions, window_size=5, sharpness_threshold=30):
    """
    Smooth the directions (in degrees) while preserving sharp angle changes.

    Args:
        directions (array-like): Input directions (in degrees).
        window_size (int): Size of the smoothing window (must be odd).
        sharpness_threshold (float): Threshold for detecting sharp angle changes (in degrees).

    Returns:
        np.ndarray: Smoothed directions (in degrees).
    """
    directions = np.asarray(directions)
    directions = np.deg2rad(directions)  # Convert degrees to radians for processing
    directions = np.unwrap(directions)  # Handle angular wrap-around

    smoothed = np.zeros_like(directions)
    half_window = window_size // 2

    for i in range(len(directions)):
        # Define the window bounds
        start = max(0, i - half_window)
        end = min(len(directions), i + half_window + 1)

        # Extract the window
        window = directions[start:end]

        # Calculate angular differences in the window
        angular_diffs = np.abs(np.diff(window))

        # Preserve sharp changes by filtering based on the sharpness threshold (converted to radians)
        if np.any(angular_diffs > np.deg2rad(sharpness_threshold)):
            smoothed[i] = directions[i]  # Keep the original value
        else:
            smoothed[i] = np.mean(window)  # Apply smoothing

    smoothed = np.rad2deg(smoothed)  # Convert back to degrees
    return np.mod(smoothed, 360)  # Wrap angles back to [0, 360]


def contour_to_direction_vector(contour):
    """
    Convert a contour into a direction vector.

    Args:
        contour (np.ndarray): Contour represented as an array of points of shape (N, 1, 2).

    Returns:
        list: A list of tuples representing direction vectors (dx, dy) for each point in the contour.
    """
    # Extract the points from the contour
    points = contour  # Shape becomes (N, 2)

    # Compute the direction vector
    direction_vector = []
    num_points = len(points)
    for i in range(num_points):
        # Compute the difference between consecutive points
        dx = points[(i + 1) % num_points][0] - points[i][0]  # Next point wraps around
        dy = points[(i + 1) % num_points][1] - points[i][1]
        direction_vector.append((dx, dy))
    angles = direction_vectors_to_angles(np.array(direction_vector))
    smoothed_angles = smooth_directions_degrees(angles, window_size=10, sharpness_threshold=80)
    # chart_utils.show_line_chart(smoothed_angles, file_name=config.output/ f"dv_angles_{10}_{80}.pdf")

    # # Compute the direction vector
    # dv_2 = []
    # num_points = len(points)
    # for i in range(num_points):
    #     # Compute the difference between consecutive points
    #     dx = points[(i + 20) % num_points][0] - points[i][0]  # Next point wraps around
    #     dy = points[(i + 20) % num_points][1] - points[i][1]
    #     dv_2.append((dx, dy))
    # angles_2 = direction_vectors_to_angles(np.array(dv_2))
    # chart_utils.show_line_chart(angles_2)
    return smoothed_angles


def shift_by_largest_gap_tensor(list_a, list_b):
    """
    Given two PyTorch tensors, list_a (1D tensor) and list_b (2D tensor of points),
    this function:
      1. Computes the differences between consecutive elements in list_a.
      2. Finds the largest gap between consecutive elements.
      3. Circularly shifts both list_a and list_b so that the element following
         the largest gap becomes the first element. This ensures that the largest
         gap appears between the last and first elements in the shifted lists.

    Args:
        list_a (torch.Tensor): A 1D tensor of numeric values.
        list_b (torch.Tensor): A 2D tensor of shape (N, D) where each row represents a point.

    Returns:
        tuple: A tuple (shifted_list_a, shifted_list_b) with the circularly shifted tensors.
    """
    # Check that list_a is 1D and list_b is 2D and they have the same number of elements
    if list_a.dim() != 1:
        raise ValueError("list_a must be a 1D tensor.")
    if list_b.dim() != 2:
        raise ValueError("list_b must be a 2D tensor.")
    if list_a.size(0) != list_b.size(0):
        raise ValueError("Both list_a and list_b must have the same number of elements.")

    n = list_a.size(0)
    if n < 2:
        # No gap to calculate if there's only one element.
        return list_a, list_b

    # Compute differences between consecutive elements in list_a
    differences = torch.abs(list_a[1:] - list_a[:-1]) % 360
    compliment_diff = torch.abs(360 - differences) % 360
    differences = torch.stack([differences, compliment_diff]).min(dim=0)[0]
    # Find the index where the difference is maximum
    max_diff_index = torch.argmax(differences).item()  # Convert tensor to Python int

    # Determine the new starting point: the element immediately after the maximum gap becomes the first element
    shift_point = max_diff_index + 1

    # Perform the circular shift on both tensors
    shifted_list_a = torch.cat((list_a[shift_point:], list_a[:shift_point]), dim=0)
    shifted_list_b = torch.cat((list_b[shift_point:], list_b[:shift_point]), dim=0)

    return shifted_list_a, shifted_list_b
