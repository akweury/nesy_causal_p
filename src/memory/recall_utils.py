# Created by x at 10.12.24
import torch
import torch.nn.functional as F  # Import F for functional operations
from tqdm import tqdm
import cv2
import numpy as np
from src.utils import data_utils


def get_shifted_matrics(img):
    # Create grid of all (x_shift, y_shift) pairs
    shifts = [(-x, -y) for x in range(img.shape[-2]) for y in range(img.shape[-1])]
    # Generate the shifted images as a batch
    shifted_images = torch.cat(
        [torch.roll(img, shifts=(row, col), dims=(-2, -1)) for row, col in shifts])
    return shifted_images


def detect_edge(matrices):
    """
    Detect edges in a batch of binary matrices.

    Args:
        matrices (torch.Tensor): A batch of binary matrices of shape (N, 1, H, W), where N is the batch size.

    Returns:
        torch.Tensor: A batch of edge-detected matrices of shape (N, 1, H, W), where edges are marked as 1 and others as 0.
    """

    # Define the edge-detection kernel
    edge_kernel = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)

    # Expand the kernel to apply it separately to each channel
    edge_kernel_repeated = edge_kernel.repeat(matrices.size(1), 1, 1, 1)

    # Apply convolution across the batch
    edges = F.conv2d(matrices, edge_kernel_repeated, groups=matrices.size(1),
                     padding=1)

    # Convert to binary (edge pixels as 1, others as 0)
    edges_binary = (edges > 0).float()
    edges_binary[:, :, 0] = 0
    edges_binary[:, :, -2:] = 0
    edges_binary[:, :, :, :2] = 0
    edges_binary[:, :, :, -2:] = 0
    # Remove the channel dimension to return a batch of (N, H, W)
    return edges_binary


def find_best_shift(scores, size_2d):
    best_values, value_indices = torch.sort(scores.flatten(), descending=True)
    best_values = best_values[:5]
    value_indices = value_indices[:5]
    if len(best_values) == 0:
        return None

    indices_2d = torch.unravel_index(value_indices, scores.shape)
    indices_2d = torch.stack(indices_2d).t()

    best_shift_indices = torch.unravel_index(indices_2d[:, 0], size_2d)
    best_shift_indices = torch.stack(best_shift_indices).t()
    # best fm idx
    best_fm_indices = indices_2d[:, 1]

    return best_fm_indices, best_shift_indices, best_values


def get_contours(input_tensor):
    """
    Given a binary 0-1 PyTorch tensor image, find contours of isolated areas
    and return an image with the contour pixels set to 1, and the rest set to 0.

    Args:
        input_tensor (torch.Tensor): A 2D binary tensor of shape (H, W) with values 0 or 1.

    Returns:
        torch.Tensor: A 2D tensor of the same shape as input, where contour pixels are 1 and rest are 0.
    """
    # Convert PyTorch tensor to numpy array
    input_array = input_tensor.astype(np.uint8)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(input_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty array to draw contours
    contour_array = np.zeros_like(input_array)

    # Draw contours on the empty array
    cv2.drawContours(contour_array, contours, -1, color=1, thickness=1)

    # Convert the numpy array back to a PyTorch tensor

    return contour_array


def resize_img(cropped_image, resize_to):
    resized_image = cv2.resize(cropped_image.astype(np.uint8), (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
    return resized_image


def compute_similarity(v1, v2):
    """
    Compute similarity between two vectors using cosine similarity.

    Args:
        v1 (tuple): A direction vector (dx, dy).
        v2 (tuple): A direction vector (dx, dy).

    Returns:
        float: Cosine similarity value.
    """
    # Convert tuples to numpy arrays
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)

    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / (magnitude + 1e-8)  # Avoid division by zero


def extract_segments(direction_vector, segment_length=8):
    """
    Extract overlapping segments from a direction vector.

    Args:
        direction_vector (list): The direction vector as a list of (dx, dy) tuples.
        segment_length (int): The length of each segment.

    Returns:
        list: List of overlapping segments, each containing `segment_length` values.
    """
    num_points = len(direction_vector)
    # Extend the vector to handle wrap-around
    extended_vector = torch.cat((direction_vector, direction_vector[:segment_length - 1]), dim=0)

    # Extract overlapping segments
    segments = [
        extended_vector[i:i + segment_length]
        for i in range(num_points)
    ]
    return segments


def compute_similarity(v1, v2):
    """
    Compute similarity between two segments using cosine similarity.

    Args:
        v1 (list): A segment represented as a list of (dx, dy) tuples.
        v2 (list): Another segment represented as a list of (dx, dy) tuples.

    Returns:
        float: Cosine similarity value for the two segments.
    """
    # Flatten segments into single vectors
    v1_flat = np.array(v1, dtype=np.float32).flatten()
    v2_flat = np.array(v2, dtype=np.float32).flatten()

    # Compute cosine similarity
    dot_product = np.dot(v1_flat, v2_flat)
    magnitude = np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat)
    return dot_product / (magnitude + 1e-8)  # Avoid division by zero


def compare_direction_vectors(known_shapes, new_vector, segment_length=3):
    """
    Compare a new contour's direction vector with known shape direction vectors.

    Args:
        known_vectors (dict): A dictionary with shape names as keys and direction vectors as values.
        new_vector (list): The direction vector of the new contour.

    Returns:
        str: The name of the shape that is most similar to the new contour.
    """
    # Extract segments for the new vector
    new_segments = extract_segments(new_vector, segment_length)

    # Track similarity scores


    # Compare each new segment with every segment of known shapes
    preds =[]
    for new_segment in new_segments:
        seg_preds = []
        for shape_name, shape_vector in known_shapes.items():
            shape_segments = extract_segments(shape_vector, segment_length)
            # Find the best match for the current segment
            best_similarity = max(
                compute_similarity(new_segment, shape_segment)
                for shape_segment in shape_segments
            )
            seg_preds.append(best_similarity)  # Accumulate similarity score
        preds.append(np.argmax(seg_preds))

    return preds
