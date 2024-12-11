# Created by jing at 10.12.24
import torch
import torch.nn.functional as F  # Import F for functional operations
from tqdm import tqdm



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

