import numpy as np
import torch
import cv2
from typing import List, Tuple, Union
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Compose
from torchvision.io import read_image
import time
import kornia.color
import kornia.morphology
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


def load_images_fast(image_paths: List[Union[str, Path]], image_size=None, device=None):
    """
    Fast image loader for small list of image paths.

    Args:
        image_paths: list of image paths (as str or Path)
        image_size: optional (H, W) for resizing
        device: 'cuda' or 'cpu'

    Returns:
        torch.Tensor: [N, 3, H, W] batch
    """
    transform = Resize(image_size) if image_size else None
    images = []

    for path in image_paths:
        img = read_image(str(path))  # [3, H, W], uint8
        if transform:
            img = transform(img)
        if device:
            img = img.to(device, non_blocking=True)
        images.append(img)

    return torch.stack(images)  # [N, 3, H, W]


def rgb_to_binary(image_rgb: np.ndarray, threshold: int = 210) -> np.ndarray:
    """
    Convert RGB image to binary format via thresholding.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary_np = np.where(gray == 211, 0, 255).astype(np.uint8)

    return binary_np


def preprocess_image_to_patch_set(binary_image: np.ndarray, num_patches: int = 6, points_per_patch: int = 16,
                                  contour_uniform=True):
    """
    Extract patch sets from binary image by contour-based object detection.
    Each patch set contains `num_patches` patches of length `points_per_patch` each.

    Returns: List of (patch_set: Tensor [P, L, 2], bounding_box: (x, y, w, h))
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    outputs = []

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < 10:
            continue

        obj_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < points_per_patch:
            continue

        contour = contours[0].squeeze(1)
        if contour.ndim != 2 or contour.shape[1] != 2:
            continue

        # Uniformly sample points along contour
        contour = torch.tensor(contour, dtype=torch.float32)
        contour = contour[torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()]
        patch_set = contour.view(num_patches, points_per_patch, 2)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        patch_set[:, :, 0] += x
        patch_set[:, :, 1] += y
        patch_set[:, :, :2] /= binary_image.shape[0]
        # patch_set_shifted = torch.stack(
        #     (patch_set[0][0][:, :, 0] + patch_set[0][1][0], patch_set[0][0][:, :, 1] + patch_set[0][1][1]), dim=2)
        # patch_set_norm = patch_set_shifted/width
        outputs.append(patch_set)

    return outputs


# def preprocess_rgb_image_to_patch_set(rgb_image: np.ndarray, num_patches: int = 6, points_per_patch: int = 16,
#                                       contour_uniform=True):
#     """
#     Extracts contour patch sets from RGB image using connected components.
#     Returns:
#         outputs: List of patch_set tensors [P, L, 7]
#         positions: List of normalized top-left positions
#         sizes: List of normalized (w, h)
#     """
#     H, W = rgb_image.shape[:2]
#     gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#     binary_mask = np.where(gray == 211, 0, 255).astype(np.uint8)
#
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#
#     outputs = []
#     positions = []
#     sizes = []
#
#     for i in range(1, num_labels):  # skip background
#         x, y, w, h, area = stats[i]
#         if area < 10:
#             continue
#
#         roi = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         if not contours or len(contours[0]) < num_patches * points_per_patch:
#             continue
#
#         contour = contours[0].squeeze(1)  # shape [N, 2]
#         if contour.ndim != 2 or contour.shape[1] != 2:
#             continue
#
#         # Sample points uniformly
#         contour = torch.from_numpy(contour)  # [N, 2]
#         idxs = torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()
#         sampled_xy = contour[idxs] + torch.tensor([x, y])  # absolute coords
#
#         # Fast batch RGB lookup
#         # Clamp each coordinate separately (Y in [0, H-1], X in [0, W-1])
#         clamp_min = torch.tensor([0, 0], dtype=sampled_xy.dtype, device=sampled_xy.device)
#         clamp_max = torch.tensor([H - 1, W - 1], dtype=sampled_xy.dtype, device=sampled_xy.device)
#         sampled_yx = torch.max(torch.min(sampled_xy[:, [1, 0]], clamp_max), clamp_min)
#         rgb_tensor = torch.from_numpy(rgb_image).float() / 255.0  # [H, W, 3]
#         flat_rgb = rgb_tensor[sampled_yx[:, 0], sampled_yx[:, 1]]  # [P*L, 3]
#
#         # Normalize coordinates and add size info
#         sampled_xy = sampled_xy.float()
#         norm_xy = sampled_xy / torch.tensor([W, H])
#         patch_set = torch.cat([norm_xy, flat_rgb], dim=1).view(num_patches, points_per_patch, 5)
#
#         size_tensor = torch.tensor([w / W, h / H], dtype=torch.float32).view(1, 1, 2).expand_as(patch_set[:, :, :2])
#         perceptual_set = torch.cat([patch_set, size_tensor], dim=-1)  # [P, L, 7]
#
#         outputs.append(perceptual_set)
#         positions.append([x / W, y / H])
#         sizes.append([w / W, h / H])
#
#     if len(outputs) == 0:
#         outputs.append(torch.zeros(num_patches, points_per_patch, 7))
#         positions.append([0.0, 0.0])
#         sizes.append([0.0, 0.0])
#
#     return outputs, positions, sizes


def preprocess_rgb_image_to_patch_set_batch(
        image_list: torch.Tensor,
        num_patches: int = 6,
        points_per_patch: int = 16,
        min_area: int = 10,
        bg_val: int = 211
) -> Tuple[List[torch.Tensor], List[List[float]], List[List[float]]]:
    """
    Args:
        image_list: list of [3, H, W] RGB tensors (uint8), can be on GPU
    Returns:
        patch_sets: list of [P, L, 7] tensors (x, y, r, g, b, w, h)
        positions: list of [x/W, y/H]
        sizes: list of [w/W, h/H]
    """
    patch_sets, positions, sizes = [], [], []

    for img in image_list:
        assert img.shape[0] == 3
        device = img.device
        H, W = img.shape[1:]

        # Convert to grayscale
        img_f = img.float() / 255.0
        gray = kornia.color.rgb_to_grayscale(img_f.unsqueeze(0))[0, 0]  # [H, W]

        # Binary mask: everything not equal to bg_val (~0.827)
        mask = (torch.abs(gray - 0.8275) > 1e-3).to(torch.uint8)  # [H, W]

        # Connected components (approximate with PyTorch ops)
        from torchvision.ops import masks_to_boxes
        bin_mask = mask.bool()
        ys, xs = torch.where(bin_mask)
        if len(xs) == 0:
            continue

        x0, y0 = xs.min().item(), ys.min().item()
        x1, y1 = xs.max().item(), ys.max().item()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w * h < min_area:
            continue

        coords = torch.stack([xs, ys], dim=1).float()
        if coords.shape[0] < num_patches * points_per_patch:
            continue
        idxs = torch.linspace(0, len(coords) - 1, steps=num_patches * points_per_patch).long()
        sampled_xy = coords[idxs]

        # RGB lookup (vectorized)
        rgb_f = img_f.permute(1, 2, 0)  # [H, W, 3]
        sampled_rgb = rgb_f[sampled_xy[:, 1].long(), sampled_xy[:, 0].long()]  # [N, 3]

        norm_xy = sampled_xy / torch.tensor([W, H], device=device)
        patch = torch.cat([norm_xy, sampled_rgb], dim=1).view(num_patches, points_per_patch, 5)
        size_tensor = torch.tensor([w / W, h / H], device=device).view(1, 1, 2).expand_as(patch[:, :, :2])
        patch_with_size = torch.cat([patch, size_tensor], dim=-1).to(device)

        patch_sets.append(patch_with_size)
        positions.append([x0 / W, y0 / H])
        sizes.append([w / W, h / H])

    return patch_sets, positions, sizes


def preprocess_rgb_image_to_patch_set_torch(rgb_image: torch.Tensor,
                                            num_patches: int = 6,
                                            points_per_patch: int = 16,
                                            contour_uniform=True):
    """
    Input:
        rgb_image: Tensor [3, H, W], dtype=torch.uint8, CPU (or GPU but converted to CPU internally)

    Returns:
        outputs: list of [P, L, 7] tensors
        positions: list of [x/W, y/H]
        sizes: list of [w/W, h/H]
    """
    t1 = time.time()

    assert rgb_image.ndim == 3 and rgb_image.shape[0] == 3, "Input must be [3, H, W]"
    if rgb_image.device.type != "cpu":
        rgb_image = rgb_image.cpu()
    rgb_np = rgb_image.permute(1, 2, 0).numpy()  # [H, W, 3]

    H, W = rgb_np.shape[:2]

    # Faster grayscale + binary mask creation
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    binary_mask = np.where(gray == 211, 0, 255).astype(np.uint8)

    t2 = time.time()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    t3 = time.time()
    d1 = t2 - t1
    d2 = t3 - t2

    outputs, positions, sizes = [], [], []

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < 10:
            continue

        obj_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < num_patches * points_per_patch:
            continue

        contour = contours[0].squeeze(1)
        if contour.ndim != 2 or contour.shape[1] != 2:
            continue

        contour = torch.from_numpy(contour).long()
        idxs = torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()
        sampled_xy = contour[idxs] + torch.tensor([x, y])

        # Efficient RGB lookup
        sampled_yx = sampled_xy[:, [1, 0]].clamp(  # clamp y and x
            min=torch.tensor([0, 0]),
            max=torch.tensor([H - 1, W - 1])
        )
        rgb_tensor = rgb_image.float().permute(1, 2, 0) / 255.0  # [H, W, 3]
        sampled_rgb = rgb_tensor[sampled_yx[:, 0], sampled_yx[:, 1]]  # [P*L, 3]

        sampled_xy = sampled_xy.float()
        norm_xy = sampled_xy / torch.tensor([W, H])
        patch_set = torch.cat([norm_xy, sampled_rgb], dim=1).view(num_patches, points_per_patch, 5)

        # Add normalized size info
        size_tensor = torch.tensor([w / W, h / H], dtype=torch.float32).view(1, 1, 2).expand_as(patch_set[..., :2])
        perceptual_set = torch.cat([patch_set, size_tensor], dim=-1)  # [P, L, 7]

        outputs.append(perceptual_set)
        positions.append([x / W, y / H])
        sizes.append([w / W, h / H])

    if len(outputs) == 0:
        outputs.append(torch.zeros(num_patches, points_per_patch, 7))
        positions.append([0.0, 0.0])
        sizes.append([0.0, 0.0])

    t4 = time.time()
    d3 = t4 - t3

    return outputs, positions, sizes


def preprocess_rgb_image_to_patch_set(rgb_image: np.ndarray, num_patches: int = 6, points_per_patch: int = 16,
                                      contour_uniform=True):
    """
    Extract patch sets from RGB image by contour-based object detection.
    Each patch set contains `num_patches` patches of length `points_per_patch` each.
    Each contour point has (x, y, R, G, B).

    Returns: List of (patch_set: Tensor [P, L, 5], bounding_box: (x, y, w, h))
    """

    # Convert RGB to grayscale and threshold to get binary mask for contour detection
    t1 = time.time()
    H, W = rgb_image.shape[:2]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary_mask = np.where(gray == 211, 0, 255).astype(np.uint8)
    t2 = time.time()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    t3 = time.time()
    d1 = t2 - t1
    d2 = t3 - t2

    outputs = []
    positions = []
    sizes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < 10:
            continue

        obj_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < points_per_patch:
            continue

        contour = contours[0].squeeze(1)  # shape [N, 2]
        if contour.ndim != 2 or contour.shape[1] != 2:
            continue

        # Sample points uniformly
        contour = torch.tensor(contour, dtype=torch.long)  # use long for indexing
        idxs = torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()
        sampled_xy = contour[idxs]

        # Get RGB values at contour points
        sampled_rgb = []

        for pt in sampled_xy:
            cx, cy = pt[0].item() + x, pt[1].item() + y  # offset to full image
            if 0 <= cy < rgb_image.shape[0] and 0 <= cx < rgb_image.shape[1]:
                color = torch.from_numpy(rgb_image[cy, cx])
            else:
                color = torch.tensor([0, 0, 0])
            sampled_rgb.append(color)
        sampled_rgb = torch.stack(sampled_rgb) / 255  # shape [P*L, 3]
        sampled_xy = sampled_xy.float()
        sampled_xy += torch.tensor([x, y], dtype=torch.float32)  # convert to absolute coords

        patch_set = torch.cat([sampled_xy, sampled_rgb], dim=1).view(num_patches, points_per_patch, 5)
        patch_set[:, :, :2] /= rgb_image.shape[0]

        size_tensor = torch.tensor([w / W, h / H], dtype=torch.float32).view(1, 1, 2).expand_as(patch_set[..., :2])
        perceptual_set = torch.cat([patch_set, size_tensor], dim=-1)  # shape: [P, L, 7]
        outputs.append(perceptual_set)
        positions.append([x / W, y / H])
        sizes.append([w / W, h / H])
    if len(outputs) == 0:
        outputs.append(torch.zeros(num_patches, points_per_patch, 7))
    if len(positions) == 0:
        positions.append([0.0, 0.0])
    if len(sizes) == 0:
        sizes.append([0.0, 0.0])

    t4 = time.time()

    d3 = t4 - t3
    return outputs, positions, sizes


def preprocess_image_to_one_patch_set(binary_image: np.ndarray, num_patches: int = 6, points_per_patch: int = 16):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    # Uniformly sample points along contour
    contour = torch.tensor(contour, dtype=torch.float32)
    contour = contour[torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()]
    patch_set = contour.view(num_patches, points_per_patch, 2)
    return patch_set

    # for i in range(1, num_labels):  # skip background
    #     x, y, w, h, area = stats[i]
    #     if area < 10:
    #         continue
    #     obj_mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    #     contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     if not contours or len(contours[0]) < points_per_patch:
    #         continue
    #
    #     contour = contours[0].squeeze(1)
    #     if contour.ndim != 2 or contour.shape[1] != 2:
    #         continue
    #
    #     # Uniformly sample points along contour
    #     contour = torch.tensor(contour, dtype=torch.float32)
    #     contour = contour[torch.linspace(0, len(contour) - 1, steps=num_patches * points_per_patch).long()]
    #     patch_set = contour.view(num_patches, points_per_patch, 2)
    #     outputs.append((patch_set, (x, y, w, h)))
    # return outputs


def detect_objects_maskrcnn(rgb_image: torch.Tensor, device):
    """
    Use pretrained Mask R-CNN to detect object masks on GPU.

    Args:
        rgb_image: [3, H, W] uint8 tensor, values in [0,255], on GPU

    Returns:
        List of [3, H, W] images with one object per image
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).eval()
    with torch.no_grad():
        img_input = F.convert_image_dtype(rgb_image, dtype=torch.float)  # [0,1]
        output = model([img_input])[0]

    results = []
    for mask, score in zip(output['masks'], output['scores']):
        if score < 0.7:
            continue
        obj_mask = mask[0] > 0.5  # [H, W]
        obj_img = torch.full_like(rgb_image, 211)  # background
        obj_img[:, obj_mask] = rgb_image[:, obj_mask]
        results.append(obj_img)

    return results


def split_image_into_objects_torch(rgb_image: torch.Tensor, bg_color=(211, 211, 211), min_area=10):
    """
    Split a [3, H, W] RGB image (uint8 tensor) into individual object masks using connected component analysis.

    Args:
        rgb_image: Tensor [3, H, W], on GPU
        bg_color: Tuple[int, int, int]
        min_area: int, minimum number of pixels for a valid object

    Returns:
        List[Tensor]: list of [3, H, W] tensors (on GPU), one per object
    """
    C, H, W = rgb_image.shape
    device = rgb_image.device
    bg = torch.tensor(bg_color, dtype=rgb_image.dtype, device=device).view(3, 1, 1)

    # Create a mask of foreground pixels
    fg_mask = torch.any(rgb_image != bg, dim=0)  # [H, W]
    if not fg_mask.any():
        return []

    # Get foreground pixels and their colors
    fg_pixels = rgb_image[:, fg_mask].T  # shape: [N, 3]
    unique_colors = torch.unique(fg_pixels, dim=0)  # [K, 3]

    outputs = []
    for color in unique_colors:
        # Create binary mask for this color
        color_mask = torch.all(rgb_image == color.view(3, 1, 1), dim=0)  # [H, W]
        if color_mask.sum() < min_area:
            continue

        # Move mask to CPU for connected components
        labeled_mask, num_labels = label(color_mask.cpu().numpy())
        for i in range(1, num_labels + 1):
            single_obj_mask = (labeled_mask == i)
            if single_obj_mask.sum() < min_area:
                continue

            single_obj_mask_torch = torch.from_numpy(single_obj_mask).to(device)

            # Create RGB image for this object
            obj_image = torch.full_like(rgb_image, fill_value=bg_color[0], device=device)
            for c in range(3):
                obj_image[c][single_obj_mask_torch] = color[c]

            outputs.append(obj_image)

    return outputs


def split_image_into_objects(rgb_image: torch.tensor):
    """
    Extract individual object regions from an RGB image, each returned as a cropped RGB image.

    Args:
        rgb_image (np.ndarray): Input image in RGB format (H, W, 3).
        min_area (int): Minimum area of object to be extracted (to skip noise).

    Returns:
        List[np.ndarray]: List of cropped RGB images, each containing one object.
    """
    min_area = 10
    bg_color = (211, 211, 211)

    H, W = rgb_image.shape[1:]
    bg_color = torch.tensor(bg_color)

    # Step 1: Create foreground mask
    fg_mask = torch.any(rgb_image != bg_color, dim=-1)

    # Step 2: Extract foreground pixels and get unique object colors
    fg_pixels = rgb_image[fg_mask]
    # if len(fg_pixels) == 0:
    #     return []

    unique_colors = torch.unique(fg_pixels.reshape(-1, 3), dim=0)

    output_images = []
    for color in unique_colors:
        # Step 3: Mask for current color
        color_mask = torch.all(rgb_image == color, dim=-1).to(torch.uint8)

        # Step 4: Connected component analysis to split disconnected shapes
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask.numpy(), connectivity=8)
        labels = torch.from_numpy(labels)
        for label in range(1, num_labels):  # skip background
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            obj_mask = (labels == label)
            obj_rgb = np.full_like(rgb_image.numpy(), fill_value=bg_color)  # start from background
            obj_rgb[obj_mask] = color  # set object color

            output_images.append(obj_rgb)

    return output_images


def load_rgb_image(image_path: Path) -> torch.tensor:
    """
    Load an RGB image from the given path and return it as a NumPy array.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Image in RGB format as a NumPy array.
    """
    image = Image.open(image_path).convert("RGB")
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float()


import numpy as np
from typing import List


def match_objects_to_labels(object_images: List[torch.tensor], gt_objects: List[dict], threshold: float = 0.1) -> List[
    int]:
    """
    Match object images to ground truth objects based on centroid proximity,
    and return the list of shape labels for each object image.

    Args:
        object_images (List[np.ndarray]): List of RGB images, each containing one object.
        gt_objects (List[dict]): List of ground truth dicts with keys 'x', 'y', and 'shape'.
        threshold (float): Tolerance for matching image centroids to GT positions (normalized).

    Returns:
        List[int]: List of predicted shape labels aligned with object_images.
    """
    labels = []
    object_images = [img.permute(1, 2, 0).numpy() for img in object_images]
    for obj_img in object_images:
        # Compute centroid of the object in this image
        mask = (obj_img != 211).any(axis=-1).astype(np.uint8)  # Non-background pixels
        if mask.sum() == 0:
            labels.append(-1)
            continue

        coords = np.argwhere(mask)
        cy, cx = coords.mean(axis=0)  # Note: (row, col) => (y, x)
        norm_x = cx / obj_img.shape[1]
        norm_y = cy / obj_img.shape[0]

        # Match to nearest GT object
        best_label = -1
        min_dist = float('inf')
        for gt in gt_objects:
            try:
                dist = (float(gt['x']) - norm_x) ** 2 + (float(gt['y']) - norm_y) ** 2
            except TypeError:
                raise TypeError
            if dist < min_dist and dist < threshold ** 2:
                min_dist = dist
                best_label = int(gt['shape'])  # Assumes shape is an int
        labels.append(best_label)

    return labels


def match_objects_to_glabels(object_images: List[np.ndarray], gt_objects: List[dict], group_labels,
                             threshold: float = 0.02) -> List[int]:
    obj_orders = []
    for obj_img in object_images:
        # Compute centroid of the object in this image
        mask = (obj_img != 211).any(axis=-1).astype(np.uint8)  # Non-background pixels
        if mask.sum() == 0:
            obj_orders.append(-1)
            continue

        coords = np.argwhere(mask)
        cy, cx = coords.mean(axis=0)  # Note: (row, col) => (y, x)
        norm_x = cx / obj_img.shape[1]
        norm_y = cy / obj_img.shape[0]

        # Match to nearest GT object
        o_id = -1
        min_dist = float('inf')
        for o_i, gt in enumerate(gt_objects):
            dist = (float(gt['x']) - norm_x) ** 2 + (float(gt['y']) - norm_y) ** 2
            if dist < min_dist and dist < threshold ** 2:
                min_dist = dist
                o_id = o_i
        obj_orders.append(o_id)

    new_group_pairs = []
    try:
        for group_pair in group_labels:
            new_group_pairs.append([obj_orders.index(id) for id in group_pair])
    except ValueError:
        raise ValueError
    return new_group_pairs


def img_path2obj_images(img_path: Path, device):
    t1 = time.time()
    image = load_rgb_image(img_path).to(device)
    t2 = time.time()
    obj_images = split_image_into_objects_torch(image)
    t3 = time.time()

    d1 = t2 - t1
    d2 = t3 - t2

    return obj_images


def img_paths2obj_images(img_path: Path, device):
    t1 = time.time()
    image = load_rgb_image(img_path).to(device)
    t2 = time.time()
    obj_images = split_image_into_objects_torch(image)
    t3 = time.time()

    d1 = t2 - t1
    d2 = t3 - t2

    return obj_images


def obj_imgs2patches(obj_images, input_type="pos_color_size"):
    # single object image to patch set
    patch_sets, obj_positions, obj_sizes = rgbs2patches(obj_images, input_type)
    # for o_i, obj_img in enumerate(obj_images):
    #     patch_set, obj_position, obj_size = rgb2patch(obj_img, input_type)
    #     patch_sets.append(patch_set)
    #     positions.append(obj_position)
    #     sizes.append(obj_size)
    return patch_sets, obj_positions, obj_sizes


def img_path2patches_and_labels(obj_images, gt_dict, device, input_type="pos_color_size"):
    objects, obj_images, permutes = align_data_and_imgs(gt_dict, obj_images)
    labels = match_objects_to_labels(obj_images, objects)
    # single object image to patch set
    # patch_sets = []
    # sorted_labels = []
    # positions = []
    # sizes = []

    patch_sets, obj_positions, obj_sizes = rgbs2patches(obj_images, input_type)
    if len(labels)!=len(patch_sets):
        print(f"len(labels)={len(labels)}, len(patch_sets)={len(patch_sets)}")
        return None, None, None, None, None
    sorted_labels = [labels[i] - 1 for i in range(len(labels)) if labels[i] != -1]
    patch_sets = [patch_sets[i] for i in range(len(labels)) if labels[i] != -1]
    positions = [obj_positions[i] for i in range(len(labels)) if labels[i] != -1]
    sizes = [obj_sizes[i] for i in range(len(labels)) if labels[i] != -1]
    # for o_i, obj_img in enumerate(obj_images):
    #     patch_set, obj_position, obj_size = rgb2patch(obj_img, input_type)
    #     if labels[o_i] == -1:
    #         continue
    #     label = labels[o_i] - 1
    #
    #     patch_sets.append(patch_set)
    #     sorted_labels.append(label)
    #     positions.append(obj_position)
    #     sizes.append(obj_size)
    return patch_sets, sorted_labels, positions, sizes, permutes


# def img_path2patches_and_glabels(image_path, gt_dict, g_labels):
#     obj_images = img_path2obj_images(image_path)
#
#     g_labels_sorted = match_objects_to_glabels(obj_images, gt_dict, g_labels)
#     # single object image to patch set
#     patch_sets = []
#     for o_i, obj_img in enumerate(obj_images):
#         binary_np = rgb_to_binary(obj_img)
#         patch_set = preprocess_image_to_patch_set(binary_np)
#         patch_sets.append(patch_set)
#
#     return patch_sets, g_labels_sorted


def img_path2one_patches(image_path):
    image = load_rgb_image(image_path)
    binary_np = rgb_to_binary(image)
    patch_set = preprocess_image_to_one_patch_set(binary_np)
    return patch_set


def shift_obj_patches_to_global_positions(obj_patch, shift):
    (x_min, y_min, w, h) = shift
    x_shift = obj_patch[:, :, 0] + w
    y_shift = obj_patch[:, :, 1] + h

    shifted_tensor = torch.stack([x_shift, y_shift], dim=2)
    return shifted_tensor



def align_gt_data_and_pred_data(
    objects: List[dict], pred_objects: List[dict]
) -> Tuple[List[dict], List[dict], List[int]]:
    """
    Align GT metadata with predicted objects based on centroid (x, y) distance using Hungarian matching.

    Args:
        objects: list of GT objects (each with 'x', 'y' keys)
        pred_objects: list of predicted objects (each with 'x', 'y' keys)

    Returns:
        aligned_gt: list of GT objects matched
        aligned_pred: list of predicted objects matched to GTs
        gt_to_pred_indices: indices such that pred_objects[i] matches objects[gt_to_pred_indices[i]]
    """
    if len(objects) == 0 or len(pred_objects) == 0:
        return [], [], []

    gt_coords = np.array([[obj["x"], obj["y"]] for obj in objects])
    pred_coords = np.array([[pred["s"]["x"], pred["s"]["y"]] for pred in pred_objects])

    # Compute pairwise distance matrix
    dists = np.linalg.norm(gt_coords[:, None, :] - pred_coords[None, :, :], axis=-1)

    # Hungarian algorithm for minimum distance matching
    gt_indices, pred_indices = linear_sum_assignment(dists)

    aligned_gt = [objects[i] for i in gt_indices]
    aligned_pred = [pred_objects[j] for j in pred_indices]
    gt_to_pred_indices = pred_indices.tolist()

    return aligned_gt, aligned_pred, gt_to_pred_indices


def align_data_and_imgs(objects: List[dict], obj_imgs: List[torch.tensor]) -> Tuple[
    List[dict], List[torch.tensor], List[int]]:
    """
    Align object metadata with object images based on centroid location matching.
    Each image contains a single colored object on a [211, 211, 211] gray background.

    :param objects: List of symbolic object dictionaries with 'x' and 'y' in [0, 1] range
    :param obj_imgs: List of 1024x1024x3 numpy arrays, one per object
    :return: Tuple of (aligned_objects, aligned_obj_imgs, old_to_new_indices)
             where old_to_new_indices[i] = index in original objects list for the i-th image
    """
    resolution = 1024
    bg_color = np.array([211, 211, 211], dtype=np.uint8)
    obj_imgs = [img.permute(1, 2, 0).to("cpu").numpy() for img in obj_imgs]

    def get_centroid(img: np.ndarray) -> Tuple[float, float]:
        mask = np.any(img != bg_color, axis=-1)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return -1, -1  # invalid
        cx = xs.mean() / resolution
        cy = ys.mean() / resolution
        return cx, cy

    # Compute centroids
    img_centroids = [get_centroid(img) for img in obj_imgs]
    obj_centroids = [torch.tensor([obj['x'], obj['y']]) for obj in objects]

    aligned_objects = []
    aligned_imgs = []
    old_to_new_indices = []

    used_obj_idxs = set()

    for i, img_c in enumerate(img_centroids):
        dists = [
            np.linalg.norm(np.array(img_c) - obj_c.numpy()) if j not in used_obj_idxs else float('inf')
            for j, obj_c in enumerate(obj_centroids)
        ]
        match_idx = int(np.argmin(dists))
        aligned_objects.append(objects[match_idx])
        aligned_imgs.append(obj_imgs[i])
        old_to_new_indices.append(match_idx)
        used_obj_idxs.add(match_idx)
    aligned_imgs = [torch.from_numpy(img).permute(2, 0, 1) for img in aligned_imgs]
    return aligned_objects, aligned_imgs, old_to_new_indices


def rgb2patch(rgb_img, input_type):
    patch_set, positions, sizes = preprocess_rgb_image_to_patch_set_torch(rgb_img)
    positions = positions[0]
    sizes = sizes[0]
    if input_type == "pos":
        patch_set = patch_set[0][:, :, :2]
    elif input_type == "pos_color":
        patch_set = patch_set[0][:, :, :5]
    elif input_type == "pos_color_size":
        patch_set = patch_set[0]
    else:
        raise ValueError

    return patch_set, positions, sizes


def rgbs2patches(rgb_imgs, input_type):
    patch_sets, positions, sizes = preprocess_rgb_image_to_patch_set_batch(rgb_imgs)
    if input_type == "pos":
        patch_sets = [p[:, :, :2] for p in patch_sets]
    elif input_type == "pos_color":
        patch_sets = [p[:, :, :5] for p in patch_sets]
    elif input_type == "pos_color_size":
        patch_sets = patch_sets
    else:
        raise ValueError
    return patch_sets, positions, sizes
