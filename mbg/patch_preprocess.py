import numpy as np
import torch
import cv2
from typing import List, Tuple
from pathlib import Path
from PIL import Image


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


def preprocess_rgb_image_to_patch_set(rgb_image: np.ndarray, num_patches: int = 6, points_per_patch: int = 16,
                                      contour_uniform=True):
    """
    Extract patch sets from RGB image by contour-based object detection.
    Each patch set contains `num_patches` patches of length `points_per_patch` each.
    Each contour point has (x, y, R, G, B).

    Returns: List of (patch_set: Tensor [P, L, 5], bounding_box: (x, y, w, h))
    """

    # Convert RGB to grayscale and threshold to get binary mask for contour detection
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary_mask = np.where(gray == 211, 0, 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
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

        H, W = rgb_image.shape[:2]

        patch_set = torch.cat([sampled_xy, sampled_rgb], dim=1).view(num_patches, points_per_patch, 5)
        patch_set[:, :, :2] /= rgb_image.shape[0]
        W, H = rgb_image.shape[:2]
        size_tensor = torch.tensor([w / W, h / H], dtype=torch.float32).view(1, 1, 2).expand_as(patch_set[..., :2])
        perceptual_set = torch.cat([patch_set, size_tensor], dim=-1)  # shape: [P, L, 7]
        outputs.append(perceptual_set)
        positions.append([x / W, y / H])
        sizes.append([w / W, h / H])
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


def split_image_into_objects(rgb_image: np.ndarray) -> List[np.ndarray]:
    """
    Given an RGB image, returns a list of RGB images where each image contains only one object.
    Objects are identified as connected components of the same color.
    """
    # Convert to grayscale and then to binary mask
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binary = np.where(gray == 211, 0, 255).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    output_images = []
    for label in range(1, num_labels):  # skip background
        # Create a mask for the current component
        mask = (labels == label).astype(np.uint8) * 255
        mask_3ch = cv2.merge([mask] * 3)

        # Mask the original image
        isolated = cv2.bitwise_and(rgb_image, mask_3ch)

        # Set the background to light gray (211,211,211)
        background = np.full_like(rgb_image, 211, dtype=np.uint8)
        final_image = np.where(mask_3ch == 0, background, isolated)

        output_images.append(final_image)

    return output_images


def load_rgb_image(image_path: Path) -> np.ndarray:
    """
    Load an RGB image from the given path and return it as a NumPy array.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Image in RGB format as a NumPy array.
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


import numpy as np
from typing import List


def match_objects_to_labels(object_images: List[np.ndarray], gt_objects: List[dict], threshold: float = 0.1) -> List[
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


def img_path2obj_images(img_path: Path) -> List[np.ndarray]:
    image = load_rgb_image(img_path)
    obj_images = split_image_into_objects(image)
    return obj_images


def img_path2patches_and_labels(image_path, gt_dict, input_type="pos_color_size"):
    obj_images = img_path2obj_images(image_path)
    objects, obj_images, permutes = align_data_and_imgs(gt_dict, obj_images)

    labels = match_objects_to_labels(obj_images, objects)
    # single object image to patch set
    patch_sets = []
    sorted_labels = []
    positions = []
    sizes = []
    for o_i, obj_img in enumerate(obj_images):
        patch_set, obj_position, obj_size = rgb2patch(obj_img, input_type)
        if labels[o_i] == -1:
            continue
        label = labels[o_i] - 1

        patch_sets.append(patch_set)
        sorted_labels.append(label)
        positions.append(obj_position)
        sizes.append(obj_size)
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


def align_data_and_imgs(objects: List[dict], obj_imgs: List[np.ndarray]) -> Tuple[List[dict], List[np.ndarray], List[int]]:
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

    return aligned_objects, aligned_imgs, old_to_new_indices


def rgb2patch(rgb_img, input_type):
    if input_type == "pos":
        patch_set, positions, sizes = preprocess_rgb_image_to_patch_set(rgb_img)
        patch_set = patch_set[0][:,:,:2]
        positions = positions[0]
        sizes=sizes[0]
    elif input_type == "pos_color":
        patch_set, positions, sizes = preprocess_rgb_image_to_patch_set(rgb_img)
        patch_set = patch_set[0][:,:,:5]
        positions = positions[0]
        sizes=sizes[0]
    elif input_type == "pos_color_size":
        patch_set, positions, sizes = preprocess_rgb_image_to_patch_set(rgb_img)
        patch_set = patch_set[0]
        positions = positions[0]
        sizes=sizes[0]
    else:
        raise ValueError

    return patch_set, positions, sizes
