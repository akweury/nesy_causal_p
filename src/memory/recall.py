# Created by jing at 10.12.24
import numpy as np
from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models
from src.reasoning import reason
from src.utils.chart_utils import van
from PIL import Image, ImageDraw
import itertools
import cv2


def recall_fms(input_fms, bk_fms, reshape=None):
    # scores = []
    # recall_fm = []
    # convolutional layer
    # input_fms = models.img2fm(img, bk_shapes["kernels"].float())
    # bk_fms = bk_shapes["fm_repo"]
    # fm similarity

    # Flatten the tensors along the spatial dimensions
    first_tensor_flat = input_fms.view(1, -1)  # Shape: 1x(Channel*16*16)
    second_tensor_flat = bk_fms.view(bk_fms.shape[0], -1)  # Shape: Nx(Channel*16*16)

    # Compute cosine similarity between first_tensor and all tensors in second_tensor
    cosine_similarities = F.cosine_similarity(first_tensor_flat, second_tensor_flat)
    most_similar_score = cosine_similarities.max().item()

    # cosine_similarities.mean()
    # Find the index of the most similar tensor
    most_similar_index = torch.argmax(cosine_similarities).item()
    # chart_utils.show_line_chart(sorted(cosine_similarities))

    # Get the most similar tensor

    most_similar_fm = bk_fms[most_similar_index]

    # sim_total = data_utils.matrix_equality(bk_fms, input_fms.unsqueeze(0))
    # scores = sim_total.max()
    # recall_fm = bk_fms[sim_total.argmax()]
    # best_shape = np.argmax(scores)
    # best_recall_fm = recall_fm[best_shape]
    return most_similar_fm, most_similar_score


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


from scipy.spatial.distance import cdist


def find_similar_patches(test_image, dataset_patches, vertex_labels, shape_labels, patch_size=32, threshold=1000):
    """
    Find patches in the test image that are similar to those in the dataset.

    Args:
        test_image (np.ndarray): Test image as a NumPy array (grayscale).
        dataset_patches (list): List of patches from the dataset as NumPy arrays.
        vertex_labels (list): List of labels corresponding to the dataset patches.
        patch_size (int): Size of the patches to extract (default is 32x32).
        threshold (float): Threshold for similarity measure (lower is more similar).

    Returns:
        list: List of matched patches' labels and their positions in the test image.
    """
    matches = []
    half_size = patch_size // 2

    # Crop the image to the valid area
    cropped_image, (x_offset, y_offset, _, _) = crop_to_valid_area(test_image, 0)
    if cropped_image.shape[0] > 32:
        resized_img = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_NEAREST)

    # Pad the test image to handle boundary cases
    padded_image = np.pad(resized_img, ((half_size, half_size), (half_size, half_size)), mode='constant',
                          constant_values=0)
    # Pre-compute flattened dataset patches for faster distance computation
    dataset_flat = np.array([p.flatten() for p in dataset_patches])

    dists = []
    shapes = []
    # Iterate through every possible patch in the test image
    for y in range(half_size, resized_img.shape[0] + half_size):
        for x in range(half_size, resized_img.shape[1] + half_size):
            # Extract the patch from the test image
            patch = padded_image[y - half_size:y + half_size, x - half_size:x + half_size]
            # Compute distances to all dataset patches
            distances = cdist(patch.flatten()[np.newaxis], dataset_flat, metric='sqeuclidean')[0]
            # Find the most similar patch in the dataset
            min_distance = np.min(distances)
            dists.append(min_distance)
            shapes.append(shape_labels[np.argmin(distances)])
            best_match_idx = np.argmin(distances)
            label = vertex_labels[best_match_idx]
            shape_label = shape_labels[best_match_idx]
            matches.append({"vertex_label": label, "position": (x - half_size, y - half_size),
                            "shape_label": shape_label,
                            "distance": min_distance})
    dists = torch.tensor(dists)
    dists_sorted, indices = torch.sort(dists)

    matches_sorted = [matches[i] for i in indices if matches[i]["shape_label"]==matches[indices[0]]["shape_label"]]

    return matches


def recall_match(args, bk_shapes, img):
    bw_img = np.array(Image.fromarray(img.numpy().astype('uint8')).convert("L"))
    bw_img[bw_img == 211] = 0
    bw_img[bw_img > 0] = 1
    # Find similar patches
    dataset_patches = torch.cat([bk_shapes[i]["fm_repo"] for i in range(len(bk_shapes))], dim=0)
    dataset_patches[dataset_patches > 0] = 1
    vertex_labels = [bk_shapes[i]["labels"] for i in range(len(bk_shapes))]
    vertex_labels = torch.cat([vertex_labels[i] for i in range(len(vertex_labels))])
    shape_labels = [[i] * len(bk_shapes[i]["fm_repo"]) for i in range(len(bk_shapes))]
    shape_labels = list(itertools.chain.from_iterable(shape_labels))

    th = 0
    matches = find_similar_patches(bw_img, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                   threshold=10)
    while len(matches) == 0:
        th += 1
        matches = find_similar_patches(bw_img, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                       threshold=th)

    most_frequent_matches = extract_most_frequent_label_matches(matches)
    match_shape_id = most_frequent_matches[0]["shape_label"] + 1

    group = gestalt_group.Group(id=0, name=match_shape_id, input_signal=segment, onside_signal=None,
                                # memory_signal=group_data['recalled_bw_img'],
                                parents=None, coverage=None, color=seg_color)

    onside_shapes = []
    onside_percents = torch.zeros(len(bk_shapes))
    cropped_data_all = []
    kernels_all = []
    for b_i, bk_shape in enumerate(bk_shapes):
        input_fms, cropped_data = models.img2fm(img, bk_shape["kernels"].float())
        mem_fms, _ = recall_fms(input_fms, bk_shape["fm_repo"], reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(input_fms, mem_fms, reshape=args.obj_fm_size)
        onside_shapes.append(group_data["onside"])
        onside_percents[b_i] = group_data["onside_percent"]
        cropped_data_all.append(cropped_data)
        kernels_all.append(bk_shape["kernels"])
    best_shape = onside_percents.argmax()
    best_cropped_data = cropped_data_all[onside_percents.argmax()]
    onside_pixels = onside_shapes[best_shape].squeeze()
    best_shape = best_shape + 1
    best_kernel = kernels_all[onside_percents.argmax()]
    return onside_pixels, best_shape, best_cropped_data, best_kernel
