# Created by X at 25/07/2024
import numpy as np
import torch
from scipy import ndimage
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from tqdm import tqdm
import torch.nn.functional as F  # Import F for functional operations
from PIL import Image, ImageDraw
from src import bk
from src.utils.chart_utils import van
from src.percept.percept_utils import *
from src.memory import recall
from src.reasoning import reason
from src.percept import gestalt_group
from src.neural import models
from src.percept import gestalt_algs
import config
from src.neural import line_arc_detector


# -----------------------------------------------------------------------------------
def find_segment_color(image: torch.Tensor) -> list:
    """
    Given a 1024x1024x3 PyTorch tensor representing an RGB image where most pixels are gray [211,211,211],
    this function finds and returns the RGB color (as a list) of the pure color connected segment.
    """
    # Create a mask for pixels that are not the gray background
    # This mask is True for pixels where at least one channel is not 211.
    mask = (image != 211).any(dim=-1)

    # Extract non-gray pixels from the image using the mask.
    non_gray_pixels = image[mask]

    # If there are no non-gray pixels, raise an error.
    if non_gray_pixels.numel() == 0:
        raise ValueError("No non-gray segment found.")

    # Since the segment is pure colored, all its pixels should have the same RGB value.
    # Get the unique colors among the non-gray pixels.
    unique_colors = non_gray_pixels.unique(dim=0)

    # If we find more than one unique non-gray color, raise an error.
    if unique_colors.size(0) != 1:
        raise ValueError("Expected a single pure color segment, but found multiple non-gray colors.")

    # Return the segment color as an RGB list.
    return unique_colors[0].tolist()


def get_most_frequent_color(args, img):
    assert img.ndim == 3

    # Find the most frequent color in the list
    if img.sum() == 0:
        return bk.no_color  # Handle empty list

    color_counts = img.reshape(3, -1).permute(1, 0).unique(return_counts=True, dim=0)
    color_sorted = sorted(zip(color_counts[0], color_counts[1]),
                          key=lambda x: x[1], reverse=True)
    if torch.all(color_sorted[0][0] == torch.tensor(bk.color_matplotlib["lightgray"]).to(args.device)):
        most_frequent = color_sorted[1][0]
    else:
        most_frequent = color_sorted[0][0]

    # Find the closest color in the dictionary
    closest_color_name = bk.color_large[0]
    smallest_distance = float('inf')
    distances = []
    for color_name, color_rgb in bk.color_matplotlib.items():
        distance = torch.sqrt((most_frequent.to(torch.uint8) - torch.tensor(color_rgb).to(args.device)).sum() ** 2)
        distances.append(distance)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_color_name = color_name

    return closest_color_name


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
    valid_rows = torch.any(image > threshold, dim=1)
    valid_cols = torch.any(image > threshold, dim=0)

    if not torch.any(valid_rows) or not torch.any(valid_cols):
        # No valid area
        return image, (0, 0, image.shape[1], image.shape[0])

    y_min, y_max = torch.where(valid_rows)[0][[0, -1]]
    x_min, x_max = torch.where(valid_cols)[0][[0, -1]]

    # Include bordering cases by padding
    padding = 0  # Add padding to account for partial patches
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, (x_min, y_min, x_max, y_max)


from scipy.spatial.distance import cdist


def find_best_matches(padded_image, dataset_flat, vertex_labels, shape_labels, threshold, half_size):
    """
    Finds the best matches for patches in a test image against a dataset using squared Euclidean distance.

    Parameters:
        padded_image (torch.Tensor): The padded input image (H, W) or (C, H, W) on GPU.
        dataset_flat (torch.Tensor): Flattened dataset patches of shape (N, D) on GPU.
        vertex_labels (list): List of vertex labels for dataset patches.
        shape_labels (list): List of shape labels for dataset patches.
        threshold (float): Distance threshold for matching.
        half_size (int): Half-size of the patch.

    Returns:
        list: A list of matching results.
    """
    device = padded_image.device  # Ensure everything is on the same device
    matches = []

    # Extract all possible patches using unfold (optimized way instead of loops)
    if padded_image.dim() == 2:  # Grayscale Image (H, W)
        padded_image = padded_image.unsqueeze(0)  # Convert to (1, H, W) for consistency

    C, H, W = padded_image.shape
    patch_size = 2 * half_size  # Full patch size

    # Use unfold to extract all patches (faster than looping)
    patches = padded_image.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C * patch_size * patch_size)  # Shape: (num_patches, D)

    # Compute squared Euclidean distances for all patches at once (batch-wise)
    distances = torch.cdist(patches.float(), dataset_flat.float(), p=2) ** 2  # Shape: (num_patches, N)

    # Find the best matches
    min_distances, best_match_indices = distances.min(dim=1)

    # Filter matches based on the threshold
    match_indices = torch.where(min_distances < threshold)[0]

    # Prepare final match results
    for idx in match_indices:
        y = idx // (W - patch_size + 1)
        x = idx % (W - patch_size + 1)
        best_match_idx = best_match_indices[idx].item()
        matches.append({
            "vertex_label": vertex_labels[best_match_idx],
            "position": (x, y),
            "shape_label": shape_labels[best_match_idx],
            "distance": min_distances[idx].item()
        })

    return matches


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
    # Pad the test image to handle boundary cases
    padded_image = F.pad(cropped_image, (half_size, half_size, half_size, half_size), mode='constant', value=0)
    # padded_image = np.pad(cropped_image, ((half_size, half_size), (half_size, half_size)), mode='constant',
    #                       constant_values=0)
    # Pre-compute flattened dataset patches for faster distance computation
    dataset_flat = dataset_patches.view(dataset_patches.shape[0], -1)
    # dataset_flat = np.array([p.flatten() for p in dataset_patches])
    matches = find_best_matches(padded_image, dataset_flat, vertex_labels, shape_labels, threshold, half_size)
    # # Iterate through every possible patch in the test image
    # for y in range(half_size, cropped_image.shape[0] + half_size):
    #     for x in range(half_size, cropped_image.shape[1] + half_size):
    #         # Extract the patch from the test image
    #         patch = padded_image[y - half_size:y + half_size, x - half_size:x + half_size]
    #         # Compute distances to all dataset patches
    #         patch_flat = patch.flatten().unsqueeze(0)  # Shape: (1, D)
    #         distances = torch.sum((dataset_flat - patch_flat) ** 2, dim=1)
    #         # distances = cdist(patch.flatten()[np.newaxis], dataset_flat, metric='sqeuclidean')[0]
    #
    #         # # Flatten the patch and compute distances to all dataset patches
    #         # patch_flat = patch.flatten()
    #         # dataset_flat = [p.flatten() for p in dataset_patches]
    #         # distances = cdist([patch_flat], dataset_flat, metric='sqeuclidean')[0]
    #
    #         # Find the most similar patch in the dataset
    #         min_distance = torch.min(distances)
    #         if min_distance < threshold:
    #             best_match_idx = torch.argmin(distances)
    #             label = vertex_labels[best_match_idx]
    #             shape_label = shape_labels[best_match_idx]
    #             matches.append({"vertex_label": label, "position": (x - half_size, y - half_size),
    #                             "shape_label": shape_label,
    #                             "distance": min_distance})
    return matches


from collections import Counter


def extract_most_frequent_label_matches(matches):
    """
    Extract the matches with the most frequent label.

    Args:
        matches (list): List of match dictionaries, each containing 'label' and 'position'.

    Returns:
        list: List of matches corresponding to the most frequent label.
    """
    if not matches:
        return []

    # Count the occurrences of each label
    label_counts = Counter(match["shape_label"] for match in matches)

    # Find the most frequent label
    most_frequent_label = label_counts.most_common(1)[0][0]

    # Filter matches with the most frequent label
    frequent_matches = [match for match in matches if match["shape_label"] == most_frequent_label]

    return frequent_matches


def percept_feature_groups(args, bk_shapes, segment):
    """ recall the memory features from the given segments """
    seg_color_rgb = find_segment_color(segment)
    seg_color = bk.color_dict_rgb2name[tuple(seg_color_rgb)]
    # rgb segment to resized bw image

    # recall the memory
    # Define the grayscale conversion weights
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=segment.device)
    # Convert to grayscale using the dot product
    test_image = torch.tensordot(segment.float(), weights, dims=([-1], [0])).round().clamp(0, 255).to(
        torch.uint8)  # Shape: (1, H, W)

    # remove the gray background, set image to black-white image
    test_image[test_image == 211] = 0
    test_image[test_image > 0] = 1
    # Find similar patches
    dataset_patches = torch.cat([bk_shapes[i]["fm_repo"] for i in range(len(bk_shapes))], dim=0).to(args.device)
    dataset_patches[dataset_patches > 0] = 1
    vertex_labels = [bk_shapes[i]["labels"] for i in range(len(bk_shapes))]
    vertex_labels = torch.cat([vertex_labels[i] for i in range(len(vertex_labels))])
    shape_labels = [[i] * len(bk_shapes[i]["fm_repo"]) for i in range(len(bk_shapes))]
    shape_labels = list(itertools.chain.from_iterable(shape_labels))

    th = 0
    matches = find_similar_patches(test_image, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                   threshold=10)
    while len(matches) == 0:
        th += 1
        matches = find_similar_patches(test_image, dataset_patches, vertex_labels, shape_labels, patch_size=32,
                                       threshold=th)
    most_frequent_matches = extract_most_frequent_label_matches(matches)
    match_shape_id = most_frequent_matches[0]["shape_label"] + 1

    group = gestalt_group.Group(id=0, name=match_shape_id, input_signal=segment, onside_signal=None,
                                # memory_signal=group_data['recalled_bw_img'],
                                parents=None, coverage=None, color=seg_color)
    return group


def merge_segments(segments):
    merged_img = segments[0].clone()
    for segment in segments:
        mask = (segment != torch.tensor(bk.color_matplotlib["lightgray"])).any(dim=-1)
        merged_img[mask] = segment[mask]
    return merged_img


def percept_closure_groups(args, segments, input_groups, bk_shapes):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    args.obj_fm_size = 32
    # all_obj_found_labels = False
    img = merge_segments(segments)
    # preprocessing img, convert rgb image to black-white image
    cropped_img, crop_data = data_utils.crop_img(img)
    bw_img = data_utils.resize_img(cropped_img, resize=args.obj_fm_size).unsqueeze(0)

    groups = []
    labels = torch.zeros(len(segments))
    label_counter = 1
    while torch.any(labels[:len(input_groups)] == 0):
        # recall the memory
        memory, group_label = recall.recall_match(args, bk_shapes, bw_img)
        # assign each object a label
        group_objs = reason.reason_labels(args, bw_img, input_groups, crop_data,
                                          labels, memory)
        if group_objs.sum() == 0:
            break

        labels[group_objs] += label_counter
        label_counter += 1
        # generate group object
        group = gen_group_tensor(input_groups, group_label, group_objs)
        groups.append(group)
    if len(groups) == 0:
        gcms = torch.zeros(1, 10)
    else:
        gcms = torch.stack(groups)

    return gcms, labels


def load_bk(args, bk_shapes):
    # load background knowledge
    bk = []
    kernel_size = config.kernel_size
    for s_i, bk_shape in enumerate(bk_shapes):
        if bk_shape == "none":
            continue
        bk_path = config.output / bk_shape
        kernel_file = bk_path / f"kernel_patches_{kernel_size}.pt"
        kernels = torch.load(kernel_file).to(args.device)

        fm_file = bk_path / f"fms_patches_{kernel_size}.pt"
        fm_data = torch.load(fm_file).to(args.device)
        fm_img = fm_data[:, 0:1]
        fm_repo = fm_data[:, 1:]

        # load pretrained autoencoder
        # ae = models.Autoencoder(fm_repo.shape[1])
        # ae.load_state_dict(torch.load(bk_path / "fm_ae.pth"))
        # # load the dimension reduced feature maps
        # ae_fm = torch.load(bk_path / "ae_fms.pt").to(args.device)

        bk.append({
            "shape": s_i,
            "kernel_size": kernel_size,
            "kernels": kernels,
            "fm_img": fm_img,
            "fm_repo": fm_repo,
            # "ae": ae,
            # "ae_fm": ae_fm,
        })
    return bk


def percept_segments(args, imgs, dtype):
    seg_file = str(args.output_file_prefix) + f"_segments_{dtype}.pt"
    if os.path.exists(seg_file):
        segments = torch.load(seg_file, map_location=torch.device(args.device))
    else:
        segments = []
        for img in imgs:
            img_torch = img.to(args.device)
            # segment the scene into separate parts
            segment = detect_connected_regions(args, img_torch).int()
            segments.append(segment)
        torch.save(segments, seg_file)
    return segments


# def cluster_by_closure(args, segments, seg_index):
#     """ group objects as a high level group, return labels of each object """
#     group_bk = load_bk(args, bk.bk_shapes)
#     # detect local feature as groups
#     loc_groups = detect_local_features(args, segments, seg_index, group_bk)
#     # loc_groups = [loc_groups[i] for i in range(len(obj_indices)) if obj_indices[i]]
#
#     # group_segs = segments[obj_indices]
#     # detect global feature as groups and seal the local feature groups into them
#     gcms, labels = percept_closure_groups(args, segments[seg_index], loc_groups, group_bk)
#
#     return gcms, labels


def gen_group_tensor(input_groups, group_shape, group_objs):
    parent_objs = [input_groups[i] for i in range(len(input_groups)) if
                   group_objs[i]]
    parent_positions = torch.stack([obj.pos for obj in parent_objs])
    x = parent_positions[:, 0]
    y = parent_positions[:, 1]
    group_x = x.mean()
    group_y = y.mean()
    # Shoelace formula
    group_size = 0.5 * torch.abs(
        torch.sum(x[:-1] * y[1:]) - torch.sum(y[:-1] * x[1:]))

    color = "none"
    shape = [0] * len(bk.bk_shapes)
    if group_shape is not None:
        shape[int(group_shape.item())] = 1.0
    color = torch.tensor(bk.color_matplotlib[color]) / 255.0
    obj_num = len(group_objs)
    tri = 0
    sq = 0
    cir = 0
    if group_shape == 0:
        tri = 1
    elif group_shape == 1:
        sq = 1
    elif group_shape == 2:
        cir = 1
    tensor = gestalt_group.gen_group_tensor(
        group_x, group_y, group_size, obj_num, color[0], color[1], color[2], tri, sq, cir
    )
    return tensor


def eval_similarity(groups):
    color_labels = groups2labels(groups, "color")
    shape_labels = groups2labels(groups, "shape")


# def eval_closure(groups):
#     positions = groups2positions(groups)
#     shapes = positions2shapes(positions)

def ocm_encoder(args, segments, dtype):
    group_bk = bk.load_bk_fms(args, bk.bk_shapes)
    # detect local feature as groups
    ocms = []
    groups = []
    for example_i in tqdm(range(len(segments)), f" ({dtype}) Example Segmentation"):
        ocm_file = str(args.output_file_prefix) + f"e{example_i}_{dtype}_ocm.pt"
        group_file = str(args.output_file_prefix) + f"e{example_i}_{dtype}_group.pt"
        example_seg = segments[example_i]
        example_groups = []
        if os.path.exists(ocm_file) and os.path.exists(group_file):
            example_ocm = torch.load(ocm_file, map_location=torch.device(args.device), weights_only=False)
            example_groups = torch.load(group_file, map_location=torch.device(args.device), weights_only=False)
        else:
            example_ocm = []
            for segment in example_seg:
                # van(segment)
                group = percept_feature_groups(args, group_bk, segment)
                # print(bk.bk_shapes[group.name])
                ocm = gestalt_group.group2tensor(group)
                example_ocm.append(ocm)
                example_groups.append(group)
            example_ocm = torch.stack(example_ocm).to(args.device)
            torch.save(example_ocm, ocm_file)
            torch.save(example_groups, group_file)
        ocms.append(example_ocm)
        groups.append(example_groups)
        args.logger.debug(f"detected local features: {len(example_ocm)}")
    return ocms, groups


def detect_connected_regions(args, input_array, pixel_num=50):
    # Find unique colors
    unique_colors, inverse = input_array.squeeze().reshape(-1, 3).unique(dim=0, return_inverse=True)
    labeled_regions = []

    width = input_array.shape[1]
    for color_idx, color in enumerate(unique_colors):
        gray_color = torch.tensor(bk.color_matplotlib["lightgray"], dtype=torch.uint8).to(args.device)
        if torch.equal(color, gray_color):
            continue
        # Create a mask for the current color
        mask = (inverse == color_idx).reshape(width, width)

        # Label connected components in the mask
        labeled_mask, num_features = ndimage.label(mask.to("cpu").numpy())

        for region_id in range(1, num_features + 1):
            # Isolate a single region
            region_mask = torch.from_numpy(labeled_mask == region_id).to(args.device)
            if region_mask.sum() > pixel_num:
                # Add the region to the labeled regions
                region_tensor = torch.zeros((3, width, width), dtype=torch.uint8).to(args.device)
                region_tensor += torch.tensor((bk.color_matplotlib["lightgray"])).reshape(3, 1, 1).to(args.device)
                for channel in range(3):
                    region_tensor[channel][region_mask] = color[channel]
                labeled_regions.append(region_tensor)

    # Stack all labeled regions into a single tensor
    if len(labeled_regions) == 0:
        print('')
    output_tensor = torch.stack(labeled_regions).permute(0, 2, 3, 1)

    args.logger.debug(f"detected connected regions: {output_tensor.shape[0]}")

    return output_tensor


def detect_local_features(args, segments, example_i, group_bk):
    if os.path.exists(group_file):
        groups = torch.load(group_file)
    else:
        groups = []
        for segment in tqdm(segments[seg_index], desc="local features detection"):
            group = percept_feature_groups(args, group_bk, segment)
            groups.append(group)
        if len(groups) != len(segments[seg_index]):
            print("")
        torch.save(groups, group_file)
    args.logger.debug(f"detected local features: {len(groups)}")

    return groups


def detect_global_features(args, loc_groups, bk, img):
    global_group_file = str(args.output_file_prefix) + f"global_groups.pt"
    if os.path.exists(global_group_file):
        labels, groups = percept_closure_groups(args, loc_groups, bk, img)
        # new_groups = torch.load(global_group_file)
    else:
        labels, groups = percept_closure_groups(args, loc_groups, bk, img)
        # while (new_groups != old_groups):
        #     new_groups = percept_gestalt_groups(args, loc_groups, bk, img)

        # torch.save(labels, global_group_file)

    return labels, groups


def percept_reward(lang):
    # # if no new group, reward is 0
    # if max([max(label) for label in labels]) <= max_label:
    #     return 0
    #
    # valid_groups = ocm.sum(axis=-1) > 0
    #
    # same_group_num = np.all(valid_groups == valid_groups[0], axis=1).all()
    #
    # group_shape = ocm[:, :, 6:].reshape(ocm.shape[0], -1)
    # same_group_shape = np.all(group_shape == group_shape[0], axis=1).all()
    #
    # group_color = ocm[:, :, 3:6].reshape(ocm.shape[0], -1)
    # same_group_color = np.all(group_color == group_color[0], axis=1).all()
    #
    # if config.gestalt_action[action] == "proximity":
    #     reward = 0
    # elif config.gestalt_action[action] == "similarity":
    #     # every group is common
    #     reward = bool(same_group_shape + same_group_color) * valid_groups[0].sum() > 1
    # elif config.gestalt_action[action] == "closure":
    #     reward = bool(same_group_num * same_group_shape)
    # else:
    #     raise ValueError("Unknown gestalt action {}".format(action))
    reward = 0
    if lang.done:
        reward = 1
    return float(reward)


def identify_kernels(args, train_loader):
    k_size = args.k_size
    kernels = []
    for (img, vertices) in tqdm(train_loader, f"Idf. Kernels (k = {k_size})"):
        bw_img = models.img2bw(img, 15)
        patches = bw_img.unfold(2, k_size, 1).unfold(3, k_size, 1)
        patches = patches.reshape(-1, k_size, k_size).unique(dim=0)
        patches = patches[~torch.all(patches == 0, dim=(1, 2))]
        kernels.append(patches)
    kernels = torch.cat(kernels, dim=0).unique(dim=0).unsqueeze(1)
    return kernels


def fm_sum_channels(fm_all):
    fm_sum = fm_all.sum(dim=1, keepdims=True)
    fm_sum = (fm_sum - fm_sum.min()) / (fm_sum.max() - fm_sum.min())
    return fm_sum


def identify_fms(shape):
    # Load metadata
    metadata_path = config.kp_base_dataset / shape / "metadata.json"
    if not os.path.exists(str(metadata_path)):
        raise FileNotFoundError(f"Metadata file not found in {str(metadata_path)}")
    metadata = data_utils.load_json(metadata_path)
    patches = []
    labels = []
    patch_set = set()
    # Iterate through the metadata
    for entry in metadata:
        for p_i, patch_filename in enumerate(entry["patches"]):
            patch_path = config.kp_base_dataset / shape / patch_filename
            if os.path.exists(patch_path):
                patch_image = Image.open(patch_path)
                patch_array = np.array(patch_image)
                patch_tuple = tuple(map(tuple, patch_array))
                if patch_tuple not in patch_set:
                    patch_set.add(patch_tuple)
                    patches.append(torch.from_numpy(patch_array))
                    labels.append(p_i)

    fm_all = torch.stack(patches)
    img_path = config.kp_base_dataset / shape / metadata[0]['image']
    image = np.array(Image.open(img_path))
    # get the contours
    contour_points = data_utils.get_contours(image)
    contour_points = contour_points.reshape(-1, 2)
    contour_img = np.zeros((512, 512, 3), dtype=np.uint8)
    from src import bk
    for i in range(len(contour_points)):
        pos = contour_points[i]
        if i < 5:
            contour_img[pos[1], pos[0]] = [255, 0, 0]
        else:
            contour_img[pos[1], pos[0]] = [255, 255, 255]
    van(contour_img)
    direction_vector = torch.tensor(data_utils.contour_to_direction_vector(contour_points))
    return fm_all, direction_vector, labels


def collect_fms(args):
    bk_shapes = bk.bk_shapes[1:]

    for bk_shape in bk_shapes:
        args.save_path = config.output / bk_shape
        args.bk_shape = bk_shape
        args.k_size = config.kernel_size
        os.makedirs(args.save_path, exist_ok=True)

        # fm identification
        fm_file = args.save_path / f'fms_patches_{args.k_size}.pt'
        if not os.path.exists(fm_file):
            fms, contours, labels = identify_fms(bk_shape)
            fms_labels = {"fms": fms, "labels": labels, "contours": contours}
            torch.save(fms_labels, fm_file)


def test_fms(args, data_loader):
    for task_id, (train_data, test_data, principle) in enumerate(data_loader):
        args.output_file_prefix = config.models / f"t{task_id}_"
        imgs = test_data["img"]
        labels_pos_gt = test_data["pos"].squeeze()
        labels_neg_gt = test_data["neg"].squeeze()
        imgs_pos = imgs[:3]
        imgs_neg = imgs[3:]
        segments_pos = percept_segments(args, imgs_pos, f"test_pos")
        segments_neg = percept_segments(args, imgs_neg, f"test_neg")

        all_labels_gt = []
        for img_i in range(len(segments_pos)):
            obj_num = segments_pos[img_i].shape[0]
            objs_labels = torch.argmax(labels_pos_gt[img_i, :obj_num,
                                       [bk.prop_idx_dict["shape_tri"], bk.prop_idx_dict["shape_sq"],
                                        bk.prop_idx_dict["shape_cir"]]], dim=1)
            all_labels_gt += objs_labels

        for img_i in range(len(segments_neg)):
            obj_num = segments_neg[img_i].shape[0]
            objs_labels = torch.argmax(labels_neg_gt[img_i, :obj_num,
                                       [bk.prop_idx_dict["shape_tri"], bk.prop_idx_dict["shape_sq"],
                                        bk.prop_idx_dict["shape_cir"]]], dim=1)
            all_labels_gt += objs_labels

        all_obj_imgs = torch.cat(segments_pos + segments_neg, dim=0)
        all_labels_gt = torch.stack(all_labels_gt)
        group_bk = bk.load_bk_fms(args, bk.bk_shapes)
        pred_labels = torch.zeros(len(all_obj_imgs))

        # remove all the similar tensors
        all_fms = torch.cat([
            group_bk[0]["fm_repo"].sum(dim=1),
            group_bk[1]["fm_repo"].sum(dim=1),
            group_bk[2]["fm_repo"].sum(dim=1),
        ], dim=0)
        # Combine all tensors into one list and remove duplicates

        unique_tensors = []
        # Assuming all_fms is your tensor
        n = all_fms.shape[0]  # Get the length of the first dimension
        perm = torch.randperm(n)  # Generate a permutation of indices
        all_fms_shuffled = all_fms[perm]  # Reindex the tensor

        for tensor in all_fms_shuffled:
            max_similar_score = 0
            for unique_tensor in unique_tensors:
                similar_score = F.cosine_similarity(tensor, unique_tensor).mean()
                if similar_score > max_similar_score:
                    max_similar_score = similar_score
            if max_similar_score < 0.95:
                unique_tensors.append(tensor)

        # Redistribute tensors back into their original sets
        bk1 = group_bk[0]["fm_repo"].sum(dim=1)
        bk2 = group_bk[1]["fm_repo"].sum(dim=1)
        bk3 = group_bk[2]["fm_repo"].sum(dim=1)
        unique_set1 = [group_bk[0]["fm_repo"][t_i] for t_i, tensor in enumerate(bk1) if
                       any((tensor == unique_tensor).all() for unique_tensor in unique_tensors)]
        unique_set2 = [group_bk[1]["fm_repo"][t_i] for t_i, tensor in enumerate(bk2) if
                       any((tensor == unique_tensor).all() for unique_tensor in unique_tensors)]
        unique_set3 = [group_bk[2]["fm_repo"][t_i] for t_i, tensor in enumerate(bk3) if
                       any((tensor == unique_tensor).all() for unique_tensor in unique_tensors)]

        fm_repos = [unique_set1, unique_set2, unique_set3]
        for img_i in range(len(all_obj_imgs)):
            scores = []
            for shape_i in range(len(group_bk)):
                fm_repo = torch.stack(fm_repos[shape_i])
                in_fms, _ = models.img2fm(all_obj_imgs[img_i], group_bk[shape_i]["kernels"])
                recalled_fms, score = recall.recall_fms(in_fms, fm_repo)
                scores.append(score)
            pred_labels[img_i] = np.argmax(scores)

        accuracy = (pred_labels == all_labels_gt).sum() / len(all_labels_gt)
        print(accuracy)


def percept_gestalt_groups(args, la_detector, ocms, segments, obj_groups, dtype, principle):
    """
    return:
    gestalt principle: the gestalt principle that can perfect grouping inputs
    gcm: group centric matrix of the groups
    labels: grouping labels of each object
    other: either thresholds (proximity) or the grouping shape (closure)
    """
    if principle == "proximity":
        labels_prox, ths, shape_proximity = gestalt_algs.cluster_by_proximity(ocms, th=0.15)
        if labels_prox is not None:
            gcm = gestalt_group.gcm_encoder(labels_prox, ocms, all_shapes=shape_proximity)
            return gcm, labels_prox, ths
    elif principle == "similarity_shape":
        labels_simi_shape, shape_similarity = gestalt_algs.cluster_by_similarity(ocms, "shape")
        if labels_simi_shape is not None:
            gcm = gestalt_group.gcm_encoder(labels_simi_shape, ocms, all_shapes=shape_similarity)
            return gcm, labels_simi_shape, None
    elif principle == "similarity_color":
        labels_simi_color, shape_similarity = gestalt_algs.cluster_by_similarity(ocms, "color")
        if labels_simi_color is not None:
            gcm = gestalt_group.gcm_encoder(labels_simi_color, ocms, all_shapes=shape_similarity)
            return gcm, labels_simi_color, None
    elif principle == "position_closure":
        labels_closure, shape_closure = gestalt_algs.cluster_by_position_closure(args, obj_groups)
        if labels_closure is not None:
            gcm = gestalt_group.gcm_encoder(labels_closure, ocms, shape_closure)
            return gcm, labels_closure, shape_closure
    elif principle == "feature_closure":
        labels_closure, shape_closure = gestalt_algs.cluster_by_feature_closure(args, la_detector, segments, obj_groups)
        if labels_closure is not None:
            gcm = gestalt_group.gcm_encoder(labels_closure, ocms, shape_closure)
            return gcm, labels_closure, shape_closure

    elif principle == "symmetry":
        labels_symmetry, th, shape_symmetry = gestalt_algs.cluster_by_symmetry(ocms)
        labels_symmetry = [torch.from_numpy(lst) for lst in labels_symmetry]
        if labels_symmetry is not None:
            gcm = gestalt_group.gcm_encoder(labels_symmetry, ocms, shape_symmetry)
            return gcm, labels_symmetry, shape_symmetry
    elif principle == "none":
        labels = [torch.tensor(list(range(len(ocm)))) for ocm in ocms]
        shapes = [torch.tensor([0]*len(ocm)) for ocm in ocms]
        gcm = gestalt_group.gcm_encoder(labels, ocms, shapes)
        return gcm, labels, shapes


    return None, None, None


def test_od_accuracy(args, train_data):
    imgs = train_data["img"]
    labels = torch.cat((train_data["pos"], train_data["neg"]), dim=1)
    obj_indices = [bk.prop_idx_dict["shape_tri"], bk.prop_idx_dict["shape_sq"], bk.prop_idx_dict["shape_cir"]]

    seg_pos = percept_segments(args, imgs, f"train_pos")
    ocm, obj_g = ocm_encoder(args, seg_pos, f"train_pos")
    ocm_pos = ocm[:10]
    ocm_neg = ocm[10:]
    obj_g_pos = obj_g[:10]
    obj_g_neg = obj_g[10:]
    # ocm_neg, obj_g_neg = ocm_encoder(args, seg_neg, f"train_neg")

    preds_pos = []
    for i in range(10):
        obj_num = len(ocm_pos[i])
        indices = torch.sort(labels[0, i, :obj_num, 0])[1]
        gt = labels[0, i, :obj_num][indices][:, obj_indices]
        pred_indices = torch.sort(ocm_pos[i][:, 0])[1]
        pred_obj = ocm_pos[i][pred_indices][:, obj_indices]
        preds_pos += torch.all(pred_obj == gt, dim=-1)
    preds_neg = []
    for i in range(10):
        obj_num = len(ocm_neg[i])
        indices = torch.sort(labels[0, i + 10, :obj_num, 0])[1]
        gt = labels[0, i + 10, :obj_num][indices][:, obj_indices]
        pred_indices = torch.sort(ocm_neg[i][:, 0])[1]
        pred_obj = ocm_neg[i][pred_indices][:, obj_indices]
        preds_neg += torch.all(pred_obj == gt, dim=-1)
    acc = (sum(preds_neg) + sum(preds_pos)) / (len(preds_neg) + len(preds_pos))
    print(acc)


def cluster_by_principle(args, imgs, mode, prin):
    """ evaluate gestalt scores, decide grouping based on which strategy

    output: NxOxP np array, N example numbers, O max group numbers, P group property numbers
    labels: NxO np array, N example numbers, O max group numbers
    - loc groups: individual objects;

    """
    # segmentation the images
    train_size = len(imgs) // 2
    imgs_pos = imgs[:train_size]
    imgs_neg = imgs[train_size:]
    seg_pos = percept_segments(args, imgs_pos, f"{mode}_pos")
    seg_neg = percept_segments(args, imgs_neg, f"{mode}_neg")
    group_label_pos = [torch.zeros(len(seg)) for seg in seg_pos]
    group_label_neg = [torch.zeros(len(seg)) for seg in seg_neg]

    # encode the segments to object centric matrix (ocm)
    ocm_pos, obj_g_pos = ocm_encoder(args, seg_pos, f"{mode}_pos")
    ocm_neg, obj_g_neg = ocm_encoder(args, seg_neg, f"{mode}_neg")
    # percept groups based on gestalt principles
    if prin == "feature_closure":
        la_detector = line_arc_detector.get_detector(args)
    else:
        la_detector = None
    group_pos, labels_pos, others_pos = percept_gestalt_groups(args, la_detector, ocm_pos, seg_pos, obj_g_pos, "pos",
                                                               prin)
    group_neg, labels_neg, others_neg = percept_gestalt_groups(args, la_detector, ocm_neg, seg_neg, obj_g_neg, "neg",
                                                               prin)

    groups = {
        "group_pos": group_pos, "label_pos": labels_pos,
        "group_neg": group_neg, "label_neg": labels_neg,
        'principle': prin
    }
    return groups
