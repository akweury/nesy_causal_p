# Created by shaji at 25/07/2024
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


# -----------------------------------------------------------------------------------

def get_most_frequent_color(img):
    assert img.ndim == 3

    # Find the most frequent color in the list
    if img.sum() == 0:
        return bk.no_color  # Handle empty list

    color_counts = img.reshape(3, -1).permute(1, 0).unique(return_counts=True, dim=0)
    color_sorted = sorted(zip(color_counts[0], color_counts[1]),
                          key=lambda x: x[1], reverse=True)
    if torch.all(color_sorted[0][0] == torch.tensor(bk.color_matplotlib["lightgray"])):
        most_frequent = color_sorted[1][0]
    else:
        most_frequent = color_sorted[0][0]

    # Find the closest color in the dictionary
    closest_color_name = bk.color_large[0]
    smallest_distance = float('inf')
    distances = []
    for color_name, color_rgb in bk.color_matplotlib.items():
        distance = torch.sqrt(sum((most_frequent - torch.tensor(color_rgb)) ** 2))
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
    # Pad the test image to handle boundary cases
    padded_image = np.pad(cropped_image, ((half_size, half_size), (half_size, half_size)), mode='constant',
                          constant_values=0)
    # Pre-compute flattened dataset patches for faster distance computation
    dataset_flat = np.array([p.flatten() for p in dataset_patches])

    # Iterate through every possible patch in the test image
    for y in range(half_size, cropped_image.shape[0] + half_size):
        for x in range(half_size, cropped_image.shape[1] + half_size):
            # Extract the patch from the test image
            patch = padded_image[y - half_size:y + half_size, x - half_size:x + half_size]
            # Compute distances to all dataset patches
            distances = cdist(patch.flatten()[np.newaxis], dataset_flat, metric='sqeuclidean')[0]

            # # Flatten the patch and compute distances to all dataset patches
            # patch_flat = patch.flatten()
            # dataset_flat = [p.flatten() for p in dataset_patches]
            # distances = cdist([patch_flat], dataset_flat, metric='sqeuclidean')[0]

            # Find the most similar patch in the dataset
            min_distance = np.min(distances)
            if min_distance < threshold:
                best_match_idx = np.argmin(distances)
                label = vertex_labels[best_match_idx]
                shape_label = shape_labels[best_match_idx]
                matches.append({"vertex_label": label, "position": (x - half_size, y - half_size),
                                "shape_label": shape_label,
                                "distance": min_distance})
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
    feature_groups = []
    seg_color = get_most_frequent_color(segment.permute(2, 0, 1))
    # rgb segment to resized bw image

    # recall the memory
    scores = []
    recalled_fms_all = []
    all_in_fms = []
    # Example test image

    test_image = np.array(Image.fromarray(segment.numpy().astype('uint8')).convert("L"))
    test_image[test_image == 211] = 0
    test_image[test_image > 0] = 1
    # Find similar patches
    dataset_patches = torch.cat([bk_shapes[i]["fm_repo"] for i in range(len(bk_shapes))], dim=0)
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
        segments = torch.load(seg_file)
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
            example_ocm = torch.load(ocm_file)
            example_groups = torch.load(group_file)
        else:
            example_ocm = []
            for segment in example_seg:
                # van(segment)
                group = percept_feature_groups(args, group_bk, segment)
                # print(bk.bk_shapes[group.name])
                ocm = gestalt_group.group2tensor(group)
                example_ocm.append(ocm)
                example_groups.append(group)
            example_ocm = torch.stack(example_ocm)
            torch.save(example_ocm, ocm_file)
            torch.save(example_groups, group_file)
        ocms.append(example_ocm)
        groups.append(example_groups)
        args.logger.debug(f"detected local features: {len(example_ocm)}")
    return ocms, groups


def detect_connected_regions(args, input_array, pixel_num=50):
    # Find unique colors
    unique_colors, inverse = np.unique(input_array.squeeze().reshape(-1, 3), axis=0,
                                       return_inverse=True)
    labeled_regions = []

    width = input_array.shape[1]
    for color_idx, color in enumerate(unique_colors):
        if np.equal(color, np.array(bk.color_matplotlib["lightgray"],
                                    dtype=np.uint8)).all():
            continue
        # Create a mask for the current color
        mask = (inverse == color_idx).reshape(width, width)

        # Label connected components in the mask
        labeled_mask, num_features = ndimage.label(mask)

        for region_id in range(1, num_features + 1):
            # Isolate a single region
            region_mask = (labeled_mask == region_id)
            if region_mask.sum() > pixel_num:
                # Add the region to the labeled regions
                region_tensor = np.zeros((3, width, width), dtype=np.float32)
                region_tensor += np.array(
                    (bk.color_matplotlib["lightgray"])).reshape(3, 1, 1)
                for channel in range(3):
                    region_tensor[channel][region_mask] = color[channel]
                labeled_regions.append(region_tensor)

    # Stack all labeled regions into a single tensor
    if len(labeled_regions) == 0:
        print('')
    output_tensor = torch.tensor(np.stack(labeled_regions), dtype=torch.float32).permute(0, 2, 3, 1)

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


def identify_fms(args, shape):
    # calculate fms
    k_size = args.k_size
    fm_all = []
    all_imgs = []

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

    # for (img) in tqdm(train_loader, desc=f"Calc. FMs (k={k_size})"):
    #     all_imgs.append(models.img2bw(img))
    #
    # for (img) in tqdm(train_loader, desc=f"Calc. FMs (k={k_size})"):
    #     fm = models.img2fm(img, kernels)
    #
    #     fm_all.append(fm)
    fm_all = torch.stack(patches)

    chart_utils.visual_batch_imgs(fm_all.unsqueeze(-1).numpy(), args.save_path, "memory_fms.png")

    return fm_all, labels


def collect_fms(args):
    bk_shapes = bk.bk_shapes[1:]
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Collecting FMs for patterns {bk_shapes}.")

    for bk_shape in bk_shapes:
        args.save_path = config.output / bk_shape
        args.bk_shape = bk_shape
        args.k_size = config.kernel_size
        os.makedirs(args.save_path, exist_ok=True)

        # load data
        train_loader, val_loader = prepare_data(args)

        # # kernel identification
        # kernel_file = args.save_path / f"kernel_patches_{args.k_size}.pt"
        # if not os.path.exists(kernel_file):
        #     kernels = identify_kernels(args, train_loader)
        #     torch.save(kernels, kernel_file)
        # else:
        #     kernels = torch.load(kernel_file)

        # fm identification
        fm_file = args.save_path / f'fms_patches_{args.k_size}.pt'
        if not os.path.exists(fm_file):
            fms, labels = identify_fms(args, bk_shape)
            fms_labels = {"fms": fms, "labels": labels}
            torch.save(fms_labels, fm_file)
        else:
            fms_labels = torch.load(fm_file)
            fms = fms_labels["fms"]
            labels = fms_labels["labels"]


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


def percept_gestalt_groups(args, ocms, segments, obj_groups, dtype, principle):
    """
    return:
    gestalt principle: the gestalt principle that can perfect grouping inputs
    gcm: group centric matrix of the groups
    labels: grouping labels of each object
    other: either thresholds (proximity) or the grouping shape (closure)
    """
    if principle == "proximity":
        labels_prox, ths = gestalt_algs.cluster_by_proximity(ocms)
        if labels_prox is not None:
            gcm = gestalt_group.gcm_encoder(labels_prox, ocms, group_shape=0)
            return gcm, labels_prox, ths
    elif principle == "similarity_shape":
        labels_simi_shape = gestalt_algs.cluster_by_similarity(ocms, "shape")
        if labels_simi_shape is not None:
            gcm = gestalt_group.gcm_encoder(labels_simi_shape, ocms, group_shape=0)
            return gcm, labels_simi_shape, None
    elif principle == "similarity_color":
        labels_simi_color = gestalt_algs.cluster_by_similarity(ocms, "color")
        if labels_simi_color is not None:
            gcm = gestalt_group.gcm_encoder(labels_simi_color, ocms, group_shape=0)
            return gcm, labels_simi_color, None
    elif principle == "closure":
        labels_closure, shape_closure = gestalt_algs.cluster_by_closure(args, segments, obj_groups)
        if labels_closure is not None:
            gcm = gestalt_group.gcm_encoder(labels_closure, ocms, group_shape=shape_closure)
            return gcm, labels_closure, shape_closure

    return None, None, None


def cluster_by_principle(args, imgs, mode, principle):
    """ evaluate gestalt scores, decide grouping based on which strategy

    output: NxOxP np array, N example numbers, O max group numbers, P group property numbers
    labels: NxO np array, N example numbers, O max group numbers
    - loc groups: individual objects;

    """
    # segmentation the images
    imgs_pos = imgs[:3]
    imgs_neg = imgs[3:]
    segments_pos = percept_segments(args, imgs_pos, f"{mode}_pos")
    segments_neg = percept_segments(args, imgs_neg, f"{mode}_neg")
    group_label_pos = [torch.zeros(len(seg)) for seg in segments_pos]
    group_label_neg = [torch.zeros(len(seg)) for seg in segments_neg]

    # encode the segments to object centric matrix (ocm)
    ocm_pos, obj_groups_pos = ocm_encoder(args, segments_pos, f"{mode}_pos")
    ocm_neg, obj_groups_neg = ocm_encoder(args, segments_neg, f"{mode}_neg")
    # percept groups based on gestalt principles
    group_pos, labels_pos, others_pos = percept_gestalt_groups(args, ocm_pos, segments_pos, obj_groups_pos, "pos",
                                                               principle)
    group_neg, labels_neg, others_neg = percept_gestalt_groups(args, ocm_neg, segments_neg, obj_groups_neg, "neg",
                                                               principle)
    groups = {
        "group_pos": group_pos, "label_pos": labels_pos,
        "group_neg": group_neg, "label_neg": labels_neg,
        'principle': principle
    }
    return groups
