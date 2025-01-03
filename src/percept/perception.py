# Created by shaji at 25/07/2024
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from src import bk
from src.utils.chart_utils import van
from src.percept.percept_utils import *
from src.memory import recall
from src.reasoning import reason
from src.percept.group import Group
from src.neural import models
import config


# -----------------------------------------------------------------------------------

def get_most_frequent_color(img):
    assert img.shape == (3, 512, 512)

    # Find the most frequent color in the list
    if img.sum() == 0:
        return bk.no_color  # Handle empty list

    color_counts = img.reshape(3, -1).permute(1, 0).unique(return_counts=True, dim=0)
    color_sorted = sorted(zip(color_counts[0], color_counts[1]),
                          key=lambda x: x[1], reverse=True)
    if torch.all(color_sorted[0][0] == torch.tensor([211, 211, 211])):
        most_frequent = color_sorted[1][0]
    else:
        most_frequent = color_sorted[0][0]

    # Find the closest color in the dictionary
    closest_color_name = bk.no_color
    smallest_distance = float('inf')
    distances = []
    for color_name, color_rgb in bk.color_matplotlib.items():
        distance = torch.sqrt(sum((most_frequent - torch.tensor(color_rgb)) ** 2))
        distances.append(distance)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_color_name = color_name

    return closest_color_name


def percept_feature_groups(args, bk_shapes, segment, img):
    """ recall the memory features from the given segments """
    feature_groups = []

    # rgb segment to resized bw image
    cropped_img, _ = data_utils.crop_img(segment.permute(1, 2, 0))
    bw_img = data_utils.resize_img(cropped_img, resize=8).unsqueeze(0)
    seg_color = get_most_frequent_color(segment)
    for b_i, bk_shape in enumerate(bk_shapes):
        # shape_str = bk.bk_shapes[bk_shape["shape"]]
        args.save_path = config.output / bk.bk_shapes[bk_shape["shape"]]
        # recall the memory
        shifted_fms, rc_fms = recall.recall_fms(args, bk_shape, bw_img)

        # reasoning recalled groups
        group_data = reason.reason_fms(rc_fms, bk_shape, bw_img)

        group = Group(id=b_i,
                      name=bk_shape["shape"],
                      input_signal=segment,
                      onside_signal=group_data["onside"],
                      memory_signal=group_data['recalled_bw_img'],
                      parents=None,
                      coverage=group_data["onside_percent"],
                      color=seg_color)
        feature_groups.append(group)

        # log
        # args.logger.debug(
        #     f" Found group: {bk_shape['name']}: "
        #     f" [coverage: {group.onside_coverage:.2f}]")

    best_idx = torch.tensor([g.onside_coverage for g in feature_groups]).argmax()
    best_group = feature_groups[best_idx]
    return best_group


def percept_closure_groups(args, input_groups, bk_shapes, img):
    """ group input groups to output groups, which are high level groups """
    # each object assigned a group id as its label
    args.obj_fm_size = 32
    # all_obj_found_labels = False

    # preprocessing img, convert rgb image to black-white image
    cropped_img, crop_data = data_utils.crop_img(img)
    bw_img = data_utils.resize_img(cropped_img, resize=args.obj_fm_size).unsqueeze(0)

    groups = []
    labels = torch.zeros(args.obj_n) - 1
    while torch.any(labels[:len(input_groups)] == -1):
        # recall the memory
        memory, group_label = recall.recall_match(args, bk_shapes, bw_img)
        # assign each object a label
        group_objs = reason.reason_labels(args, bw_img, input_groups, crop_data,
                                          labels, memory)
        if group_objs.sum() == 0:
            break

        labels[group_objs] = group_label

        # generate group object
        group = gen_group_tensor(input_groups, group_label, group_objs)
        groups.append(group)
    group_ocm = torch.stack(groups)
    return labels, group_ocm


def cluster_by_proximity(object_matrix, threshold):
    # Function to compute distance or difference

    obj_n = object_matrix.shape[0]
    labels = np.full((obj_n,), -1, dtype=np.int32)
    visited = np.zeros(obj_n, dtype=np.bool_)
    current_label = 0
    for i in range(obj_n):
        if not visited[i]:
            # BFS or DFS
            stack = [i]
            visited[i] = True
            labels[i] = current_label

            while stack:
                top = stack.pop()
                for j in range(obj_n):
                    if not visited[j]:
                        dist = proximity_distance(object_matrix[top, :2],
                                                  object_matrix[j, :2])
                        if dist <= threshold:
                            visited[j] = True
                            labels[j] = current_label
                            stack.append(j)
            current_label += 1
    return labels


def cluster_by_similarity(object_matrix, threshold, weights):
    obj_n = object_matrix.shape[0]
    labels = np.full((obj_n,), -1, dtype=np.int32)
    visited = np.zeros(obj_n, dtype=np.bool_)
    current_label = 0
    for i in range(obj_n):
        if not visited[i]:
            # BFS or DFS
            stack = [i]
            visited[i] = True
            labels[i] = current_label

            while stack:
                top = stack.pop()
                for j in range(obj_n):
                    if not visited[j]:
                        dist = similarity_distance(object_matrix[top],
                                                   object_matrix[j], weights)
                        if dist <= threshold:
                            visited[j] = True
                            labels[j] = current_label
                            stack.append(j)
            current_label += 1
    return labels


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


def cluster_by_closure(args, object_matrix, img, threshold):
    """ group objects as a high level group, return labels of each object """

    # convert rgb image to black-white image
    group_bk = load_bk(args, bk.bk_shapes)

    img_torch = torch.from_numpy(img).to(args.device)
    # segment the scene into separate parts
    segments = detect_connected_regions(args, img_torch)
    # detect local feature as groups
    loc_groups = detect_local_features(args, segments, group_bk, img_torch)
    # detect global feature as groups and seal the local feature groups into them
    labels, groups = percept_closure_groups(args, loc_groups, group_bk, img_torch)

    groups= torch.cat([groups, torch.zeros(len(object_matrix)-len(groups), 10)],dim=0)
    return labels, groups


def gen_group_tensor(input_groups, group_label, group_objs):
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
    shape[int(group_label.item())] = 1.0
    color = list(bk.color_matplotlib[color])
    others = [group_x, group_y, group_size]
    tensor = torch.tensor(others + color + shape)
    tensor[3:6] /= 255
    return tensor


def eval_similarity(groups):
    color_labels = groups2labels(groups, "color")
    shape_labels = groups2labels(groups, "shape")


# def eval_closure(groups):
#     positions = groups2positions(groups)
#     shapes = positions2shapes(positions)


def cluster_by_principle(args, ocms, imgs, action, threshold, weights):
    """ evaluate gestalt scores, decide grouping based on which strategy
    - loc groups: individual objects;
    """
    labels = np.zeros(ocms.shape[:2])
    groups = []
    action = 2
    for o_i, ocm in enumerate(ocms):
        if config.gestalt_action[action] == "proximity":
            labels[o_i] = cluster_by_proximity(ocm, threshold)
        elif config.gestalt_action[action] == "similarity":
            labels[o_i] = cluster_by_similarity(ocm, threshold, weights)
        elif config.gestalt_action[action] == "closure":
            labels[o_i], new_groups = cluster_by_closure(args, ocm, imgs[o_i],
                                                         threshold)
            groups.append(new_groups)
        else:
            raise ValueError("Unknown gestalt action {}".format(action))
    groups = torch.stack(groups)
    return groups, labels

    #     # strategy: proximity
    #     labels = cluster_by_proximity(ocm, threshold)
    # elif config.gestalt_action[action] == "similarity":
    #     # strategy: similarity
    #     labels = cluster_by_similarity(ocm, threshold)
    # elif config.gestalt_action[action] == "closure":
    #     # strategy: closure
    #     labels = cluster_by_closure(ocm, threshold)
    # else:
    #     raise ValueError
    # return labels


def percept_gestalt_groups(args, loc_groups, bk, img):
    gestalt_groups = percept_closure_groups(args, loc_groups, bk, img)
    return gestalt_groups


def detect_connected_regions(args, input_array, pixel_num=50):
    # Find unique colors
    unique_colors, inverse = np.unique(input_array.reshape(-1, 3), axis=0,
                                       return_inverse=True)
    labeled_regions = []

    for color_idx, color in enumerate(unique_colors):
        if np.equal(color, np.array(bk.color_matplotlib["lightgray"],
                                    dtype=np.uint8)).all():
            continue
        # Create a mask for the current color
        mask = (inverse == color_idx).reshape(512, 512)

        # Label connected components in the mask
        labeled_mask, num_features = ndimage.label(mask)

        for region_id in range(1, num_features + 1):
            # Isolate a single region
            region_mask = (labeled_mask == region_id)
            if region_mask.sum() > pixel_num:
                # Add the region to the labeled regions
                region_tensor = np.zeros((3, 512, 512), dtype=np.float32)
                region_tensor += np.array(
                    (bk.color_matplotlib["lightgray"])).reshape(3, 1, 1)
                for channel in range(3):
                    region_tensor[channel][region_mask] = color[channel]
                labeled_regions.append(region_tensor)

    # Stack all labeled regions into a single tensor
    if len(labeled_regions) == 0:
        print('')
    output_tensor = torch.tensor(np.stack(labeled_regions), dtype=torch.float32)

    args.logger.debug(f"detected connected regions: {output_tensor.shape[0]}")

    return output_tensor


def detect_local_features(args, segments, group_bk, img):
    group_file = args.output_file_prefix + f"_feature_groups.pt"
    if os.path.exists(group_file):
        groups = torch.load(group_file)
    else:
        groups = []
        for segment in tqdm(segments, desc="local features detection"):
            group = percept_feature_groups(args, group_bk, segment, img)
            groups.append(group)
        torch.save(groups, group_file)
    args.logger.debug(f"detected local features: {len(groups)}")
    return groups


def detect_global_features(args, loc_groups, bk, img):
    global_group_file = args.output_file_prefix + f"_global_groups.pt"
    if os.path.exists(global_group_file):
        labels, groups = percept_closure_groups(args, loc_groups, bk, img)
        # new_groups = torch.load(global_group_file)
    else:
        labels, groups = percept_closure_groups(args, loc_groups, bk, img)
        # while (new_groups != old_groups):
        #     new_groups = percept_gestalt_groups(args, loc_groups, bk, img)

        # torch.save(labels, global_group_file)

    return labels, groups


def percept_groups(args, idx, group_bk, img):
    # segment the scene into separate parts
    segments = detect_connected_regions(args, img)
    # detect local feature as groups
    loc_groups = detect_local_features(args, segments, group_bk, img)
    # detect global feature as groups and seal the local feature groups into them
    glo_groups = detect_global_features(args, loc_groups, group_bk, img)

    return glo_groups


def percept_reward(groups):
    return np.all(groups == groups[0]).astype(np.float32)


def identify_kernels(args, train_loader):
    k_size = args.k_size
    kernels = []
    for (bw_img) in tqdm(train_loader, f"Idf. Kernels (k = {k_size})"):
        patches = bw_img.unfold(2, k_size, 1).unfold(3, k_size, 1)
        patches = patches.reshape(-1, k_size, k_size).unique(dim=0)
        patches = patches[~torch.all(patches == 0, dim=(1, 2))]
        kernels.append(patches)
    kernels = torch.cat(kernels, dim=0).unique(dim=0).unsqueeze(1)
    return kernels


def identify_fms(args, train_loader, kernels):
    # calculate fms
    k_size = args.k_size
    fm_all = []
    data_shift_all = []
    for (bw_img) in tqdm(train_loader, desc=f"Calc. FMs (k={k_size})"):
        fms = models.one_layer_conv(bw_img, kernels)
        fms, row_shift, col_shift = data_utils.shift_content_to_top_left(fms)

        bw_img, _, _ = data_utils.shift_content_to_top_left(bw_img,
                                                            row_shift,
                                                            col_shift)
        fm_all.append(fms)
        data_shift_all.append(bw_img)

    fm_all = torch.cat(fm_all, dim=0)

    data_shift_all = torch.cat(data_shift_all, dim=0)
    data_all = torch.cat((data_shift_all, fm_all), dim=1).unique(dim=0)

    # visual memory fms
    fm_np_array = data_all[:, 1:].sum(dim=1, keepdims=True)
    fm_np_array = (fm_np_array - fm_np_array.min()) / (
            fm_np_array.max() - fm_np_array.min())
    fm_np_array = fm_np_array.permute(0, 2, 3, 1).numpy()

    chart_utils.visual_batch_imgs(fm_np_array, args.save_path, "memory_fms.png")

    return data_all


def collect_fms(args):
    bk_shapes = args.exp_setting["bk_groups"]
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

        # kernel identification
        kernel_file = args.save_path / f"kernel_patches_{args.k_size}.pt"
        if not os.path.exists(kernel_file):
            kernels = identify_kernels(args, train_loader)
            torch.save(kernels, kernel_file)
        else:
            kernels = torch.load(kernel_file)

        # fm identification
        fm_file = args.save_path / f'fms_patches_{args.k_size}.pt'
        if not os.path.exists(fm_file):
            fms = identify_fms(args, train_loader, kernels)
            torch.save(fms, fm_file)
        else:
            fms = torch.load(fm_file)

        # log
        args.logger.debug(f"#Kernels: {len(kernels)}, "
                          f"#Data: {len(train_loader)}, "
                          f"Ratio: {len(kernels) / len(train_loader):.2f}"
                          f"#FM: {len(fms)}. "
                          f"#Data: {len(train_loader)}, "
                          f"ratio: {len(fms) / len(train_loader):.2f} "
                          f"feature maps have been saved to "
                          f"{args.save_path}/f'fms_patches_{args.k_size}.pt'")
