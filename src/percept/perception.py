# Created by shaji at 25/07/2024
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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
    assert img.shape == (3, 512, 512)

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


def percept_feature_groups(args, bk_shapes, segment):
    """ recall the memory features from the given segments """
    feature_groups = []

    # rgb segment to resized bw image
    cropped_img, _ = data_utils.crop_img(segment)
    bw_img = data_utils.resize_img(cropped_img, resize=8).unsqueeze(0)
    seg_color = get_most_frequent_color(segment.permute(2, 0, 1))
    for b_i, bk_shape in enumerate(bk_shapes):
        # shape_str = bk.bk_shapes[bk_shape["shape"]]
        args.save_path = config.output / bk.bk_shapes[bk_shape["shape"]]
        # recall the memory
        shifted_fms, rc_fms = recall.recall_fms(args, bk_shape, bw_img)

        # reasoning recalled groups
        group_data = reason.reason_fms(rc_fms, bk_shape, bw_img)

        group = gestalt_group.Group(id=b_i,
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
        ocm_file = str(args.output_file_prefix) + f"ocm_{dtype}_example_{example_i}.pt"
        group_file = str(args.output_file_prefix) + f"group_{dtype}_example_{example_i}.pt"
        example_seg = segments[example_i]
        example_groups = []
        if os.path.exists(ocm_file) and os.path.exists(group_file):
            example_ocm = torch.load(ocm_file)
            example_groups = torch.load(group_file)
        else:
            example_ocm = []
            for segment in example_seg:
                group = percept_feature_groups(args, group_bk, segment)
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


def percept_gestalt_groups(args, ocms, segments, obj_groups, dtype):
    gcms = []
    group_labels = []
    for example_i in range(len(ocms)):
        example_gcm = []
        example_group_labels = []
        example_ocm = ocms[example_i]

        labels_prox = gestalt_algs.cluster_by_proximity(example_ocm, 0.1)
        example_group_labels.append(labels_prox)
        gcm_prox = gestalt_group.gcm_encoder(labels_prox, example_ocm)
        example_gcm.append(gcm_prox)

        labels_simi_shape = gestalt_algs.cluster_by_similarity(example_ocm, 0, "shape")
        example_group_labels.append(labels_simi_shape)
        gcm_simi_shape = gestalt_group.gcm_encoder(labels_simi_shape, example_ocm)
        example_gcm.append(gcm_simi_shape)

        labels_simi_color = gestalt_algs.cluster_by_similarity(example_ocm, 0, "color")
        example_group_labels.append(labels_simi_color)
        gcm_simi_color = gestalt_group.gcm_encoder(labels_simi_color, example_ocm)
        example_gcm.append(gcm_simi_color)

        labels_closure = gestalt_algs.percept_closure_groups(args, segments, obj_groups)
        example_group_labels.append(labels_closure)
        gcm_closure = gestalt_group.gcm_encoder(labels_closure, example_ocm)
        example_gcm.append(gcm_closure)

        gcms.append(example_gcm)
        group_labels.append(example_group_labels)
    return gcms, group_labels


def cluster_by_principle(args, imgs):
    """ evaluate gestalt scores, decide grouping based on which strategy

    output: NxOxP np array, N example numbers, O max group numbers, P group property numbers
    labels: NxO np array, N example numbers, O max group numbers
    - loc groups: individual objects;

    """
    # segmentation the images
    imgs_pos = imgs[:3]
    imgs_neg = imgs[3:]
    segments_pos = percept_segments(args, imgs_pos, "pos")
    segments_neg = percept_segments(args, imgs_neg, "neg")
    group_label_pos = [torch.zeros(len(seg)) for seg in segments_pos]
    group_label_neg = [torch.zeros(len(seg)) for seg in segments_neg]

    # encode the segments to object centric matrix (ocm)
    ocm_pos, obj_groups_pos = ocm_encoder(args, segments_pos, "pos")
    ocm_neg, obj_groups_neg = ocm_encoder(args, segments_neg, "neg")

    group_pos, group_labels_pos = percept_gestalt_groups(args, ocm_pos, segments_pos, obj_groups_pos, "pos")

    gcms = None
    group_labels = None

    return gcms, group_labels
