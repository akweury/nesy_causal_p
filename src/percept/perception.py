# Created by shaji at 25/07/2024
import numpy as np
import torch
from scipy import ndimage

from src import bk
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
    seg_np = segment.permute(1, 2, 0).numpy().astype(np.uint8)
    bw_img = data_utils.rgb2bw(seg_np, crop=True, resize=8).unsqueeze(0)
    seg_color = get_most_frequent_color(segment)
    for b_i, bk_shape in enumerate(bk_shapes):
        args.save_path = config.output / bk_shape["name"]
        # recall the memory
        shifted_fms, rc_fms = recall.recall_fms(args, bk_shape, bw_img)

        # reasoning recalled groups
        group_data = reason.reason_fms(args, segment, rc_fms, bk_shape, img,
                                       bw_img)

        group = Group(id=b_i,
                      name=bk_shape["name"],
                      input_signal=img,
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


def percept_object_groups(args, input_groups, bk_shapes, img):
    """ group objects as a high level group """

    # convert rgb image to black-white image
    args.obj_fm_size = 32
    bw_img = data_utils.rgb2bw(img.numpy(), crop=False,
                               resize=args.obj_fm_size).unsqueeze(0)

    segment = img.permute(2, 0, 1)

    obj_groups = []
    onside_shapes = []
    for b_i, bk_shape in tqdm(enumerate(bk_shapes), desc="grouping objects"):
        # recall the memory
        args.save_path = config.output / bk_shape["name"]
        shifted_fms, rc_fms = recall.recall_fms(args, bk_shape, bw_img,
                                                reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(args, segment, rc_fms, bk_shape, img, bw_img,
                                   reshape=args.obj_fm_size)


        onside_shapes.append(group_data["onside"])

    onside_shapes = torch.stack(onside_shapes, dim=0)
    onside_argsmax = onside_shapes.float().argmax(dim=0)

    for b_i, bk_shape in enumerate(bk_shapes):
        onside_mask = onside_argsmax == b_i
        shape_mask = torch.zeros_like(onside_argsmax)
        for loc_group in input_groups:
            input_seg = loc_group.input
            seg_np = input_seg.numpy().astype(np.uint8)
            seg_img = data_utils.rgb2bw(seg_np, crop=False,
                                        resize=args.obj_fm_size).unsqueeze(0)
            seg_mask = seg_img > 0
            shape_mask += onside_mask * seg_mask.squeeze()

        group_data = {
            "onside": shape_mask,
            "recalled_bw_img": shape_mask.unsqueeze(0).unsqueeze(0),
            "parents": None,
            "onside_percent": 0,
        }

        # # convert data to group object
        group = Group(id=b_i,
                      name=bk_shape["name"],
                      input_signal=img,
                      onside_signal=group_data["onside"],
                      memory_signal=group_data['recalled_bw_img'],
                      parents=input_groups,
                      coverage=group_data["onside_percent"],
                      color=None)
        obj_groups.append(group)

    best_idx = torch.tensor([g.onside_coverage for g in obj_groups]).argmax()
    best_group = obj_groups[best_idx]
    return best_group


def detect_connected_regions(input_array, pixel_num=50):
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
    return groups


def detect_global_features(args, loc_groups, bk, img):
    global_group_file = args.output_file_prefix + f"_global_groups.pt"
    if os.path.exists(global_group_file):
        groups = torch.load(global_group_file)
    else:
        groups = percept_object_groups(args, loc_groups, bk, img)
        torch.save(groups, global_group_file)

    return [groups]


def percept_gestalt_groups(args, idx, group_bk, img):
    # segment the scene into separate parts
    segments = detect_connected_regions(img)
    # detect local feature as groups
    loc_groups = detect_local_features(args, segments, group_bk, img)
    # detect global feature as groups and seal the local feature groups into them
    glo_groups = detect_global_features(args, loc_groups, group_bk, img)

    return glo_groups


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
