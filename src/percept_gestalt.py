# Created by X at 26.11.24

import torch
import cv2
import os

import config
import train_common_features
from utils import file_utils, args_utils, data_utils
from src.alpha import alpha
from src import bk


def load_bk(args, bk_shapes):
    # load background knowledge
    bk = []
    for bk_shape in bk_shapes:
        for kernel_size in [3]:
            if bk_shape == "none":
                continue
            kernels = torch.load(config.output / bk_shape / f"kernel_patches_{kernel_size}.pt").to(args.device)
            fm_data = torch.load(config.output / bk_shape / f"fms_patches_{kernel_size}.pt").to(args.device)
            fm_img = fm_data[:, 0:1]
            fm_repo = fm_data[:, 1:]
            bk.append({
                "name": bk_shape,
                "kernel_size": kernel_size,
                "kernels": kernels,
                "fm_img": fm_img,
                "fm_repo": fm_repo
            })

    return bk


def load_data(args, image_path):
    patch = data_utils.load_bw_img(image_path, size=64)
    img = file_utils.load_img(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, patch


def obj2tensor(shape, color, pos, group_name, group_count_conf):
    obj_tensor = torch.zeros(len(bk.obj_ohc))
    i = 0
    obj_tensor[i] = bk.color_large.index(color)  # color
    i += 1
    obj_tensor[i] = bk.shape_extend.index(shape)  # shape
    i += 1
    obj_tensor[i] = pos[0]  # x position
    i += 1
    obj_tensor[i] = pos[1]  # y position
    i += 1
    obj_tensor[i] = bk.group_name_extend.index(group_name)  # group label
    i += 1
    obj_tensor[i] = group_count_conf  # group confidence according to the count of objects
    return obj_tensor


def group2ocm(data, groups):
    """ return the object centric matrix of the groups """
    group_max_num = 25
    group_ocms = []
    positions = data_utils.data2positions(data)
    for g_i, group in enumerate(groups):
        # group
        group_ocm = torch.zeros(group_max_num, len(bk.obj_ohc))
        group_name = group["name"]
        group_obj_positions = group["onside"]
        group_count_conf = group["count_conf"]
        pos_count = 0
        for p_i, pos in enumerate(positions):
            if group_obj_positions[pos[1].item(), pos[0].item()] > 0:
                shape = data[p_i]["shape"]
                color = data[p_i]["color_name"]
                obj_tensor = obj2tensor(shape, color, pos, group_name, group_count_conf)
                group_ocm[pos_count] = obj_tensor
                pos_count += 1
        group_ocms.append(group_ocm)
    group_ocms = torch.stack(group_ocms, dim=0)
    return group_ocms


# def percept_gestalt_pattern(args, image_paths, out_path):
#     group_bk = load_bk(args, bk.group_name_solid)
#
#     for idx in range(min(2, len(image_paths))):
#         args.logger.debug(f"\n =========== Analysis Image {idx + 1}/{min(2, len(image_paths))} ==============")
#         file_name, file_extension = image_paths[idx].split(".")
#         data = file_utils.load_json(f"{file_name}.json")
#         img, img_resized = load_data(args, image_paths[idx])
#         # img_resized = 1-img_resized
#
#         group_saved_file = f"{out_path}/group_2nd.pt"
#         if os.path.exists(group_saved_file):
#             pixel_groups_2nd = torch.load(group_saved_file)
#         else:
#             output_file_prefix = str(out_path / f'img_{idx}')
#             pixel_groups = train_common_features.percept_feature_groups(args, group_bk, img, output_file_prefix)
#             pixel_groups_2nd = train_common_features.percept_objects(args, pixel_groups, group_bk, img,
#                                                                      output_file_prefix)
#             torch.save(pixel_groups_2nd, group_saved_file)
#
#         lang = alpha.alpha(args, pixel_groups_2nd)

#
# if __name__ == "__main__":
#     logger = args_utils.init_logger()
#     args = args_utils.get_args(logger)
#     image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)[:500]
#     out_path = config.output / args.exp_name
#     percept_gestalt_pattern(args, image_paths, out_path)
