# Created by jing at 17.06.24
import logging

from tqdm import tqdm
import torch
import cv2
import os

import config
import train_common_features
from utils import file_utils, args_utils, data_utils
from src.alpha import alpha
from src.alpha.fol import bk
from src import llama_call


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
    file_name, file_extension = image_path.split(".")
    data = file_utils.load_json(f"{file_name}.json")
    patch = data_utils.oco2patch(data).unsqueeze(0).to(args.device)
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


# def group2ocm(data, groups):
#     """ return the object centric matrix of the groups """
#     group_max_num = 25
#     group_ocms = []
#     positions = data_utils.data2positions(data)
#     for g_i, group in enumerate(groups):
#         # group
#         group_ocm = torch.zeros(group_max_num, len(bk.obj_ohc))
#         group_name = group["name"]
#         group_obj_positions = group["onside"]
#         group_count_conf = group["count_conf"]
#         pos_count = 0
#         for p_i, pos in enumerate(positions):
#             if group_obj_positions[pos[1].item(), pos[0].item()] > 0:
#                 shape = data[p_i]["shape"]
#                 color = data[p_i]["color_name"]
#                 obj_tensor = obj2tensor(shape, color, pos, group_name, group_count_conf)
#                 group_ocm[pos_count] = obj_tensor
#                 pos_count += 1
#         group_ocms.append(group_ocm)
#     group_ocms = torch.stack(group_ocms, dim=0)
#     return group_ocms

def has_found_global_group(groups):
    has_found = False
    for group in groups:
        if group.onside_coverage > 0.9:
            has_found = True
    return has_found


def percept_gestalt_groups(args, group_bk, img, output_file_prefix):
    group_file = output_file_prefix + f"_feature_groups.pt"
    if os.path.exists(group_file):
        groups = torch.load(group_file)
    else:
        groups = train_common_features.percept_feature_groups(args, group_bk, img, output_file_prefix)
        torch.save(groups, group_file)


    global_group_file = output_file_prefix + f"_global_groups.pt"
    if os.path.exists(global_group_file):
        groups = torch.load(global_group_file)
    else:
        groups = train_common_features.percept_object_groups(args, groups, group_bk, img, output_file_prefix)
        torch.save(groups, global_group_file)

    groups.ocm[0, -1] = 1
    return [groups]


def train_clauses(args, image_paths, out_path):
    save_file = config.output / args.exp_name / f'learned_lang.pkl'
    if os.path.exists(save_file):
        lang_data = torch.load(save_file)
        if lang_data is not None:
            lang = alpha.load_lang(args, lang_data)
            if lang is not None:
                return lang

    # load background knowledge
    lang = None
    all_clauses = []
    group_bk = load_bk(args, bk.group_name_solid)
    for idx in range(min(2, len(image_paths))):
        args.logger.debug(f"\n =========== Analysis Image {idx + 1}/{min(2, len(image_paths))} ==============")
        file_name, file_extension = image_paths[idx].split(".")
        data = file_utils.load_json(f"{file_name}.json")
        img, img_resized = load_data(args, image_paths[idx])

        output_file_prefix = str(out_path / f'img_{idx}')
        groups = percept_gestalt_groups(args, group_bk, img, output_file_prefix)
        lang = alpha.alpha(args, groups)
        # rename predicates
        # update clauses
        all_clauses += lang.clauses
    # remove the less occurred clauses

    frequency = {}
    for item in all_clauses:
        frequency[item] = frequency.get(item, 0) + 1
    most_frequency_value = max(frequency.values())
    most_frequent_clauses = [key for key, value in frequency.items() if value == most_frequency_value]
    lang.clauses = most_frequent_clauses

    # convert machine clause to final clause
    merged_clauses = lang.rewrite_clauses(args)
    llm_clauses, name_dict = llama_call.rewrite_clauses(args, merged_clauses)
    lang.llm_clauses = llm_clauses

    lang_dict = {
        "atoms": lang.atoms,
        "clauses": lang.clauses,
        "consts": lang.consts,
        "preds": lang.predicates,
        "g_num": lang.group_variable_num,
        "attrs": lang.attrs,
        "llm_clauses": llm_clauses,
        "name_dict": name_dict
    }
    torch.save(lang_dict, save_file)
    return lang


if __name__ == "__main__":
    logger = args_utils.init_logger()
    args = args_utils.get_args(logger)
    image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)[:500]
    out_path = config.output / args.exp_name
    train_clauses(args, image_paths, out_path)
