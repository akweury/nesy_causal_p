# Created by jing at 17.06.24
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import cv2
from collections import Counter

import config
import train_common_features
from percept import perception
from utils import visual_utils, file_utils, args_utils, data_utils
from src.alpha import alpha
from src.alpha.fol import bk


def load_bk(args, bk_shapes):
    # load background knowledge
    bk = []
    for bk_shape in bk_shapes:
        kernels = torch.load(config.output / bk_shape / f"kernels.pt").to(args.device)
        fm_data = torch.load(config.output / bk_shape / f"fms.pt").to(args.device)
        fm_img = fm_data[:, 0:1]
        fm_repo = fm_data[:, 1:]
        bk.append({
            "name": bk_shape,
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
    obj_tensor[i] = bk.color.index(color)  # color
    i += 1
    obj_tensor[i] = bk.shape.index(shape)  # shape
    i += 1
    obj_tensor[i] = pos[0]  # x position
    i += 1
    obj_tensor[i] = pos[1]  # y position
    i += 1
    obj_tensor[i] = bk.group_name.index(group_name)  # group label
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


def main():
    args = args_utils.get_args()

    bk_shapes = {
        "data_diamond",
        "data_circle",
        "data_square",
        "data_triangle"
    }
    image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)[:500]

    # load background knowledge
    bk = load_bk(args, bk_shapes)

    clause_all = []
    for idx in tqdm(range(min(4, len(image_paths)))):
        file_name, file_extension = image_paths[idx].split(".")
        data = file_utils.load_json(f"{file_name}.json")

        img, obj_pos = load_data(args, image_paths[idx])
        groups = train_common_features.img2groups(args, bk, obj_pos, idx, img)
        group_tensors = group2ocm(data, groups)
        clauses = alpha.alpha(args, group_tensors)
        clause_all.append(clauses)

    clause_list = [c for cc in clause_all for c in cc]
    frequency = {}
    for item in clause_list:
        frequency[item] = frequency.get(item, 0) + 1
    most_frequency_value = max(frequency.values())
    most_frequent_clauses = [key for key, value in frequency.items() if value == most_frequency_value]
    return most_frequent_clauses


if __name__ == "__main__":
    main()
