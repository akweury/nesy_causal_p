# Created by jing at 17.06.24
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import cv2

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


def obj2tensor(shape, color, pos):
    obj_tensor = torch.zeros(len(bk.obj_ohc))
    i = 0
    obj_tensor[i] = bk.color.index(color)
    i += 1
    obj_tensor[i] = bk.shape.index(shape)
    i += 1
    obj_tensor[i] = pos[0]
    i += 1
    obj_tensor[i] = pos[1]
    return obj_tensor


def group2ocm(data, groups):
    """ return the object centric matrix of the groups """
    group_ocms = {}
    positions = data_utils.data2positions(data)
    for group in groups:
        # group to
        group_name = group["name"]
        group_obj_positions = group["onside"]

        ocm = []
        for p_i, pos in enumerate(positions):
            if group_obj_positions[pos[1].item(), pos[0].item()] > 0:
                shape = data[p_i]["shape"]
                color = data[p_i]["color_name"]
                obj_tensor = obj2tensor(shape, color, pos)
                ocm.append(obj_tensor)
        group_ocms[group_name] = torch.stack(ocm)
    return group_ocms


def main():
    args = args_utils.get_args()
    idx = 0
    bk_shapes = {
        "data_circle",
        "data_square",
        "data_triangle"
    }
    image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)

    # load background knowledge
    bk = load_bk(args, bk_shapes)
    file_name, file_extension = image_paths[idx].split(".")
    data = file_utils.load_json(f"{file_name}.json")

    img, obj_pos = load_data(args, image_paths[idx])
    groups = train_common_features.img2groups(args, bk, obj_pos, idx, img)
    group_tensors = group2ocm(data, groups)
    print(f"{idx}: {len(groups)}")


if __name__ == "__main__":
    main()