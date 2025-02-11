# Created by X at 03.02.25

from PIL import Image, ImageDraw
import numpy as np
import os
import json
from pathlib import Path
import torch
from fontTools.svgLib.path import shapes

import config
from tqdm import tqdm
from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.ShapeOnShapes import ShapeOnShape
from src import bk
from src.utils import chart_utils, args_utils
from src.percept import gestalt_group
from kandinsky_generator import gestalt_patterns

u = KandinskyUniverse.SimpleUniverse()

def kf2data(kf, width):
    data = []
    for obj in kf:
        data.append({"x": obj.x,
                     "y": obj.y,
                     "size": obj.size,
                     "color_name": bk.color_large.index(obj.color),
                     "color_r": bk.color_matplotlib[obj.color][0],
                     "color_g": bk.color_matplotlib[obj.color][1],
                     "color_b": bk.color_matplotlib[obj.color][2],
                     "shape": bk.bk_shapes.index(obj.shape),
                     "width": width
                     })
    return data


def kf2tensor(kf, max_length):
    tensors = []
    for obj in kf:
        color = np.array((bk.color_matplotlib[obj.color])) / 255
        tri = 1 if obj.shape == "triangle" else 0
        sq = 1 if obj.shape == "square" else 0
        cir = 1 if obj.shape == "circle" else 0
        tensor = gestalt_group.gen_group_tensor(obj.x, obj.y, obj.size, 1,
                                                color[0], color[1], color[2], tri, sq, cir)
        tensors.append(tensor)
    if len(tensors) < max_length:
        tensors = tensors + [torch.zeros(len(tensors[0]))] * (max_length - len(tensors))
    else:
        raise ValueError
    tensors = torch.stack(tensors)
    return tensors


def get_task_names(principle):
    if principle == "good_figure":
        task_names = ["good_figure_two_groups",
                      "good_figure_three_groups",
                      "good_figure_always_three"]
    elif principle == "proximity":
        task_names = ["proximity_red_triangle"]
    # elif principle == "similarity_shape":
    #     task_names = ["similarity_triangle_circle"]
    # elif principle == "similarity_color":
    #     task_names = ["similarity_two_pairs"]
    # elif principle == "closure":
    #     task_names = ["gestalt_triangle",
    #                   "gestalt_square",
    #                   "gestalt_circle",
    #                   "tri_group",
    #                   "square_group",
    #                   "triangle_square"]
    # elif principle == "continuity":
    #     task_names = ["continuity_one_splits_two",
    #                   "continuity_one_splits_three"]
    # elif principle == "symmetry":
    #     task_names = ["symmetry_pattern"]
    else:
        raise ValueError
    return task_names

def gen_and_save(path, args, mode):
    width = args.width
    example_num = args.example_num
    max_length = 64
    all_tensors = {"positive": [], "negative": []}
    task_counter = 0
    principles = bk.gestalt_principles
    for principle in ["proximity"]:
        task_names = get_task_names(principle)
        for t_i, task_name in enumerate(task_names):
            print("Generating training patterns for task {}".format(task_name))
            img_data = []
            kfs = []
            for dtype in [True, False]:
                for example_i in range(example_num):
                    kfs.append(gestalt_patterns.gen_patterns(task_name, dtype))  # pattern generation
            tensors = []
            images = []
            for kf in kfs:
                img = np.asarray(KandinskyUniverse.kandinskyFigureAsImage(kf, width)).copy()
                images.append(img)
                img_data.append(kf2data(kf, width))
                tensors.append(kf2tensor(kf, max_length))
            tensors = torch.stack(tensors)

            # save image
            os.makedirs(path / ".." / f"{mode}_all", exist_ok=True)
            os.makedirs(path / ".." / f"{mode}_all" / f"{task_counter}", exist_ok=True)
            for img_i in range(len(images)):
                Image.fromarray(images[img_i]).save(
                    path / ".." / f"{mode}_all" / f"{task_counter}" / f"sep_{task_counter:06d}_{img_i}.png")
            images = chart_utils.hconcat_imgs(images)
            Image.fromarray(images).save(path / f"{task_counter:06d}.png")
            # save data
            data = {"principle": principle,
                    "img_data": img_data}
            with open(path / f"{task_counter:06d}.json", 'w') as f:
                json.dump(data, f)

            # save tensor
            all_tensors["positive"].append(tensors[:3])
            all_tensors["negative"].append(tensors[3:])

            task_counter += 1
    return all_tensors


def genGestaltTraining(args):
    base_path = config.kp_gestalt_dataset_all
    os.makedirs(base_path, exist_ok=True)
    for mode in ['train', "test"]:
        data_path = base_path / mode
        os.makedirs(data_path, exist_ok=True)
        tensor_file = data_path / f"{mode}.pt"
        if os.path.exists(tensor_file):
            continue
        tensors = gen_and_save(data_path, args, mode)
        torch.save(tensors, tensor_file)
    print("")


if __name__ == "__main__":
    args = args_utils.get_args()
    args.width = 1024
    args.example_num = 10
    genGestaltTraining(args)