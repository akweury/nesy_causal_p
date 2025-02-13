# Created by X at 13.02.25

import os
import torch
import json
from PIL import Image, ImageDraw
import numpy as np

from src.utils import chart_utils

import config
from src.utils import args_utils
from gpg import task_settings
from gpg.generate_patterns_utils import *
from gpg.task_settings import *


def gen_patterns(pattern_name, dtype):
    so = 0.1
    overlap_patterns = ["gestalt_triangle", "gestalt_circle", "gestalt_square", "closure_square_red_yellow",
                        "closure_four_squares"]

    # proximity
    if pattern_name == "proximity_red_triangle":
        g = lambda so, truth: proximity_red_triangle(so, dtype)
    elif pattern_name == "proximity_one_shape":
        g = lambda so, truth: proximity_one_shape(so, dtype)

    # feature closures
    elif pattern_name == "gestalt_square":
        g = lambda so, truth: closure_classic_square(so, dtype)
    elif pattern_name == "closure_square_red_yellow":
        g = lambda so, truth: closure_square_red_yellow(so, dtype)
    elif pattern_name == "closure_four_squares":
        g = lambda so, truth: closure_four_squares(so, dtype)

    elif pattern_name == "gestalt_triangle_and_noise":
        g = lambda so, truth: closure_classic_triangle_and_noise(so, dtype)
    # elif pattern_name == "gestalt_circle":
    #     g = lambda so, truth: closure_classic_circle(so, dtype)
    # elif pattern_name == "gestalt_triangle":
    #     g = lambda so, truth: closure_classic_triangle(so, dtype)

    # position closures
    # elif pattern_name == "square_group":
    #     so = 0.1
    #     g = lambda so, truth: closure_big_square(so, dtype)
    # elif pattern_name == "tri_group":
    #     so = 0.1
    #     g = lambda so, truth: closure_big_triangle(so, dtype)
    # elif pattern_name == "triangle_square":
    #     so = 0.1
    #     g = lambda so, truth: closure_big_square(so, dtype) + closure_big_triangle(so, dtype)


    # similarity
    # elif pattern_name == "fixed_number":
    #     g = lambda so, truth: generate_random_clustered_circles(so, dtype)
    # elif pattern_name == "similarity_triangle_circle":
    #     g = lambda so, truth: similarity_two_colors(so, dtype)
    #
    # # symmetry
    # elif pattern_name == "symmetry_pattern":
    #     g = lambda so, truth: symmetry_pattern(so, dtype)

    # continuity
    # elif pattern_name == "continuity_one_splits_two":
    #     g = lambda so, truth: continuity_one_splits_n(so, dtype, n=2)
    # elif pattern_name == "continuity_one_splits_three":
    #     g = lambda so, truth: continuity_one_splits_n(so, dtype, n=3)


    else:
        raise ValueError
    kf = g(so, dtype)
    t = 0
    tt = 0
    max_try = 1000
    if pattern_name not in overlap_patterns:
        while (KandinskyUniverse.overlaps(kf) or KandinskyUniverse.overflow(kf)) and (t < max_try):
            kf = g(so, dtype)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
    return kf


def gen_and_save(args, path, task_name, task_principle):
    all_tensors = {"positive": [], "negative": []}
    print("Generating training patterns for task {}".format(task_name))
    kfs = []
    for dtype in [True, False]:
        for example_i in range(args.example_num):
            kfs.append(gen_patterns(task_name, dtype))  # pattern generation
    # encode symbolic object tensors
    tensors, images, img_data = encode_symbolic_features(args, kfs)
    save_img(path, task_name, task_principle, img_data, images)
    # save tensor
    all_tensors["positive"].append(tensors[:3])
    all_tensors["negative"].append(tensors[3:])

    return all_tensors


def genGestaltTraining(args):
    base_path = config.kp_gestalt_dataset

    task_names = task_settings.get_task_names()
    for (task_name, task_principle) in task_names.items():
        for mode in ['train', "test"]:
            data_path = base_path / mode
            os.makedirs(data_path, exist_ok=True)
            tensor_file = data_path / f"{mode}.pt"
            if os.path.exists(tensor_file):
                continue
            tensors = gen_and_save(args, data_path, task_name, task_principle)
            torch.save(tensors, tensor_file)
    print("all patterns generated")


if __name__ == "__main__":
    args = args_utils.get_args()
    args.width = 1024
    args.max_length = 64
    args.example_num = 10
    args.gestalt_principles = [
        'closure',
        "proximity",
        "symmetry",
        "similarity_shape",
        # "similarity_color",
        # "continuity",
    ]
    genGestaltTraining(args)
