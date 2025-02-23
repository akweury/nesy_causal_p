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
from gpg import same_numbers


def gen_patterns(pattern_name, dtype):
    so = 0.1
    overlap_patterns = ["feature_proximity_circle_two", "feature_proximity_circle_three",
                        "feature_proximity_circle_four",
                        "feature_closure_one_circle", "feature_closure_two_circles", "feature_closure_three_circles"]

    # proximity
    if pattern_name == "position_proximity_red_triangle_one":
        g = lambda so, truth: proximity_red_triangle(so, dtype, cluster_num=1)
    elif pattern_name == "position_proximity_red_triangle_two":
        g = lambda so, truth: proximity_red_triangle(so, dtype, cluster_num=2)
    elif pattern_name == "position_proximity_red_triangle_three":
        g = lambda so, truth: proximity_red_triangle(so, dtype, cluster_num=3)
    elif pattern_name == "feature_proximity_circle_two":
        so = 0.05
        g = lambda so, truth: proximity_circle(so, dtype, cluster_num=2)
    elif pattern_name == "feature_proximity_circle_three":
        so = 0.05
        g = lambda so, truth: proximity_circle(so, dtype, cluster_num=3)
    elif pattern_name == "feature_proximity_circle_four":
        so = 0.05
        g = lambda so, truth: proximity_circle(so, dtype, cluster_num=4)
    elif pattern_name == "grid_2":
        g = lambda so, truth: proximity_grid(so, dtype, cluster_num=2)
    elif pattern_name == "grid_3":
        g = lambda so, truth: proximity_grid(so, dtype, cluster_num=3)
    elif pattern_name == "grid_4":
        g = lambda so, truth: proximity_grid(so, dtype, cluster_num=4)
    elif pattern_name == "proximity_one_shape":
        g = lambda so, truth: proximity_one_shape(so, dtype)

    # similarity
    elif pattern_name == "fixed_number_two":
        # Example for three groups: yellow, blue, and red.
        g = lambda so, truth: same_numbers.generate_scene(so, dtype, g_num=2, grid_size=3, min_circles=3,
                                                          max_circles=5, diameter=0.08, image_size=(1, 1))
    elif pattern_name == "fixed_number_three":
        g = lambda so, truth: same_numbers.generate_scene(so, dtype, g_num=3, grid_size=3, min_circles=3,
                                                          max_circles=5, diameter=0.08, image_size=(1, 1))
    elif pattern_name == "fixed_number_four":
        g = lambda so, truth: same_numbers.generate_scene(so, dtype, g_num=4, grid_size=3, min_circles=3,
                                                          max_circles=5, diameter=0.08, image_size=(1, 1))
    elif pattern_name == "similarity_pacman_one":
        g = lambda so, truth: similarity_pacman(so, dtype, clu_num=1)
    elif pattern_name == "similarity_pacman_two":
        g = lambda so, truth: similarity_pacman(so, dtype, clu_num=2)
    elif pattern_name == "similarity_pacman_three":
        g = lambda so, truth: similarity_pacman(so, dtype, clu_num=3)

    # symbolic feature closure
    elif pattern_name == "tri_group_one":
        g = lambda so, truth: closure_big_triangle(so, dtype)
    elif pattern_name == "tri_group_two":
        g = lambda so, truth: closure_big_triangle(so, dtype) + closure_big_triangle(so, dtype)
    elif pattern_name == "tri_group_three":
        so = 0.05
        g = lambda so, truth: closure_big_triangle(so, dtype) + closure_big_triangle(so, dtype) + closure_big_triangle(
            so, dtype)

    elif pattern_name == "square_group_one":
        g = lambda so, truth: closure_big_square(so, dtype)
    elif pattern_name == "square_group_two":
        so = 0.05
        g = lambda so, truth: closure_big_square(so, dtype) + closure_big_square(so, dtype)
    elif pattern_name == "square_group_three":
        so = 0.05
        g = lambda so, truth: closure_big_square(so, dtype) + closure_big_square(so, dtype) + closure_big_square(
            so, dtype)
    elif pattern_name == "circle_group_one":
        g = lambda so, truth: closure_big_circle(so, dtype)
    elif pattern_name == "circle_group_two":
        so = 0.05
        g = lambda so, truth: closure_big_circle(so, dtype) + closure_big_circle(so, dtype)
    elif pattern_name == "circle_group_three":
        so = 0.05
        g = lambda so, truth: closure_big_circle(so, dtype) + closure_big_circle(so, dtype) + closure_big_circle(
            so, dtype)

    # feature closures
    elif pattern_name == "feature_closure_one_square":
        g = lambda so, truth: closure_classic_square(so, dtype)
    elif pattern_name == "feature_closure_two_squares":
        g = lambda so, truth: feature_closure_two_squares(so, dtype)
    elif pattern_name == "feature_closure_four_squares":
        g = lambda so, truth: feature_closure_four_squares(so, dtype)
    elif pattern_name == "feature_closure_four_squares":
        g = lambda so, truth: feature_closure_four_squares(so, dtype)

    elif pattern_name == "feature_closure_one_circle":
        g = lambda so, truth: feature_closure_circle_one(so, dtype)
    elif pattern_name == "feature_closure_two_circles":
        g = lambda so, truth: feature_closure_circle_two(so, dtype)
    elif pattern_name == "feature_closure_three_circles":
        g = lambda so, truth: feature_closure_circle_three(so, dtype)

    elif pattern_name == "feature_closure_one_triangle":
        g = lambda so, truth: feature_closure_triangle_one(so, dtype)
    elif pattern_name == "feature_closure_two_triangles":
        g = lambda so, truth: feature_closure_triangle_two(so, dtype)
    elif pattern_name == "feature_closure_three_triangles":
        g = lambda so, truth: feature_closure_triangle_three(so, dtype)

    # # symmetry
    # elif pattern_name == "symmetry_pattern":
    #     g = lambda so, truth: symmetry_pattern(so, dtype)

    # continuity
    elif pattern_name == "continuity_one_splits_two":
        g = lambda so, truth: continuity_one_splits_n(so, dtype)

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
            os.makedirs(data_path / task_principle, exist_ok=True)
            os.makedirs(data_path / task_principle / task_name, exist_ok=True)
            tensor_file = data_path / task_principle / task_name / f"{mode}_{task_name}.pt"
            # if os.path.exists(tensor_file):
            #     continue
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
        "similarity_shape"
    ]
    genGestaltTraining(args)
