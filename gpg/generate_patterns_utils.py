# Created by X at 13.02.25


import os
import json
from PIL import Image
import numpy as np
import torch

from src import bk
from src.utils import chart_utils
from kandinsky_generator.src.kp import KandinskyUniverse


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
                     "shape": 0,
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
        tensor = bk.gen_group_tensor(obj.x, obj.y, obj.size, 1, color[0], color[1], color[2], tri, sq, cir)
        tensors.append(tensor)
    if len(tensors) < max_length:
        tensors = tensors + [torch.zeros(len(tensors[0]))] * (max_length - len(tensors))
    else:
        raise ValueError
    tensors = torch.stack(tensors)
    return tensors


def save_img(path, task_name, principle, img_data, images):
    # save image
    for img_i in range(len(images)):
        Image.fromarray(images[img_i]).save(
            path / principle / f"{task_name}" / f"{img_i:06d}.png")
    images = chart_utils.hconcat_imgs(images)
    Image.fromarray(images).save(path / principle / f"{task_name}.png")
    # save data
    data = {"principle": principle, "img_data": img_data}
    with open(path / principle / f"{task_name}.json", 'w') as f:
        json.dump(data, f)


def encode_symbolic_features(args, kfs):
    images = []
    tensors = []
    img_data = []

    for kf in kfs:
        img = np.asarray(KandinskyUniverse.kandinskyFigureAsImage(kf, args.width)).copy()
        images.append(img)
        img_data.append(kf2data(kf, args.width))
        tensors.append(kf2tensor(kf, args.max_length))

    tensors = torch.stack(tensors)
    return tensors, images, img_data
