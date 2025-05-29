# evaluate_patch_classifier_on_image.py

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from src import bk
from mbg.object.patch_classifier_model import PatchClassifier
import mbg.mbg_config as param
from mbg import patch_preprocess


def load_model(device):
    model_path = param.OBJ_MODEL_SAVE_PATH
    model = PatchClassifier(
        num_patches=param.PATCHES_PER_SET,
        patch_len=param.POINTS_PER_PATCH,
        num_classes=len(param.LABEL_NAMES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


def evaluate_shapes(model, data):
    device = next(model.parameters()).device
    patch_sets, _, positions, sizes = patch_preprocess.img_path2patches_and_labels(data["image_path"][0],
                                                                      data["symbolic_data"])
    predictions = []
    for o_i in range(len(patch_sets)):
        with torch.no_grad():
            logits = model(patch_sets[o_i][:, :, :2].unsqueeze(0).to(device))
            pred_label = logits.argmax(dim=1).item()
            predictions.append(pred_label)

    return predictions, patch_sets, positions, sizes


def evaluate_colors(data):
    colors = []
    obj_images = patch_preprocess.img_path2obj_images(data["image_path"][0])
    bk_color = np.array([211, 211, 211], dtype=np.uint8)
    for o_i in range(len(obj_images)):
        img = obj_images[o_i]
        mask = np.any(img != bk_color, axis=-1)
        filtered_pixels = img[mask]
        color = filtered_pixels.mean(axis=0).astype(int)
        colors.append(color)
    return colors


def evaluate_image(model, data):
    labels, patches, positions, sizes = evaluate_shapes(model, data)
    colors = evaluate_colors(data)
    objs = []
    for i in range(len(labels)):
        shape_one_hot = torch.zeros(len(bk.bk_shapes), dtype=torch.float32)
        shape_one_hot[labels[i] + 1] = 1.0
        obj = {
            "id": i,
            "s": {"shape": shape_one_hot,
                  "color": [int(colors[i][0]), int(colors[i][1]), int(colors[i][2])],
                  "x": positions[i][0],
                  "y": positions[i][1],
                  "w": sizes[i][0],
                  "h": sizes[i][1],
                  },
            "h": patches[i]
        }
        objs.append(obj)

    return objs
