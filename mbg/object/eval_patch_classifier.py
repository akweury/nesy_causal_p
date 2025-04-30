# evaluate_patch_classifier_on_image.py

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

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
    device = torch.device(param.DEVICE)
    patch_sets, labels = patch_preprocess.img_path2patches_and_labels(data["image_path"][0],
                                                                      data["symbolic_data"]["objects"])
    predictions = []
    positions = []
    sizes = []
    for o_i in range(len(patch_sets)):
        x, y, w, h = patch_sets[o_i][0][1]
        positions.append([x, y])
        sizes.append([w, h])

        patch_tensor = patch_sets[o_i][0][0].unsqueeze(0).to(device)  # (1, P, L, 2)
        with torch.no_grad():
            logits = model(patch_tensor)
            pred_label = logits.argmax(dim=1).item()
            predictions.append(pred_label)

    return positions, labels, sizes, patch_sets


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
    positions, labels, sizes, patches = evaluate_shapes(model, data)
    colors = evaluate_colors(data)
    objs = []
    for i in range(len(positions)):
        obj = {
            "id": i,
            "s": {"shape":  labels[i],
                  "color": [int(colors[i][0]),int(colors[i][1]),int(colors[i][2])],
                  "x":float(positions[i][0]),
                  "y":float(positions[i][1]),
                  "w":sizes[i][0],
                  "h":sizes[i][1],
                  },
            "h": patches[i]
        }
        objs.append(obj)

    return objs
