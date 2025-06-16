# evaluate_patch_classifier_on_image.py

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import config
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


def evaluate_shapes(model, obj_images):
    device = next(model.parameters()).device

    patch_sets, positions, sizes = patch_preprocess.obj_imgs2patches(obj_images)
    predictions = []
    for o_i in range(len(patch_sets)):
        with torch.no_grad():
            logits = model(patch_sets[o_i][:, :, :2].unsqueeze(0).to(device))
            pred_label = logits.argmax(dim=1).item()
            predictions.append(pred_label)

    return predictions, patch_sets, positions, sizes


def evaluate_colors(obj_images):
    # obj_images = patch_preprocess.img_path2obj_images(img_path)

    colors = []
    bk_color = np.array([211, 211, 211], dtype=np.uint8)
    for o_i in range(len(obj_images)):
        img = obj_images[o_i]
        mask = np.any(img != bk_color, axis=-1)
        filtered_pixels = img[mask]
        color = filtered_pixels.mean(axis=0).astype(int)
        colors.append(color)
    return colors


def evaluate_image(model, img_path):
    obj_images = patch_preprocess.img_path2obj_images(img_path)
    labels, patches, positions, sizes = evaluate_shapes(model, obj_images)
    colors = evaluate_colors(obj_images)
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
    show_images_horizontally(obj_images)
    return objs


from matplotlib import pyplot as plt


def show_images_horizontally(image_list):
    # Ensure all images have the same height
    # Check height consistency
    heights = [img.shape[0] for img in image_list]
    if len(set(heights)) != 1:
        raise ValueError("All images must have the same height")

    height = heights[0]
    channels = image_list[0].shape[2] if len(image_list[0].shape) == 3 else 1

    # Create padding
    pad_shape = (height, 5, channels)
    padding = np.full(pad_shape, (255, 255, 255), dtype=np.uint8)

    # Interleave images with padding
    padded_images = []
    for i, img in enumerate(image_list):
        padded_images.append(img)
        if i != len(image_list) - 1:
            padded_images.append(padding)

    # Concatenate horizontally
    combined = np.concatenate(padded_images, axis=1)

    # If using OpenCV for display
    # cv2.imshow("Combined Image", combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Or using matplotlib (if images are RGB)
    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(combined)

    # Save to PDF
    image_pil.save(config.output / "object.pdf", "PDF", resolution=300.0)
