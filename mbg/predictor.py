# Created by MacBook Pro at 16.04.25


import numpy as np
import cv2
import torch
import faiss
from torchvision import transforms
import config
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from PIL import Image
from src import bk
import mbg.mbg_config as param
from mbg.object import dataset_patch
from mbg.contour_utils import extract_object_contours, generate_patch_set_from_contour

# 图像预处理（如需灰度 -> RGB 转换可加 transform）
transform = transforms.Compose([
    transforms.ToTensor()  # 保持为 Tensor 格式，可选
])


# 轮廓编码函数
def sample_contour_patches(contour, num_patches=20, patch_size=5):
    contour = np.asarray(contour)
    N = len(contour)
    if N < patch_size:
        return []
    indices = np.linspace(0, N - patch_size, num_patches, dtype=int)
    return [contour[i:i + patch_size] for i in indices]


def encode_patch(patch):
    patch = patch - np.mean(patch, axis=0)
    norm = np.linalg.norm(patch)
    return (patch / norm).flatten() if norm > 0 else patch.flatten()


def encode_shape_outline(contour, num_patches=20, patch_size=5):
    patches = sample_contour_patches(contour, num_patches, patch_size)
    if len(patches) == 0:
        return None
    descriptors = [encode_patch(p) for p in patches]
    return np.mean(descriptors, axis=0)


def extract_normalized_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=len).squeeze()
    if contour.ndim != 2 or contour.shape[0] < 3:
        return None
    centroid = np.mean(contour, axis=0)
    contour = contour - centroid
    max_dist = np.max(np.linalg.norm(contour, axis=1))
    return contour / (2 * max_dist + 1e-6)


def extract_dominant_color_rgb(mask, rgb_img):
    masked_pixels = rgb_img[mask > 0]
    if masked_pixels.size == 0:
        return {"r": 0, "g": 0, "b": 0, "label": 0}
    avg_color = np.mean(masked_pixels, axis=0).astype(int)
    return {
        "r": int(avg_color[0]),
        "g": int(avg_color[1]),
        "b": int(avg_color[2]),
        "label": 0  # TODO: 可选颜色分类器
    }


def visual_patch_sets(patch_set):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(len(patch_set)):
        ax = axes[i]
        p_set = patch_set[i].squeeze(0).numpy()  # shape: [set_size, patch_size, 2]

        for patch in p_set:
            ax.plot(patch[:, 0], patch[:, 1], marker='o')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # 清理 ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

        # ax.set_title(f"Label: {labels[i].item()}", fontsize=10)

    # 调整子图间距
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()

def visual_img(img_tensor):
    plt.imshow(img_tensor)
    plt.show()

# Predictor
@torch.no_grad()
def predictor(image_tensor, model, device='cpu', shape_names=["triangle", "rectangle", "ellipse"]):
    """
    image_tensor: torch.Tensor of shape [3, H, W], RGB image, values in [0,1]
    patch_classifier: trained PatchSetClassifier
    return: list of dicts per object with shape prediction and other attributes
    """
    assert image_tensor.shape[0] == 3
    visual_img(image_tensor.permute(1, 2, 0))
    H, W = image_tensor.shape[1:]
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    binary = np.where(gray == 211, 0, 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results = []

    for i in range(1, num_labels):  # skip background

        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        mask = (labels == i).astype(np.uint8)

        contours = extract_object_contours(Image.fromarray(mask.astype(np.uint8), mode="L"))
        # if len(contours) < 3:
        #     continue
        selected = random.sample(contours, k=min(3, len(contours)))
        merged = np.concatenate(selected, axis=0)
        patch_set = generate_patch_set_from_contour(merged,  num_patches=param.PATCHES_PER_SET,
                                                    points_per_patch=param.POINTS_PER_PATCH)

        logits = model(torch.from_numpy(patch_set).unsqueeze(0))

        pred_label = logits.argmax(1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_label].item()
        shape_pred = param.LABEL_NAMES[pred_label]
        if shape_pred == "ellipse":
            shape_pred = "circle"
        shape_pred = bk.bk_shapes.index(shape_pred)
        color = image_np[mask > 0].mean(axis=0).astype(int)
        s_obj = {
            "x": float(cx / W),
            "y": float(cy / H),
            "size": float(area / (H * W)),
            "color_r": int(color[0]),
            "color_g": int(color[1]),
            "color_b": int(color[2]),
            "shape": shape_pred
        }
        plt.imshow(mask)
        plt.show()
        results.append(s_obj)

    return results