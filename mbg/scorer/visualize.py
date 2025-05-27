# Created by MacBook Pro at 24.04.25


import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from mbg.scorer.proximity_pair_dataset import ProximityPairDataset  # 或使用你的实际路径
from mbg.scorer import scorer_config
import config

SAVE_DIR = config.mb_outlines / "proximity_pair_vis"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def render_patch(patch_a,patch_b, image_size=1024):
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    pts = np.concatenate([patch_a,patch_b]).astype(int)
    pts = np.clip(pts, 0, image_size - 1)
    pts = pts.reshape(-1, 2)
    for i in range(len(pts)):
        cv2.circle(img, tuple(pts[i]), radius=2, color=(0, 0, 0), thickness=-1)
    return img

def draw_pair_image(patch_i, patch_j, label, idx):
    img = render_patch(patch_i, patch_j)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), f"Label: {label}", fill=(255, 0, 0))

    pil_img.save(SAVE_DIR / f"pair_{idx:04d}_label{label}.png")

def main():
    dataset = ProximityPairDataset(scorer_config.proximity_path)

    for idx in tqdm(range(min(len(dataset), 200))):  # 可调整数量
        c_i, c_j, label = dataset[idx]
        draw_pair_image(c_i.numpy(), c_j.numpy(), int(label.item()), idx)

if __name__ == "__main__":
    main()