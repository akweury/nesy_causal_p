# Created by MacBook Pro at 20.04.25

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import config
import math

IMAGE_SIZE = 256
SHAPE_COLOR = (0, 0, 0)
BG_COLOR = (211, 211, 211)
SAVE_DIR = config.kp_gestalt_dataset / "pam_synthetic"
NUM_IMAGES_PER_CLASS = 1000
SHAPE_SIZE = 30

def draw_rectangle(image, draw, inverse_mask=False):
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    group_scale = random.uniform(0.85, 1.2)
    aspect_ratio = random.uniform(0.5, 2.0)  # width / height

    base = 60 * group_scale
    w = base * aspect_ratio
    h = base

    shape_scales = [random.uniform(0.8, 1.2) for _ in range(4)]

    # Rectangle corners
    corners = [
        (cx - w, cy - h),  # Top Left
        (cx + w, cy - h),  # Top Right
        (cx + w, cy + h),  # Bottom Right
        (cx - w, cy + h),  # Bottom Left
    ]
    start_angles = [90, 180, 270, 0]

    for i, (x, y) in enumerate(corners):
        r = SHAPE_SIZE * shape_scales[i]
        draw.pieslice([x - r, y - r, x + r, y + r], start=0, end=360, fill=SHAPE_COLOR)

    # ==== 生成遮挡区域 ====
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0 if not inverse_mask else 255)
    draw_mask = ImageDraw.Draw(mask)

    # 使用略小矩形区域遮挡或保留（模拟 closure）
    margin_w = (w + SHAPE_SIZE) * random.uniform(0.65, 0.8)
    margin_h = (h + SHAPE_SIZE) * random.uniform(0.65, 0.8)
    rect_box = [cx - margin_w, cy - margin_h, cx + margin_w, cy + margin_h]

    fill_value = 255 if not inverse_mask else 0
    draw_mask.rectangle(rect_box, fill=fill_value)

    background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (211, 211, 211))
    composed = Image.composite(image, background, mask)
    image.paste(composed)


def draw_ellipse(image, draw, inverse_mask=False):
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    group_scale = random.uniform(0.85, 1.2)
    square_scale = [random.uniform(0.8, 1.2) for _ in range(4)]

    offset = 60 * group_scale
    base_square_size = SHAPE_SIZE

    centers = [
        (cx - offset, cy - offset),
        (cx + offset, cy - offset),
        (cx + offset, cy + offset),
        (cx - offset, cy + offset),
    ]

    for i, (x, y) in enumerate(centers):
        r = base_square_size * square_scale[i]
        draw.rectangle([x - r, y - r, x + r, y + r], fill=SHAPE_COLOR)

    # 构建遮罩层
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0 if not inverse_mask else 255)
    draw_mask = ImageDraw.Draw(mask)

    ellipse_margin = 30 * group_scale
    bbox = [ellipse_margin, ellipse_margin, IMAGE_SIZE - ellipse_margin, IMAGE_SIZE - ellipse_margin]

    # mask 区域：正向为白色椭圆，反向为黑色椭圆
    fill_value = 255 if not inverse_mask else 0
    draw_mask.ellipse(bbox, fill=fill_value)

    # 合成遮挡效果
    background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (211, 211, 211))
    composed = Image.composite(image, background, mask)
    image.paste(composed)

def draw_triangle(image, draw, inverse_mask=False):
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    group_scale = random.uniform(0.85, 1.2)
    shape_scales = [random.uniform(0.8, 1.2) for _ in range(3)]

    size = 60 * group_scale
    r = size

    # 基于中心点计算等边三角形三个顶点
    p1 = (cx, cy - r)
    dx = r * math.sin(math.radians(60))
    dy = r * math.cos(math.radians(60))
    p2 = (cx + dx, cy + dy)
    p3 = (cx - dx, cy + dy)
    corners = [p1, p2, p3]

    start_angles = [180, 300, 60]

    for i, (x, y) in enumerate(corners):
        rr = SHAPE_SIZE * shape_scales[i]
        draw.pieslice([x - rr, y - rr, x + rr, y + rr], start=0, end=360, fill=SHAPE_COLOR)

    # ==== 添加遮挡区域 ====
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0 if not inverse_mask else 255)
    draw_mask = ImageDraw.Draw(mask)

    # 定义包含整个三角形的大三角形包围框（略大于实际顶点范围）
    margin = size + SHAPE_SIZE
    draw_mask.polygon([
        (cx, cy - margin),
        (cx + margin * math.sin(math.radians(60)), cy + margin * math.cos(math.radians(60))),
        (cx - margin * math.sin(math.radians(60)), cy + margin * math.cos(math.radians(60))),
    ], fill=255 if not inverse_mask else 0)

    background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (211, 211, 211))
    composed = Image.composite(image, background, mask)
    image.paste(composed)

def draw_symmetry(draw, center):
    x, y = center
    draw_triangle(draw, (x - 40, y))
    draw_triangle(draw, (x + 40, y))


def draw_random(image, draw):
    shape_func = random.choice([draw_rectangle])
    shape_func(image, draw)


def generate_image(label_name):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(image)

    inverse_mask = random.choice([True, False])

    if label_name == "triangle":
        draw_triangle(image, draw, inverse_mask=inverse_mask)
    elif label_name == "rectangle":
        draw_rectangle(image, draw, inverse_mask=inverse_mask)
    elif label_name == "ellipse":
        draw_ellipse(image, draw, inverse_mask=inverse_mask)
    return image


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    gt_all = {}

    for label in ["triangle", "rectangle", "ellipse"]:
        folder = SAVE_DIR / label
        folder.mkdir(exist_ok=True)
        for i in range(NUM_IMAGES_PER_CLASS):
            img = generate_image(label)
            name = f"{label}_{i:05d}"
            img.save(folder / f"{name}.png")
            gt_all[name] = label

    with open(SAVE_DIR / "gt.json", "w") as f:
        json.dump(gt_all, f, indent=2)


if __name__ == "__main__":
    main()
