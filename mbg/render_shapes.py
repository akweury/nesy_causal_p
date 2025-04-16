# Created by MacBook Pro at 15.04.25

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import config

# Rendering settings
img_size = 64
line_thickness = 2
render_dir = config.mb_outlines
os.makedirs(render_dir, exist_ok=True)


def render_shape_to_image(points, img_size=64, thickness=2):
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255  # white background
    pts = (points * (img_size * 0.8) + img_size / 2).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=0, thickness=thickness)
    return canvas


output_dir = config.mb_outlines
# Load shapes and render
rendered_counts = {}
for shape_name in ["triangle", "rectangle", "ellipse"]:
    shape_array = np.load(os.path.join(output_dir, f"{shape_name}_shapes.npy"), allow_pickle=True)
    shape_dir = os.path.join(render_dir, shape_name)
    os.makedirs(shape_dir, exist_ok=True)

    count = 0
    for i, pts in enumerate(shape_array):
        img = render_shape_to_image(pts, img_size=img_size, thickness=line_thickness)
        path = os.path.join(shape_dir, f"{shape_name}_{i:04d}.png")
        cv2.imwrite(path, img)
        count += 1
    rendered_counts[shape_name] = count
