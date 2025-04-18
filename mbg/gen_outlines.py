import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import config

# 路径配置
root_dir = config.kp_gestalt_dataset / "train"
output_dir = config.mb_outlines
os.makedirs(output_dir, exist_ok=True)


def extract_contour_points(image, num_points=100):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=len).squeeze()
    if contour.ndim != 2 or len(contour) < num_points:
        return None
    indices = np.linspace(0, len(contour) - 1, num=num_points, dtype=np.int32)
    contour = contour[indices].astype(np.float32)
    centroid = contour.mean(axis=0)
    contour -= centroid
    max_extent = np.linalg.norm(contour, axis=1).max()
    contour = contour / (2 * max_extent + 1e-6)
    return contour


def match_label(centroid_xy, gt_objects, img_width):
    min_dist = float("inf")
    matched_label = None
    matched_idx = -1
    for idx, obj in enumerate(gt_objects):
        gt_x = obj["x"] * obj["width"]
        gt_y = obj["y"] * obj["width"]
        dist = np.sqrt((centroid_xy[0] - gt_x) ** 2 + (centroid_xy[1] - gt_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            matched_label = obj["shape"]
            matched_idx = idx
    return matched_label, matched_idx


# 初始化每类 shape 的轮廓列表
shape_data = {1: [], 2: [], 3: []}

# 处理每个 task
task_dirs = [d for d in os.listdir(root_dir) if
             os.path.isdir(os.path.join(root_dir, d)) and "closure" not in d and "gestalt" not in d]

for task in tqdm(task_dirs, desc="Processing tasks"):
    task_path = os.path.join(root_dir, task)
    gt_path = os.path.join(task_path, "gt.json")
    if not os.path.exists(gt_path):
        continue
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    for img_name, obj_list in gt_data["img_data"].items():
        img_path = os.path.join(task_path, img_name + ".png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = np.where(gray < 210, 255, 0).astype(np.uint8)
        # plt.imshow(binary)
        # plt.show()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        matched = set()
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            if w == 0 or h == 0:
                continue
            patch = binary[y-2:y + h+2, x-2:x + w+2]
            # plt.imshow(patch)
            # plt.show()
            contour = extract_contour_points(patch)
            if contour is None:
                continue

            cx, cy = centroids[i]
            label, gt_idx = match_label((cx, cy), obj_list, obj_list[0]["width"])
            if label is not None and gt_idx not in matched:
                matched.add(gt_idx)
                shape_data[label].append(contour)

# 保存
for shape_id, contours in shape_data.items():
    np.save(os.path.join(output_dir, f"shape_{shape_id}_contours.npy"), np.stack(contours))

# 可视化前 10 个
fig, axs = plt.subplots(1, 3, figsize=(16, 16))
titles = ['Triangle', 'Rectangle', 'Ellipse']
for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    ax.axis('equal')
    ax.axis('off')
    for c in shape_data[i+1][:10]:
        ax.plot(c[:, 0], c[:, 1])
plt.tight_layout()
plt.show()
