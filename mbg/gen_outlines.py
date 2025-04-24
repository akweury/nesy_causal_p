# import os
# import json
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from mbg import mbg_config as cfg
#
# output_dir = os.path.join(cfg.CONTOUR_DATASET_DIR, "real_extracted")
# os.makedirs(output_dir, exist_ok=True)
#
# shape_data = {i: [] for i in range(cfg.NUM_CLASSES)}  # 0–circle, 1–triangle, etc.
#
#
# def extract_contour(binary_img, num_points=100):
#     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         return None
#     contour = max(contours, key=len).squeeze()
#     if contour.ndim != 2 or len(contour) < num_points:
#         return None
#     idx = np.linspace(0, len(contour) - 1, num=num_points, dtype=np.int32)
#     contour = contour[idx].astype(np.float32)
#
#     # normalize to [-0.5, 0.5]
#     centroid = contour.mean(axis=0)
#     contour -= centroid
#     max_extent = np.linalg.norm(contour, axis=1).max()
#     contour /= (2 * max_extent + 1e-6)
#     return contour
#
#
# def match_gt_label(cx, cy, gt_objects):
#     min_dist, best_label = float("inf"), None
#     for obj in gt_objects:
#         gx = obj["x"] * obj["width"]
#         gy = obj["y"] * obj["width"]
#         d = np.hypot(cx - gx, cy - gy)
#         if d < min_dist:
#             min_dist = d
#             best_label = obj["shape"]
#     if min_dist>1:
#         return None
#     return best_label - 1
#
#
# task_dirs = [
#     d for d in os.listdir(cfg.ROOT_DATASET_DIR)
#     if os.path.isdir(os.path.join(cfg.ROOT_DATASET_DIR, d)) and "closure" not in d and "gestalt" not in d
# ]
#
# for task in tqdm(task_dirs, desc="Extracting contours"):
#     task_path = os.path.join(cfg.ROOT_DATASET_DIR, task)
#     gt_path = os.path.join(task_path, cfg.GT_EXTENSION)
#     if not os.path.exists(gt_path):
#         continue
#
#     with open(gt_path, "r") as f:
#         gt_data = json.load(f)["img_data"]
#
#     for img_name, objects in gt_data.items():
#         img_path = os.path.join(task_path, img_name + ".png")
#         img = cv2.imread(img_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         binary = np.where(gray != 211, 255, 0).astype(np.uint8)  # update: handle background correctly
#
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
#
#         for i in range(1, num_labels):
#             x, y, w, h, _ = stats[i]
#             if w == 0 or h == 0: continue
#             obj_mask = (labels == i).astype(np.uint8) * 255
#             cx, cy = centroids[i]
#             label = match_gt_label(cx, cy, objects)
#             if label is None: continue
#
#             contour = extract_contour(obj_mask)
#             if contour is not None:
#                 shape_data[label].append(contour)
#
# # ===== 保存 .npy =====
# for label, contours in shape_data.items():
#     save_path = os.path.join(output_dir, f"shape_{label}_contours.npy")
#     np.save(save_path, np.stack(contours))
#     print(f"Saved {len(contours)} contours to: {save_path}")
#
# # ===== 可视化每类轮廓 =====
# fig, axs = plt.subplots(1, cfg.NUM_CLASSES, figsize=(4 * cfg.NUM_CLASSES, 4))
# for label in range(cfg.NUM_CLASSES):
#     ax = axs[label]
#     ax.set_title(cfg.LABEL_NAMES[label])
#     ax.axis("off")
#     ax.set_aspect('equal')
#     for c in shape_data[label][:10]:
#         ax.plot(c[:, 0], -c[:, 1], alpha=0.8)
# plt.tight_layout()
# vis_path = os.path.join(output_dir, "visualized_contours.png")
# plt.savefig(vis_path)
# print(f"✅ Saved visualization to: {vis_path}")
# plt.show()
