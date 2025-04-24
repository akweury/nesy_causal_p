# # Created by MacBook Pro at 18.04.25
# import os
# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import cv2
# from mbg import mbg_config as cfg
# from tqdm import tqdm
#
#
# def generate_random_patch_set(contour, patch_size=16, num_patches=6):
#     total_len = len(contour)
#     assert total_len >= patch_size, "Contour too short for patching"
#     patch_set = []
#     selected = set()
#     attempts = 0
#     while len(patch_set) < num_patches and attempts < 10 * num_patches:
#         start = np.random.randint(0, total_len - patch_size)
#         if any(abs(start - s) < patch_size for s in selected):
#             attempts += 1
#             continue
#         patch = contour[start:start + patch_size]
#         patch = patch.T
#         patch_set.append(patch)
#         selected.add(start)
#     while len(patch_set) < num_patches:
#         patch_set.append(patch_set[-1].copy())
#     return np.stack(patch_set, axis=0)
#
#
# def extract_contour(mask, num_points=100):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         return None
#     contour = max(contours, key=len).squeeze()
#     if contour.ndim != 2 or len(contour) < num_points:
#         return None
#     idx = np.linspace(0, len(contour) - 1, num=num_points, dtype=np.int32)
#     contour = contour[idx].astype(np.float32)
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
#     return best_label - 1
#
#
# class PatchSetDataset(Dataset):
#     def __init__(self, root_dir, max_samples=None):
#         self.data = []  # list of (patch_set, label)
#         task_dirs = [
#             d for d in os.listdir(root_dir)
#             if os.path.isdir(os.path.join(root_dir, d)) and "closure" not in d
#         ]
#
#         for task in tqdm(task_dirs, desc="Building dataset"):
#             task_path = os.path.join(root_dir, task)
#             gt_path = os.path.join(task_path, cfg.GT_EXTENSION)
#             if not os.path.exists(gt_path):
#                 continue
#
#             with open(gt_path, "r") as f:
#                 gt_data = json.load(f)["img_data"]
#
#             for img_name, obj_list in gt_data.items():
#                 img_path = os.path.join(task_path, img_name + ".png")
#                 img = cv2.imread(img_path)
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 binary = np.where(gray != 211, 255, 0).astype(np.uint8)
#
#                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
#
#                 for i in range(1, num_labels):
#                     x, y, w, h, _ = stats[i]
#                     if w == 0 or h == 0: continue
#                     obj_mask = (labels == i).astype(np.uint8) * 255
#                     cx, cy = centroids[i]
#                     label = match_gt_label(cx, cy, obj_list)
#                     if label is None: continue
#
#                     # contour = extract_contour(obj_mask)
#                     contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                     contour = contours[0][:, 0, :]
#                     contour = contour.astype(np.float32)
#
#                     if contour is None: continue
#
#                     patch_set = generate_random_patch_set(
#                         contour,
#                         patch_size=cfg.POINTS_PER_PATCH,
#                         num_patches=cfg.PATCHES_PER_SET
#                     )
#                     self.data.append((patch_set, label))
#
#                     if max_samples and len(self.data) >= max_samples:
#                         break
#                 if max_samples and len(self.data) >= max_samples:
#                     break
#             if max_samples and len(self.data) >= max_samples:
#                 break
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         patch, label = self.data[idx]
#         return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)