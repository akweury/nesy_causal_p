# Created by MacBook Pro at 20.04.25

import numpy as np
import cv2

def extract_object_contours(pil_image, min_area=30):
    """从 PIL 图像中提取所有 object 的轮廓（返回 resampled 的 contour）"""
    rgb = np.array(pil_image)
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        binary = np.where(gray < 210, 255, 0).astype(np.uint8)
    else:
        binary = rgb
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    contours = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour and len(contour[0]) >= 10:
            resampled = resample_contour(contour[0][:, 0, :], 100)
            contours.append(resampled)
    return contours

def resample_contour(contour, num_points):
    """将轮廓按轮廓长度均匀插值为固定数量的点"""
    contour = contour.astype(np.float32)
    distances = np.sqrt(((np.diff(contour, axis=0))**2).sum(axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative[-1]
    uniform_dist = np.linspace(0, total_length, num_points)
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    for i, d in enumerate(uniform_dist):
        idx = np.searchsorted(cumulative, d) - 1
        idx = np.clip(idx, 0, len(contour) - 2)
        t = (d - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx] + 1e-8)
        resampled[i] = (1 - t) * contour[idx] + t * contour[idx + 1]
    return resampled

def generate_patch_set_from_contour(contour, num_patches=6, points_per_patch=16):
    """将 resampled 轮廓切分为多个 patch（每个 patch 长度为 points_per_patch）"""
    L = len(contour)
    step = L // num_patches
    patch_set = []
    for i in range(num_patches):
        start = (i * step) % L
        patch = [contour[(start + j) % L] for j in range(points_per_patch)]
        patch = np.array(patch).T  # (2, patch_len)
        patch_set.append(patch)
    return np.stack(patch_set)  # (num_patches, 2, points_per_patch)
