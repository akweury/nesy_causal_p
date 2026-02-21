import time
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, r'E:\projects\nesy_causal_p')
from src import dataset as ds

IMG = r'E:\projects\nesy_causal_p\src\test.png'
object_data = {'x': 0.5, 'y': 0.5, 'size': 0.2}
num_iters = 50

def profile_once():
    timings = {}
    t0 = time.time()
    img = cv2.imread(IMG)
    timings['imread'] = time.time() - t0
    if img is None:
        raise RuntimeError('test image not found')
    h, w = img.shape[:2]

    # coords
    t0 = time.time()
    center_x_norm = float(object_data.get('x', 0.5))
    center_y_norm = float(object_data.get('y', 0.5))
    size_norm = float(object_data.get('size', 0.1))
    center_x = int(center_x_norm * w)
    center_y = int(center_y_norm * h)
    size = int(size_norm * min(w, h))
    if size <= 0:
        size = max(4, int(min(w, h) * 0.05))
    half = max(1, size // 2)
    padding = max(2, int(size * 0.3))
    x1 = max(0, center_x - half - padding)
    y1 = max(0, center_y - half - padding)
    x2 = min(w, center_x + half + padding)
    y2 = min(h, center_y + half + padding)
    timings['coords'] = time.time() - t0

    # ROI
    t0 = time.time()
    roi = img[y1:y2, x1:x2]
    timings['roi'] = time.time() - t0

    # gray
    t0 = time.time()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    timings['cvtColor'] = time.time() - t0

    # binary mask by bg value
    t0 = time.time()
    binary = np.where(gray == 211, 0, 255).astype(np.uint8)
    timings['binary_mask'] = time.time() - t0

    # fallback adaptive threshold if almost empty
    t0 = time.time()
    fallback = False
    if np.count_nonzero(binary) < 3:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        fallback = True
    timings['adaptive'] = time.time() - t0

    # findContours
    t0 = time.time()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    timings['findContours'] = time.time() - t0

    # choose largest
    t0 = time.time()
    if not contours:
        timings['choose'] = time.time() - t0
        return timings
    try:
        largest_contour = max(contours, key=cv2.contourArea)
    except Exception:
        largest_contour = max(contours, key=lambda c: c.shape[0])
    timings['choose'] = time.time() - t0

    # reshape & adjust coords
    t0 = time.time()
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    contour_points[:, 0] += x1
    contour_points[:, 1] += y1
    timings['reshape_adjust'] = time.time() - t0

    # resample
    t0 = time.time()
    res = ds.resample_contour(contour_points, 16)
    timings['resample'] = time.time() - t0

    timings['fallback_used'] = fallback
    return timings


if __name__ == '__main__':
    sums = {}
    flags = {'fallback_used': 0}
    for i in range(num_iters):
        t = profile_once()
        for k, v in t.items():
            if k == 'fallback_used':
                flags['fallback_used'] += 1 if v else 0
                continue
            sums[k] = sums.get(k, 0.0) + v
    print('Averaged timings over', num_iters, 'runs:')
    for k in sorted(sums.keys()):
        print(f"{k}: {sums[k] / num_iters:.6f} s")
    print('adaptive fallback used in', flags['fallback_used'], 'runs')

