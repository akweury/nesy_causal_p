import time
import sys
sys.path.insert(0, r'E:\projects\nesy_causal_p')
from src import dataset as ds
import cv2

IMG = r'E:\projects\nesy_causal_p\src\test.png'
object_data = {'x': 0.5, 'y': 0.5, 'size': 0.2}

num_iters = 200

# Warmup
img = cv2.imread(IMG)
if img is None:
    raise RuntimeError('test image not found')
_ = ds.extract_object_contour_from_image(img, object_data, num_points=16)
_ = ds.extract_object_contour(IMG, object_data, num_points=16)

# Measure extract_object_contour (reads image each call)
start = time.perf_counter()
for i in range(num_iters):
    _ = ds.extract_object_contour(IMG, object_data, num_points=16)
end = time.perf_counter()
print(f"extract_object_contour (with imread) avg: {(end-start)/num_iters*1000:.3f} ms per call")

# Measure extract_object_contour_from_image (preloaded image)
start = time.perf_counter()
for i in range(num_iters):
    _ = ds.extract_object_contour_from_image(img, object_data, num_points=16)
end = time.perf_counter()
print(f"extract_object_contour_from_image (preloaded) avg: {(end-start)/num_iters*1000:.3f} ms per call")

# Also measure resample alone
import numpy as np
contour = np.random.rand(200,2)*100
start = time.perf_counter()
for i in range(num_iters):
    _ = ds.resample_contour(contour, 16)
end = time.perf_counter()
print(f"resample_contour avg: {(end-start)/num_iters*1000:.3f} ms per call")

