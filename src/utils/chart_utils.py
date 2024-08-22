# Created by shaji at 24/07/2024


import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import cm


def zoom_matrix_to_image_cv(matrix, zoom_factor=8, colormap='plasma', padding=True):
    """
    Zooms a 64x64 matrix to a larger image using a specified colormap and OpenCV.

    Parameters:
    - matrix: 2D NumPy array of size 64x64.
    - zoom_factor: Factor by which to zoom the image (default is 8 for 512x512 output).
    - colormap: Colormap to apply (default is 'plasma').

    Returns:
    - zoomed_image_matrix: 2D NumPy array representing the zoomed image with colormap applied.
    """
    # Ensure matrix is in the correct range for a colormap

    # Apply the colormap using Matplotlib's colormap functions
    colormap_func = cm.get_cmap(colormap)
    colored_matrix = colormap_func(matrix)

    # Convert the RGBA colormap output to BGR for OpenCV (ignore alpha channel)
    colored_matrix_bgr = (colored_matrix[:, :, :3] * 255).astype(np.uint8)
    colored_matrix_bgr = cv2.cvtColor(colored_matrix_bgr, cv2.COLOR_RGB2BGR)

    # Resize (zoom) using OpenCV with nearest neighbor interpolation
    zoomed_image = cv2.resize(colored_matrix_bgr,
                              (matrix.shape[1] * zoom_factor, matrix.shape[0] * zoom_factor),
                              interpolation=cv2.INTER_NEAREST)

    pad_width = 2
    zoomed_image = np.pad(zoomed_image, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                          constant_values=255)

    return zoomed_image


def zoom_img(matrix, zoom_factor=8, add_border=True):
    """
    Zooms a 64x64 matrix to a larger matrix by replicating each pixel into a block of size zoom_factor x zoom_factor.

    Parameters:
    - matrix: 2D NumPy array of size 64x64.
    - zoom_factor: Factor by which to zoom the matrix (default is 8 for 512x512 output).

    Returns:
    - zoomed_matrix: 2D NumPy array of the zoomed matrix (512x512 for zoom_factor=8).
    """
    # Check the input matrix dimensions

    pad_width = 2
    if matrix.shape != (64, 64):
        raise ValueError("Input matrix must be of size 64x64.")

    # Resize (zoom) the matrix using nearest neighbor interpolation to keep pixels sharp
    zoomed_matrix = cv2.resize(matrix, (matrix.shape[1] * zoom_factor, matrix.shape[0] * zoom_factor),
                               interpolation=cv2.INTER_NEAREST)
    rgb_image = np.stack((zoomed_matrix,) * 3, axis=-1)

    rgb_image = np.pad(rgb_image, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                       constant_values=255)
    return rgb_image
