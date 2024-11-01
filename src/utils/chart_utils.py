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
    zoomed_image = np.zeros((1, 64, 64))

    # Convert to numpy for applying the colormap
    input_matrix_np = matrix.numpy() / 5
    assert matrix.max() <= 5

    # Choose a colormap, e.g., 'viridis'
    cmap = plt.get_cmap('viridis')

    for i in range(len(matrix)):
        # zoomed_image[i] = cv2.resize(cmap(input_matrix_np[i])[:, :, :3], (512, 512), interpolation=cv2.INTER_NEAREST)
        only_matching_region = input_matrix_np[i]
        only_matching_region[only_matching_region != 1] = 0
        zoomed_image[-1] += only_matching_region
    zoomed_image /= zoomed_image.max()
    zoomed_image = cv2.resize(cmap(zoomed_image[-1])[:, :, :3], (512, 512), interpolation=cv2.INTER_NEAREST)
    return ((zoomed_image) * 255).astype(np.uint8)


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
    zoom_factor = 512 // len(matrix)
    pad_width = 2
    if matrix.shape != (64, 64):
        raise ValueError("Input matrix must be of size 64x64.")

    # Resize (zoom) the matrix using nearest neighbor interpolation to keep pixels sharp
    zoomed_matrix = cv2.resize(matrix, (matrix.shape[1] * zoom_factor, matrix.shape[0] * zoom_factor),
                               interpolation=cv2.INTER_NEAREST)
    rgb_image = np.stack((zoomed_matrix,) * 3, axis=-1)

    # rgb_image = np.pad(rgb_image, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
    #                    constant_values=255)
    return rgb_image


def color_mapping(matrix, norm_factor, text=None):
    # Choose a colormap, e.g., 'viridis'
    cmap = plt.get_cmap('viridis')
    zoomed_image = cv2.resize(cmap(matrix / norm_factor)[:, :, :3], (512, 512), interpolation=cv2.INTER_NEAREST)
    zoomed_image = (zoomed_image * 255).astype(np.uint8)

    if text is not None:
        position = (zoomed_image.shape[0] - 250, zoomed_image.shape[1] - 50)
        cv2.putText(zoomed_image, text=text, org=position, color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, lineType=cv2.LINE_AA)
    return zoomed_image


def concat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        pad_width = 2
        pad_img = np.pad(img, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)), constant_values=255)
        padding_imgs.append(pad_img)
    img = np.hstack(padding_imgs).astype(np.uint8)

    return img


def vconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        pad_width = 2
        pad_img = np.pad(img, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)), constant_values=255)
        padding_imgs.append(pad_img)
    img = np.vstack(padding_imgs).astype(np.uint8)

    return img


def visual_np_array(array):
    plt.imshow(array)
    plt.axis('off')
    plt.show()