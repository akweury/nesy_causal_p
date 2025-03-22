# Created by X at 24/07/2024


import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image

import config
from src import bk


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
    zoomed_image = cv2.resize(cmap(zoomed_image[-1])[:, :, :3], (512, 512),
                              interpolation=cv2.INTER_NEAREST)
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
    zoomed_matrix = cv2.resize(matrix, (
        matrix.shape[1] * zoom_factor, matrix.shape[0] * zoom_factor),
                               interpolation=cv2.INTER_NEAREST)
    rgb_image = np.stack((zoomed_matrix,) * 3, axis=-1)

    # rgb_image = np.pad(rgb_image, pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
    #                    constant_values=255)
    return rgb_image


def color_mapping(matrix, norm_factor, text=None):
    # Choose a colormap, e.g., 'viridis'
    cmap = plt.get_cmap('viridis')
    zoomed_image = cv2.resize(cmap(matrix / norm_factor)[:, :, :3], (512, 512),
                              interpolation=cv2.INTER_NEAREST)
    zoomed_image = (zoomed_image * 255).astype(np.uint8)

    if text is not None:
        position = (zoomed_image.shape[0] - 250, zoomed_image.shape[1] - 50)
        cv2.putText(zoomed_image, text=text, org=position, color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                    lineType=cv2.LINE_AA)
    return zoomed_image


def add_text(text, img):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    if text is not None:
        position = (img.shape[0] - 250, img.shape[1] - 50)
        cv2.putText(img, text=text, org=position, color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                    lineType=cv2.LINE_AA)
    return img


def hconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        padding_imgs.append(img_padding(img))
    img = np.hstack(padding_imgs).astype(np.uint8)

    return img


def img_padding(img, pad_width=2):
    if img.ndim == 3:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
                         constant_values=255)
    elif img.ndim == 2:
        pad_img = np.pad(img, pad_width=(
            (pad_width, pad_width), (pad_width, pad_width)),
                         constant_values=255)

    else:
        raise ValueError()

    return pad_img


def vconcat_imgs(img_list):
    padding_imgs = []
    for img in img_list:
        padding_imgs.append(img_padding(img))
    img = np.vstack(padding_imgs).astype(np.uint8)

    return img


def visual_np_array(array, filename=None):
    if filename is not None:
        # save the image
        # Convert array to image
        image = Image.fromarray(array)
        # Save as PNG
        image.save(filename)
    plt.axis('off')


def van(array, file_name=None):
    plt.clf()  # Clear current figure
    if isinstance(array, list):
        hconcat = hconcat_imgs(array)
        visual_np_array(hconcat.squeeze(), file_name)
    elif len(array.shape) == 2:
        visual_np_array(array.squeeze(), file_name)
    elif len(array.shape) == 3:
        visual_np_array(array.squeeze(), file_name)
    elif len(array.shape) == 4:
        visual_np_array(array[0].squeeze(), file_name)


def get_black_img():
    black_img = np.zeros((512, 512)).astype(np.uint8)
    return black_img


def array2img(array):
    img = cv2.resize(array, (512, 512), interpolation=cv2.INTER_NEAREST)
    img = color_mapping(img, 1, "BW")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_img(img, file_name):
    cv2.imwrite(str(file_name), img)


def visual_batch_imgs(batch_imgs, path, name):
    if len(batch_imgs) > 10:
        row_num = len(batch_imgs) // 10
        col_num = 10
    else:
        row_num = 1
        col_num = len(batch_imgs)
    batch_imgs = [array2img(img) for img in batch_imgs]
    row_imgs = []
    for i in range(row_num):
        row_img = batch_imgs[i * col_num:(i + 1) * col_num]
        if len(row_img) < col_num:
            row_img += [get_black_img()] * (col_num - len(row_img))

        row_img = hconcat_imgs(row_img)
        row_imgs.append(row_img)
    col_imgs = vconcat_imgs(row_imgs)
    save_img(col_imgs, path / name)


def resize_img(img, new_size):
    resized_img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
    return resized_img


def shadow_obj(img, obj_tensor, label, action):
    color = bk.color_matplotlib["gray"]
    alpha = 0.5
    size = int(obj_tensor[2] * 512 * 0.6)
    x_pos = int(obj_tensor[0] * 512)
    y_pos = int(obj_tensor[1] * 512)
    top_left = (x_pos - size, y_pos - size)
    bottom_right = (x_pos + size, y_pos + size)
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text=str(label), org=(x_pos, y_pos), color=(255, 255, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                lineType=cv2.LINE_AA)

    return img


def ocm2img(labels, ocm, input_img, action):
    unique_labels = np.unique(labels).astype(int)

    img = add_text(f"Group", input_img)
    for label in unique_labels:
        group_ocm = ocm[labels == label]
        for obj_tensor in group_ocm:
            img = shadow_obj(img, obj_tensor, label, action)
    cv2.putText(img, text=str(action), org=(50, 100), color=(0, 0, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                lineType=cv2.LINE_AA)
    return img


def visual_rl_step(task_img, ocms, output_labels, action, reward):
    output_imgs = []
    for i, labels in enumerate(output_labels):
        labels_trun = labels[:len(ocms[i])]
        output_img = ocm2img(labels_trun, ocms[i], task_img[i], action)
        output_imgs.append(output_img)
    output_imgs = hconcat_imgs(output_imgs)
    cv2.putText(output_imgs, text=f"Reward {reward}", org=(50, 50), color=(0, 0, 255),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                lineType=cv2.LINE_AA)
    return output_imgs


def show_line_chart(data, title="", file_name=None):
    plt.clf()  # Clear current figure

    plt.plot(data)
    # Beautify the plot
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    # Customize the plot
    plt.legend()
    # Save as a PDF
    if file_name is not None:
        plt.savefig(file_name, format="pdf")
    plt.show()

def visual_multiple_segments(labels, data):
    # Visualize the segments
    plt.figure(figsize=(10, 6))

    for i, segment in enumerate(labels):
        indices = segment
        values = data[indices]
        if len(values) > 10:
            color = bk.color_large[i]
        else:
            color = 'k'  # Edge case (single-point segment)
            label = 'Single point'
        plt.plot(indices, values, color=color)
    # Beautify the plot
    plt.title("Segments of Tensor Trends")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def visual_labeled_contours(width, all_contour_segs, contour_points, contour_labels):
    rgb_colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.0, 1.0)  # Magenta
    ]
    seg_img = np.zeros((width, width, 3))
    lind_width = 5
    for contour_i, contour_segs in enumerate(all_contour_segs):
        for seg_i, seg in enumerate(contour_segs):
            points = contour_points[contour_i][seg]
            label = contour_labels[contour_i][seg_i]
            color = rgb_colors[0] if label == "line" else rgb_colors[1]
            # Color the given positions
            for pos in points:
                seg_img[pos[1] - lind_width:pos[1] + lind_width, pos[0] - lind_width:pos[0] + lind_width] = color

    seg_img = (seg_img * 255).astype(np.uint8)
    van(seg_img, file_name=config.output / "contour_segs.png")


def show_convex_hull(points, hull_points):
    width = 400
    points_integer = (points * width).astype(np.int32)
    hull_points_integer = (hull_points * width).astype(np.int32)
    # Create a blank (black) image; adjust size as needed
    # Here, we assume the coordinates fit in a 400x400 region
    img = np.zeros((width, width, 3), dtype=np.uint8)

    # Draw original points in white
    for i, (x, y) in enumerate(points_integer):
        cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)  # White color (B, G, R)
        thickness = 1
        # Add text to the image
        cv2.putText(img, str(i), (x + 10, y - 10), font, font_scale, color, thickness)

    # Convert the hull array to the correct shape for polylines
    # hull is already Nx1x2 in shape if directly from cv2.convexHull
    # but to be safe in polylines, we can reshape:
    hull_reshaped = hull_points_integer.reshape(-1, 1, 2)

    # Draw the hull polygon in green with thickness of 2
    cv2.polylines(img, [hull_reshaped], isClosed=True, color=(0, 255, 0), thickness=2)

    for x, y in hull_points_integer:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
    van(img)
    return img


def add_lines_to_img(img, lines):
    for line in lines:
        points = (line["points"] * img.shape[0]).astype(np.int32)
        current_start = min(points, key=lambda p: (p[0], p[1]))
        current_end = max(points, key=lambda p: (p[0], p[1]))
        cv2.line(img, (current_start[0], current_start[1]), (current_end[0], current_end[1]), (255, 0, 0), 2)
    van(img)
    return img
