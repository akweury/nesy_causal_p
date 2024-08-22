# Created by jing at 19.08.24


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

import config
from utils import chart_utils, args_utils


def draw_triangle(points):
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)
    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())
    # Draw a triangle
    draw = ImageDraw.Draw(image)
    draw.polygon(points, outline=1, fill=0)
    # Convert PIL image back to tensor

    return np.array(image)


def draw_angle():
    ls_up = [(2, 2), (2, 0)]
    ls_upright = [(2, 2), (4, 0)]
    ls_right = [(2, 2), (4, 2)]
    ls_downright = [(2, 2), (4, 4)]
    ls_down = [(2, 2), (2, 4)]
    ls_downleft = [(2, 2), (0, 4)]
    ls_left = [(2, 2), (0, 2)]
    ls_topleft = [(2, 2), (0, 0)]

    directions = [ls_up, ls_upright, ls_right, ls_downright, ls_down, ls_downleft, ls_left, ls_topleft]

    angle_imgs = []
    for d_i in range(len(directions) - 1):
        for d_j in range(d_i + 1, len(directions)):
            # Create a 64x64 tensor with zeros
            tensor = torch.zeros((64, 64), dtype=torch.uint8)
            # Convert tensor to PIL image
            image = Image.fromarray(tensor.numpy())

            draw = ImageDraw.Draw(image)
            draw.line((directions[d_i][0], directions[d_i][1]), fill="white", width=1)
            draw.line((directions[d_j][0], directions[d_j][1]), fill="white", width=1)
            # Convert PIL image back to tensor
            img = torch.from_numpy(np.array(image))
            img = img.to(torch.bool).to(torch.uint8)
            angle_imgs.append(img)

    return angle_imgs


def draw_line(points):
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)
    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())
    # Draw a triangle
    draw = ImageDraw.Draw(image)
    draw.line(points, fill="white", width=1)
    # Convert PIL image back to tensor

    return np.array(image)


def get_cover_percent(img, fm):
    fm_points = fm.sum()
    img_points = img.sum()
    cover_points = (img[fm] > 0).sum()
    cover_percent = cover_points / fm_points
    return cover_percent


def draw_circle(d=1.0):
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)

    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())

    # Draw a circle
    draw = ImageDraw.Draw(image)
    draw.ellipse([(20, 20), (int(44 * d), int(44 * d))], outline=1, fill=0)

    # Convert PIL image back to tensor

    return np.array(image)


def draw_rect(r=1.0):
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)

    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())

    # Draw a circle
    draw = ImageDraw.Draw(image)
    draw.rectangle([(20, 20), (int(44 * r), 44)], outline=1, fill=0)

    # Convert PIL image back to tensor

    return np.array(image)


def correlation(img_fix, fm, name):
    cover_percents = np.zeros((64, 64))
    cover_percent = get_cover_percent(img_fix, fm)
    cover_percents[0, 0] = cover_percent

    for i in range(64):
        up_shifted_img = torch.roll(fm, shifts=i, dims=0)  # Shift all rows up
        for j in range(64):
            left_shifted_img = torch.roll(up_shifted_img, shifts=j, dims=1)  # Shift all columns to the left
            percent = get_cover_percent(img_fix, left_shifted_img)
            cover_percents[i, j] = percent
    # Generate a 64x64      matrix (example)
    hm_cover_percents = chart_utils.zoom_matrix_to_image_cv(cover_percents)
    input_img = chart_utils.zoom_img((fm * 255).to(torch.uint8).numpy())
    fm_mask_img = chart_utils.zoom_img((img_fix * 255).to(torch.uint8).numpy())
    # Vertically concatenate the two images
    concatenated_image = np.vstack((input_img, fm_mask_img, hm_cover_percents))
    image_array = concatenated_image.astype(np.uint8)
    # Save the array as an image using OpenCV
    cv2.imwrite(name, image_array)
    print("finish")


def main():
    args = args_utils.get_args()
    rotate_triangle_1_big = [(8, 10), (40, 16), (30, 30)]
    rotate_triangle_1_small = [(8, 10), (24, 9), (20, 20)]
    triangle_1_big = [(10, 10), (50, 10), (30, 50)]
    triangle_1_small = [(20, 10), (40, 10), (30, 30)]
    line_1 = torch.from_numpy(draw_line([(7, 15), (25, 6)]))
    line_2 = torch.from_numpy(draw_line([(7, 15), (5, 50)]))

    rect = torch.from_numpy(draw_rect())
    cir = torch.from_numpy(draw_circle(d=1.0))
    tri = torch.from_numpy(draw_triangle(rotate_triangle_1_big))
    fm_angles = draw_angle()
    cir_fm = torch.from_numpy(draw_circle(d=1.2))
    rect_fm = torch.from_numpy(draw_rect(r=1.2))

    pred_patterns = [[rect, cir], [cir, tri], [tri, rect]]
    features = fm_angles
    idx = 50
    for fp in pred_patterns:
        for mp in features:
            idx += 1
            pattern = (fp[0] + fp[1]).to(torch.bool).to(torch.uint8)
            name = str(config.output / f"{args.exp_name}" / f'{str(idx)}.png')
            correlation(pattern, mp, name)


if __name__ == "__main__":
    main()
