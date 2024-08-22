# Created by jing at 19.08.24


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import torch.nn.functional as F

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
            tensor = torch.zeros((5, 5), dtype=torch.uint8)
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


def correlation(img_fix, fm):
    # cover_percents = np.zeros((64, 64))
    # cover_percent = get_cover_percent(img_fix, fm)
    # cover_percents[0, 0] = cover_percent
    correlation = F.conv2d(img_fix, fm, padding=2)
    convolution = F.conv2d(img_fix, fm.flip([2, 3]), padding=2)
    correlation[:, :, :5, :5] = fm
    convolution[:, :, :5, :5] = fm
    return correlation, convolution
    # for i in range(64):
    #     up_shifted_img = torch.roll(fm, shifts=i, dims=0)  # Shift all rows up
    #     for j in range(64):
    #         left_shifted_img = torch.roll(up_shifted_img, shifts=j, dims=1)  # Shift all columns to the left
    #         percent = get_cover_percent(img_fix, left_shifted_img)
    #         cover_percents[i, j] = percent


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
        pattern = (fp[0] + fp[1]).to(torch.bool).to(torch.uint8).unsqueeze(0).unsqueeze(0)
        cors = torch.zeros((len(features), 64, 64))
        cons = torch.zeros((len(features), 64, 64))
        for f_i, mp in enumerate(features):
            kernel = mp.unsqueeze(0).unsqueeze(0)
            cor, con = correlation(pattern, kernel)
            cors[f_i] = cor
            cons[f_i] = con

        # visual
        # Generate a 64x64      matrix (example)

        cor_imgs = chart_utils.zoom_matrix_to_image_cv(cors)
        con_imgs = chart_utils.zoom_matrix_to_image_cv(cons)
        # input_img = chart_utils.zoom_img((fm * 255).to(torch.uint8).numpy())
        fm_mask_img = chart_utils.zoom_img((pattern.squeeze() * 255).to(torch.uint8).numpy())
        # Vertically concatenate the two images
        cor_imgs = np.vstack((fm_mask_img, cor_imgs)).astype(np.uint8)
        # Save the array as an image using OpenCV
        cv2.imwrite(str(config.output / f"{args.exp_name}" / f'cor.png'), cor_imgs)

        con_imgs = np.vstack((fm_mask_img, con_imgs)).astype(np.uint8)
        cv2.imwrite(str(config.output / f"{args.exp_name}" / f'con.png'), con_imgs)

        print("finish")


if __name__ == "__main__":
    main()
