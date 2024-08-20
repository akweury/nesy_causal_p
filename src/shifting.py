# Created by jing at 19.08.24


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

import config
from utils import chart_utils, args_utils


def draw_triangle(matrix_size=64, width=100):
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)
    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())
    # Draw a triangle
    draw = ImageDraw.Draw(image)
    draw.polygon([(10, 10), (50, 10), (30, 50)], outline=1, fill=0)
    # Convert PIL image back to tensor

    return np.array(image)


def get_cover_percent(mask, img):
    fm_points = mask.sum()
    img_points = img.sum()
    cover_points = (img[mask] > 0).sum()
    cover_percent = cover_points / fm_points
    return cover_percent


def draw_circle():
    # Create a 64x64 tensor with zeros
    tensor = torch.zeros((64, 64), dtype=torch.uint8)

    # Convert tensor to PIL image
    image = Image.fromarray(tensor.numpy())

    # Draw a circle
    draw = ImageDraw.Draw(image)
    draw.ellipse([(20, 20), (44, 44)], outline=1, fill=0)

    # Convert PIL image back to tensor

    return np.array(image)


def main():
    args = args_utils.get_args()
    triangle_fix = draw_triangle(matrix_size=64, width=20)
    triangle_fix = torch.from_numpy(triangle_fix)

    circle_fix = draw_circle()
    circle_fix = torch.from_numpy(circle_fix)

    img_fix = (triangle_fix + circle_fix).to(torch.bool).to(torch.uint8)
    # img_fix = circle_fix.to(torch.bool).to(torch.uint8)

    circle = draw_circle()
    img_moving = torch.from_numpy(circle)

    cover_percents = np.zeros((64, 64))
    cover_percent = get_cover_percent(img_fix, img_moving)
    cover_percents[-1, -1] = cover_percent

    for i in reversed(range(img_moving.shape[0])):
        up_shifted_img = torch.roll(img_moving, shifts=-i, dims=0)  # Shift all rows up
        for j in reversed(range(img_moving.shape[1])):
            left_shifted_img = torch.roll(up_shifted_img, shifts=-j, dims=1)  # Shift all columns to the left
            percent = get_cover_percent(img_fix, left_shifted_img)
            cover_percents[(32 + i) % 64, (32 + j) % 64] = percent
            if percent > 0.06:
                # Generate a 64x64      matrix (example)
                hm_cover_percents = chart_utils.zoom_matrix_to_image_cv(cover_percents)
                input_img = chart_utils.zoom_img((left_shifted_img * 255).to(torch.uint8).numpy())
                fm_mask_img = chart_utils.zoom_img((img_fix * 255).to(torch.uint8).numpy())
                # Vertically concatenate the two images
                concatenated_image = np.vstack((input_img, fm_mask_img, hm_cover_percents))
                image_array = concatenated_image.astype(np.uint8)
                # Save the array as an image using OpenCV
                cv2.imwrite(
                    str(config.output / f"{args.exp_name}" / f'cir_tri_{64 - i}_{64 - j}_{percent:.2f}.png'),
                    image_array)
                print(f"saved an image :  f'{64 - i}_{64 - j}_{percent:.2f}.png'")

    print("finish")


if __name__ == "__main__":
    main()
