# Created by shaji at 03/08/2024

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageDraw

import config
from src.percept import perception
from src.utils import args_utils, data_utils, chart_utils


def prepare_kp_sy_data(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = perception.ShapeDataset(args, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def img2nesy_tensor(args, data, fms):
    _, nzs_patches, nz_positions = data_utils.find_submatrix(data.squeeze(), args.kernel)
    nz_positions[:, 0] -= nz_positions[0, 0].item()
    nz_positions[:, 1] -= nz_positions[0, 1].item()
    image_tensor = torch.zeros((50, 3)) + len(fms)

    for p_i, p in enumerate(nzs_patches):
        for f_i, fm in enumerate(fms):
            if torch.equal(fm, p):
                image_tensor[p_i, 0] = f_i
        if image_tensor[p_i, 0] == len(fms):
            image_tensor[p_i, 0] = -1  # unknown fm
        image_tensor[p_i, 1:] = nz_positions[p_i]
    return image_tensor


def get_cover_percent(mask, img):
    fm_points = mask.sum()
    img_points = img.sum()
    cover_points = (img[mask] > 0).sum()
    cover_percent = cover_points / fm_points
    return cover_percent


def check_fm_in_img(args, fm_mask, img):
    cover_percents = torch.zeros_like(img)
    cover_percent = get_cover_percent(fm_mask.squeeze(), img)
    cover_percents[-1, -1] = cover_percent

    for i in reversed(range(img.shape[0])):
        up_shifted_img = torch.roll(img, shifts=-i, dims=0)  # Shift all rows up
        for j in reversed(range(img.shape[1])):
            left_shifted_img = torch.roll(up_shifted_img, shifts=-j, dims=1)  # Shift all columns to the left
            percent = get_cover_percent(fm_mask, left_shifted_img)
            cover_percents[i, j] = percent
            if percent > 0.05:
                # Generate a 64x64      matrix (example)
                hm_cover_percents = chart_utils.zoom_matrix_to_image_cv(cover_percents.numpy())
                input_img = chart_utils.zoom_img((left_shifted_img * 255).to(torch.uint8).numpy())
                fm_mask_img = chart_utils.zoom_img((fm_mask * 255).to(torch.uint8).numpy())
                # Vertically concatenate the two images
                concatenated_image = np.vstack((input_img, fm_mask_img, hm_cover_percents))
                image_array = concatenated_image.astype(np.uint8)
                # Save the array as an image using OpenCV
                cv2.imwrite(
                    str(config.output / f"{args.exp_name}" / f'saved_image_{64 - i}_{64 - j}_{percent:.2f}.png'),
                    image_array)
                print(f"saved an image :  f'saved_image_{64 - i}_{64 - j}_{percent:.2f}.png'")
    return cover_percents


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
            angle_imgs.append(img.unsqueeze(0))

    return torch.cat((angle_imgs), dim=0)


def similarity(img_fm, fm_repo):
    non_zero_mask = fm_repo != 0
    total_non_zero_comparisons = torch.sum(non_zero_mask, dim=[1, 2, 3]).float()
    best_indices = torch.zeros((img_fm.shape[-2], img_fm.shape[-1]), dtype=torch.uint8)
    best_values = torch.zeros((img_fm.shape[-2], img_fm.shape[-1]), dtype=torch.uint8)
    for shift_i in tqdm(range(img_fm.shape[-2] - 50), desc="shift row"):
        for shift_j in range(img_fm.shape[-1] - 50):
            shifted_img = torch.roll(img_fm, shifts=(-shift_i, -shift_j), dims=(-2, -1))  # Shift all rows up
            if shifted_img[:, :, shift_i, :].sum() == 0 or shifted_img[:, :, :, shift_j].sum() == 0:
                continue
            equal_items = (shifted_img == fm_repo) * non_zero_mask
            num_equal_non_zero = torch.sum(equal_items, dim=[1, 2, 3]).float()
            percentage_equal_non_zero = (num_equal_non_zero / (total_non_zero_comparisons + 1e-20))
            best_indices[shift_i, shift_j] = torch.argmax(percentage_equal_non_zero)
            best_values[shift_i, shift_j] = int(torch.max(percentage_equal_non_zero) * 100)
    max_idx = torch.argmax(best_values)
    shift_idx = torch.unravel_index(max_idx, best_values.shape)
    best_value = best_values[shift_idx].item()
    best_shift_img_fm = torch.roll(img_fm, shifts=(-torch.stack(shift_idx)).tolist(),
                                   dims=(-2, -1))  # Shift all rows up
    match_percent = f"{best_value}%"
    fm_best_idx = best_indices[shift_idx].item()
    fm_best = fm_repo[fm_best_idx]
    fm_mask = non_zero_mask[fm_best_idx]
    fm_best_diff = ((fm_best != best_shift_img_fm.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)
    fm_best_same = ((fm_best == best_shift_img_fm.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)

    exact_matching_matrix = (fm_best == best_shift_img_fm.squeeze()).sum(dim=0).to(torch.float32)
    exact_matching_matrix = (exact_matching_matrix == fm_best.shape[0]).to(torch.float32)
    exact_matching_matrix = torch.roll(exact_matching_matrix, shifts=torch.stack(shift_idx).tolist(),
                                       dims=(-2, -1))  # Shift all rows up
    return fm_best, fm_best_diff, fm_best_same, match_percent, exact_matching_matrix


def main():
    args = args_utils.get_args()
    args.batch_size = 1

    # load learned triangle fms
    fm_repo = torch.load(config.output / f"data_triangle" / f"fms.pt").to(args.device)
    # load test dataset
    args.data_types = args.exp_name
    train_loader, val_loader = prepare_kp_sy_data(args)
    os.makedirs(config.output / f"{args.exp_name}", exist_ok=True)
    kernels = torch.load(config.output / f"data_triangle" / f"kernels.pt").to(args.device)

    for data, labels in tqdm(train_loader):
        img_fm = perception.extract_fm(data, kernels)
        data_fm_shifted, shift_row, shift_col = data_utils.shift_content_to_top_left(img_fm)
        fm_best, fm_best_diff, fm_best_same, match_percent, exact_matrix = similarity(data_fm_shifted, fm_repo)

        exact_matrix = torch.roll(exact_matrix, shifts=(shift_row, shift_col), dims=(-2, -1))  # Shift all rows up
        data_mask = data.squeeze() != 0
        data_onside = exact_matrix * data_mask
        data_offside = (1 - exact_matrix) * data_mask
        # visual fm and the input image
        data_img = chart_utils.color_mapping(data.squeeze(), 1, "IN")

        data_fm_shifted = data_fm_shifted.sum(dim=1).squeeze()
        fm_best = fm_best.sum(dim=0).squeeze()
        norm_factor = max(data_fm_shifted.max(), fm_best.max(), fm_best_diff.max())
        data_onside_img = chart_utils.color_mapping(data_onside, norm_factor, "Onside")
        data_offside_img = chart_utils.color_mapping(data_offside, norm_factor, "Offside")

        data_fm_img = chart_utils.color_mapping(data_fm_shifted, norm_factor, "IN_FM")
        repo_fm_img = chart_utils.color_mapping(fm_best, norm_factor, "Match_FM")
        repo_fm_best_same = chart_utils.color_mapping(fm_best_same, norm_factor, f"SAME {match_percent}")
        repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff, norm_factor, "DIFF")

        compare_img = chart_utils.concat_imgs(
            [data_img, data_fm_img, repo_fm_img, repo_fm_best_same, repo_fm_best_diff, data_onside_img,
             data_offside_img])

        cv2.imwrite(str(config.output / f"{args.exp_name}" / f'compare.png'), compare_img)
        print("image done")

    print("program is finished.")


if __name__ == "__main__":
    main()
