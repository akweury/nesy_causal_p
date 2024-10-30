# Created by shaji at 03/08/2024

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw

import config
from src.percept import perception
from src.utils import args_utils, data_utils, chart_utils, file_utils


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


def get_shifted_versions(img):
    # Create grid of all (x_shift, y_shift) pairs
    shifts = [(-x, -y) for x in range(img.shape[-2]) for y in range(img.shape[-1])]
    # Generate the shifted images as a batch
    shifted_images = torch.cat([torch.roll(img, shifts=(row, col), dims=(-2, -1)) for row, col in shifts])
    return shifted_images


def similarity_fast(img_fm, fm_repo):
    row_num, col_num = img_fm.shape[-2], img_fm.shape[-1]

    non_zero_mask = fm_repo != 0
    img_fm_all_shift = get_shifted_versions(img_fm)
    # Flatten the matrices from (192, 64, 64) to (192*64*64)
    group1_flat = img_fm_all_shift.view(img_fm_all_shift.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    group2_flat = fm_repo.view(fm_repo.size(0), -1)  # Shape: (N2, 192 * 64 * 64)

    # Compute cosine similarity between all pairs (N1 x N2 matrix)
    similarity_matrix = torch.mm(group1_flat, group2_flat.t()) / group2_flat.sum(dim=1).unsqueeze(0)  # Shape: (N1, N2)

    # Find the maximum similarity value and the corresponding indices
    max_similarity = int(torch.max(similarity_matrix).item() * 100)
    max_idx = torch.argmax(similarity_matrix)

    max_idx_2d = torch.unravel_index(max_idx, similarity_matrix.shape)
    max_idx_shift = torch.unravel_index(max_idx_2d[0], img_fm.shape[-2:])
    max_idx_fm = max_idx_2d[1]

    max_img_fm_shift = torch.roll(img_fm, shifts=(-torch.stack(max_idx_shift)).tolist(), dims=(-2, -1))
    max_fm = fm_repo[max_idx_fm]
    match_percent = f"{max_similarity}%"

    fm_mask = non_zero_mask[max_idx_fm]
    fm_best_diff = ((max_fm != max_img_fm_shift.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)
    fm_best_same = ((max_fm == max_img_fm_shift.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)

    exact_matching_matrix = (max_fm == max_img_fm_shift.squeeze()).sum(dim=0).to(torch.float32)
    exact_matching_matrix = (exact_matching_matrix == max_fm.shape[0]).to(torch.float32)
    exact_matching_matrix = torch.roll(exact_matching_matrix, shifts=torch.stack(max_idx_shift).tolist(),
                                       dims=(-2, -1))  # Shift all rows up
    return max_fm, fm_best_diff, fm_best_same, match_percent, exact_matching_matrix


def similarity(img_fm, fm_repo):
    row_num, col_num = img_fm.shape[-2], img_fm.shape[-1]
    non_zero_mask = fm_repo != 0
    total_non_zero_comparisons = torch.sum(non_zero_mask, dim=[1, 2, 3]).float()
    shift_best_fmt = torch.zeros((row_num, col_num), dtype=torch.uint8)
    shift_best_fm_score = torch.zeros((row_num, col_num), dtype=torch.uint8)

    img_fm_all_shift = get_shifted_versions(img_fm)

    for shift_i in tqdm(range(row_num)):
        for shift_j in range(col_num):
            shifted_img = torch.roll(img_fm, shifts=(-shift_i, -shift_j), dims=(-2, -1))
            if shifted_img[:, :, shift_i, :].sum() == 0 or shifted_img[:, :, :, shift_j].sum() == 0:
                continue
            equal_items = (shifted_img == fm_repo) * non_zero_mask
            num_equal_non_zero = torch.sum(equal_items, dim=[1, 2, 3]).float()
            percentage_equal_non_zero = (num_equal_non_zero / (total_non_zero_comparisons + 1e-20))
            shift_best_fmt[shift_i, shift_j] = torch.argmax(percentage_equal_non_zero)
            shift_best_fm_score[shift_i, shift_j] = int(torch.max(percentage_equal_non_zero) * 100)
    max_idx = torch.argmax(shift_best_fm_score)
    shift_idx = torch.unravel_index(max_idx, shift_best_fm_score.shape)
    best_value = shift_best_fm_score[shift_idx].item()
    best_shift_img_fm = torch.roll(img_fm, shifts=(-torch.stack(shift_idx)).tolist(),
                                   dims=(-2, -1))  # Shift all rows up
    match_percent = f"{best_value}%"
    fm_best_idx = shift_best_fmt[shift_idx].item()
    fm_best = fm_repo[fm_best_idx]
    fm_mask = non_zero_mask[fm_best_idx]
    fm_best_diff = ((fm_best != best_shift_img_fm.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)
    fm_best_same = ((fm_best == best_shift_img_fm.squeeze()) * fm_mask).sum(dim=0).to(torch.float32)

    exact_matching_matrix = (fm_best == best_shift_img_fm.squeeze()).sum(dim=0).to(torch.float32)
    exact_matching_matrix = (exact_matching_matrix == fm_best.shape[0]).to(torch.float32)
    exact_matching_matrix = torch.roll(exact_matching_matrix, shifts=torch.stack(shift_idx).tolist(),
                                       dims=(-2, -1))  # Shift all rows up
    return fm_best, fm_best_diff, fm_best_same, match_percent, exact_matching_matrix


def get_pair(img_fm, fm_repo):
    img_fm_shifts = get_shifted_versions(img_fm)
    # Flatten the matrices from (192, 64, 64) to (192*64*64)
    g1_flat = img_fm_shifts.view(img_fm_shifts.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    g2_flat = fm_repo.view(fm_repo.size(0), -1)  # Shape: (N2, 192 * 64 * 64)
    # Compute cosine similarity between all pairs (N1 x N2 matrix)
    similarity_matrix = torch.mm(g1_flat, g2_flat.t()) / g2_flat.sum(dim=1).unsqueeze(0)  # Shape: (N1, N2)

    top_values, top_indices = torch.topk(similarity_matrix.flatten(), 5)
    top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()

    max_idx_shift = torch.stack(torch.unravel_index(top_indices_2d[:, 0], img_fm.shape[-2:])).t()
    max_idx_fm = top_indices_2d[:, 1]

    return max_idx_shift, max_idx_fm, top_values


def visual_all(args, idx, img, data, data_fm_shifted, fm_best, max_value, fm_best_same, fm_best_diff, data_onside,
               data_offside, img_onside_uncertain, bk_shape, out_path):
    in_fm_img = data_fm_shifted.squeeze().sum(dim=0)
    compare_imgs = []
    for i in range(len(fm_best)):
        best_fm_img = fm_best[i].sum(dim=0)
        norm_factor = max([in_fm_img.max(), best_fm_img.max()])

        match_percent = f"{int(max_value[i].item() * 100)}%"

        pos_img = chart_utils.color_mapping(data.squeeze(), 1, "IN")
        data_fm_img = chart_utils.color_mapping(in_fm_img, norm_factor, "IN_FM_SHIFT")
        repo_fm_img = chart_utils.color_mapping(best_fm_img, norm_factor, "Match_FM")
        repo_fm_best_same = chart_utils.color_mapping(fm_best_same[i], norm_factor, f"SAME {match_percent}")
        repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff[i], norm_factor, "DIFF")
        data_onside_img = chart_utils.color_mapping(data_onside[i], norm_factor, "Onside")
        data_onside_uncertain_img = chart_utils.color_mapping(img_onside_uncertain[i], norm_factor, "Onside_Uncertain")
        data_offside_img = chart_utils.color_mapping(data_offside[i], norm_factor, "Offside")

        compare_imgs.append(chart_utils.concat_imgs(
            [img, pos_img, data_fm_img, repo_fm_img, repo_fm_best_same, repo_fm_best_diff, data_onside_img,
             data_onside_uncertain_img, data_offside_img]))
    compare_imgs = chart_utils.vconcat_imgs(compare_imgs)
    compare_imgs = cv2.cvtColor(compare_imgs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_path / f'{idx}_{bk_shape["name"]}.png'), compare_imgs)
    print(f"- grouping result: " + str(config.output / f"{args.exp_name}" / f'c_{bk_shape["name"]}_{idx}.png'))


def get_match_detail(target_fm, shift_in_fm, max_value):
    fm_mask = target_fm != 0

    # similarity_value = [torch.mm(shift_in_fm.flatten().unsqueeze(0),
    #                              target_fm[i].flatten().unsqueeze(0).t()) / target_fm[i].sum().unsqueeze(0) for i in
    #                     range(len(fm_mask))]
    same_fm = ((target_fm == shift_in_fm) * fm_mask).sum(dim=1).to(torch.float32)
    mask_full_match = torch.all(target_fm == shift_in_fm, dim=1) * torch.any(fm_mask, dim=1)
    mask_any_mismatch = torch.any((target_fm == shift_in_fm) * fm_mask, dim=1) * torch.any(fm_mask,
                                                                                           dim=1) * ~mask_full_match
    all_same_fm = same_fm * mask_full_match
    any_diff_fm = same_fm * mask_any_mismatch
    same_percent = mask_full_match.sum(dim=[1, 2]) / (target_fm.sum(dim=1).bool().sum(dim=[1, 2]) + 1e-20)
    return all_same_fm, any_diff_fm, same_percent


def get_siding(args, data, match_same, match_diff, match_fm_img):
    # exact_matching_matrix = (max_fm == max_img_fm_shift.squeeze()).sum(dim=0).to(torch.float32)
    # exact_matching_matrix = (exact_matching_matrix == max_fm.shape[0]).to(torch.float32)
    # exact_matrix = F.affine(match_same.unsqueeze(0), angle=0,
    #                         translate=(-torch.tensor(max_shift_idx)).tolist(), scale=1.0, shear=[0, 0]).squeeze()

    # exact_matrix = torch.roll(exact_matrix, shifts=(shift_row, shift_col), dims=(-2, -1))  # Shift all rows up
    # shift_match = F.affine(match_same.unsqueeze(0), angle=0, translate=shift, scale=1.0, shear=[0, 0]).squeeze()
    top_k = args.top_fm_k
    data_mask = data.squeeze() != 0

    data_onside = [(match_fm_img[i].squeeze() * data_mask) for i in range(top_k)]
    data_onside.append((match_fm_img.squeeze().sum(dim=0).float() * data_mask))
    data_onside = torch.stack(data_onside)

    data_onside_uncertain = [(match_diff[i] * data_mask) for i in range(top_k)]
    data_onside_uncertain.append((match_diff.sum(dim=0).float() * data_mask))
    data_onside_uncertain = torch.stack(data_onside_uncertain)

    data_offside = [((match_fm_img[i].squeeze() == 0) * data_mask) for i in range(top_k)]
    data_offside.append(((match_fm_img.squeeze().sum(dim=0) == 0).float() * data_mask))
    data_offside = torch.stack(data_offside)
    return data_onside, data_offside, data_onside_uncertain


def load_shift(args, bk_shape, idx, in_fm, fm_repo, out_path):
    buffer_file = out_path / f"shift_{bk_shape['name']}_{idx}.pt"
    if os.path.exists(buffer_file):
        data = torch.load(buffer_file)
        match_fm_shift = data['match_fm_shift']
        match_fm_idx = data['match_fm_idx']
        top_values = data["top_values"]
    else:
        match_fm_shift, match_fm_idx, top_values = get_pair(in_fm, fm_repo)
        data = {"match_fm_shift": match_fm_shift,
                "match_fm_idx": match_fm_idx,
                "top_values": top_values}
        torch.save(data, buffer_file)
    top_values = torch.cat((top_values, torch.zeros(1)))
    return match_fm_shift, match_fm_idx, top_values


def img2groups(args, bk, data, idx, img, out_path):
    groups = []
    for bk_shape in bk:
        in_fm = perception.extract_fm(data.unsqueeze(0), bk_shape["kernels"])
        fm_repo = bk_shape["fm_repo"]
        fm_img = bk_shape["fm_img"]
        match_fm_shift, match_fm_idx, match_fm_value = load_shift(args, bk_shape, idx, in_fm, fm_repo, out_path)
        match_fm = fm_repo[match_fm_idx]
        match_fm_img = fm_img[match_fm_idx]
        shift_mfm = [torch.roll(match_fm[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
                     range(args.top_fm_k)]
        shift_mfm.append(torch.zeros_like(shift_mfm[0]))
        shift_mfm = torch.stack(shift_mfm)
        shift_mfm_img = torch.stack(
            [torch.roll(match_fm_img[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
             range(args.top_fm_k)])
        match_same, match_diff, same_percent = get_match_detail(shift_mfm, in_fm.squeeze(), match_fm_value)
        img_onside, img_offside, img_onside_uncertain = get_siding(args, data, match_same, match_diff,
                                                                   shift_mfm_img)
        visual_all(args, idx, img, data, in_fm, shift_mfm, same_percent, match_same, match_diff, img_onside,
                   img_offside, img_onside_uncertain, bk_shape, out_path)
        if (img_onside[-1] > 0).sum() > fm_repo[0, 0].sum():
            print("")
        group_count_conf = ((img_onside[-1] > 0).sum() / fm_repo[0, 0].sum()).item()
        print(f"{bk_shape['name']} group conf: {group_count_conf:.2f}, th: {args.group_count_conf_th}")
        if group_count_conf < args.group_count_conf_th:
            continue

        groups.append({
            "name": bk_shape["name"],
            "onside": img_onside[-1],
            "count_conf": group_count_conf
        })
    return groups


def main():
    args = args_utils.get_args()
    bk_shapes = {"data_circle", "data_square", "data_triangle"}
    args.batch_size = 1
    args.data_types = args.exp_name
    # train_loader, val_loader = prepare_kp_sy_data(args)
    os.makedirs(config.output / f"{args.exp_name}", exist_ok=True)
    image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)

    # load background knowledge
    bk = []
    for bk_shape in bk_shapes:
        kernels = torch.load(config.output / bk_shape / f"kernels.pt").to(args.device)
        fm_data = torch.load(config.output / bk_shape / f"fms.pt").to(args.device)
        fm_img = fm_data[:, 0:1]
        fm_repo = fm_data[:, 1:]
        bk.append({
            "name": bk_shape,
            "kernels": kernels,
            "fm_img": fm_img,
            "fm_repo": fm_repo
        })

    for idx in tqdm(range(len(image_paths)), "matching image"):
        groups = img2groups(args, bk, image_paths[idx], idx, img)
        print(f"{idx}: {len(groups)}")


if __name__ == "__main__":
    main()
