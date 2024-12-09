# Created by shaji at 03/08/2024
import logging
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

import config
from src.percept.group import Group
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


def get_shifted_matrics(img):
    # Create grid of all (x_shift, y_shift) pairs
    shifts = [(-x, -y) for x in range(img.shape[-2]) for y in range(img.shape[-1])]
    # Generate the shifted images as a batch
    shifted_images = torch.cat([torch.roll(img, shifts=(row, col), dims=(-2, -1)) for row, col in shifts])
    return shifted_images


def similarity_fast(img_fm, fm_repo):
    row_num, col_num = img_fm.shape[-2], img_fm.shape[-1]

    non_zero_mask = fm_repo != 0
    img_fm_all_shift = get_shifted_matrics(img_fm)
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

    img_fm_all_shift = get_shifted_matrics(img_fm)

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


def get_pair(args, img_fm, fm_repo):
    img_fm_shifts = get_shifted_matrics(img_fm)
    # Flatten the matrices from (192, 64, 64) to (192*64*64)
    g1_flat = img_fm_shifts.view(img_fm_shifts.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    g2_flat = fm_repo.view(fm_repo.size(0), -1)  # Shape: (N2, 192 * 64 * 64)
    # Compute cosine similarity between all pairs (N1 x N2 matrix)
    similarity_matrix = torch.mm(g1_flat, g2_flat.t()) / g2_flat.sum(dim=1).unsqueeze(0)  # Shape: (N1, N2)

    top_values, top_indices = torch.topk(similarity_matrix.flatten(), args.top_fm_k)
    # Filter out values below the threshold
    mask = top_values >= args.fm_th
    top_values = top_values[mask]
    top_indices = top_indices[mask]
    if len(top_values) == 0:
        return None, None, None
    # top_indices_2d: (best_shift_idx_1d, best_fm_idx)
    top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()
    # best shift idx 2d
    best_shift_idx = torch.stack(torch.unravel_index(top_indices_2d[:, 0], img_fm.shape[-2:])).t()
    # best fm idx
    best_fm_idx = top_indices_2d[:, 1]

    return best_shift_idx, best_fm_idx, top_values


# shifted_images = torch.cat([torch.roll(img, shifts=(row, col), dims=(-2, -1)) for row, col in shifts])


def fm_intersection(img_fm, fm_repo):
    img_fm_shifts = get_shifted_matrics(img_fm)

    # img_fm_shifts, row_shifts, col_shifts = data_utils.shift_content_to_top_left(img_fm)
    chart_utils.visual_np_array(img_fm_shifts[0, 0].squeeze().numpy())
    # Flatten the matrices from (192, 64, 64) to (192*64*64)
    g1_flat = img_fm_shifts.view(img_fm_shifts.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    g2_flat = fm_repo.view(fm_repo.size(0), -1)  # Shape: (N2, 192 * 64 * 64)

    similarity_matrix = torch.mm(g1_flat, g2_flat.t()) / g2_flat.sum(dim=1).unsqueeze(0)  # Shape: (N1, N2)
    top_values, top_indices = torch.topk(similarity_matrix.flatten(), 5)
    # top_indices_2d: (best_shift_idx_1d, best_fm_idx)
    top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()
    # best shift idx 2d
    best_shift_idx = torch.stack(torch.unravel_index(top_indices_2d[:, 0], img_fm.shape[-2:])).t()
    # best fm idx
    best_fm_idx = top_indices_2d[:, 1]
    return best_shift_idx, best_fm_idx, top_values

    # g2_flat = torch.where(g2_flat != 0, torch.tensor(1), g2_flat)
    # g1_flat_mask = g1_flat > 0
    # equal_matrix = torch.zeros_like(g1_flat)
    # equal_matrix[g1_flat_mask] = (g1_flat == g2_flat)[g1_flat_mask].float()
    # equal_counts = equal_matrix.sum(dim=1)
    # fm_counts = img_fm.sum(dim=[1, 2, 3])
    # equal_percents = equal_counts / (fm_counts + 1e-20)
    # top_fm_similar_percents, top_fm_indices = torch.topk(equal_percents, 5)
    # top_fm_shifts = torch.stack([row_shifts[top_fm_indices], col_shifts[top_fm_indices]], dim=1)
    # top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()
    # max_idx_shift = torch.stack(torch.unravel_index(top_indices_2d[:, 0], img_fm.shape[-2:])).t()
    # max_idx_fm = top_indices_2d[:, 1]

    # return top_fm_shifts, top_fm_indices, top_fm_similar_percents


def visual_all(group_img_name, img, data, data_fm_shifted, fm_best, max_value, fm_best_same, fm_best_diff, data_onside,
               data_offside):
    in_fm_img = data_fm_shifted.squeeze().sum(dim=0)
    mask_img = chart_utils.color_mapping(data.squeeze(), 1, "IN")
    norm_factor = max([in_fm_img.max(), fm_best.sum(dim=1).max()])
    in_fm_norm_img = chart_utils.color_mapping(in_fm_img, norm_factor, "IN_FM")
    blank_img = chart_utils.color_mapping(torch.zeros_like(data[0]), norm_factor, "")
    compare_imgs = []

    for i in range(min(10, len(fm_best))):
        best_fm_img = fm_best[i].sum(dim=0)
        # norm_factor = max([in_fm_img.max(), best_fm_img.max()])
        match_percent = f"{int(max_value[i].item() * 100)}%"

        repo_fm_img = chart_utils.color_mapping(best_fm_img, norm_factor, "RECALL_FM")
        repo_fm_best_same = chart_utils.color_mapping(fm_best_same[i], norm_factor, f"SAME FM {match_percent}")
        repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff[i], norm_factor, "DIFF FM")
        data_onside_img = chart_utils.color_mapping(data_onside[i], 1, "Onside Objs")
        data_offside_img = chart_utils.color_mapping(data_offside[i], 1, "Offside Objs")

        compare_imgs.append(chart_utils.concat_imgs(
            [img, mask_img, in_fm_norm_img, repo_fm_img, repo_fm_best_same, repo_fm_best_diff, data_onside_img,
             data_offside_img]))
    # last row: combined result
    data_mask = data.squeeze() != 0
    onside_comb = (data_onside.sum(dim=0).float() * data_mask)
    onside_comb_img = chart_utils.color_mapping(onside_comb, 1, "Onside Comb.")

    offside_comb = (data_offside.sum(dim=0).float() * data_mask)
    offside_comb_img = chart_utils.color_mapping(offside_comb, 1, "Offside Comb.")

    compare_imgs.append(chart_utils.concat_imgs(
        [img, mask_img, in_fm_norm_img, blank_img, blank_img, blank_img, onside_comb_img, offside_comb_img]))
    compare_imgs = chart_utils.vconcat_imgs(compare_imgs)
    compare_imgs = cv2.cvtColor(compare_imgs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(group_img_name, compare_imgs)


def get_match_detail(mem_fm, visual_fm):
    mask_mem = mem_fm != 0
    same_fm = ((mem_fm == visual_fm) * mask_mem).sum(dim=1).to(torch.float32)
    mask_full_match = torch.all(mem_fm == visual_fm, dim=1) * torch.any(mask_mem, dim=1)
    mask_any_mismatch = torch.any((mem_fm == visual_fm) * mask_mem, dim=1) * torch.any(
        mask_mem, dim=1) * ~mask_full_match
    all_same_fm = same_fm * mask_full_match
    any_diff_fm = same_fm * mask_any_mismatch
    same_percent = mask_full_match.sum(dim=[1, 2]) / (mem_fm.sum(dim=1).bool().sum(dim=[1, 2]) + 1e-20)
    return all_same_fm, any_diff_fm, same_percent


def get_siding(data, match_fm_img):
    # data_mask = data.squeeze() != 0
    data_onside = torch.stack([(match_fm_img[i].squeeze()) for i in range(len(match_fm_img))])
    data_offside = torch.stack([((match_fm_img[i].squeeze() == 0)) for i in range(len(match_fm_img))])
    # data_onside.append((match_fm_img.squeeze().sum(dim=0).float() * data_mask))
    # data_onside = torch.stack(data_onside)
    # data_onside_uncertain = [(match_diff[i] * data_mask) for i in range(len(match_diff))]
    # data_onside_uncertain.append((match_diff.sum(dim=0).float() * data_mask))
    # data_onside_uncertain = torch.stack(data_onside_uncertain)
    # data_offside.append(((match_fm_img.squeeze().sum(dim=0) == 0).float() * data_mask))
    # data_offside = torch.stack(data_offside)
    return data_onside, data_offside


def load_shift(args, bk_shape, idx, in_fm, fm_repo, out_path):
    buffer_file = out_path / f"shift_{bk_shape['name']}_{idx}.pt"
    if os.path.exists(buffer_file):
        data = torch.load(buffer_file)
        match_fm_shift = data['match_fm_shift']
        match_fm_idx = data['match_fm_idx']
        top_values = data["top_values"]
    else:
        # match_fm_shift, match_fm_idx, top_values = fm_intersection(in_fm, fm_repo)
        match_fm_shift, match_fm_idx, top_values = get_pair(args, in_fm, fm_repo)
        # chart_utils.visual_np_array(in_fm[0, 0].squeeze().numpy())
        data = {"match_fm_shift": match_fm_shift,
                "match_fm_idx": match_fm_idx,
                "top_values": top_values}
        torch.save(data, buffer_file)
    top_values = torch.cat((top_values, torch.zeros(1)))
    return match_fm_shift, match_fm_idx, top_values


def calculate_non_zero_distances(tensor):
    # Get coordinates of all non-zero elements
    non_zero_coords = torch.nonzero(tensor, as_tuple=False)

    if non_zero_coords.size(0) == 0:
        return torch.tensor([])  # Return an empty tensor if no non-zero elements

    # Calculate the centroid of non-zero elements
    center = non_zero_coords.float().mean(dim=0)

    # Calculate distances from each non-zero point to the centroid
    distances = torch.norm(non_zero_coords.float() - center, dim=1)

    return distances


def tensor_similarity(onside_tensor, ref_tensor):
    # Calculate non-zero distances for each tensor
    ref_coords = torch.nonzero(ref_tensor, as_tuple=False).float()
    onside_coords = torch.nonzero(onside_tensor, as_tuple=False).float()

    # If either tensor has no non-zero elements, return an empty tensor for distances
    if ref_coords.size(0) == 0 or onside_coords.size(0) == 0:
        return torch.tensor(0)  # No distances to calculate

    # Step 2: Calculate pairwise distances between points in coords2 to coords1
    # Expand dimensions to (N, M, 2) and (M, N, 2) for broadcasting
    onside_coords_exp = onside_coords.unsqueeze(0)  # Shape: (1, M, 2)
    ref_coords_exp = ref_coords.unsqueeze(1)  # Shape: (N, 1, 2)

    # Calculate Euclidean distances
    distances = torch.norm(ref_coords_exp - onside_coords_exp, dim=2)  # Shape: (N, M)

    # Step 3: Find the minimum distance for each point in coords2 to any point in coords1
    min_distances = torch.min(distances, dim=1).values  # Shape: (N,)
    conf = (100 - min_distances.sum()) * 0.01
    if conf < 0:
        conf = torch.tensor(0)
    return conf


def eval_onside_conf(args, data, onsides, fm_imgs, bk_shape):
    fm_imgs = fm_imgs.squeeze()
    data_mask = data.squeeze() != 0
    onside_img = (onsides.sum(dim=0).float() * data_mask)
    onside_mask = onside_img > 0
    onside_mask = onside_mask.unsqueeze(0)
    onside_mask = torch.repeat_interleave(onside_mask, len(fm_imgs), dim=0)
    same = onside_mask.to(torch.float) == fm_imgs
    same[~onside_mask] = False
    group_count_conf = torch.zeros(len(onside_mask))
    for i in range(len(onside_mask)):
        onside_matrix = onside_mask.to(torch.float)[i].unsqueeze(0).unsqueeze(0)
        fm_matrix = fm_imgs[i].unsqueeze(0).unsqueeze(0)
        group_count_conf[i] = perception.matrix_equality(onside_matrix, fm_matrix)
        # group_count_conf[i] = tensor_similarity(onside_mask.to(torch.float)[i], fm_imgs[i])
    group_count_conf = torch.mean(group_count_conf)

    show_imgs = []
    for i in range(onside_mask.shape[0]):
        show_imgs.append(chart_utils.color_mapping(same[i].squeeze(), 1, f"TOP FM {i}"))
    show_imgs.append(chart_utils.color_mapping(onside_img.squeeze(), 1, f"Conf:{group_count_conf:.2f}"))

    return group_count_conf


def img2groups(args, bk, data, idx, img, out_path):
    groups = []
    for bk_shape in bk:
        in_fm = perception.one_layer_conv(data.unsqueeze(0), bk_shape["kernels"])
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
        match_same, match_diff, same_percent = get_match_detail(shift_mfm, in_fm.squeeze())

        img_onside, img_offside, img_onside_uncertain = get_siding(args, data, match_same, match_diff,
                                                                   shift_mfm_img, same_percent)

        visual_all(args, idx, img, data, in_fm, shift_mfm, same_percent, match_same, match_diff, img_onside,
                   img_offside, img_onside_uncertain, bk_shape, out_path)
        if (img_onside[-1] > 0).sum() > fm_repo[0, 0].sum():
            print("")

        group_count_conf = eval_onside_conf(args, img_onside[-1], shift_mfm_img.squeeze(), bk_shape, out_path)

        # print(f"{bk_shape['name']} group conf: {group_count_conf:.2f}, th: {args.group_count_conf_th}")
        if group_count_conf < args.group_count_conf_th:
            continue

        groups.append({
            "name": bk_shape["name"],
            "onside": img_onside[-1],
            "count_conf": group_count_conf
        })
    return groups


def calculate_center_position(tensor):
    # Find indices of nonzero elements
    nonzero_indices = torch.nonzero(tensor, as_tuple=False)
    # Calculate the mean position along each dimension (rows and columns)
    return nonzero_indices.float().mean(dim=0).numpy()


def get_individual_groups(shift_mfm_imgs):
    # Calculate center positions for each tensor
    centers = np.array([calculate_center_position(img) for img in shift_mfm_imgs])
    # Calculate pairwise distances between centers
    distances = cdist(centers, centers, metric='euclidean')
    # Use a list of clusters to group indices of tensors that are within the threshold
    clusters = []
    visited = set()

    # Iterate over each tensor's center to cluster them
    for i in range(len(centers)):
        if i in visited:
            continue
        # Start a new cluster
        cluster = [i]
        visited.add(i)

        # Find all tensors within the threshold distance
        for j in range(i + 1, len(centers)):
            radius_sum = data_utils.find_valid_radius(shift_mfm_imgs[i, 0]) + data_utils.find_valid_radius(
                shift_mfm_imgs[j, 0])
            if j not in visited and distances[i, j] < radius_sum * 1.5:
                cluster.append(j)
                visited.add(j)

        clusters.append(cluster)

    # Group tensors by clusters
    return clusters
    # clustered_tensors = [[shift_mfm_imgs[idx] for idx in cluster] for cluster in clusters]

    # return clustered_tensors


def img2groups_flexible(args, bk, data, data_idx, img, out_path):
    groups = []
    group_count = 0
    for bk_shape in bk:
        in_fm = perception.one_layer_conv(data.unsqueeze(0), bk_shape["kernels"])
        fm_repo = bk_shape["fm_repo"]  # (#fm, #kernel, row, col)
        fm_img = bk_shape["fm_img"]  # (#fm, #channel, row, col)
        match_fm_shift, match_fm_idx, match_fm_value = get_pair(args, in_fm, fm_repo)
        if match_fm_shift is None:
            continue
        match_fm = fm_repo[match_fm_idx]
        match_fm_img = fm_img[match_fm_idx]
        shift_mfm = torch.stack([torch.roll(match_fm[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
                                 range(len(match_fm_value))])
        shift_mfm_img = torch.stack(
            [torch.roll(match_fm_img[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
             range(len(match_fm_value))])
        match_same, match_diff, same_percent = get_match_detail(shift_mfm, in_fm.squeeze())

        # check number of groups
        individual_indices = get_individual_groups(shift_mfm_img)
        for indices in individual_indices:
            clustered_tensors = shift_mfm_img[indices]
            clustered_fm = shift_mfm[indices]
            clustered_same_percent = same_percent[indices]
            clustered_fm_same = match_same[indices]
            clustered_fm_diff = match_diff[indices]
            onside, offside = get_siding(data, clustered_tensors)
            group_img_name = str(out_path / f'img_{data_idx}_group_{group_count}_{bk_shape["name"]}.png')
            group_count_conf = eval_onside_conf(args, data, onside, clustered_tensors, bk_shape)

            if group_count_conf < args.group_count_conf_th:
                continue
            visual_all(group_img_name, img, data, in_fm, clustered_fm, clustered_same_percent, clustered_fm_same,
                       clustered_fm_diff, onside, offside)
            groups.append({
                "id": group_count,
                "name": bk_shape["name"],
                "onside": (onside.sum(dim=0) > 0).float(),
                "count_conf": group_count_conf
            })
            group_count += 1

    return groups


def recall_mem_fms(args, in_fm, mem_fms, core_img, mem_imgs):
    shifted_fms = get_shifted_matrics(in_fm)
    shifted_imgs = get_shifted_matrics(core_img.unsqueeze(0))

    # edge similarity
    img_edge = perception.detect_edge(shifted_imgs.float())
    repo_edge = perception.detect_edge(mem_imgs.float())
    similarity_edge_fms = perception.matrix_equality(repo_edge, img_edge)

    # fm similarity
    similarity_fms = perception.matrix_equality(mem_fms, shifted_fms)

    similarity_matrix = (similarity_edge_fms + similarity_fms * similarity_edge_fms.max()).permute(1, 0)
    # similarity_scale = perception.matrix_scale_similarity(img_edge, repo_edge)
    # similarity_matrix[similarity_scale > 1.5] = 0
    # similarity_matrix[similarity_scale < 0.5] = 0
    # args.fm_th = 0.1
    # torch.sort(similarity_matrix.flatten(), descending=True)
    # top_values, top_indices = torch.topk(similarity_matrix.flatten(), args.top_fm_k)
    top_values, top_indices = torch.sort(similarity_matrix.flatten(), descending=True)
    # Filter out values below the threshold
    # mask = top_values >= args.fm_th
    top_values = top_values[:5]
    top_indices = top_indices[:5]
    if len(top_values) == 0:
        return None
    # top_indices_2d: (best_shift_idx_1d, best_fm_idx)
    top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()
    # best shift idx 2d
    best_shift_idx = torch.stack(torch.unravel_index(top_indices_2d[:, 0], shifted_fms.shape[-2:])).t()
    # best fm idx
    best_fm_idx = top_indices_2d[:, 1]

    args.logger.debug(f"recall {len(top_values)} possible fms, highest conf: {top_values[0]:.2f}")
    return best_fm_idx, best_shift_idx, top_values


def fit_group_fm_to_mem_fm(args, shifted_fms, mem_fms, shifted_imgs, mem_imgs):
    # edge fm
    img_edge = perception.detect_edge(shifted_imgs.float())
    repo_edge = perception.detect_edge(mem_imgs.float())
    similarity_edge_fms = perception.matrix_equality(repo_edge, img_edge).permute(1, 0)
    similarity_fms = perception.matrix_similarity(mem_fms, shifted_fms).permute(1, 0)
    similarity_matrix = similarity_edge_fms + similarity_fms * similarity_edge_fms.max()

    # args.fm_th = 0.1
    top_values, top_indices = torch.sort(similarity_matrix.flatten(), descending=True)
    # top_values, top_indices = torch.topk(similarity_fms.flatten(), args.top_fm_k)
    # Filter out values below the threshold

    mask = top_values >= args.fm_th * 0.8
    top_values = top_values[mask][:10]
    top_indices = top_indices[mask][:10]
    if len(top_values) == 0:
        return None, None, None
    # top_indices_2d: (best_shift_idx_1d, best_fm_idx)
    top_indices_2d = torch.stack(torch.unravel_index(top_indices, similarity_matrix.shape)).t()
    # best shift idx 2d
    best_shift_idx = torch.stack(torch.unravel_index(top_indices_2d[:, 0], shifted_fms.shape[-2:])).t()
    # best fm idx
    best_fm_idx = top_indices_2d[:, 1]

    return best_shift_idx, best_fm_idx, top_values


# def percept_gestalt_groups(args, group_bk, img, output_file_prefix):
#     pixel_groups = percept_pixel_groups(args, group_bk, img, output_file_prefix)
#     pixel_groups_2nd = percept_2nd_pixel_groups(args, pixel_groups, group_bk, img)
#     return pixel_groups_2nd


def fit_groups(args, b_i, img, core_image, mem_fm_data, output_file_prefix, fms, bk, parent_groups=None):
    mem_fm_idx, mem_fm_shift, mem_fm_conf = mem_fm_data
    mem_fms = bk["fm_repo"][mem_fm_idx]
    mem_imgs = bk["fm_img"][mem_fm_idx]
    mem_fms = torch.stack(
        [torch.roll(mem_fms[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1)) for i in range(len(mem_fm_conf))])
    mem_imgs = torch.stack(
        [torch.roll(mem_imgs[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1)) for i in range(len(mem_fm_conf))])

    match_same, match_diff, same_percent = get_match_detail(mem_fms, fms.squeeze())

    onside, offside = get_siding(core_image, mem_imgs)
    group_img_name = output_file_prefix + str(f'_group_{bk["name"]}.png')
    group_count_conf = eval_onside_conf(args, core_image, onside, mem_imgs, bk)

    visual_all(group_img_name, img, core_image, fms, mem_fms, same_percent, match_same,
               match_diff, onside, offside)
    # instantiate detected group
    onside_coverage = calc_onside_coverage(core_image.squeeze(), onside.squeeze())
    group = Group(id=b_i, name=bk["name"],
                  input_signal=img,
                  onside_signal=(onside.sum(dim=0) > 0),
                  memory_signal=mem_imgs,
                  parents=parent_groups,
                  onside_conf=group_count_conf,
                  coverage=onside_coverage)
    args.logger.debug(
        f" Found group: {bk['name']}: [conf|th  {group_count_conf:.2f}|{args.group_count_conf_th}]"
        f" [coverage: {group.onside_coverage:.2f}]")

    # individual_indices = get_individual_groups(mem_imgs)
    # groups = []
    # for i, indices in enumerate(individual_indices):
    #     clustered_tensors = mem_imgs[indices]
    #     clustered_fm = mem_fms[indices]
    #     clustered_same_percent = same_percent[indices]
    #     clustered_fm_same = match_same[indices]
    #     clustered_fm_diff = match_diff[indices]
    #     onside, offside = get_siding(core_image, clustered_tensors)
    #     group_img_name = output_file_prefix + str(f'_group_{i}_{bk["name"]}.png')
    #     group_count_conf = eval_onside_conf(args, core_image, onside, clustered_tensors, bk)
    #     onside_coverage = calc_onside_coverage(core_image.squeeze(), onside.squeeze())
    #     if group_count_conf >= args.group_count_conf_th:
    #         visual_all(group_img_name, img, core_image, fms, clustered_fm,
    #                    clustered_same_percent, clustered_fm_same, clustered_fm_diff, onside, offside)
    #         # instantiate detected group
    #         groups.append(Group(id=i, name=bk["name"],
    #                             input_signal=img,
    #                             onside_signal=(onside.sum(dim=0) > 0),
    #                             memory_signal=clustered_tensors,
    #                             parents=None,
    #                             onside_conf=group_count_conf,
    #                             coverage=onside_coverage))
    #         args.logger.debug(
    #             f" Found group: {bk['name']}: [conf|th  {group_count_conf:.2f}|{args.group_count_conf_th}]"
    #             f" [coverage: {groups[-1].onside_coverage:.2f}]")
    # args.logger.debug(f"clustering fms to {len(groups)} groups")
    return group


def calc_onside_coverage(core_image, onside):
    onside_mask = onside.sum(dim=0).bool()
    onside_coverage = core_image[onside_mask].bool().sum() / core_image.bool().sum()
    return onside_coverage


def percept_feature_groups(args, bk, img, output_file_prefix):
    groups = []
    # segment the scene into separate parts
    segments = perception.detect_connected_regions(img)
    # convert to fms
    for segment in segments:
        segment_groups = []
        for b_i, bk_shape in enumerate(bk):
            core_image = data_utils.rgb2bw(segment.permute(1, 2, 0).numpy(), resize=64)
            fms = perception.one_layer_conv(core_image.unsqueeze(0), bk_shape["kernels"].float())
            # recall the memory
            mem_fms_data = recall_mem_fms(args, fms, bk_shape["fm_repo"], core_image, bk_shape["fm_img"])

            if mem_fms_data is not None:
                # check number of groups
                bk_group = fit_groups(args, b_i, img, core_image, mem_fms_data, output_file_prefix, fms, bk_shape)
                segment_groups.append(bk_group)
                # grouping the memory to new groups
        best_idx = torch.tensor([g.conf for g in segment_groups]).argmax()
        groups.append(segment_groups[best_idx])
    return groups


def percept_objects(args, input_groups, bk, img, output_file_prefix):
    """clustering feature groups together based on its position
        if the common area of two groups still related to one memory fm, then merge these two groups as one
    """
    output_groups = []
    for i in range(len(input_groups)):
        base_groups = input_groups[i]

        pass
    pass


def percept_object_groups(args, input_groups, bk, img, output_file_prefix):
    """ group objects as a high level group """
    output_groups = []
    group_count = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int32)
    gray = cv2.resize(gray.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    gray[gray == 211] = 0
    core_image = torch.from_numpy(gray).unsqueeze(0)

    # merge images in the type groups into one image
    onside_mask = torch.cat([g.onside.unsqueeze(0) for g in input_groups], dim=0)
    onside_mask = onside_mask.sum(dim=0, keepdim=True).bool()

    for b_i, bk_shape in enumerate(bk):
        fms = perception.one_layer_conv(onside_mask.unsqueeze(0).to(torch.float32), bk_shape["kernels"].float())
        # recall the memory
        mem_fms_data = recall_mem_fms(args, fms, bk_shape["fm_repo"], core_image, bk_shape["fm_img"])
        if mem_fms_data is not None:
            # check number of groups
            bk_group = fit_groups(args, b_i, img, core_image, mem_fms_data, output_file_prefix, fms, bk_shape,
                                  parent_groups=input_groups)
            output_groups.append(bk_group)
            # grouping the memory to new groups
    best_idx = torch.tensor([g.conf for g in output_groups]).argmax()
    best_group = output_groups[best_idx]

    # onside_img = torch.zeros_like(core_image)
    # onside_img[onside_mask] = core_image[onside_mask]
    # in_fm = perception.one_layer_conv(onside_img.unsqueeze(0), bk[i]["kernels"])
    # fm_repo = bk[i]["fm_repo"]  # (#fm, #kernel, row, col)
    # fm_img = bk[i]["fm_img"]  # (#fm, #channel, row, col)
    # img_fm_shifts = get_shifted_matrics(in_fm)
    # img_shifts = get_shifted_matrics(core_image.unsqueeze(0))

    # match_fm_shift, match_fm_idx, match_fm_value = fit_group_fm_to_mem_fm(args, img_fm_shifts, fm_repo, img_shifts,
    #                                                                       fm_img)
    # if match_fm_shift is not None:
    #     match_fm = fm_repo[match_fm_idx]
    #     match_fm_img = fm_img[match_fm_idx]
    #     shift_mfm = torch.stack([torch.roll(match_fm[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
    #                              range(len(match_fm_value))])
    #     shift_mfm_img = torch.stack(
    #         [torch.roll(match_fm_img[i], shifts=tuple(match_fm_shift[i]), dims=(-2, -1)) for i in
    #          range(len(match_fm_value))])
    #     match_same, match_diff, same_percent = get_match_detail(in_fm, shift_mfm)
    #
    #     onside, offside = get_siding(core_image, shift_mfm_img)
    #     group_img_name = output_file_prefix + str(f'2nd_group_{group_count}_{bk[i]["name"]}.png')
    #     group_count_conf = eval_onside_conf(args, core_image, onside, shift_mfm_img, bk[i])
    #
    #     visual_all(group_img_name, img, core_image, in_fm, shift_mfm, same_percent, match_same,
    #                match_diff, onside, offside)
    #     onside_coverage = calc_onside_coverage(core_image.squeeze(), onside.squeeze())
    #
    #     new_group = Group(id=group_count,
    #                       onside_signal=(onside.sum(dim=0) > 0),
    #                       input_signal=img,
    #                       memory_signal=shift_mfm_img,
    #                       parents=input_groups,
    #                       name=bk[i]["name"],
    #                       onside_conf=1,
    #                       coverage=onside_coverage
    #                       )
    #
    #     new_group.generate_tensor()
    #     output_groups.append(new_group)
    #     group_count += 1

    return best_group


def main():
    args = args_utils.get_args()
    bk_shapes = {"data_circle", "data_square", "data_triangle"}
    args.batch_size = 1
    args.data_types = args.exp_name
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
