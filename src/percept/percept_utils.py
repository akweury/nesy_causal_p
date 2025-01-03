# Created by jing at 10.12.24
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F  # Import F for functional operations
from tqdm import tqdm
from PIL import Image, ImageDraw

from src.utils import chart_utils, data_utils
from src import dataset, bk


def matrix_similarity(mat1, mat2):
    # Flatten the matrices from (N, W, H) to (N*W*H)
    g1_flat = mat1.view(mat1.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    g2_flat = mat2.view(mat2.size(0), -1)  # Shape: (N2, 192 * 64 * 64)
    # Compute cosine similarity between all pairs (N1 x N2 matrix)
    similarity_matrix = torch.mm(g1_flat, g2_flat.t()) / (
            g2_flat.sum(dim=1).unsqueeze(0) + 1e-20)  # Shape: (N1, N2)
    return similarity_matrix


def get_match_detail(mem_fm, visual_fm):
    mask_mem = mem_fm != 0
    same_fm = ((mem_fm == visual_fm) * mask_mem).sum(dim=1).to(torch.float32)
    mask_full_match = torch.all(mem_fm == visual_fm, dim=1) * torch.any(mask_mem,
                                                                        dim=1)
    mask_any_mismatch = torch.any((mem_fm == visual_fm) * mask_mem,
                                  dim=1) * torch.any(
        mask_mem, dim=1) * ~mask_full_match
    all_same_fm = same_fm * mask_full_match
    any_diff_fm = same_fm * mask_any_mismatch
    same_percent = mask_full_match.sum(dim=[1, 2]) / (
            mem_fm.sum(dim=1).bool().sum(dim=[1, 2]) + 1e-20)
    return all_same_fm, any_diff_fm, same_percent


def get_siding(data, match_fm_img):
    data_onside = torch.stack(
        [(match_fm_img[i].squeeze()) for i in range(len(match_fm_img))])
    data_offside = torch.stack(
        [((match_fm_img[i].squeeze() == 0)) for i in range(len(match_fm_img))])
    return data_onside, data_offside


def prepare_data(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    continue_shape_dataset = dataset.BasicShapeDataset(args, transform=transform)
    train_size = int(0.8 * len(continue_shape_dataset))
    val_size = len(continue_shape_dataset) - train_size
    train_dataset, val_dataset = random_split(continue_shape_dataset,
                                              [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def visual_all(group_img_name, img, bw_img, data_fm_shifted, fm_best, max_value,
               fm_best_same, fm_best_diff, data_onside, data_offside):
    in_fm_img = data_fm_shifted.squeeze().sum(dim=0)
    mask_img = chart_utils.color_mapping(bw_img, 1, "IN")
    norm_factor = max([in_fm_img.max(), fm_best.sum(dim=1).max()])
    in_fm_norm_img = chart_utils.color_mapping(in_fm_img, norm_factor, "IN_FM")
    blank_img = chart_utils.color_mapping(torch.zeros_like(bw_img), norm_factor,
                                          "")
    compare_imgs = []

    for i in range(min(10, len(fm_best))):
        best_fm_img = fm_best[i].sum(dim=0)
        # norm_factor = max([in_fm_img.max(), best_fm_img.max()])
        match_percent = f"{int(max_value[i].item() * 100)}%"

        repo_fm_img = chart_utils.color_mapping(best_fm_img, norm_factor,
                                                "RECALL_FM")
        repo_fm_best_same = chart_utils.color_mapping(fm_best_same[i],
                                                      norm_factor,
                                                      f"SAME FM {match_percent}")
        repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff[i],
                                                      norm_factor,
                                                      "DIFF FM")
        data_onside_img = chart_utils.color_mapping(data_onside[i], 1,
                                                    "Onside Objs")
        data_offside_img = chart_utils.color_mapping(data_offside[i], 1,
                                                     "Offside Objs")

        compare_imgs.append(chart_utils.hconcat_imgs(
            [img, mask_img, in_fm_norm_img, repo_fm_img, repo_fm_best_same,
             repo_fm_best_diff, data_onside_img,
             data_offside_img]))
    # last row: combined result
    data_mask = bw_img != 0
    onside_comb = (data_onside.sum(dim=0).float() * data_mask)
    onside_comb_img = chart_utils.color_mapping(onside_comb, 1, "Onside Comb.")

    offside_comb = (data_offside.sum(dim=0).float() * data_mask)
    offside_comb_img = chart_utils.color_mapping(offside_comb, 1,
                                                 "Offside Comb.")

    compare_imgs.append(chart_utils.hconcat_imgs(
        [img, mask_img, in_fm_norm_img, blank_img, blank_img, blank_img,
         onside_comb_img, offside_comb_img]))
    compare_imgs = chart_utils.vconcat_imgs(compare_imgs)
    compare_imgs = cv2.cvtColor(compare_imgs, cv2.COLOR_BGR2RGB)

    cv2.imwrite(group_img_name, compare_imgs)


def groups2positions(groups):
    positions = torch.zeros((len(groups), 2))
    for g_i, group in enumerate(groups):
        positions[g_i] = group.pos

    return positions


def groups2labels(groups, label_type):
    labels = torch.zeros((len(groups), 1))
    for g_i, group in enumerate(groups):
        if label_type == "color":
            labels[g_i] = group.color
        elif label_type == "shape":
            labels[g_i] = group.name
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    return labels


def proximity_distance(u, v):
    return np.linalg.norm(u - v)


def similarity_distance(a, b, weights):
    """Compute weighted distance between object i and j."""
    a_dict = bk.tensor2dict(a)
    b_dict = bk.tensor2dict(b)
    (x1, y1) = a_dict["position"]
    (x2, y2) = b_dict["position"]
    shape_i = a_dict["shape"]
    shape_j = a_dict["shape"]
    color_i = a_dict["color"]
    color_j = b_dict["color"]
    w_pos = weights[0]
    w_shape = weights[1]
    w_color = weights[2]

    # Positional distance (Euclidean)
    pos_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Shape distance
    shape_dist = 0 if (shape_i == shape_j) else 1

    # Color distance
    color_dist = 0 if (color_i == color_j) else 1

    # Weighted sum
    dist = w_pos * pos_dist + w_shape * shape_dist + w_color * color_dist

    return dist
