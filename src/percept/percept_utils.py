# Created by jing at 10.12.24
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F  # Import F for functional operations
from tqdm import tqdm
from PIL import Image, ImageDraw

from src.utils import chart_utils, data_utils
from src import dataset


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


def bw_layer(segment):
    bw_img = data_utils.rgb2bw(segment.permute(1, 2, 0).numpy(), resize=64)
    bw_img = bw_img.unsqueeze_(0)
    return bw_img


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
