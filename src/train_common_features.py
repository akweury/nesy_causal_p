# Created by shaji at 03/08/2024

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb
from rtpt import RTPT

import os
import config
from src.percept import perception
from src.utils import file_utils, args_utils, data_utils, log_utils


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


def main(exp_name):
    args = args_utils.get_args()
    # data file
    args.data_types = ["data_trianglesquare", "data_trianglecircle", "data_triangle"]
    train_loader, val_loader = prepare_kp_sy_data(args)
    model = perception.SimpleCNN().to(args.device)
    model_dict_path = config.output / exp_name / "detector_model.pth"
    model.load_state_dict(torch.load(model_dict_path))

    # Assume val_loader is the DataLoader for the validation set
    input_dim = 128  # This should match the dimensionality of the FC layer input
    target_label = 0  # We want to maximize the logit for label 0

    log_utils.init_wandb(pj_name=f"FM-{dataset_name}-mask", archi="FCN")

    mask_optimizer = perception.MaskOptimizer(input_dim, target_label)
    perception.optimize_mask(model, train_loader,val_loader, mask_optimizer, target_label)
    torch.save(mask_optimizer.mask, config.output / exp_name / "mask.pth")

if __name__ == "__main__":
    dataset_name = "kp_sy"
    main(dataset_name)
