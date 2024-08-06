# Created by jing at 17.06.24
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import config
import grouping
from percept import perception
from utils import visual_utils, file_utils, args_utils, data_utils
from src.alpha import alpha


def prepare_kp_sy_data(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = perception.ShapeDataset(args, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def main():
    args = args_utils.get_args()
    # data file
    args.data_types = ["data_triangle"]
    train_loader, val_loader = prepare_kp_sy_data(args)
    os.makedirs(config.output / f"kp_sy_{args.exp_name}", exist_ok=True)

    for images, labels in train_loader:
        fms = perception.extract_fm(images.to(args.device))
        relations = alpha.alpha(args, fms, images)
    print("program finished")


if __name__ == "__main__":
    main()
