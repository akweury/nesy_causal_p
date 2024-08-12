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


def prepare_fms(args):
    fms = torch.load(config.output / f"kp_sy_{args.exp_name}" / f"fms.pt").to(args.device)
    return fms


def main():
    args = args_utils.get_args()
    os.makedirs(config.output / f"kp_sy_{args.exp_name}", exist_ok=True)
    # data file
    args.data_types = ["data_triangle"]
    train_loader, val_loader = prepare_kp_sy_data(args)
    fms = prepare_fms(args)
    image_tensors_all = []
    for data, labels in tqdm(train_loader):
        image_tensors = []
        for img_i, image in enumerate(data):
            non_zero_patches, non_zero_positions = data_utils.find_submatrix(image.squeeze())
            non_zero_positions[:, 0] -= non_zero_positions[0, 0].item()
            non_zero_positions[:, 1] -= non_zero_positions[0, 1].item()
            image_tensor = torch.zeros((25, 3))
            for p_i, p in enumerate(non_zero_patches):
                for f_i, fm in enumerate(fms):
                    if torch.equal(fm, p):
                        image_tensor[p_i, 0] = f_i
                image_tensor[p_i, 1:] = non_zero_positions[p_i]

            image_tensors.append(image_tensor.unsqueeze(0))
        image_tensors = torch.cat(image_tensors, dim=0)
        image_tensors_all.append(image_tensors)
    torch.save(torch.cat(image_tensors_all,dim=0), config.output / f"kp_sy_{args.exp_name}" / f"img_tensors.pt")

    print("program finished")


if __name__ == "__main__":
    main()
