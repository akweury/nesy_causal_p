# Created by shaji at 03/08/2024

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import os
import config
from src.percept import perception
from src.utils import args_utils, data_utils


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
            image_tensor[p_i, 0] = -1 # unknown fm
        image_tensor[p_i, 1:] = nz_positions[p_i]
    return image_tensor

def main():
    args = args_utils.get_args()
    args.batch_size = 1

    # load learned triangle fms
    tri_fms = torch.load(config.output / f"kp_sy_triangle_only" / f"fms.pt").to(args.device)
    # tri_nesy_img = torch.load(config.output / f"kp_sy_triangle_only" / f"img_tensors.pt").to(args.device)
    # tri_nesy_img = tri_nesy_img.unique(dim=0)

    # load test dataset
    args.data_types = args.exp_name
    train_loader, val_loader = prepare_kp_sy_data(args)
    os.makedirs(config.output / f"kp_sy_{args.exp_name}", exist_ok=True)
    for data, labels in tqdm(train_loader):
        nesy_tensor = img2nesy_tensor(args, data, tri_fms)


        print("image done.")
        # _, nzs_patches, nz_positions = data_utils.find_submatrix(data.squeeze(), args.kernel)
        # nz_positions[:, 0] -= nz_positions[0, 0].item()
        # nz_positions[:, 1] -= nz_positions[0, 1].item()
        # image_tensor = torch.zeros((25, 3)) + len(tri_fms)
        #
        # for p_i, p in enumerate(nzs_patches):
        #     for f_i, fm in enumerate(tri_fms):
        #         if torch.equal(fm, p):
        #             image_tensor[p_i, 0] = f_i
        #     if image_tensor[p_i, 0] == len(tri_fms):
        #         raise ValueError("fm not found in patches")
        #     image_tensor[p_i, 1:] = nz_positions[p_i]

    print("program is finished.")


if __name__ == "__main__":
    main()
