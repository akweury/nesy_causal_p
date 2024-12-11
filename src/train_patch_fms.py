# Created by jing at 28.11.24

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os

import config
from src.percept import perception
from src.utils import args_utils, data_utils
from src import dataset


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


def prepare_mnist_data(args):
    import torchvision.datasets as datasets
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    minst_dataset = datasets.MNIST(root='./data', train=True, transform=transform,
                                   download=True)
    train_size = int(0.8 * len(minst_dataset))
    val_size = len(minst_dataset) - train_size
    train_dataset, val_dataset = random_split(minst_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader


def draw_training_history(train_losses, val_losses, val_accuracies, path):
    # Plotting the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(path / 'loss_history.png')  # Save the figure

    # Plotting the validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(path / 'accuracy_history.png')  # Save the figure


def draw_angle():
    ls_up = [(2, 2), (2, 0)]
    ls_upright = [(2, 2), (4, 0)]
    ls_right = [(2, 2), (4, 2)]
    ls_downright = [(2, 2), (4, 4)]
    ls_down = [(2, 2), (2, 4)]
    ls_downleft = [(2, 2), (0, 4)]
    ls_left = [(2, 2), (0, 2)]
    ls_topleft = [(2, 2), (0, 0)]

    directions = [ls_up, ls_upright, ls_right, ls_downright, ls_down, ls_downleft,
                  ls_left, ls_topleft]

    angle_imgs = []
    for d_i in range(len(directions) - 1):
        for d_j in range(d_i + 1, len(directions)):
            # Create a 64x64 tensor with zeros
            tensor = torch.zeros((5, 5), dtype=torch.uint8)
            # Convert tensor to PIL image
            image = Image.fromarray(tensor.numpy())

            draw = ImageDraw.Draw(image)
            draw.line((directions[d_i][0], directions[d_i][1]), fill="white",
                      width=1)
            draw.line((directions[d_j][0], directions[d_j][1]), fill="white",
                      width=1)
            # Convert PIL image back to tensor
            img = torch.from_numpy(np.array(image))
            img = img.to(torch.bool).to(torch.uint8)
            angle_imgs.append(img.unsqueeze(0))

    return torch.cat((angle_imgs), dim=0)


def train_fm_stack():
    args = args_utils.get_args()

    patch_size = 5
    bk_shapes = ["triangle"]
    for bk_shape in bk_shapes:
        args.exp_name = bk_shape
        train_loader, val_loader = prepare_kp_sy_data(args)
        os.makedirs(config.output / f"{args.exp_name}", exist_ok=True)
        kernels = []

        for data in tqdm(train_loader):
            patches = data.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
            patches = patches.reshape(-1, patch_size, patch_size).unique(dim=0)
            patches = patches[~torch.all(patches == 0, dim=(1, 2))]
            kernels.append(patches)
        kernels = torch.cat(kernels, dim=0).unique(dim=0).unsqueeze(1)
        torch.save(kernels, config.output / f"{args.exp_name}" / f"kernels.pt")

        fm_all = []
        data_shift_all = []
        for data in tqdm(train_loader):
            fms = perception.one_layer_conv(data, kernels)
            fms, rs, cs = data_utils.shift_content_to_top_left(fms)
            data_shift, _, _ = data_utils.shift_content_to_top_left(data, rs, cs)
            fm_all.append(fms)
            data_shift_all.append(data_shift)
        fm_all = torch.cat(fm_all, dim=0)
        data_shift_all = torch.cat(data_shift_all, dim=0)

        data_all = torch.cat((data_shift_all, fm_all), dim=1).unique(dim=0)

        print(f"#FM: {len(data_all)}. #Data: {len(train_loader)}")
        torch.save(data_all, config.output / f"{args.exp_name}" / f"fms.pt")
        print(
            f"feature maps have been saved to {config.output / f'{args.exp_name}' / 'fms.pt'}")


def train_fm_cloud(logger, bk_shape):
    args = args_utils.get_args(logger)
    args.exp_name = bk_shape

    save_path = config.output / f"{args.exp_name}"
    os.makedirs(save_path, exist_ok=True)

    train_loader, val_loader = prepare_data(args)

    # kernel size
    k_size = config.kernel_size
    # find kernels
    kernels = []
    for (rgb_img, bw_img) in tqdm(train_loader, f"Calc. Kernels (k={k_size})"):
        patches = bw_img.unfold(2, k_size, 1).unfold(3, k_size, 1)
        patches = patches.reshape(-1, k_size, k_size).unique(dim=0)
        patches = patches[~torch.all(patches == 0, dim=(1, 2))]
        kernels.append(patches)
    kernels = torch.cat(kernels, dim=0).unique(dim=0).unsqueeze(1)
    logger.debug(f"#Kernels: {len(kernels)}, "
                 f"#Data: {len(train_loader)}, "
                 f"Ratio: {len(kernels) / len(train_loader):.2f}")
    torch.save(kernels, save_path / f"kernel_patches_{k_size}.pt")

    # calculate fms
    fm_all = []
    data_shift_all = []
    for (rgb_img, bw_img) in tqdm(train_loader, desc=f"Calc. FMs (k={k_size})"):
        fms = perception.one_layer_conv(bw_img, kernels)
        fms, row_shift, col_shift = data_utils.shift_content_to_top_left(fms)

        data_shift, _, _ = data_utils.shift_content_to_top_left(bw_img,
                                                                row_shift,
                                                                col_shift)
        fm_all.append(fms)
        data_shift_all.append(data_shift)

    fm_all = torch.cat(fm_all, dim=0)
    data_shift_all = torch.cat(data_shift_all, dim=0)
    data_all = torch.cat((data_shift_all, fm_all), dim=1).unique(dim=0)
    torch.save(data_all, save_path / f'fms_patches_{k_size}.pt')
    logger.debug(
        f"#FM: {len(data_all)}. "
        f"#Data: {len(train_loader)}, "
        f"ratio: {len(data_all) / len(train_loader):.2f} "
        f"feature maps have been saved to "
        f"{save_path}/f'fms_patches_{k_size}.pt'")


if __name__ == "__main__":
    # Create a color handler
    logger = args_utils.init_logger()
    bk_shapes = ["circle", "triangle", "square"]

    for bk_shape in bk_shapes:
        train_fm_cloud(logger, bk_shape)
