# Created by jing at 11.12.24

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F for functional operations
import cv2
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from src.neural.neural_utils import *

import config
from src import bk


# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # (in, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (32, 32, 32) -> (16, 16, 16)
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (16, 16, 16) -> (8, 8, 8)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Latent space
        self.latent = nn.Linear(16 * 8 * 8, 128)  # Example latent size: 128

        # Decoder
        self.decoder_fc = nn.Linear(128, 16 * 8 * 8)
        self.decoder = nn.Sequential(
            # (16, 8, 8) -> (32, 16, 16)
            nn.ConvTranspose2d(16, 32,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (32, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(32, 64,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (64, 32, 32) -> (in, 64, 64)
            nn.ConvTranspose2d(64, in_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent(x)

        # Decoder
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 16, 8, 8)
        x = self.decoder(x)
        return x


def train_autoencoder(args, bk_shapes):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Training Autoencoder for patterns {bk_shapes}.")

    for bk_shape in bk_shapes:
        save_path = config.output / f"{bk_shape}"
        os.makedirs(save_path, exist_ok=True)
        model_file = save_path / "fm_ae.pth"
        ae_fm_file = save_path / "fm_ae.pt"
        train_loader, fm_channels = prepare_fm_data(args)
        if not os.path.exists(model_file):
            # Initialize the model, loss, and optimizer
            model = Autoencoder(fm_channels)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            epochs = 20
            loss_history = []
            for epoch in range(epochs):
                epoch_loss = 0
                for batch, in train_loader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(batch)

                    # Compute the loss
                    loss = criterion(outputs, batch)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # visual
                img_file = save_path / f"train_output.png"
                original_img = batch[0].sum(dim=0)
                output_img = outputs[0].sum(dim=0)
                visual_ae_compare(original_img, output_img, img_file)
                avg_loss = epoch_loss / len(train_loader)
                loss_history.append(avg_loss)
                args.logger.debug(f"Train AE ({bk_shape}) "
                                  f"Epoch [{epoch + 1}/{epochs}], "
                                  f"Loss: {avg_loss:.4f}")

            # Save the trained model
            torch.save(model.state_dict(), save_path / "fm_ae.pth")
            args.logger.info("Feature map autoencoder is saved as 'fm_ae.pth'")

            # Visualize the training history
            visual_train_history(save_path, epochs, loss_history)

            args.logger.debug(f"Training {bk_shape} autoencoder completed!")
        if not os.path.exists(ae_fm_file):
            # Test the trained model
            model = Autoencoder(fm_channels)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            for b_i, (batch,) in enumerate(train_loader):
                test_output = model(batch)

                img_file = save_path / f"ae_test_{b_i}.png"
                original_img = batch[0].sum(dim=0)
                output_img = test_output[0].sum(dim=0)
                visual_ae_compare(original_img, output_img, img_file)


def one_layer_conv(data, kernels):
    data = data.to(torch.float32)
    kernels = kernels.to(torch.float32)
    if kernels.shape[-1] == 3:
        padding = 1
    elif kernels.shape[-1] == 5:
        padding = 2
    elif kernels.shape[-1] == 7:
        padding = 3
    elif kernels.shape[-1] == 9:
        padding = 4
    else:
        raise ValueError("kernels has to be 3/5/7/9 dimensional")
    output = F.conv2d(data, kernels, stride=1, padding=padding)
    # max_value = kernels.sum(dim=[1, 2, 3])
    # max_value = max_value.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    # max_value = torch.repeat_interleave(max_value, output.shape[2], dim=-2)
    # max_value = torch.repeat_interleave(max_value, output.shape[3], dim=-1)
    # mask = (max_value == output).to(torch.float32)
    output = output / 9
    return output


def crop_img(img, crop_data=None):
    rgb = img.numpy().astype(np.uint8)
    bg_mask = np.all(rgb == bk.color_matplotlib["lightgray"], axis=-1)
    rgb[bg_mask] = [0, 0, 0]
    bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bw_img = torch.from_numpy(bw_img).squeeze()
    if crop_data is None:
        height, width = bw_img.shape[-2], bw_img.shape[-1]
        # Find the bounding box of the nonzero values
        nonzero_coords = torch.nonzero(bw_img)
        if nonzero_coords.numel() == 0:  # Handle completely empty images
            return bw_img, [0, 0, 0, 0]

        min_y, min_x = nonzero_coords.min(dim=0).values
        max_y, max_x = nonzero_coords.max(dim=0).values

        # Compute the side length of the square
        side_length = max(max_y - min_y + 1, max_x - min_x + 1)

        # Adjust the bounding box to make it square
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        half_side = side_length // 2 + 5

        # Compute the new square bounding box
        new_min_y = max(center_y - half_side, 0)
        new_max_y = min(center_y + half_side + 1, height)
        new_min_x = max(center_x - half_side, 0)
        new_max_x = min(center_x + half_side + 1, width)
    else:
        new_min_y, new_max_y, new_min_x, new_max_x = crop_data
    # Crop the image
    cropped_image = bw_img[new_min_y:new_max_y, new_min_x:new_max_x]

    # if resize is not None:
    #     cropped_image = cv2.resize(cropped_image.numpy(), (resize, resize),
    #                                interpolation=cv2.INTER_AREA)
    #     cropped_image = torch.from_numpy(cropped_image)
    cropped_image = cropped_image.unsqueeze(0)
    return cropped_image, [new_min_y, new_max_y, new_min_x, new_max_x]


def resize_img(img, resize):
    # rgb = rgb_np.numpy().astype(np.uint8)
    # bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # bw_img[bw_img != 211] = 1
    # bw_img[bw_img == 211] = 0
    # if crop:
    #     # bw image to cropped bw image
    #     bw_img, _ = crop_img(torch.from_numpy(bw_img).squeeze(), resize=resize)
    # else:
    #     if resize:
    img = img.squeeze().numpy()
    resized_img = cv2.resize(img, (resize, resize),
                             interpolation=cv2.INTER_LINEAR)
    resized_img = torch.from_numpy(resized_img).unsqueeze(0)
    # else:
    #     bw_img = torch.from_numpy(bw_img).unsqueeze(0)
    return resized_img


def to_bw_img(image):
    # Load an image
    image[image > 0] = 1
    return image


def img2bw(img, cropped_data=None, resize=16):
    cropped_img, cropped_data = crop_img(img.squeeze(), crop_data=cropped_data)
    resized_img = resize_img(cropped_img, resize=resize)
    bw_img = to_bw_img(resized_img)
    return bw_img, cropped_data


def img2fm(img, kernels, cropped_data=None):
    bw_img, cropped_data = img2bw(img, cropped_data)
    fms = one_layer_conv(bw_img, kernels)
    if fms.ndim == 3:
        fms = fms.unsqueeze(0)
    return fms, cropped_data


def fm_merge(fms):
    if fms.ndim == 3:
        in_fms = fms.sum(dim=0).squeeze()
    elif fms.ndim == 4:
        in_fms = fms.sum(dim=1).squeeze()
    else:
        raise ValueError
    merged_fm = (in_fms - in_fms.min()) / ((in_fms.max() - in_fms.min()) + 1e-20)
    return merged_fm
