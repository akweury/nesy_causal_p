# Created by jing at 11.12.24

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F for functional operations
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from src.neural.neural_utils import *

import config


# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # (in, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (32, 32, 32) -> (16, 16, 16)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (16, 16, 16) -> (8, 8, 8)
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Latent space
        self.latent = nn.Linear(8 * 8 * 8, 128)  # Example latent size: 128

        # Decoder
        self.decoder_fc = nn.Linear(128, 8 * 8 * 8)
        self.decoder = nn.Sequential(
            # (16, 8, 8) -> (32, 16, 16)
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (32, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(16, 32,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (64, 32, 32) -> (in, 64, 64)
            nn.ConvTranspose2d(32, in_channels,
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
        x = x.view(x.size(0), 8, 8, 8)
        x = self.decoder(x)
        return x


def train_autoencoder(args, bk_shapes):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Training Autoencoder for patterns {bk_shapes}.")

    for bk_shape in bk_shapes:
        save_path = config.output / f"{bk_shape}"
        os.makedirs(save_path, exist_ok=True)
        model_path = save_path / "fm_ae.pth"
        if os.path.exists(model_path):
            continue

        train_loader, fm_channels = prepare_fm_data(args)

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
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            args.logger.debug(f"Epoch [{epoch + 1}/{epochs}], "
                              f"Loss: {avg_loss:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), save_path / "fm_ae.pth")
        args.logger.info("Feature map autoencoder is saved as 'fm_ae.pth'")

        # Visualize the training history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), loss_history, marker='o',
                 label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path / f"loss_history.png")

        args.logger.debug(f"Training {bk_shape} autoencoder completed!")

def one_layer_conv(data, kernels):
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
    return output
