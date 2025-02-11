# Created by x at 11.12.24

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def prepare_fm_data(args):

    # Sample data
    fm_file = args.save_path / f'fms_patches_{args.k_size}.pt'
    data = torch.load(fm_file)

    # Normalize the data
    data = (data - data.min()) / (data.max() - data.min())

    # Define a PyTorch Dataset and DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    fm_channels = data.shape[1]
    return dataloader,  fm_channels


def visual_ae_compare(original_img, output_img, save_file):
    # Visualize the original and reconstructed images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Original image

    original_img = original_img.cpu().detach().numpy()
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Reconstructed image

    test_img = output_img.cpu().detach().numpy()
    axes[1].imshow(test_img)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    plt.savefig(save_file)

def visual_train_history(save_path, epochs, loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='o',
             label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"loss_history.png")