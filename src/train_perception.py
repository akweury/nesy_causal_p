# Created by shaji at 24/07/2024

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
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

    minst_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_size = int(0.8 * len(minst_dataset))
    val_size = len(minst_dataset) - train_size
    train_dataset, val_dataset = random_split(minst_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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


def main(dataset_name):
    args = args_utils.get_args()
    # data file
    args.data_types = ["data_triangle"]
    train_loader, val_loader = prepare_kp_sy_data(args)
    # train_loader, val_loader = prepare_mnist_data(args)
    os.makedirs(config.output / f"kp_sy_{args.exp_name}", exist_ok=True)

    perception.learn_fm(args, train_loader, val_loader)


    log_utils.init_wandb(pj_name=f"FM-{dataset_name}-{args.exp_name}", archi="FCN")

    perception.train_percept(args, model, train_loader, val_loader)

    file_utils.save_model(model, dataset_name, f'detector_model.pth')


if __name__ == "__main__":
    dataset_name = "triangle"
    main(dataset_name)
