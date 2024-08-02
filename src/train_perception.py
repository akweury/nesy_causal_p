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
from src.percept.perception import FCN
from src.utils import file_utils, args_utils, data_utils, log_utils


# Define the neural network model
def prepare_data(label_name, top_data):
    data_path = config.output / f'train_cha_{label_name}_groups'
    data = file_utils.load_json(data_path / "data.json")
    dataset = []

    for file_type in ["true", "false"]:
        if file_type == "true":
            label = torch.tensor(config.obj_true)
        else:
            label = torch.tensor(config.obj_false)
        files = file_utils.get_all_files(data_path / file_type, "png", True)
        indices = np.random.choice(len(files), size=top_data, replace=False)

        for f_i in range(len(files)):
            if f_i not in indices:
                continue
            task_id, example_id, group_type, group_id, data_type = files[f_i].split("_")
            data_type = data_type.split(".")[0]
            matrix = data[task_id][data_type][int(example_id)][group_type][int(group_id)]
            matrix = data_utils.patch2tensor(matrix)
            rows, cols = matrix.shape
            if rows > 4 and cols > 4:
                dataset.append((matrix, label))

    return dataset


def prepare_kp_data(label_name, top_data):
    data_path = config.kp_dataset / label_name
    dataset = []
    transform = transforms.ToTensor()
    for file_type in ["true", "false"]:
        if file_type == "true":
            label = torch.tensor(config.obj_true)
        else:
            label = torch.tensor(config.obj_false)
        files = file_utils.get_all_files(data_path / file_type, "png", True)
        indices = np.random.choice(len(files), size=top_data, replace=False)

        for f_i in range(len(files)):
            if f_i not in indices:
                continue
            file_name, file_extension = files[f_i].split(".")
            # data = file_utils.load_json(data_path / file_type / f"{file_name}.json")
            img = Image.open(data_path / file_type / f"{file_name}.{file_extension}")

            # Apply the transformation
            image_tensor = transform(img)
            dataset.append((image_tensor, label))

    return dataset


def prepare_kp_sy_data(label_name, top_data):
    data_path = config.kp_dataset / f"percept_{label_name}"
    dataset = []

    for file_type in ["true", "false"]:
        if file_type == "true":
            label = torch.tensor(config.obj_true)
        else:
            label = torch.tensor(config.obj_false)
        files = file_utils.get_all_files(data_path / file_type, "png", True)
        indices = np.random.choice(len(files), size=top_data, replace=False)

        for f_i in range(len(files)):
            if f_i not in indices:
                continue
            file_name, file_extension = files[f_i].split(".")
            data = file_utils.load_json(data_path / file_type / f"{file_name}.json")
            if len(data) > 16:
                patch = data_utils.oco2patch(data).unsqueeze(0)
                dataset.append((patch, label))

    return dataset


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
    label_name = args.exp_name
    # prepare the dataset
    if dataset_name == "kp-ne":
        dataset = prepare_kp_data(label_name, args.top_data)
    elif dataset_name == "kp_sy":
        dataset = prepare_kp_sy_data(label_name, args.top_data)
    elif dataset_name == "arc":
        dataset = prepare_data(label_name, args.top_data)
    else:
        raise ValueError

    ####### init monitor board ########
    log_utils.init_wandb(pj_name=f"percp-{dataset_name}-{label_name}", archi="FCN")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = FCN(in_channels=dataset[0][0].shape[0]).to(args.device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    val_accuracies = []
    # Training loop

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(args.device))
            labels = labels.float().to(args.device)  # Make sure labels are the same shape as outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(args.device))
                labels = labels.float().to(args.device)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                pred_labels = predicted.argmax(dim=1)
                gt_labels = labels.argmax(dim=1)
                correct += (pred_labels == gt_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        wandb.log({'train_loss': avg_train_loss,
                   'val_loss': avg_val_loss,
                   'val_accuracy': accuracy})

    wandb.finish()
    # Save the model
    folder_name = f'{dataset_name}_{label_name}'
    model_name = f'{label_name}_detector_model.pth'
    os.makedirs(config.output / folder_name, exist_ok=True)
    torch.save(model.state_dict(), config.output / folder_name / model_name)
    draw_training_history(train_losses, val_losses, val_accuracies, config.output / folder_name)


if __name__ == "__main__":
    dataset_name = "kp_sy"
    main(dataset_name)
