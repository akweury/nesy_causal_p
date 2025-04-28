# train_pam_synthetic_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from mbg.group.pam_patchset_dataset import PAMPatchSetDataset
import mbg.mbg_config as param
from mbg.object import patch_classifier_model
device = torch.device(param.DEVICE)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

wandb.init(
    project="pam_synthetic_patch_classifier",
    config={k: v for k, v in vars(param).items() if is_jsonable(v)}
)

dataset = PAMPatchSetDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=param.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=param.BATCH_SIZE)

model = patch_classifier_model.init_patch_classifier(device)
optimizer = optim.Adam(model.parameters(), lr=param.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def visualize_prediction(patch_set, image_path, prediction, label, logits):
    patch_set = patch_set.cpu().numpy()
    image = np.array(to_pil_image(torch.tensor(np.array(Image.open(image_path)).transpose(2, 0, 1)) / 255.0))
    fig, ax = plt.subplots()
    ax.imshow(image)
    for patch in patch_set:
        patch = patch
        ax.plot(patch[0, :], patch[1, :], linewidth=2, alpha=0.5)
    ax.set_title(f"Pred: {prediction}, Label: {label}, Conf: {logits.max():.2f}")
    ax.axis('off')
    return wandb.Image(fig)

for epoch in range(param.EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y, img_path in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    train_acc = correct / len(train_loader.dataset)
    wandb.log({"train_loss": total_loss / len(train_loader.dataset), "train_acc": train_acc}, step=epoch)

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y, img_path in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()

    val_acc = correct / len(val_loader.dataset)
    wandb.log({"val_loss": val_loss / len(val_loader.dataset), "val_acc": val_acc}, step=epoch)

    if (epoch + 1) % 10 == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if (epoch + 1) % 10 == 0:
        x_vis, y_vis, _ = next(iter(val_loader))
        x_vis, y_vis = x_vis.to(device), y_vis.to(device)
        with torch.no_grad():
            logits = model(x_vis)
            preds = logits.argmax(dim=1)
            images = []
            for i in range(min(4, len(x_vis))):
                img_path = dataset.data[val_set.indices[i]][2]  # access image path
                images.append(visualize_prediction(x_vis[i], img_path, preds[i].item(), y_vis[i].item(), logits[i]))
            wandb.log({"Patch Prediction Visualization": images}, step=epoch)

torch.save(model.state_dict(), param.MODEL_SAVE_PATH)
print(f"Saved model to {param.MODEL_SAVE_PATH}")

