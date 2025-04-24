# train_patch_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import json

import mbg.mbg_config as param
from mbg.object.obj_patchset_dataset import ObjPatchSetDataset
from mbg.object.patch_classifier_model import PatchClassifier

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


device = torch.device(param.DEVICE)

wandb.init(
    project="pam_synthetic_patch_classifier",
    config={k: v for k, v in vars(param).items() if is_jsonable(v)}
)

dataset = ObjPatchSetDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=param.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=param.BATCH_SIZE)

model = PatchClassifier(
    num_patches=param.PATCHES_PER_SET,
    patch_len=param.POINTS_PER_PATCH,
    num_classes=param.NUM_CLASSES
).to(device)

optimizer = optim.Adam(model.parameters(), lr=param.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(param.EPOCHS):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in train_loader:
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
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()

    val_acc = correct / len(val_loader.dataset)
    wandb.log({"val_loss": val_loss / len(val_loader.dataset), "val_acc": val_acc}, step=epoch)

    if (epoch + 1) % 10 == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), param.OBJ_MODEL_SAVE_PATH)
print(f"Saved model to {param.OBJ_MODEL_SAVE_PATH}")
