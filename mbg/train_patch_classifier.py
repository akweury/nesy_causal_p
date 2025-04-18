# Created by MacBook Pro at 17.04.25


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

import config
from patch_set_model import PatchSetClassifier
from mbg.torch_patch_dataset import TorchClosurePatchSetDataset  # 替换为你的数据集模块

def train_model(device='cpu', epochs=10, batch_size=64, lr=1e-3):
    dataset = TorchClosurePatchSetDataset(config.mb_outlines, patch_size=5, set_size=3, num_sets=5)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = PatchSetClassifier(input_dim=3*5*2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for patch_set, label in train_loader:
            patch_set, label = patch_set.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(patch_set)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(label).sum().item()
            total += label.size(0)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for patch_set, label in val_loader:
                patch_set, label = patch_set.to(device), label.to(device)
                logits = model(patch_set)
                val_loss += criterion(logits, label).item()
                val_correct += (logits.argmax(1) == label).sum().item()

        val_acc = 100 * val_correct / len(val_set)
        print(f"[VAL] Epoch {epoch + 1} | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), config.mb_outlines/"patch_set_classifier.pt")
    print("✅ Model saved to patch_set_classifier.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_model(device=args.device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)