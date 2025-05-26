# Created by MacBook Pro at 23.05.25

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path
import random
import json
from tqdm import tqdm
from src import bk
from mbg.scorer import similarity_scorer
from mbg.scorer import scorer_config


def context_collate_fn(batch):
    """
    Collate function for ContextSimilarityDataset.
    """
    ci_list, cj_list, ctx_list, label_list = zip(*batch)
    return list(ci_list), list(cj_list), list(ctx_list), torch.stack(label_list)


def shape_to_onehot(shape_str):
    # Normalize shape name
    if shape_str == "pac_man":
        shape = "circle"
    elif shape_str == "square":
        shape = "rectangle"
    else:
        shape = shape_str
    # Find index and construct one-hot vector
    if shape not in bk.bk_shapes:
        raise ValueError(f"Unknown shape '{shape}' not found in bk.bk_shapes: {bk.bk_shapes}")

    shape_id = bk.bk_shapes.index(shape)
    onehot = torch.zeros(len(bk.bk_shapes), dtype=torch.float32)
    onehot[shape_id] = 1.0
    return onehot


class ContextSimilarityDataset(Dataset):
    def __init__(self, root_dir=scorer_config.SIMILARITY_PATH):
        self.root_dir = Path(root_dir)
        self.data = []
        self._load()

    def _load(self):
        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        task_dirs = random.sample(task_dirs, 20)
        for task_dir in task_dirs:
            for label_dir in ["positive", "negative"]:
                labeled_dir = task_dir / label_dir
                if not labeled_dir.exists():
                    continue

                json_files = sorted(labeled_dir.glob("*.json"))

                for json_file in json_files:
                    with open(json_file) as f:
                        metadata = json.load(f)
                    objects = metadata.get("img_data", [])
                    if len(objects) < 2 or len(objects)>20:
                        continue


                    for i in range(len(objects)):
                        for j in range(len(objects)):
                            if i == j:
                                continue

                            obj_i = objects[i]
                            obj_j = objects[j]

                            # Extract features
                            c_i = {
                                "color": torch.tensor([obj_i["color_r"], obj_i["color_g"], obj_i["color_b"]],
                                                      dtype=torch.float32) / 255,
                                "size": torch.tensor([obj_i["size"]], dtype=torch.float32),
                                "shape": shape_to_onehot(obj_i["shape"])
                            }

                            c_j = {
                                "color": torch.tensor([obj_j["color_r"], obj_j["color_g"], obj_j["color_b"]],
                                                      dtype=torch.float32) / 255,
                                "size": torch.tensor([obj_j["size"]], dtype=torch.float32),
                                "shape": shape_to_onehot(obj_j["shape"])
                            }

                            # Context (exclude i and j)
                            others = []
                            for k, obj_k in enumerate(objects):
                                if k != i and k != j:
                                    others.append({
                                        "color": torch.tensor([obj_k["color_r"], obj_k["color_g"], obj_k["color_b"]],
                                                              dtype=torch.float32) / 255,
                                        "size": torch.tensor([obj_k["size"]], dtype=torch.float32),
                                        "shape": shape_to_onehot(obj_k["shape"])
                                    })

                            if not others:
                                others = [{
                                    "color": torch.zeros(3),
                                    "size": torch.zeros(1),
                                    "shape": torch.zeros(len(bk.bk_shapes))
                                }]

                            label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
                            self.data.append((c_i, c_j, others, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i, c_j, others, torch.tensor(label, dtype=torch.float32)

def train_model():

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 1
    LR = 1e-3

    # Model and Data
    dataset = ContextSimilarityDataset()
    model = similarity_scorer.ContextualSimilarityScorer()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=context_collate_fn)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for ci, cj, ctx, label in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            # Forward pass
            logits = model(ctx[0], ci, cj)  # ctx[0]: context for B=1
            loss = criterion(logits, label)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item() * len(label)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label).sum().item()
            total += len(label)

        print(f"[Epoch {epoch + 1}] Loss: {total_loss / total:.4f} | Acc: {correct / total:.4f}")

    # Save model
    torch.save(model.state_dict(), scorer_config.SIMILARITY_MODEL)
    print(f"Model saved to {scorer_config.SIMILARITY_MODEL}")


if __name__ == '__main__':
    train_model()