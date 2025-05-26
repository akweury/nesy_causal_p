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
from mbg.group import grouping_similarity


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
        self._load_balanced()

    def _load_balanced(self):
        task_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])[:20]
        pos_data, neg_data = [], []

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
                    if len(objects) < 2 or len(objects) > 10:
                        continue

                    # enrich objects
                    for o in objects:
                        o["color"] = [o["color_r"], o["color_g"], o["color_b"]]
                        o["w"] = o["size"] * 1024 + random.uniform(-5, 5)
                        o["shape"] = shape_to_onehot(o["shape"])

                    for i in range(len(objects)):
                        for j in range(len(objects)):
                            if i == j:
                                continue
                            obj_i = objects[i]
                            obj_j = objects[j]
                            c_i, c_j, others = grouping_similarity.obj2pair_data(objects, i, j)
                            label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
                            item = (c_i, c_j, others, label)
                            (pos_data if label == 1 else neg_data).append(item)

        # Balance dataset
        min_len = min(len(pos_data), len(neg_data))
        random.shuffle(pos_data)
        random.shuffle(neg_data)
        self.data = pos_data[:min_len] + neg_data[:min_len]
        random.shuffle(self.data)

        print(f"Loaded {len(self.data)} samples (balanced: {min_len} pos, {min_len} neg)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i, c_j, others, torch.tensor(label, dtype=torch.float32)


def train_model():
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 1
    LR = 1e-3

    # Model and Data
    dataset = ContextSimilarityDataset()
    model = similarity_scorer.ContextualSimilarityScorer()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=context_collate_fn)

    pos = sum(label for _, _, _, label in dataset)
    neg = len(dataset) - pos
    print(f"Positive: {pos}, Negative: {neg}, Ratio: {pos / len(dataset):.2f}")

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for ci, cj, ctx, label in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            # Forward pass
            logits = model(ctx[0], ci[0], cj[0])  # ctx[0]: context for B=1
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
