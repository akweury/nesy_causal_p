# Created by MacBook Pro at 27.05.25

import torch

from slot_attention import SlotAttention  # import your saved module
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
from torch.utils.data import Dataset
from pathlib import Path
import json
from mbg import patch_preprocess  # your custom patch extraction module
from mbg.scorer import scorer_config
import torch.nn.functional as F


class SlotGroupingDataset(Dataset):
    def __init__(self, root_dir, max_objects=100):
        """
        Args:
            root_dir: path to your Gestalt dataset (task folders)
            max_objects: skip images with too many objects
        """
        self.root_dir = Path(root_dir)
        self.data = []
        self.max_objects = max_objects
        self._load()

    def _load(self):
        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for task_dir in task_dirs:
            for label_dir in ["positive", "negative"]:
                if len(self.data) > 100:
                    continue
                labeled_dir = task_dir / label_dir
                if not labeled_dir.exists():
                    continue

                json_files = sorted(labeled_dir.glob("*.json"))
                png_files = sorted(labeled_dir.glob("*.png"))

                for f_i, json_file in enumerate(json_files):
                    with open(json_file) as f:
                        metadata = json.load(f)
                    objects = metadata.get("img_data", [])
                    if len(objects) < 2 or len(objects) > self.max_objects:
                        continue

                    # Extract patches for all objects
                    obj_imgs = patch_preprocess.img_path2obj_images(png_files[f_i])
                    if len(obj_imgs) != len(objects):
                        continue

                    objects, obj_imgs = patch_preprocess.align_data_and_imgs(objects, obj_imgs)

                    patch_feats = [patch_preprocess.rgb2patch(img).flatten() for img in obj_imgs]
                    group_ids = [obj["group_id"] for obj in objects]
                    patch_feats = torch.stack(patch_feats)  # shape [N, D]
                    group_ids = torch.tensor(group_ids)  # shape [N]
                    self.data.append((patch_feats, group_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def hungarian_loss(pred_attn, gt_group_ids, mask):
    B, N, K = pred_attn.shape
    loss = 0
    for b in range(B):
        n_valid = mask[b].sum().item()
        if n_valid < 2:
            continue
        pred = pred_attn[b, mask[b]]  # [n_valid, K]
        gt = gt_group_ids[b, mask[b]]  # [n_valid]
        true_ids = list(set(gt.tolist()))
        M = len(true_ids)
        cost = torch.zeros(K, M)
        for k in range(K):
            for j, gid in enumerate(true_ids):
                match = (gt == gid).float()
                cost[k, j] = - (pred[:, k] * match).sum()  # maximize overlap
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        best_cost = cost[row, col].sum()
        loss += -best_cost
    return loss / B


def hungarian_loss_and_accuracy(pred_attn, gt_group_ids, mask):
    B, N, K = pred_attn.shape
    total_loss = 0
    total_acc = 0
    count = 0

    for b in range(B):
        n_valid = mask[b].sum().item()
        if n_valid < 2:
            continue

        pred = pred_attn[b, mask[b]]        # [n_valid, K]
        gt = gt_group_ids[b, mask[b]]       # [n_valid]
        true_ids = list(set(gt.tolist()))
        M = len(true_ids)

        cost = torch.zeros(K, M, device=pred.device)
        for k in range(K):
            for j, gid in enumerate(true_ids):
                match = (gt == gid).float()
                cost[k, j] = - (pred[:, k] * match).sum()

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        aligned_pred = pred[:, row].argmax(dim=1)
        aligned_gt = torch.tensor([col[true_ids.index(g.item())] for g in gt], device=gt.device)

        acc = (aligned_pred == aligned_gt).float().mean()
        total_acc += acc.item()
        total_loss += cost[row, col].sum()
        count += 1

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    return avg_loss, avg_acc

def collate_fn(batch):
    max_len = max(x[0].size(0) for x in batch)
    feats, gids, mask = [], [], []
    for f, g in batch:
        pad_len = max_len - f.size(0)
        feats.append(F.pad(f, (0, 0, 0, pad_len)))  # [N, D] → [max_N, D]
        gids.append(F.pad(g, (0, pad_len), value=-1))  # [N] → [max_N]
        mask.append(F.pad(torch.ones_like(g, dtype=torch.bool), (0, pad_len), value=False))  # [N] → [max_N]
    return torch.stack(feats), torch.stack(gids), torch.stack(mask)


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SlotAttention(num_slots=10, dim=192).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20
data_path = scorer_config.closure_path
dataset = SlotGroupingDataset(data_path)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0
    for feats, gids, mask in loader:
        feats, gids, mask = feats.to(device), gids.to(device), mask.to(device)
        _, attn = model(feats)  # attn: [B, N, K]
        loss, acc = hungarian_loss_and_accuracy(attn, gids, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        count += 1

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

torch.save({
    "model_state": model.state_dict(),
}, scorer_config.CLOSURE_MODEL)

print("✅ Model saved to slot_attention_grouping.pt")
