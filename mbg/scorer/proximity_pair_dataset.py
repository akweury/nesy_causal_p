# Created by MacBook Pro at 24.04.25
# proximity_pair_dataset.py

import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset



class ProximityPairDataset(Dataset):
    def __init__(self, root_dir, num_points=100, balance_classes=True):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.balance_classes = balance_classes
        self.data = []
        self._load()

    def _load(self):
        positive_samples = []
        negative_samples = []

        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for task_dir in task_dirs:
            gt_path = task_dir / "gt.json"
            if not gt_path.exists():
                continue
            with open(gt_path) as f:
                gt_data = json.load(f)
            images_data = gt_data.get("img_data", {})

            for img_name, img_data in images_data.items():
                patches = img_data.get("patches", [])
                labels = img_data.get("proximity", [])
                if len(patches) < 2:
                    continue

                for i in range(len(patches)):
                    for j in range(i + 1, len(patches)):

                        c_i = torch.tensor(patches[i][0][0])
                        c_i[:, :, 0] += patches[i][0][1][0]
                        c_i[:, :, 1] += patches[i][0][1][1]

                        c_j = torch.tensor(patches[j][0][0])
                        c_j[:, :, 0] += patches[j][0][1][0]
                        c_j[:, :, 1] += patches[j][0][1][1]

                        if [i, j] in labels or [j, i] in labels:
                            positive_samples.append((c_i, c_j, 1))
                        else:
                            negative_samples.append((c_i, c_j, 0))

        if self.balance_classes:
            n = min(len(positive_samples), len(negative_samples))
            self.data = random.sample(positive_samples, n) + random.sample(negative_samples, n)
        else:
            self.data = positive_samples + negative_samples

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, label = self.data[idx]
        return c_i, c_j, torch.tensor(label, dtype=torch.float32)
