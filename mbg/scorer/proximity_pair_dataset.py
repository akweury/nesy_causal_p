# Created by MacBook Pro at 24.04.25
# proximity_pair_dataset.py

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset


class ProximityPairDataset(Dataset):
    def __init__(self, root_dir, num_points=100):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.data = []
        self._load()

    def _load(self):
        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for task_dir in task_dirs:
            gt_path = task_dir / "gt.json"
            if not gt_path.exists():
                continue
            with open(gt_path) as f:
                gt_data = json.load(f)
            images_data = gt_data.get("img_data", {})
            for img_name, img_data in images_data.items():
                image_path = task_dir / (img_name + ".png")
                if not image_path.exists():
                    continue
                patches = img_data["patches"]
                labels = img_data["proximity"]
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
                            label = 1
                        else:
                            label = 0
                        self.data.append((c_i, c_j, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, label = self.data[idx]
        return c_i, c_j, torch.tensor(label, dtype=torch.float32)
