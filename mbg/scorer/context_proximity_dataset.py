# =============================
# Dataset: context_proximity_dataset.py
# =============================
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

def obj2context_pair_data(i, j, patches, pairs):
    c_i = patches[i][0][0].clone()
    c_i[:, :, 0] += patches[i][0][1][0]
    c_i[:, :, 1] += patches[i][0][1][1]

    c_j = patches[j][0][0].clone()
    c_j[:, :, 0] += patches[j][0][1][0]
    c_j[:, :, 1] += patches[j][0][1][1]

    others = []
    for k in range(len(patches)):
        if k != i and k != j:
            c_k = patches[k][0][0].clone()
            c_k[:, :, 0] += patches[k][0][1][0]
            c_k[:, :, 1] += patches[k][0][1][1]
            c_k = c_k.tolist()
            others.append(c_k)

    if [i, j] in pairs or [j, i] in pairs:
        label = 1
    else:
        label = 0
    others = torch.tensor(others)
    return c_i/1024, c_j/1024, others/1024, label


class ContextProximityDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
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
                patches = img_data["patches"]
                labels = img_data.get("proximity", [])

                if len(patches) < 2:
                    continue

                for i in range(len(patches)):
                    for j in range(len(patches)):
                        if i == j:
                            continue
                        c_i = torch.tensor(patches[i][0][0])
                        c_i[:, :, 0] += patches[i][0][1][0]
                        c_i[:, :, 1] += patches[i][0][1][1]

                        c_j = torch.tensor(patches[j][0][0])
                        c_j[:, :, 0] += patches[j][0][1][0]
                        c_j[:, :, 1] += patches[j][0][1][1]

                        others = []
                        for k in range(len(patches)):
                            if k != i and k != j:
                                c_k = torch.tensor(patches[k][0][0])
                                c_k[:, :, 0] += patches[k][0][1][0]
                                c_k[:, :, 1] += patches[k][0][1][1]
                                c_k = c_k.tolist()
                                others.append(c_k)

                        if [i, j] in labels or [j, i] in labels:
                            label = 1
                        else:
                            label = 0
                        others = torch.tensor(others)
                        self.data.append((c_i, c_j, others, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i/1024, c_j/1024, others/1024, torch.tensor(label, dtype=torch.float32)

def context_collate_fn(batch):
    contour_i, contour_j, context, labels = zip(*batch)
    contour_i = torch.stack(contour_i)
    contour_j = torch.stack(contour_j)
    labels = torch.stack(labels)
    return contour_i, contour_j, context, labels