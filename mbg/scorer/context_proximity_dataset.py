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
import random


def obj2context_pair_data(i, j, patches):
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

    others = torch.tensor(others)
    return c_i / 1024, c_j / 1024, others / 1024


class ContextProximityDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.data = []
        self._load()

    def _get_bbox_corners(self, x, y, size):
        half_size = size / 2
        # Return 4 corners: top-left, top-right, bottom-right, bottom-left
        return [
            [x - half_size, y - half_size],
            [x + half_size, y - half_size],
            [x + half_size, y + half_size],
            [x - half_size, y + half_size],
        ]

    def _load(self):
        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        task_dirs = random.sample(task_dirs, 50)
        for task_dir in task_dirs:
            for label_dir in ["positive", "negative"]:
                labeled_dir = task_dir / label_dir
                if not labeled_dir.exists():
                    continue

                json_files = sorted(labeled_dir.glob("*.json"))
                json_files = random.sample(json_files, 3)
                for json_file in json_files:
                    with open(json_file) as f:
                        metadata = json.load(f)
                    objects = metadata.get("img_data", [])
                    if len(objects) < 2:
                        continue

                    for i in range(len(objects)):
                        for j in range(len(objects)):
                            if i == j:
                                continue

                            obj_i = objects[i]
                            obj_j = objects[j]

                            # Convert to tensors of shape (4, 2)
                            c_i = torch.tensor(self._get_bbox_corners(obj_i["x"], obj_i["y"], obj_i["size"]))
                            c_j = torch.tensor(self._get_bbox_corners(obj_j["x"], obj_j["y"], obj_j["size"]))

                            # Context objects (excluding i and j)
                            others = []
                            for k in range(len(objects)):
                                if k != i and k != j:
                                    obj_k = objects[k]
                                    c_k = self._get_bbox_corners(obj_k["x"], obj_k["y"], obj_k["size"])
                                    others.append(c_k)
                            if others:
                                others_tensor = torch.tensor(others)  # (N_ctx, 4, 2)
                            else:
                                others_tensor = torch.zeros((1, 4, 2))  # placeholder if no context

                            label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
                            self.data.append((c_i, c_j, others_tensor, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i, c_j, others, torch.tensor(label, dtype=torch.float32)

        # gt_path = task_dir / "gt.json"
        # if not gt_path.exists():
        #     continue
        # with open(gt_path) as f:
        #     gt_data = json.load(f)
        # images_data = gt_data.get("img_data", {})
        #
        # for img_name, img_data in images_data.items():
        #     patches = img_data["patches"]
        #     labels = img_data.get("proximity", [])
        #     if len(patches) < 2:
        #         continue
        #
        #     for i in range(len(patches)):
        #         for j in range(len(patches)):
        #             if i == j:
        #                 continue
        #             c_i = torch.tensor(patches[i][0][0])
        #             c_i[:, :, 0] += patches[i][0][1][0]
        #             c_i[:, :, 1] += patches[i][0][1][1]
        #
        #             c_j = torch.tensor(patches[j][0][0])
        #             c_j[:, :, 0] += patches[j][0][1][0]
        #             c_j[:, :, 1] += patches[j][0][1][1]
        #
        #             others = []
        #             for k in range(len(patches)):
        #                 if k != i and k != j:
        #                     c_k = torch.tensor(patches[k][0][0])
        #                     c_k[:, :, 0] += patches[k][0][1][0]
        #                     c_k[:, :, 1] += patches[k][0][1][1]
        #                     c_k = c_k.tolist()
        #                     others.append(c_k)
        #
        #             if [i, j] in labels or [j, i] in labels:
        #                 label = 1
        #             else:
        #                 label = 0
        #             others = torch.tensor(others)
        #             self.data.append((c_i, c_j, others, label))
    #
    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, idx):
    #     c_i, c_j, others, label = self.data[idx]
    #     return c_i/1024, c_j/1024, others/1024, torch.tensor(label, dtype=torch.float32)


def context_collate_fn(batch):
    contour_i, contour_j, context, labels = zip(*batch)
    contour_i = torch.stack(contour_i)
    contour_j = torch.stack(contour_j)
    labels = torch.stack(labels)
    return contour_i, contour_j, context, labels
