# Created by MacBook Pro at 22.04.25
# pam_patchset_dataset.py

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from mbg import mbg_config as param
from mbg import patch_preprocess


class ObjPatchSetDataset(Dataset):
    def __init__(self, root_dir=param.ROOT_DATASET_DIR, num_patches=param.PATCHES_PER_SET,
                 points_per_patch=param.POINTS_PER_PATCH):
        self.root_dir = Path(root_dir)
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.data = []
        self._build()

    def _build(self):
        task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        for task_dir in task_dirs:
            gt_path = task_dir / "gt.json"
            if not gt_path.exists():
                continue
            with open(gt_path) as f:
                gt_dict = json.load(f)["img_data"]

            for img_path in task_dir.iterdir():
                print(img_path)
            for img_name in gt_dict:
                image_path = task_dir / (img_name + ".png")

                patch_sets, labels, _, _ = patch_preprocess.img_path2patches_and_labels(image_path, gt_dict[img_name]["objects"])
                image_paths = [image_path] * len(patch_sets)
                self.data += zip(patch_sets, labels, image_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_set, label, _ = self.data[idx]
        return patch_set[:,:,:2], torch.tensor(label)
