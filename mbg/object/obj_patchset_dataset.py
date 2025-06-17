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
from src import bk


class ObjPatchSetDataset(Dataset):
    def __init__(self, root_dir=param.ROOT_DATASET_DIR, num_patches=param.PATCHES_PER_SET,
                 points_per_patch=param.POINTS_PER_PATCH):
        self.root_dir = Path(root_dir)
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.data = []
        self._build()

    def _build(self):
        task_dirs = [d / "positive" for d in self.root_dir.iterdir() if d.is_dir()]
        task_dirs += [d / "negative" for d in self.root_dir.iterdir() if d.is_dir()]
        task_dirs = random.sample(task_dirs, 100)
        for task_dir in task_dirs:
            json_files = sorted(task_dir.glob("*.json"))
            png_files = sorted(task_dir.glob("*.png"))
            for f_i, json_file in enumerate(json_files):
                with open(json_file) as f:
                    metadata = json.load(f)
                objects = metadata.get("img_data", [])
                image_path = png_files[f_i]
                for o in objects:
                    o["shape"] = bk.bk_shapes_2.index(o['shape'])
                patch_sets, labels, _, _, _ = patch_preprocess.img_path2patches_and_labels(image_path, objects,
                                                                                           device="cpu")
                image_paths = [image_path] * len(patch_sets)
                self.data += zip(patch_sets, labels, image_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_set, label, _ = self.data[idx]
        return patch_set[:, :, :2], torch.tensor(label)
