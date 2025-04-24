# Created by MacBook Pro at 20.04.25
# pam_patchset_dataset.py
import json
from pathlib import Path
from torch.utils.data import Dataset

import config
import mbg.mbg_config as param
from mbg import patch_preprocess


class PAMPatchSetDataset(Dataset):
    def __init__(self, root_dir=None, num_patches=param.PATCHES_PER_SET, points_per_patch=param.POINTS_PER_PATCH):
        self.root_dir = Path(root_dir or (config.kp_gestalt_dataset / "pam_synthetic"))
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.data = []
        self._build()

    def _build(self):
        gt_path = self.root_dir / "gt.json"
        with open(gt_path) as f:
            gt = json.load(f)

        for name, label_str in gt.items():
            if label_str not in param.LABEL_NAMES.values():
                continue
            img_path = self.root_dir / label_str / f"{name}.png"
            # img path to patch set
            patch_sets= patch_preprocess.img_path2one_patches(img_path)
            label = list(param.LABEL_NAMES.values()).index(label_str)
            self.data.append((patch_sets, label, str(img_path)))

            # image = Image.open(img_path).convert("RGB")
            # contours = extract_object_contours(image)
            # if len(contours) < 3:
            #     continue
            # selected = random.sample(contours, k=min(3, len(contours)))
            # merged = np.concatenate(selected, axis=0)
            # patch_set = generate_patch_set_from_contour(merged, self.num_patches, self.points_per_patch)
            # self.data.append((patch_set, label, str(img_path)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch_set, label, path = self.data[idx]
        return patch_set, label, path