# Created by MacBook Pro at 17.04.25

import torch
from torch.utils.data import Dataset
import os
import numpy as np

import config


class TorchClosurePatchSetDataset(Dataset):
    def __init__(self, contour_dir, shape_classes=None, patch_size=5, set_size=3, num_sets=10, num_patches=20):
        self.data = []
        self.labels = []
        self.label2idx = {}
        self.patch_size = patch_size
        self.set_size = set_size
        self.num_sets = num_sets
        self.num_patches = num_patches

        if shape_classes is None:
            shape_classes = ["triangle", "rectangle", "ellipse"]

        for idx, shape in enumerate(shape_classes):
            self.label2idx[shape] = idx
            shape_file = os.path.join(contour_dir, f"shape_{idx+1}_contours.npy")
            if not os.path.exists(shape_file):
                continue
            np_contours = np.load(shape_file, allow_pickle=True)
            contours = [torch.tensor(c, dtype=torch.float32) for c in np_contours]
            for contour in contours:
                patch_sets = self.generate_random_patch_sets(contour)
                self.data.extend(patch_sets)
                self.labels.extend([idx] * len(patch_sets))

    def generate_random_patch_sets(self, contour):
        N = contour.size(0)
        if N < self.patch_size:
            return []

        patch_start_indices = torch.linspace(0, N - self.patch_size, steps=self.num_patches, dtype=torch.long)
        patches = [contour[i:i + self.patch_size] for i in patch_start_indices]

        patch_sets = []
        for _ in range(self.num_sets):
            if len(patches) < self.set_size:
                continue
            idxs = torch.randperm(len(patches))[:self.set_size]
            patch_set = torch.stack([patches[i] for i in idxs])
            patch_sets.append(patch_set)
        return patch_sets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def visualize_patch_sets(dataset, num_figures=6, set_size=3):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    counter = 0
    for i, (patch_set, label) in enumerate(loader):
        if label==2:
            counter += 1
        else:
            continue

        if counter>5:
            break
        ax = axes[counter]
        patch_set = patch_set.squeeze(0).numpy()  # shape: [set_size, patch_size, 2]

        for patch in patch_set:
            ax.plot(patch[:, 0], patch[:, 1], marker='o')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # 清理 ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

        ax.set_title(f"Label: {label.item()}", fontsize=10)

    # 调整子图间距
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


dataset = TorchClosurePatchSetDataset(config.mb_outlines, patch_size=5, set_size=3, num_sets=5)
visualize_patch_sets(dataset)