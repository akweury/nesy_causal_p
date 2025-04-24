# Created by MacBook Pro at 22.04.25


# patch_classifier_model.py
import torch
import torch.nn as nn
from mbg import mbg_config as param


class PatchClassifier(nn.Module):
    def __init__(self, num_patches, patch_len, num_classes):
        super().__init__()
        self.num_patches = num_patches
        self.patch_len = patch_len
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_patches * patch_len * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_patch_classifier(model_path, device, num_patches, patch_len, num_classes):
    model = PatchClassifier(
        num_patches=num_patches,
        patch_len=patch_len,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def init_patch_classifier(device):
    model = PatchClassifier(
        num_patches=param.PATCHES_PER_SET,
        patch_len=param.POINTS_PER_PATCH,
        num_classes=len(param.LABEL_NAMES)
    ).to(device)
    return model
