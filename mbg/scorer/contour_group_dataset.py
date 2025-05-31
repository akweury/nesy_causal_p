# Created by MacBook Pro at 27.05.25


from torch.utils.data import Dataset
import torch
import json
from pathlib import Path
from mbg import patch_preprocess

# class ContourGroupDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = Path(root_dir)
#         self.data = []
#         self._load()
#
#     def _load(self):
#         task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
#         for task_dir in task_dirs:
#             for label_dir in ["positive", "negative"]:
#                 if len(self.data)>10:
#                     continue
#                 labeled_dir = task_dir / label_dir
#                 if not labeled_dir.exists():
#                     continue
#
#                 json_files = sorted(labeled_dir.glob("*.json"))
#                 png_files = sorted(labeled_dir.glob("*.png"))
#
#                 for f_i, json_file in enumerate(json_files):
#                     with open(json_file) as f:
#                         metadata = json.load(f)
#                     objects = metadata.get("img_data", [])
#                     if len(objects) < 2:
#                         continue
#                     obj_imgs = patch_preprocess.img_path2obj_images(png_files[f_i])
#                     if len(objects) != len(obj_imgs):
#                         continue
#                     objects, obj_imgs, permutes = patch_preprocess.align_data_and_imgs(objects, obj_imgs)
#                     patches = [patch_preprocess.rgb2patch(img) for img in obj_imgs]
#                     group_ids = [obj["group_id"] for obj in objects]
#                     self.data.append((patches, group_ids))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         patches, group_ids = self.data[idx]
#         patches = torch.stack(patches, dim=0)  # (N, C, H, W)
#         group_ids = torch.tensor(group_ids, dtype=torch.long)
#         return patches, group_ids


def collate_group_batch(batch):
    max_len = max(p[0].size(0) for p in batch)
    padded_patches = []
    padded_masks = []
    group_labels = []
    for patches, gids in batch:
        N, C, H, W = patches.shape
        pad_len = max_len - N
        pad_patch = torch.zeros((pad_len, C, H, W))
        padded_patches.append(torch.cat([patches, pad_patch], dim=0))

        mask = torch.cat([torch.ones(N), torch.zeros(pad_len)])
        padded_masks.append(mask)

        group_labels.append(torch.cat([gids, -1 * torch.ones(pad_len, dtype=torch.long)]))

    return (
        torch.stack(padded_patches),     # [B, max_N, C, H, W]
        torch.stack(padded_masks),       # [B, max_N]
        torch.stack(group_labels)        # [B, max_N]
    )