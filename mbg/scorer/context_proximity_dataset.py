# =============================
# Dataset: context_proximity_dataset.py
# =============================
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
import random
from tqdm import tqdm

from mbg import patch_preprocess


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


class ContextContourDataset(Dataset):
    def __init__(self, root_dir, input_type,device, task_num=20, data_num=100000):
        self.root_dir = Path(root_dir)
        self.data = []
        self.input_type = input_type
        self.data_num = data_num
        self.task_num = task_num
        self._load(device)
        self.balance_labels()

    def _get_bbox_corners(self, x, y, size):
        half_size = size / 2
        # Return 4 corners: top-left, top-right, bottom-right, bottom-left
        return [
            [x - half_size, y - half_size],
            [x + half_size, y - half_size],
            [x + half_size, y + half_size],
            [x - half_size, y + half_size],
        ]
    def _get_task_dirs(self):
        task_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        sampled_dirs =random.sample(task_dirs, self.task_num)
        # sampled_dirs = task_dirs[96:99]
        return sampled_dirs

    def _process_label_dir(self, labeled_dir, device, file_sample_size=3):
        json_files = sorted(labeled_dir.glob("*.json"))
        png_files = sorted(labeled_dir.glob("*.png"))
        file_indices = list(range(len(json_files)))
        if len(file_indices) > file_sample_size:
            file_indices = random.sample(file_indices, file_sample_size)

        for f_i in file_indices:
            with open(json_files[f_i]) as f:
                metadata = json.load(f)
            objects = metadata.get("img_data", [])
            obj_imgs = patch_preprocess.img_path2obj_images(png_files[f_i], device)
            if len(objects) != len(obj_imgs) or len(objects) < 2:
                continue
            objects, obj_imgs, permutes = patch_preprocess.align_data_and_imgs(objects, obj_imgs)
            self._process_object_pairs(objects, obj_imgs, device)

    def _process_object_pairs(self, objects, obj_imgs, device="cpu", sample_size=10):
        patches = [patch_preprocess.rgb2patch(img, self.input_type)[0].to(device) for img in obj_imgs]
        pair_indices = [(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j]
        if len(pair_indices) > sample_size:
            pair_indices = random.sample(pair_indices, sample_size)
        for i, j in pair_indices:
            obj_i = objects[i]
            obj_j = objects[j]
            c_i = patches[i]
            c_j = patches[j]
            others = [patches[k] for k in range(len(objects)) if k != i and k != j]
            if others:
                others_tensor = torch.stack(others, dim=0).to(device)
            else:
                others_tensor = torch.zeros((1, 6, 16, c_i.shape[-1]), device=device)
            label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
            self.data.append((c_i, c_j, others_tensor, label))


    def _load(self, device="cpu"):
        task_dirs = self._get_task_dirs()
        for task_dir in task_dirs:
            if len(self.data) > self.data_num:
                break
            print(f"Processing task directory: {task_dir}, current data size: {len(self.data)}")
            for label_dir in ["positive", "negative"]:
                labeled_dir = task_dir / label_dir
                if not labeled_dir.exists():
                    continue
                self._process_label_dir(labeled_dir, device)

    def balance_labels(self):
        # Separate data by label
        positives = [d for d in self.data if d[3] == 1]
        negatives = [d for d in self.data if d[3] == 0]
        min_count = min(len(positives), len(negatives))
        # Sample min_count from each
        positives = random.sample(positives, min_count)
        negatives = random.sample(negatives, min_count)
        # Combine and shuffle
        self.data = positives + negatives
        random.shuffle(self.data)

    # def _load(self, device="cpu"):
    #     task_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
    #     task_dirs = random.sample(task_dirs, self.task_num)
    #     for task_dir in tqdm(task_dirs):
    #         for label_dir in ["positive", "negative"]:
    #             if len(self.data) > self.data_num:
    #                 continue
    #             labeled_dir = task_dir / label_dir
    #             if not labeled_dir.exists():
    #                 continue
    #
    #             json_files = sorted(labeled_dir.glob("*.json"))
    #             png_files = sorted(labeled_dir.glob("*.png"))
    #
    #             for f_i, json_file in enumerate(json_files):
    #                 with open(json_file) as f:
    #                     metadata = json.load(f)
    #                 objects = metadata.get("img_data", [])
    #                 obj_imgs = patch_preprocess.img_path2obj_images(png_files[f_i], device)
    #                 if len(objects) != len(obj_imgs):
    #                     continue
    #                 if len(objects) < 2:
    #                     continue
    #                 objects, obj_imgs, permutes = patch_preprocess.align_data_and_imgs(objects, obj_imgs)
    #                 for i in range(len(objects)):
    #                     for j in range(len(objects)):
    #                         if i == j:
    #                             continue
    #
    #                         obj_i = objects[i]
    #                         obj_j = objects[j]
    #                         c_i, _, _ = patch_preprocess.rgb2patch(obj_imgs[i], self.input_type)
    #                         c_j, _, _ = patch_preprocess.rgb2patch(obj_imgs[j], self.input_type)
    #                         # Context objects (excluding i and j)
    #                         others = []
    #                         for k in range(len(objects)):
    #                             if k != i and k != j:
    #                                 c_k, _, _ = patch_preprocess.rgb2patch(obj_imgs[k], self.input_type)
    #                                 others.append(c_k)
    #                         if others:
    #                             others_tensor = torch.stack(others, dim=0)  # (N_ctx, 4, 2)
    #                         else:
    #                             others_tensor = torch.zeros((1, 6, 16, c_i.shape[-1]))  # placeholder if no context
    #
    #                         label = 1 if obj_i["group_id"] == obj_j["group_id"] and obj_i["group_id"] != -1 else 0
    #                         self.data.append((c_i, c_j, others_tensor, label))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c_i, c_j, others, label = self.data[idx]
        return c_i, c_j, others, torch.tensor(label, dtype=torch.float32)


def context_collate_fn(batch):
    contour_i, contour_j, context, labels = zip(*batch)
    contour_i = torch.stack(contour_i)
    contour_j = torch.stack(contour_j)
    labels = torch.stack(labels)
    return contour_i, contour_j, context, labels
