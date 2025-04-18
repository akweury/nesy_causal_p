# Created by X at 10.12.24

import torch
import cv2
import json
from torch.utils.data import Dataset, DataLoader
import config
from src.utils import data_utils, file_utils, chart_utils
import numpy as np
import os
import glob
from PIL import Image


class BasicShapeDataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.image_paths = []
        # self.labels = []
        self.device = args.device
        folder = config.kp_base_dataset / args.bk_shape
        self.image_paths = sorted(file_utils.get_all_files(folder, "png", False)[:1000])
        self.vertices = data_utils.load_json(folder / 'metadata.json')

    def __len__(self):
        return len(self.image_paths)

    def extract_patches(self, image, vertices, patch_size=32):
        """
        Extract patches of size patch_size x patch_size around each vertex.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            vertices (list): List of (x, y) tuples representing vertex coordinates.
            patch_size (int): Size of the patch to extract (default is 32).

        Returns:
            list: List of extracted patches as NumPy arrays.
        """
        patches = []
        half_size = patch_size // 2
        padded_image = np.pad(image, ((half_size, half_size), (half_size, half_size), (0, 0)), mode='constant',
                              constant_values=0)

        for (x, y) in vertices:
            x_p, y_p = x + half_size, y + half_size
            patch = padded_image[y_p - half_size:y_p + half_size, x_p - half_size:x_p + half_size]
            patches.append(torch.from_numpy(patch).unsqueeze(0))

        return torch.cat(patches, dim=0)

    def __getitem__(self, idx):
        rgb_image = torch.from_numpy(cv2.imread(self.image_paths[idx]))
        vertices = self.vertices[idx]["vertices"]
        patches = self.extract_patches(rgb_image.numpy(), vertices)

        return patches, vertices


class GestaltDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = []

        # Scan each task folder
        task_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        for task_folder in task_folders:
            gt_path = os.path.join(task_folder, "gt.json")
            if not os.path.isfile(gt_path):
                continue
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
            principle = gt_data["principle"]
            img_data = gt_data["img_data"]

            # Collect all images and their metadata
            for img_name in sorted(glob.glob(os.path.join(task_folder, "*.png"))):
                img_id = os.path.basename(img_name).split(".")[0]
                if img_id not in img_data:
                    continue
                self.samples.append({
                    "image_path": img_name,
                    "symbolic_data": img_data[img_id],
                    "principle": principle,
                    "task": os.path.basename(task_folder)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        image = self.transform(image)

        return {
            "image": image,
            "symbolic_data": sample["symbolic_data"],
            "principle": sample["principle"],
            "task": sample["task"],
            "filename": os.path.basename(sample["image_path"])
        }


# class GestaltDataset(Dataset):
#     def __init__(self, args, imgs):
#         self.args = args
#         self.device = args.device
#         self.imgs = imgs
#         #
#         # self.image_paths = []
#         #
#         # folder = config.kp_base_dataset / args.exp_name
#         # imgs = file_utils.get_all_files(folder, "png", False)[:1000]
#         # # labels = [self.get_label(args.exp_name) for img in imgs]
#         # self.image_paths += imgs
#         # # self.labels += labels
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def load_data(self, idx):
#         file_name, file_extension = self.imgs[idx].split(".")
#         data = file_utils.load_json(f"{file_name}.json")
#         img = file_utils.load_img(self.imgs[idx])
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         return img, data, file_name.split("/")[-4:]
#
#         # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.args.device)
#
#     def __getitem__(self, idx):
#         img, data, file_name = self.load_data(idx)
#         self.args.logger.debug(
#             f"\n =========== Analysis Image {file_name} {idx + 1}/{len(self.imgs)} ==============")
#
#         return img, data
#         # img = data_utils.load_bw_img(self.image_paths[idx], size=64)
#         # resize
#         # file_name, file_extension = self.image_paths[idx].split(".")
#         # data = file_utils.load_json(f"{file_name}.json")
#         # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.device)
#
#         # return img

#
# class GSDataset(Dataset):
#     def __init__(self):
#         self.data_train = torch.load(config.kp_gestalt_dataset / "train" / "train.pt")
#         self.imgs_train = self.load_imgs(config.kp_gestalt_dataset / "train")
#
#         self.data_test = torch.load(config.kp_gestalt_dataset / "test" / "test.pt")
#         self.imgs_test = self.load_imgs(config.kp_gestalt_dataset / "test")
#
#     def __len__(self):
#         return len(self.data_train["positive"])
#
#     def load_imgs(self, path):
#         img_files = file_utils.get_all_files(path, '.png')
#         img_files = sorted(img_files)
#         imgs = []
#         for file in img_files:
#             img = file_utils.load_img(file)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_list = []
#             for i in range(img.shape[1] // img.shape[0]):
#                 img_list.append(img[:, i * img.shape[0]:(i + 1) * img.shape[0], :])
#             img_list = [im[2:-2, 2:-2] for im in img_list]
#             imgs.append(img_list)
#         return imgs
#
#     def __getitem__(self, idx):
#         train_data = {
#             "pos": self.data_train["positive"][idx],
#             "neg": self.data_train["negative"][idx],
#             "img": self.imgs_train[idx],
#         }
#         test_data = {
#             "pos": self.data_test["positive"][idx],
#             "neg": self.data_test["negative"][idx],
#             "img": self.imgs_test[idx],
#         }
#
#         principle = file_utils.load_json(str(config.kp_gestalt_dataset / "train" / f"{idx:06d}.json"))["principle"]
#         return train_data, test_data, principle


def load_dataset(args, mode):
    data_path = config.kp_gestalt_dataset / mode
    _dataset = GestaltDataset(data_path)
    data_loader = DataLoader(_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    return data_loader


import torchvision.transforms as transforms

IMAGE_SIZE = 1024  # ViT default input size
import torchvision.datasets as datasets
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset


def get_dataloader(data_dir, label):
    img_files = file_utils.get_all_files(data_dir / label, '.png')
    img_files = sorted(img_files)
    img_list = []
    for file in img_files:
        img = file_utils.load_img(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(torch.from_numpy(img))
    return img_list


def get_imgs(dataloader, device):
    imgs = []
    for images, labels in dataloader:
        imgs.extend(images.to(device))
    return imgs


def load_elvis_dataset(args):
    principle_path = config.storage / "res_1024" / args.run_principle
    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    return pattern_folders
