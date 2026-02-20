# Created by X at 10.12.24

import torch
import cv2
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import config
from src.utils import data_utils, file_utils, chart_utils
import numpy as np
import os
import glob
from PIL import Image
import re
from src import bk


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
            transforms.Resize((1024, 1024)),
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
            all_names = sorted(glob.glob(os.path.join(task_folder, "*.png")), key=self.extract_idx)
            for img_i, img_name in enumerate(all_names):
                img_id = os.path.basename(img_name).split(".")[0]
                if img_id not in img_data:
                    continue
                img_label = img_i < len(all_names) // 2
                self.samples.append({
                    "image_path": img_name,
                    "img_label": img_label,
                    "symbolic_data": img_data[img_id],
                    "principle": principle,
                    "task": os.path.basename(task_folder)})

    def extract_idx(self, path: str) -> int:
        # 匹配最后一个下划线到 .png 之间的数字
        m = re.search(r'_(\d+)\.png$', path)
        return int(m.group(1)) if m else -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def shape_to_id(name):
    if name == "pac_man":
        n = bk.bk_shapes_2.index("circle")
    else:
        n = bk.bk_shapes_2.index(name)
    return n

def shape_to_id_clevr(name):
    n = bk.bk_shapes_clevr.index(name)
    return n


def extract_object_contour(image_path, object_data, num_points=64):
    """
    Extract contour keypoints for an object from an image and normalize to fixed length.

    Args:
        image_path: Path to the image file
        object_data: Dictionary containing object information (x, y, size, etc.)
                    All values are normalized (0-1 range)
        num_points: Number of points to sample from the contour (default: 64)

    Returns:
        numpy.ndarray: Array of shape (num_points, 2) containing normalized (x, y) coordinates in 0-1 range
                      Returns zeros if contour extraction fails
    """
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros((num_points, 2), dtype=np.float32)

        img_height, img_width = img.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        center_x_norm = float(object_data['x'])
        center_y_norm = float(object_data['y'])
        size_norm = float(object_data['size'])

        center_x = int(center_x_norm * img_width)
        center_y = int(center_y_norm * img_height)
        # Size is normalized, convert to pixels (assuming size is relative to image dimensions)
        size = int(size_norm * min(img_width, img_height))

        # Define ROI around object (with some padding)
        padding = int(size * 0.3)  # 30% padding
        x1 = max(0, center_x - size - padding)
        y1 = max(0, center_y - size - padding)
        x2 = min(img_width, center_x + size + padding)
        y2 = min(img_height, center_y + size + padding)

        # Extract ROI
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((num_points, 2), dtype=np.float32)

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        # Use adaptive threshold for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.zeros((num_points, 2), dtype=np.float32)

        # Get the largest contour (assumed to be the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Reshape contour to (N, 2)
        contour_points = largest_contour.reshape(-1, 2).astype(np.float32)

        # Adjust coordinates back to original image space (pixel coordinates)
        contour_points[:, 0] += x1
        contour_points[:, 1] += y1

        # Resample contour to fixed number of points (still in pixel coordinates)
        contour_resampled = resample_contour(contour_points, num_points)

        # Normalize contour coordinates to 0-1 range
        contour_normalized = contour_resampled.copy()
        contour_normalized[:, 0] /= img_width   # Normalize x
        contour_normalized[:, 1] /= img_height  # Normalize y

        return contour_normalized

    except Exception as e:
        # If any error occurs, return zeros
        print(f"Warning: Failed to extract contour from {image_path}: {e}")
        return np.zeros((num_points, 2), dtype=np.float32)


def resample_contour(contour_points, num_points):
    """
    Resample a contour to a fixed number of points using interpolation.

    Args:
        contour_points: Array of shape (N, 2) with original contour points
        num_points: Desired number of points

    Returns:
        numpy.ndarray: Array of shape (num_points, 2) with resampled points
    """
    if len(contour_points) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    # If contour has fewer points than desired, use interpolation
    if len(contour_points) < num_points:
        # Compute cumulative distance along contour
        distances = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        # Create evenly spaced sample points
        total_length = cumulative_distances[-1]
        if total_length == 0:
            # Degenerate case: all points are the same
            return np.tile(contour_points[0], (num_points, 1))

        sample_distances = np.linspace(0, total_length, num_points)

        # Interpolate x and y separately
        x_interp = np.interp(sample_distances, cumulative_distances, contour_points[:, 0])
        y_interp = np.interp(sample_distances, cumulative_distances, contour_points[:, 1])

        resampled = np.stack([x_interp, y_interp], axis=1).astype(np.float32)

    else:
        # If contour has more points, sample uniformly
        indices = np.linspace(0, len(contour_points) - 1, num_points).astype(int)
        resampled = contour_points[indices].astype(np.float32)

    return resampled


class GrbDataset(Dataset):
    def __init__(self, folder_path: str, mode: str, val_split: float = 0.4, task_num=None):
        assert mode in ["train", "val", "test"]
        self.samples = []
        if task_num is None:
            task_num = len(os.listdir(folder_path))

        for task_folder in sorted(os.listdir(folder_path))[:task_num]:
            full_task_path = os.path.join(folder_path, task_folder)
            if not os.path.isdir(full_task_path):
                continue

            task_data = {"task": task_folder, "positive": [], "negative": []}
            meta_data_loaded = False
            for class_label, class_name in enumerate(["negative", "positive"]):
                class_folder = os.path.join(full_task_path, class_name)
                if not os.path.isdir(class_folder):
                    continue

                image_files = sorted([f for f in os.listdir(class_folder) if f.endswith(".png")])
                split_idx = int(len(image_files) * (1 - val_split))

                if mode == "train":
                    selected_files = image_files[:split_idx]
                elif mode == "val":
                    selected_files = image_files[split_idx:]
                elif mode == "test":
                    selected_files = image_files  # full set

                for fname in selected_files:
                    img_path = os.path.join(class_folder, fname)
                    json_path = img_path.replace(".png", ".json")

                    if not os.path.exists(json_path):
                        continue

                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                    if not meta_data_loaded:
                        task_data["principle"] = json_data["principle"]
                        task_data["non_overlap"] = json_data["non_overlap"]
                        task_data["qualifier_all"] = json_data["qualifier_all"]
                        task_data["qualifier_exist"] = json_data["qualifier_exist"]
                        task_data["prop_shape"] = json_data["prop_shape"]
                        task_data["prop_color"] = json_data["prop_color"]
                        task_data["prop_size"] = json_data["prop_size"]
                        task_data["prop_count"] = json_data["prop_count"]
                        
                        meta_data_loaded = True

                    sym_data = [{
                        'x': od['x'],
                        'y': od['y'],
                        'size': od['size'],
                        'color_r': od['color_r'],
                        'color_g': od['color_g'],
                        'color_b': od['color_b'],
                        'shape': shape_to_id(od["shape"]),
                        "group_id": od["group_id"] if "group_id" in od else None,
                        # "contour": extract_object_contour(img_path, od),  # Now enabled
                    } for od in json_data["img_data"]]

                    entry = {
                        "image_path": img_path,
                        "img_label": class_label,
                        "symbolic_data": sym_data,
                        "principle": json_data["principle"]
                    }
                    task_data[class_name].append(entry)

            if task_data["positive"] or task_data["negative"]:
                self.samples.append(task_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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


def load_dataset(args, data, mode=None):
    _dataset = GrbDataset(data, mode)
    data_loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=False)
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


def load_train_val_dataset(args, train_loader, val_loader):
    train_val_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
    train_val_loader = DataLoader(
        train_val_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn
    )
    return train_val_loader


class MultiSplitTaskLoader(Dataset):
    def __init__(self, train_set, val_set, test_set):
        min_len = min(len(train_set), len(val_set), len(test_set))
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.length = min_len  # to allow zipping

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.train_set[idx],
            self.val_set[idx],
            self.test_set[idx]
        )


def load_combined_dataset(principle_path, task_num=None):
    combined_dataset = MultiSplitTaskLoader(GrbDataset(principle_path / "train", "train", task_num=task_num),
                                            GrbDataset(principle_path / "train", "val", task_num =task_num),
                                            GrbDataset(principle_path / "test", "test", task_num = task_num))
    combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    return combined_loader
