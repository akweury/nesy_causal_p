# Created by jing at 10.12.24

import torch
import cv2

from torch.utils.data import Dataset, DataLoader
import config
from src.utils import data_utils, file_utils, chart_utils


class BasicShapeDataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.image_paths = []
        # self.labels = []
        self.device = args.device
        folder = config.kp_base_dataset / args.bk_shape
        self.image_paths = file_utils.get_all_files(folder, "png", False)[:1000]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(self.image_paths[idx])
        cropped_img, _ = data_utils.crop_img(rgb_image)
        bw_img = data_utils.resize_img(cropped_img, resize=8)
        return bw_img


class GestaltDataset(Dataset):
    def __init__(self, args, imgs):
        self.args = args
        self.device = args.device
        self.imgs = imgs
        #
        # self.image_paths = []
        #
        # folder = config.kp_base_dataset / args.exp_name
        # imgs = file_utils.get_all_files(folder, "png", False)[:1000]
        # # labels = [self.get_label(args.exp_name) for img in imgs]
        # self.image_paths += imgs
        # # self.labels += labels

    def __len__(self):
        return len(self.imgs)

    def load_data(self, idx):
        file_name, file_extension = self.imgs[idx].split(".")
        data = file_utils.load_json(f"{file_name}.json")
        img = file_utils.load_img(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, data, file_name.split("/")[-4:]

        # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.args.device)

    def __getitem__(self, idx):
        img, data, file_name = self.load_data(idx)
        self.args.logger.debug(
            f"\n =========== Analysis Image {file_name} {idx + 1}/{len(self.imgs)} ==============")

        return img, data
        # img = data_utils.load_bw_img(self.image_paths[idx], size=64)
        # resize
        # file_name, file_extension = self.image_paths[idx].split(".")
        # data = file_utils.load_json(f"{file_name}.json")
        # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.device)

        # return img


class GSDataset(Dataset):
    def __init__(self):
        self.data = torch.load(config.kp_gestalt_dataset / "train" / "train.pt")
        self.imgs = self.load_imgs(config.kp_gestalt_dataset / 'train')

    def __len__(self):
        return len(self.data)

    def load_imgs(self, path):
        img_files = file_utils.get_all_files(path, '.png')
        img_files = sorted(img_files)
        imgs = []
        for file in img_files:
            img = file_utils.load_img(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list = []
            for i in range(img.shape[1] // 516):
                img_list.append(img[:, i * 516:(i + 1) * 516, :])
            img_list = [im[2:-2, 2:-2] for im in img_list]
            imgs.append(img_list)
        return imgs

    def __getitem__(self, idx):
        return self.data["positive"][idx], self.data["negative"][idx], self.imgs[idx]


def load_dataset(args):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Importing training and testing data.")

    train_dataset = GSDataset()
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    return train_loader
