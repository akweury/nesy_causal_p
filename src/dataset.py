# Created by jing at 10.12.24

import torch
import cv2
import json
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
        rgb_image = torch.from_numpy(cv2.imread(self.image_paths[idx]))

        return rgb_image


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
        self.data_train = torch.load(config.kp_gestalt_dataset / "train" / "train.pt")
        self.imgs_train = self.load_imgs(config.kp_gestalt_dataset / "train")

        self.data_test = torch.load(config.kp_gestalt_dataset / "test" / "test.pt")
        self.imgs_test = self.load_imgs(config.kp_gestalt_dataset / "test")

    def __len__(self):
        return len(self.data_train["positive"])

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
        train_data = {
            "pos": self.data_train["positive"][idx],
            "neg": self.data_train["negative"][idx],
            "img": self.imgs_train[idx],
        }
        test_data = {
            "pos": self.data_test["positive"][idx],
            "neg": self.data_test["negative"][idx],
            "img": self.imgs_test[idx],
        }


        principle = file_utils.load_json(str(config.kp_gestalt_dataset / "train" / f"{idx:06d}.json"))["principle"]
        return train_data, test_data, principle


def load_dataset(args):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Importing training and testing data.")

    _dataset = GSDataset()
    data_loader = DataLoader(_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    return data_loader
