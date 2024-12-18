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
        bw_img = data_utils.rgb2bw(rgb_image, crop=True, resize=8)
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
        return img, data

        # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.args.device)

    def __getitem__(self, idx):
        self.args.logger.debug(
            f"\n =========== Analysis Image {idx + 1}/{len(self.imgs)} ==============")
        img, data = self.load_data(idx)
        return img, data
        # img = data_utils.load_bw_img(self.image_paths[idx], size=64)
        # resize
        # file_name, file_extension = self.image_paths[idx].split(".")
        # data = file_utils.load_json(f"{file_name}.json")
        # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.device)

        # return img


def load_dataset(args):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Importing training and testing data.")

    # get data file names
    train_imges = file_utils.get_all_files(args.train_folder,
                                           "png",
                                           False)[:500]
    positive_images = file_utils.get_all_files(args.test_true_folder,
                                               "png",
                                               False)[:500]
    random_imges = file_utils.get_all_files(args.test_random_folder,
                                            "png",
                                            False)[:500]
    counterfactual_imges = file_utils.get_all_files(args.test_cf_folder,
                                                    "png",
                                                    False)[:500]

    train_dataset = GestaltDataset(args, train_imges)
    test_pos_dataset = GestaltDataset(args, positive_images)
    test_rand_dataset = GestaltDataset(args, random_imges)
    test_cf_dataset = GestaltDataset(args, counterfactual_imges)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    test_pos_loader = DataLoader(test_pos_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)
    test_rand_loader = DataLoader(test_rand_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    test_cf_dataloader = DataLoader(test_cf_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

    return train_loader, test_pos_loader, test_rand_loader, test_cf_dataloader
