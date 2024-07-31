# Created by shaji at 25/07/2024


import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F for functional operations
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks

import config
from src.utils import data_utils


class FCN(nn.Module):
    def __init__(self, in_channels):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Final fully connected layer for classification
        self.fc = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = torch.sigmoid(self.fc(x))
        return x.view(-1, 2)


class PerceptLine(nn.Module):
    def __init__(self, path, device):
        super(PerceptLine, self).__init__()
        self.model = FCN().to(device)
        self.device = device
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set the model to evaluation mode

    def find_lines(self, matrix):
        # Perform Hough Transform
        h, theta, d = hough_line(matrix.numpy())

        # Extract the angle and distance for the most prominent line
        accum, angles, dists = hough_line_peaks(h, theta, d)
        lines = [(angle, dist) for angle, dist in zip(angles, dists)]
        return lines

    def forward(self, x):
        # check lines
        g_patch = data_utils.group2patch(x["group_patch"], x["tile_pos"])
        g_tensor = data_utils.patch2tensor(g_patch)
        has_line = False
        line_conf = self.model(g_tensor.to(self.device).unsqueeze(0))
        lines = []
        if config.obj_true[line_conf.argmax()] == 1:
            has_line = True
            # find lines inside the patch
            lines = self.find_lines(g_tensor)
        return has_line, lines


class PerceptBB(nn.Module):
    def __init__(self, path, device):
        super(PerceptBB, self).__init__()
        self.model = FCN().to(device)
        self.device = device
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set the model to evaluation mode

    def generate_anchor_boxes(self, matrix):
        """
        Generate anchor boxes for a feature map.

        :param matrix: Tuple of (height, width) of the feature map
        :param base_size: The base size of the anchor boxes
        :param scales: List of scales to use for the anchor boxes
        :param aspect_ratios: List of aspect ratios (width/height) for the anchor boxes
        :return: List of anchor boxes, each represented by (center_x, center_y, width, height)
        """
        anchors = []
        for row in range(matrix[0] - 2):
            for col in range(matrix[1] - 2):
                # Center of the current feature map cell
                for w in range(3, matrix[1] - col + 1):
                    for h in range(3, matrix[0] - row + 1):
                        bb = (row, col, w, h)
                        if bb not in anchors and 2 < bb[2] <= matrix[1] and 2 < bb[3] <= matrix[0]:
                            anchors.append(bb)
        return anchors

    def find_rects(self, matrix, anchor_boxes):
        rect_confs = []
        for box in anchor_boxes:
            row, col, w, h = box
            top = matrix[row, col:col + w]
            left = matrix[row + 1:row + h, col]
            bottom = matrix[row + h - 1, col + 1:col + w]
            try:
                right = matrix[row + 1:row + h - 1, col + w - 1]
            except IndexError:
                raise IndexError
            bb_tiles = np.concatenate((top, left, bottom, right))
            non_zero_tiles = bb_tiles[bb_tiles != 0]
            if len(non_zero_tiles) == 0:
                bb_conf = 0
            else:
                _, counts = np.unique(non_zero_tiles, return_counts=True)
                bb_conf = counts[0] / len(bb_tiles)
            rect_confs.append(bb_conf)
        indices = [i for i, v in enumerate(rect_confs) if v > 0.99]
        rects = [anchor_boxes[i] for i in indices]
        return rects

    def forward(self, x):
        # check lines
        g_patch = data_utils.group2patch(x["group_patch"], x["tile_pos"])
        g_tensor = data_utils.patch2tensor(g_patch)
        has_rect = False
        rect_conf = self.model(g_tensor.to(self.device).unsqueeze(0))
        rects = []
        if rect_conf[0, np.argmax(config.obj_true)] > 0.8:
            has_rect = True
            # Generate anchor boxes
            anchor_boxes = self.generate_anchor_boxes(g_tensor.shape)
            rects = self.find_rects(g_tensor, anchor_boxes)
        return has_rect, rects


def percept_objs(args, example_features):
    perceptor_line = PerceptLine(config.output / f'train_cha_line_groups' / 'line_detector_model.pth', args.device)
    perceptor_bb = PerceptBB(config.output / f'train_cha_rect_groups' / 'rect_detector_model.pth', args.device)
    objs = {"input_groups": [], "output_groups": []}
    for group_type in ["input_groups", "output_groups"]:
        for group in example_features[group_type]:
            # line
            is_line, lines = perceptor_line(group)
            # rectangle
            is_rect, rects = perceptor_bb(group)

            objs[group_type].append({"is_rect": is_rect, "rects": rects, "is_line": is_line, "lines": lines})

    return objs
