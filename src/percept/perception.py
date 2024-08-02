# Created by shaji at 25/07/2024


import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F for functional operations
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
import cv2 as cv

import config
from src.utils import data_utils, visual_utils


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
        self.model = FCN(in_channels=1).to(device)
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


class PerceptTriangle(nn.Module):
    def __init__(self, args, device):
        super(PerceptTriangle, self).__init__()
        self.device = device
        self.args = args
        # self.model = FCN(1).to(device)
        # self.model.load_state_dict(torch.load(config.model_group_kp_triangle, map_location=device))
        # self.model.eval()  # Set the model to evaluation mode

        self.model_only = FCN(1).to(device)
        self.model_only.load_state_dict(torch.load(config.model_group_kp_triangle_only, map_location=device))
        self.model_only.eval()

    def merge_overlapping_lines(self, lines):
        merged_lines = []

        # Function to check if two lines overlap
        def lines_overlap(line1, line2):
            return any(point in line1 for point in line2)

        while lines:
            current_line = lines.pop(0)
            overlap_found = False
            for i, merged_line in enumerate(merged_lines):
                if lines_overlap(current_line, merged_line):
                    merged_lines[i] = merged_lines[i] | set(current_line)
                    overlap_found = True
                    break
            if not overlap_found:
                merged_lines.append(set(current_line))

        return [list(merged_line) for merged_line in merged_lines]

    def find_lines(self, matrix):
        # Use Canny edge detection
        edges = np.uint8(matrix > 0) * 255  # Edge detection based on non-zero elements
        # Hough Transform parameters
        rho_resolution = 0.5  # distance resolution in pixels of the Hough grid
        theta_resolution = np.pi / 180  # angle resolution in radians of the Hough grid
        threshold = 5  # minimum number of votes (intersections in Hough grid cell)
        # Detect lines using the Hough Transform
        lines = cv.HoughLines(edges, rho_resolution, theta_resolution, threshold)
        # List to store removable metrics for each line
        lines_with_removable_metrics = []
        rows, cols = matrix.shape
        if lines is not None:
            for rho_theta in lines:
                rho, theta = rho_theta[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                line_metrics = []

                for i in range(-max(rows, cols), max(rows, cols)):
                    x = int(x0 + i * (-b))
                    y = int(y0 + i * (a))
                    if 0 <= x < cols and 0 <= y < rows:
                        if matrix[y, x] > 0:
                            line_metrics.append((y, x))

                # Append the line and its corresponding points
                if line_metrics:
                    lines_with_removable_metrics.append(line_metrics)
        return lines_with_removable_metrics

    def remove_point_from_matrix(self, matrix):
        sub_matrics = []
        # Iterate over all elements in the matrix
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                # Check if the current element is non-zero
                if matrix[0, i, j] != 0:
                    # Create a copy of the original matrix
                    new_matrix = matrix.clone()
                    # Set the specific element to zero
                    new_matrix[0, i, j] = 0
                    # Append the modified matrix to the list
                    sub_matrics.append(new_matrix.unsqueeze(0))
        return torch.cat(sub_matrics, dim=0)

    def extract_triangle(self, matrix):
        tri_only_confs = []
        tri_only_conf = self.model_only(matrix.to(self.device))
        tri_only_conf = tri_only_conf[0, np.argmax(config.obj_false)].tolist()
        tri_only_confs.append(tri_only_conf)

        # while conf above threshold or no lines can be removed
        new_matrix = matrix.clone()

        try_count = 0
        while tri_only_conf < self.args.th_group:
            sub_matrices = self.remove_point_from_matrix(new_matrix)
            tri_conf = self.model_only(sub_matrices.to(self.device))
            tri_best_conf = tri_conf[:, np.argmax(config.obj_false)].min()
            best_idx = tri_conf[:, np.argmax(config.obj_false)].argmin()
            new_matrix = sub_matrices[best_idx]
            tri_only_conf = tri_best_conf.tolist()
            tri_only_confs.append(tri_only_conf)

            img = visual_utils.patch2img(new_matrix.squeeze().to(torch.int).tolist())
            img_file = config.output / f"kp_sy_{self.args.exp_name}" / f"extracted_patch_{try_count}.png"
            visual_utils.save_image(img, str(img_file))
            try_count += 1
        triangle = None
        return triangle

    def forward(self, x):
        # check triangle
        # triangle_conf = self.model(x.to(self.device))
        # has_triangle = False
        # triangle = None
        # if triangle_conf[0, np.argmax(config.obj_true)] > 0.8:
        #     has_triangle = True
        lines = self.find_lines(x.squeeze())
        triangle = self.extract_triangle(x)

        return triangle


def percept_objs(args, example_features):
    # visualize patch
    img = visual_utils.patch2img(example_features.squeeze().to(torch.int).tolist())
    img_file = config.output / f"kp_sy_{args.exp_name}" / f"patch.png"
    visual_utils.save_image(img, str(img_file))
    # extract objects
    perceptor_triangle = PerceptTriangle(args, args.device)
    triangles = perceptor_triangle(example_features)
    objs = {"triangle": triangles}
    return objs
