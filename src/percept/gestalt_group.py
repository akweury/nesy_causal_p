# Created by jing at 01.12.24
import torch
import cv2
import numpy as np
from src import bk


class Group():
    def __init__(self, id, name, input_signal, onside_signal, parents,
                 color, coverage):
        self.id = id
        self.name = name
        self.input = input_signal
        self.onside = onside_signal
        # self.memory = memory_signal
        self.parents = parents
        self.color = self.search_color(input_signal, onside_signal, color)
        self.pos, self.size = self.find_center()
        self.onside_coverage = coverage

    def __str__(self):
        # return self.name
        return self.name  # + "_" + str(self.id)

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == Group:
            return self.name == other.name
        else:
            return False

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def search_color(self, input_signal, onside_signal, color):
        if color is not None:
            return bk.color_matplotlib[color]
        mask = onside_signal > 0
        input_tensor = input_signal.numpy()
        zoomed_image = cv2.resize(input_tensor, onside_signal.size(),
                                  interpolation=cv2.INTER_NEAREST)
        zoomed_image = torch.from_numpy(zoomed_image)
        valid_pixels = zoomed_image[mask]

        # Find the most frequent color in the list
        if len(valid_pixels) == 0:
            return bk.color_matplotlib[bk.no_color]  # Handle empty list

        color_counts = valid_pixels.unique(return_counts=True, dim=0)
        color_sorted = sorted(zip(color_counts[0], color_counts[1]),
                              key=lambda x: x[1], reverse=True)
        most_frequent = color_sorted[0][0]

        # Find the closest color in the dictionary
        closest_color_name = bk.no_color
        smallest_distance = float('inf')

        for color_name, color_rgb in bk.color_matplotlib.items():
            distance = torch.sqrt(sum((c1 - c2) ** 2 for c1, c2 in
                                      zip(most_frequent, torch.tensor(color_rgb))))
            if distance < smallest_distance:
                smallest_distance = distance
                closest_color_name = color_name

        color = bk.color_matplotlib[closest_color_name]
        return color

    def find_center(self):
        matrix = self.input.sum(axis=-1)
        matrix[matrix == 633] = 0
        # Get the indices of all nonzero elements
        nonzero_indices = torch.argwhere(matrix != 0).float()

        if len(nonzero_indices) == 0:
            return torch.tensor([torch.nan,
                                 torch.nan])  # Return nan if there are no nonzero elements

        # Calculate the average row and column indices
        row_center = torch.mean(nonzero_indices[:, 0])
        col_center = torch.mean(nonzero_indices[:, 1])
        pos = torch.tensor([row_center, col_center])

        x_size = (nonzero_indices[:, 0].max() - nonzero_indices[:, 0].min()) / self.input.shape[0]
        y_size = (nonzero_indices[:, 1].max() - nonzero_indices[:, 1].min()) / self.input.shape[1]
        obj_size = x_size * y_size
        if pos.max() > 1:
            pos = pos / 512
        return pos, obj_size


def group_tensor2dict(group_tensor):
    group_dict = {"x": group_tensor[0],
                  "y": group_tensor[1],
                  "size": group_tensor[2],
                  "obj_num": group_tensor[3],
                  "color_r": group_tensor[4],
                  "color_g": group_tensor[5],
                  "color_b": group_tensor[6],
                  "shape_tri": group_tensor[7],
                  "shape_sq": group_tensor[8],
                  "shape_cir": group_tensor[9]}
    return group_dict


def gen_group_tensor(x, y, size, obj_num, r, g, b, tri, sq, cir):
    return torch.tensor([x, y, size, obj_num, r, g, b, tri, sq, cir])


def gen_extended_group_tensor(ocm):
    extended_group_tensor = torch.zeros(4, 3, 32, 14)
    extended_group_tensor[:, :, :, :10] = torch.repeat_interleave(torch.from_numpy(ocm).unsqueeze(0), 4, dim=0)
    return extended_group_tensor


def group_dict2tensor(group_dict):
    group_tensor = torch.tensor([
        group_dict["x"],
        group_dict["y"],
        group_dict["size"],
        group_dict["obj_num"],
        group_dict["color_r"],
        group_dict["color_g"],
        group_dict["color_b"],
        group_dict["shape_tri"],
        group_dict["shape_sq"],
        group_dict["shape_cir"]
    ])
    return group_tensor


def group2tensor(group):
    obj_num = 1 if group.parents is None else group.parents.shape[0]

    tri = 1 if bk.bk_shapes[group.name] == "triangle" else 0
    sq = 1 if bk.bk_shapes[group.name] == "square" else 0
    cir = 1 if bk.bk_shapes[group.name] == "circle" else 0
    color = torch.tensor(group.color) / 255
    tensor = gen_group_tensor(group.pos[0], group.pos[1], group.size, obj_num,
                              color[0], color[1], color[2], tri, sq, cir)
    return tensor


def gcm_encoder(labels, ocms, group_shape=0):
    shape = bk.bk_shapes[group_shape]
    groups = []
    for example_i in range(len(ocms)):
        ocm = ocms[example_i]
        example_labels = labels[example_i]
        example_groups = []
        for l_i, label in enumerate(example_labels.unique()):
            group_ocms = ocm[example_labels == label]
            parent_positions = group_ocms[:, :2]
            x = parent_positions[:, 0]
            y = parent_positions[:, 1]
            group_x = x.mean()
            group_y = y.mean()
            obj_num = len(group_ocms)
            if len(group_ocms) == 1:
                gcm = group_ocms[0]
            else:
                group_size = 0.5 * (x.max() - x.min() + y.max() - y.min())
                color_r = 0
                color_g = 0
                color_b = 0
                shape_tri = 1 if shape == "triangle" else 0
                shape_sq = 1 if shape == "square" else 0
                shape_cir = 1 if shape == "circle" else 0
                gcm = gen_group_tensor(group_x, group_y, group_size, obj_num, color_r, color_g, color_b,
                                       shape_tri, shape_sq, shape_cir)
            group_ocms = group_ocms.reshape(-1, 10)
            gcm = gcm.reshape(-1, 10)
            group = {"gcm": gcm, "ocm": group_ocms}
            example_groups.append(group)
        groups.append(example_groups)

    return groups
