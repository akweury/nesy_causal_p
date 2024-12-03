# Created by jing at 01.12.24
import torch
import cv2
from collections import Counter

from src.alpha.fol import bk
from src.utils import data_utils


class Group():
    def __init__(self, id, name, input_signal, onside_signal, memory_signal, parents, conf):
        self.id = id
        self.name = name
        self.input = torch.from_numpy(input_signal)
        self.onside = onside_signal
        self.memory = memory_signal
        self.parents = parents
        self.conf = conf
        self.color = self.search_color(input_signal, onside_signal)
        self.ocm = None
        self.pos = None

    def __str__(self):
        # return self.name
        return self.name + "_" + str(self.id)

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

    def search_color(self, input_signal, onside_signal):
        mask = onside_signal > 0
        zoomed_image = cv2.resize(input_signal, onside_signal.size(), interpolation=cv2.INTER_NEAREST)
        zoomed_image = torch.from_numpy(zoomed_image)
        valid_pixels = zoomed_image[mask]

        # Find the most frequent color in the list
        if len(valid_pixels) == 0:
            return bk.no_color  # Handle empty list

        color_counts = valid_pixels.unique(return_counts=True, dim=0)
        color_sorted = sorted(zip(color_counts[0], color_counts[1]), key=lambda x: x[1], reverse=True)
        most_frequent = color_sorted[0][0]

        # Find the closest color in the dictionary
        closest_color_name = bk.no_color
        smallest_distance = float('inf')

        for color_name, color_rgb in bk.color_matplotlib.items():
            distance = torch.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(most_frequent, torch.tensor(color_rgb))))
            if distance < smallest_distance:
                smallest_distance = distance
                closest_color_name = color_name
        return closest_color_name

    def obj2tensor(self, shape, color, pos, group_name, group_count_conf):
        obj_tensor = torch.zeros(len(bk.obj_ohc))
        i = 0
        obj_tensor[i] = bk.color_large.index(color)  # color
        i += 1
        obj_tensor[i] = bk.shape_extend.index(shape)  # shape
        i += 1
        obj_tensor[i] = pos[0]  # x position
        i += 1
        obj_tensor[i] = pos[1]  # y position
        i += 1
        obj_tensor[i] = bk.group_name_extend.index(group_name)  # group label
        i += 1
        obj_tensor[i] = group_count_conf  # group confidence according to the count of objects
        return obj_tensor

    def generate_tensor(self):
        self.pos = self.find_center()
        self.ocm = torch.stack([self.obj2tensor(self.name, self.color, self.pos, self.name, self.conf)])

    def find_center(self):
        matrix = self.memory.sum(dim=0)

        # Get the indices of all nonzero elements
        nonzero_indices = torch.argwhere(matrix != 0).float()

        if len(nonzero_indices) == 0:
            return torch.tensor([torch.nan, torch.nan])  # Return nan if there are no nonzero elements

        # Calculate the average row and column indices
        row_center = torch.mean(nonzero_indices[:, 0])
        col_center = torch.mean(nonzero_indices[:, 1])
        return torch.tensor([row_center, col_center])
