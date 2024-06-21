# Created by jing at 17.06.24

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from collections import defaultdict

from utils import tile_utils, file_utils, visual_utils
from src import visual
"""
    dim_in == dim_out 
        - group individual --> match
    dim_in > dim_out 
        - find identical groups in dim_in 
    dim_in < dim_out 
        - find identical groups in dim_out
"""


# def highlight_patch(small_patch, big_patch, pos):
#     patch = np.zeros_like(big_patch)
#     group_region = big_patch[pos[0]:pos[0] + len(small_patch), pos[1]:pos[1] + len(small_patch[0])]
#     patch[pos[0]:pos[0] + len(small_patch), pos[1]:pos[1] + len(small_patch[0])] = group_region
#
#     return patch


# def hier_data2group_img(hier_data):
#     patches = []
#     for group in hier_data['identical_groups']:
#         # add group images
#         patch_array = highlight_patch(hier_data['output'], hier_data['input'], group['position'])
#         patches.append(patch_array)
#
#     return patches


# def id_data2patches(input_patch, output_patch, id_groups):
#     patches = []
#     for group in id_groups:
#         patch = highlight_patch(input_patch, output_patch, group['position'])
#         patches.append(patch)
#     return patches


# def group_with_identical(args, input_patches, output_patches):
#     data = []
#     all_patches = []
#     for input_patch, output_patch in zip(input_patches, output_patches):
#         input_patch = np.array(input_patch)
#         output_patch = np.array(output_patch)
#         if input_patch.shape == output_patch.shape:
#             # find identical patch in state B
#             id_groups = tile_utils.find_identical_groups(space_patch=output_patch, target_patch=input_patch)
#             id_patches = id_data2patches(input_patch, output_patch, id_groups)
#
#         elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
#             # input shape is bigger than output
#             id_groups = tile_utils.find_identical_groups(input_patch, output_patch)
#             id_patches = id_data2patches(output_patch, input_patch, id_groups)
#         else:
#             # if output contains input?
#             id_groups = tile_utils.find_identical_groups(output_patch, input_patch)
#             id_patches = id_data2patches(input_patch, output_patch, id_groups)
#         data.append(id_groups)
#         all_patches.append(id_patches)
#     # all_patches = np.array(all_patches)
#     # all_patches = np.moveaxis(all_patches, 0, 1)
#     return data, all_patches

def find_connected_components(matrix):
    # To store all components for different colors
    all_components = []

    # Iterate over each possible color (0-9)
    for color in range(10):
        # Create a binary mask where the current color is 1 and all others are 0
        binary_mask = (matrix == color).astype(int)

        # Label the connected components in the binary mask
        labeled_array, num_features = label(binary_mask, structure=np.ones((3, 3)))

        # Extract components for the current color
        components = []
        for i in range(1, num_features + 1):
            component = list(zip(*np.where(labeled_array == i)))
            if component:
                components.append(component)

        # Append components of the current color to the all_components list
        if components:
            all_components.extend(components)
    all_components = sorted(all_components, key=lambda x: len(x), reverse=True)
    return all_components

def find_color_components(matrix):
    color_groups = defaultdict(list)

    # Collect coordinates of each color
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            color = matrix[x, y]
            color_groups[color].append((x, y))
    color_groups = sorted(list(color_groups.values()), key=lambda x: len(x), reverse=True)
    return color_groups



def group_by_color(train_cha):
    # return groups for each task
    train_cha_groups = []
    for task in tqdm(train_cha, "Grouping by color"):
        task_train_groups = []
        for example in task["train"]:
            data_input = example["input"]
            data_output = example["output"]
            # grouping by color

            input_groups_color = find_color_components(np.array(data_input))
            output_groups_color = find_color_components(np.array(data_output))
            # grouping by ...
            task_train_groups.append([input_groups_color, output_groups_color])
        train_cha_groups.append(task_train_groups)
    return train_cha_groups

def get_belong_relations(color_groups_cha):
    pass


# data file
raw_data = file_utils.get_raw_data()

# grouping by color
color_groups_cha = group_by_color(raw_data["train_cha"])

# visualization
# visual.export_groups_as_images(raw_data["train_cha"], color_groups_cha, "color")

# find relations between input and output groups


belong_group_pairs = get_belong_relations(color_groups_cha)

# group can be further divided into groups with another algorithm
# connected_groups_cha = group_by_connection(raw_data["train_cha"])



print("program finished")


