# Created by jing at 17.06.24

import numpy as np

from utils import tile_utils, hierarchical_utils, visual_utils

"""
    dim_in == dim_out 
        - group individual --> match
    dim_in > dim_out 
        - find identical groups in dim_in 
    dim_in < dim_out 
        - find identical groups in dim_out
"""


def highlight_patch(small_patch, big_patch, pos):
    patch = np.zeros_like(big_patch)
    group_region = big_patch[pos[0]:pos[0] + len(small_patch), pos[1]:pos[1] + len(small_patch[0])]
    patch[pos[0]:pos[0] + len(small_patch), pos[1]:pos[1] + len(small_patch[0])] = group_region

    return patch


def hier_data2group_img(hier_data):
    patches = []
    for group in hier_data['identical_groups']:
        # add group images
        patch_array = highlight_patch(hier_data['output'], hier_data['input'], group['position'])
        patches.append(patch_array)

    return patches


def id_data2patches(input_patch, output_patch, id_groups):
    patches = []
    for group in id_groups:
        patch = highlight_patch(input_patch, output_patch, group['position'])
        patches.append(patch)
    return patches


def group_with_identical(args, input_patches, output_patches):
    data = []
    all_patches = []
    for input_patch, output_patch in zip(input_patches, output_patches):
        input_patch = np.array(input_patch)
        output_patch = np.array(output_patch)
        if input_patch.shape == output_patch.shape:
            # find identical patch in state B
            id_groups = tile_utils.find_identical_groups(space_patch=output_patch, target_patch=input_patch)
            id_patches = id_data2patches(input_patch, output_patch, id_groups)

        elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
            # input shape is bigger than output
            id_groups = tile_utils.find_identical_groups(input_patch, output_patch)
            id_patches = id_data2patches(output_patch, input_patch, id_groups)
        else:
            # if output contains input?
            id_groups = tile_utils.find_identical_groups(output_patch, input_patch)
            id_patches = id_data2patches(input_patch, output_patch, id_groups)
        data.append(id_groups)
        all_patches.append(id_patches)
    # all_patches = np.array(all_patches)
    # all_patches = np.moveaxis(all_patches, 0, 1)
    return data, all_patches


def data2patch(args, data):
    input_patches = []
    output_patches = []
    for example in data:
        input_patch = example['input']
        output_patch = example['output']
        input_patches.append(input_patch)
        output_patches.append(output_patch)
    return input_patches, output_patches
