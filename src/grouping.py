# Created by X at 17.06.24

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from collections import defaultdict

from utils import tile_utils, file_utils, log_utils, args_utils
import group_section


def find_color_components(matrix):
    color_groups = defaultdict(list)

    # Collect coordinates of each color
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            color = matrix[x, y]
            color_groups[color].append((x, y))
    color_groups = sorted(list(color_groups.values()), key=lambda x: len(x), reverse=True)
    return color_groups


def group_by_color_single(example):
    data_input = example["input"]
    data_output = example["output"]
    # grouping by color
    input_groups_color = find_color_components(np.array(data_input))
    output_groups_color = find_color_components(np.array(data_output))
    return input_groups_color, output_groups_color


def group_by_section_single(example):
    data_input = example["input"]
    data_output = example["output"]
    # grouping by section

    # split by a bar
    # split by a grid
    # split by a bounding box
    input_groups_section = group_section.connection_splitting(np.array(data_input))
    input_groups_section = group_section.grid_splitting(np.array(data_input))
    input_groups_section = group_section.bar_splitting(np.array(data_input))

    output_groups_section = None

    return input_groups_section, output_groups_section


def group_by_connection_single(example):
    data_input = example["input"]
    data_output = example["output"]
    # grouping by connection
    raise NotImplementedError

    # input_groups_connection = is_splitting(np.array(data_input))
    # input_groups_connection = get_blocks(np.array(data_input))
    # output_groups_connection = get_blocks(np.array(data_output))
    # return input_groups_connection, output_groups_connection


def group_by_color(data):
    cha_data = data["cha"]
    sol_data = data["sol"]
    # return groups for each task
    train_groups = []
    test_groups = []
    for t_i in tqdm(range(len(cha_data)), "Grouping by color"):
        task_train_groups = []
        for example in cha_data[t_i]["train"]:
            data_input = example["input"]
            data_output = example["output"]
            # grouping by color
            input_groups_color = find_color_components(np.array(data_input))
            output_groups_color = find_color_components(np.array(data_output))
            # grouping by ...
            task_train_groups.append({"input": input_groups_color,
                                      "output": output_groups_color})
        train_groups.append(task_train_groups)

        task_test_groups = []
        for e_i in range(len(sol_data[t_i])):
            data_input = cha_data[t_i]["test"][e_i]["input"]
            data_output = sol_data[t_i][e_i]

            input_groups_color = find_color_components(np.array(data_input))
            output_groups_color = find_color_components(np.array(data_output))
            task_test_groups.append({"input": input_groups_color,
                                     "output": output_groups_color})
        test_groups.append(task_test_groups)

    return [train_groups, test_groups]


def group2patch(group):
    min_x, max_x, min_y, max_y = get_bounding_box(group)
    patch = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    group = np.array(group)
    group[:, 0] = group[:, 0] - min_x
    group[:, 1] = group[:, 1] - min_y
    for pos in group:
        patch[pos[0], pos[1]] = 1
    return patch


def get_bounding_box(coords):
    min_x = min(coords, key=lambda t: t[0])[0]
    max_x = max(coords, key=lambda t: t[0])[0]
    min_y = min(coords, key=lambda t: t[1])[1]
    max_y = max(coords, key=lambda t: t[1])[1]
    return min_x, max_x, min_y, max_y


def is_inside(inner_box, outer_box):
    return (inner_box[0] > outer_box[0] and
            inner_box[1] < outer_box[1] and
            inner_box[2] > outer_box[2] and
            inner_box[3] < outer_box[3])


def inputinoutput(input_groups, output_groups):
    # convert to patch
    relations = np.zeros((len(input_groups), len(output_groups)))

    for ig_i in range(len(input_groups)):
        input_patch = group2patch(input_groups[ig_i])
        for og_i in range(len(output_groups)):
            output_patch = group2patch(output_groups[og_i])
            relations[ig_i, og_i] = tile_utils.part_of_patch(input_patch, output_patch)
    return relations


def get_belong_relations(color_groups_cha):
    data_relations = []
    for task_i in tqdm(range(len(color_groups_cha)), desc="Finding Group Relations"):
        task_relations = []
        for e_i in range(len(color_groups_cha[task_i])):
            input_groups_color = color_groups_cha[task_i][e_i][0]
            output_groups_color = color_groups_cha[task_i][e_i][1]
            # input in output
            task_relations.append(inputinoutput(input_groups_color, output_groups_color))
        data_relations.append(task_relations)
    return data_relations


# color_groups_eval_cha = group_by_color(raw_data["eval_cha"])

# visualization

# visual.export_groups_as_images(raw_data["eval_cha"], color_groups_eval_cha, "eval_cha")

# find relations between input and output groups
# belong_group_pairs = get_belong_relations(color_groups_cha)

def reason_shape_relation(igs, og):
    # convert to patch
    relations = np.zeros(len(igs))
    for ig_i in range(len(igs)):
        input_patch = group2patch(igs[ig_i])
        output_patch = group2patch(og)
        relations[ig_i] = tile_utils.part_of_patch(input_patch, output_patch)

    return relations


def find_identical_shape(space_patch, target_patch):
    """ the small_array can be x% identical to large_array
    return: top n identical patches in large array, its position, width, and different tiles if any
    """
    space_patch = np.array(space_patch)
    target_patch = np.array(target_patch)
    # map all number to 1 (bw mode)
    space_patch = space_patch / (space_patch + 1e-20)
    target_patch = target_patch / (target_patch + 1e-20)
    # Get sliding windows of shape (3, 3) from the large array
    windows = tile_utils.sliding_window_view(space_patch, target_patch.shape)
    # Calculate the similarity percentage for each window
    match_counts = np.sum(windows == target_patch, axis=(2, 3))
    similarity = match_counts / target_patch.size * 100
    # Generate the positions and differences
    positions = np.argwhere(similarity == 100)
    return positions


def io2st_patch(input_patch, output_patch):
    if input_patch.shape == output_patch.shape:
        # find identical patch in state B
        space_patch = output_patch
        target_patch = input_patch
    elif input_patch.shape[0] >= output_patch.shape[0] and input_patch.shape[1] >= output_patch.shape[1]:
        # input shape is bigger than output
        space_patch = input_patch
        target_patch = output_patch
    else:
        space_patch = output_patch
        target_patch = input_patch
    return space_patch, target_patch


def find_common_prop(example, ig, og):
    ig = np.array(ig)
    og = np.array(og)
    input_patch = np.array(example["input"])
    output_patch = np.array(example["output"])

    ig_mask = np.zeros_like(input_patch, dtype=bool)
    og_mask = np.zeros_like(output_patch, dtype=bool)
    for pos in ig:
        ig_mask[pos[0], pos[1]] = True
    for pos in og:
        og_mask[pos[0], pos[1]] = True
    input_patch[~ig_mask] = 0
    output_patch[~og_mask] = 0
    ig_color = np.unique(input_patch)
    og_color = np.unique(output_patch)
    space_patch, target_patch = io2st_patch(input_patch, output_patch)
    duplicate_pos = find_identical_shape(space_patch, target_patch)
    scale_io_ratio = len(output_patch) / len(input_patch)
    # task 00:
    ig2og_relations = {
        "scale_io_ratio": scale_io_ratio,
        "duplicate_pos": duplicate_pos,
        "io_color_mapping": [ig_color, og_color]
    }
    return ig2og_relations


def progs_check(ig2og_relations, ig):
    io_duplicate_pos = ig2og_relations["io_duplicate_pos"]
    scale_io_ratio = ig2og_relations["scale_io_ratio"]

    progs = {
        "input_scaling": np.all(io_duplicate_pos == ig * scale_io_ratio),
    }


def get_prop_igs2og(example, og, igs):
    igs2og = []
    for ig in igs:
        ig2og = find_common_prop(example, ig, og)
        igs2og.append(ig2og)
    return igs2og


def _fun(example, og, igs):
    og_shape_ig = None
    og_color_ig = None
    # find a way to map ig to og.
    igs2og = get_prop_igs2og(example, og, igs)

    return igs2og


def get_groups_code(example, groups):
    groups_code = []
    for group in groups:
        group_patch = np.array(example)
        ig_mask = np.zeros_like(group_patch, dtype=bool)
        for pos in group:
            ig_mask[pos[0], pos[1]] = True
        group_patch[~ig_mask] = 0
        group_color = np.unique(group_patch)[1:]

        # space_patch, target_patch = io2st_patch(input_patch, output_patch)
        # duplicate_pos = find_identical_shape(space_patch, target_patch)
        # scale_io_ratio = len(output_patch) / len(input_patch)
        group_code = {
            "color": group_color,
            "tile_pos": group,
            "width": group_patch.shape[0],
            'group_patch': group_patch,
        }

        groups_code.append(group_code)
    return groups_code


def percept_task_features(args, example):
    example["input"] = np.array(example["input"]) + 1
    example["output"] = np.array(example["output"]) + 1

    # splitting by colors how to decide the group strategy?
    # So grouping with fixed way and estimated by 
    igs_color, ogs_color = group_by_color_single(example)
    igs_color_data = get_groups_code(example["input"], igs_color)
    ogs_color_data = get_groups_code(example["output"], ogs_color)

    # splitting by sections
    # igs_section, ogs_section = group_by_section_single(example)
    # igs_section_data = get_groups_code(example["input"], igs_section)
    # ogs_section_data = get_groups_code(example["output"], ogs_section)
    #
    # # splitting by connections
    # igs_connection, ogs_connection = group_by_connection_single(example)
    # igs_connection_data = get_groups_code(example["input"], igs_connection)
    # ogs_connection_data = get_groups_code(example["output"], ogs_connection)

    example_features = {"ig": igs_color_data,
                        "og": ogs_color_data}

    args.ig_num = len(igs_color)
    args.og_num = len(ogs_color)

    return example_features


def percept_task_features_sections(args, example):
    example["input"] = np.array(example["input"]) + 1
    example["output"] = np.array(example["output"]) + 1
    # grouping the tiles in example
    igs, ogs = group_by_section_single(example)
    ig_codes = get_groups_code(example["input"], igs)
    og_codes = get_groups_code(example["output"], ogs)
    example_features = {"input_groups": ig_codes, "output_groups": og_codes}
    args.ig_num = len(igs)
    args.og_num = len(ogs)
