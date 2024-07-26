# Created by jing at 16.06.24
import os
import numpy as np
from tqdm import tqdm

import grouping
import src.utils.data_utils
from utils import visual_utils, file_utils, data_utils

import config


def export_data_as_images(raw_data, data_type):
    # export data as images
    output_folder = config.output / data_type
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visual_utils.export_task_img(raw_data[data_type], output_folder)


def export_line_groups(raw_data, g_data, group_type, line_indices):
    output_path = config.output / f'{group_type}_line_groups'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_cha, data_sol = raw_data["cha"], raw_data["sol"]
    g_train, g_test = g_data
    line_group_data = {}
    # init data structure
    for task_i in line_indices:
        line_group_data[task_i] = {"train": [], "test": []}
        for data_type in ["train", "test"]:
            for example_i in range(len(data_cha[int(task_i)][data_type])):
                line_group_data[task_i][data_type].append({"input": [], "output": []})
                for group_type in ["input", "output"]:
                    for ig_i in range(len(g_train[int(task_i)][example_i][group_type])):
                        line_group_data[task_i][data_type][example_i][group_type] = []

    for task_i in tqdm(line_indices, desc="Exporting Line patch images"):
        data_type = "train"
        train_data = data_cha[int(task_i)][data_type]
        for example_i in range(len(train_data)):
            for group_type in ["input", "output"]:
                group_positions = g_train[int(task_i)][example_i][group_type]
                group_matrix = train_data[example_i][group_type]
                for ig_i in range(len(group_positions)):
                    patch = src.utils.data_utils.group2patch(group_matrix, group_positions[ig_i])
                    img = visual_utils.patch2img(patch)
                    img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"")
                    img_file = output_path / f"{task_i}_{example_i:01d}_{group_type}_{ig_i:01d}_{data_type}.png"
                    visual_utils.save_image(img, str(img_file))
                    line_group_data[task_i][data_type][example_i][group_type].append(patch.tolist())
        # test image
        data_type = "test"
        for example_i in range(len(data_cha[int(task_i)][data_type])):
            # input data
            for group_type in ["input", "output"]:
                if group_type == "input":
                    test_matrix = data_cha[int(task_i)][data_type][example_i][group_type]
                    test_positions = g_test[int(task_i)][example_i][group_type]
                elif group_type == "output":
                    test_matrix = data_sol[int(task_i)][example_i]
                    test_positions = g_test[int(task_i)][example_i][group_type]
                else:
                    raise ValueError
                for g_i in range(len(test_positions)):
                    patch = src.utils.data_utils.group2patch(test_matrix, test_positions[g_i])
                    img = visual_utils.patch2img(patch)
                    img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"")
                    img_file = output_path / f"{task_i}_{example_i:01d}_{group_type}_{g_i:01d}_{data_type}.png"
                    visual_utils.save_image(img, str(img_file))
                    line_group_data[task_i][data_type][example_i][group_type].append(patch.tolist())
    file_utils.save_json(line_group_data, output_path / "data.json")
    return line_group_data


def export_obj_groups(raw_data, g_data, group_type, obj_indices):
    output_path = config.output / f'{group_type}_groups'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_cha, data_sol = raw_data["cha"], raw_data["sol"]
    g_train, g_test = g_data
    obj_group_data = {}
    # init data structure
    for task_i in obj_indices:
        obj_group_data[task_i] = {"train": [], "test": []}
        for data_type in ["train", "test"]:
            for example_i in range(len(data_cha[int(task_i)][data_type])):
                obj_group_data[task_i][data_type].append({"input": [], "output": []})
                for group_type in ["input", "output"]:
                    for ig_i in range(len(g_train[int(task_i)][example_i][group_type])):
                        obj_group_data[task_i][data_type][example_i][group_type] = []

    for task_i in tqdm(obj_indices, desc="Exporting Line patch images"):
        data_type = "train"
        train_data = data_cha[int(task_i)][data_type]
        for example_i in range(len(train_data)):
            for group_type in ["input", "output"]:
                group_positions = g_train[int(task_i)][example_i][group_type]
                group_matrix = train_data[example_i][group_type]
                for ig_i in range(len(group_positions)):
                    patch = src.utils.data_utils.group2patch(group_matrix, group_positions[ig_i])
                    img = visual_utils.patch2img(patch)
                    img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"")
                    img_file = output_path / f"{task_i}_{example_i:01d}_{group_type}_{ig_i:01d}_{data_type}.png"
                    visual_utils.save_image(img, str(img_file))
                    obj_group_data[task_i][data_type][example_i][group_type].append(patch.tolist())
        # test image
        data_type = "test"
        for example_i in range(len(data_cha[int(task_i)][data_type])):
            # input data
            for group_type in ["input", "output"]:
                if group_type == "input":
                    test_matrix = data_cha[int(task_i)][data_type][example_i][group_type]
                    test_positions = g_test[int(task_i)][example_i][group_type]
                elif group_type == "output":
                    test_matrix = data_sol[int(task_i)][example_i]
                    test_positions = g_test[int(task_i)][example_i][group_type]
                else:
                    raise ValueError
                for g_i in range(len(test_positions)):
                    patch = src.utils.data_utils.group2patch(test_matrix, test_positions[g_i])
                    img = visual_utils.patch2img(patch)
                    img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"")
                    img_file = output_path / f"{task_i}_{example_i:01d}_{group_type}_{g_i:01d}_{data_type}.png"
                    visual_utils.save_image(img, str(img_file))
                    obj_group_data[task_i][data_type][example_i][group_type].append(patch.tolist())
    file_utils.save_json(obj_group_data, output_path / "data.json")
    return obj_group_data


def export_groups_as_images(raw_data, g_data, group_type):
    output_path = config.output / f'{group_type}_groups'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_cha, data_sol = raw_data["cha"], raw_data["sol"]
    g_train, g_test = g_data

    for task_i in tqdm(range(len(data_cha)), desc="Exporting group images"):
        train_data = data_cha[task_i]["train"]
        task_img = []
        for example_i in range(len(train_data)):
            input_data_train = train_data[example_i]["input"]
            img = visual_utils.patch2img(input_data_train)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"input-{example_i}")
            input_g_img = [img]
            input_groups = g_train[task_i][example_i][0]
            for ig_i in range(len(input_groups)):
                patch = src.utils.data_utils.group2patch(input_data_train, input_groups[ig_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"in-g-{ig_i}")
                input_g_img.append(img)

            output_data_train = train_data[example_i]["output"]
            img = visual_utils.patch2img(output_data_train)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"output-{example_i}")
            output_g_img = [img]
            output_groups = g_train[task_i][example_i][1]
            for og_i in range(len(output_groups)):
                patch = src.utils.data_utils.group2patch(output_data_train, output_groups[og_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"out-g-{og_i}")
                output_g_img.append(img)

            input_g_img, output_g_img = visual_utils.align_white_imgs(input_g_img, output_g_img)
            input_g_img = visual_utils.hconcat_resize(input_g_img)
            output_g_img = visual_utils.hconcat_resize(output_g_img)

            train_img = visual_utils.vconcat_resize([input_g_img, output_g_img])
            task_img.append(train_img)
        # test image
        for example_i in range(len(data_cha[task_i]["test"])):
            test_input_data = data_cha[task_i]["test"][example_i]["input"]
            img = visual_utils.patch2img(test_input_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"test-i")
            test_input_img = [img]
            test_input_groups = g_test[task_i][example_i][0]
            for g_i in range(len(test_input_groups)):
                patch = src.utils.data_utils.group2patch(test_input_data, test_input_groups[g_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"test-i-g-{g_i}")
                test_input_img.append(img)

            test_output_data = data_sol[task_i][example_i]
            img = visual_utils.patch2img(test_output_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text="test-o")
            test_output_img = [img]
            test_output_groups = g_test[task_i][example_i][1]
            for g_i in range(len(test_output_groups)):
                patch = src.utils.data_utils.group2patch(test_output_data, test_output_groups[g_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"test-o-g-{g_i}")
                test_output_img.append(img)

            test_input_img, test_output_img = visual_utils.align_white_imgs(test_input_img, test_output_img)
            test_input_img = visual_utils.hconcat_resize(test_input_img)
            test_output_img = visual_utils.hconcat_resize(test_output_img)
            test_img = visual_utils.vconcat_resize([test_input_img, test_output_img])

            task_img.append(test_img)
        task_img = visual_utils.vconcat_resize(task_img)
        img_file = output_path / f"{task_i:03d}_{group_type}.png"
        visual_utils.save_image(task_img, str(img_file))


# export_groups_as_images(file_utils.get_raw_data(), "train_cha")
# export_data_as_images(file_utils.get_raw_data(), "train_cha")
# export_data_as_images(file_utils.get_raw_data(), "eval_cha")
# export_data_as_images(file_utils.get_raw_data(), "test_cha")
# export_data_as_images(file_utils.get_raw_data(), "train_sol")
# export_data_as_images(file_utils.get_raw_data(), "eval_sol")
def export_belong_relation_as_images(data, groups, group_type, belong_group_pairs):
    output_path = config.output / 'relations'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for task_i in tqdm(range(len(data)), desc="Exporting relation images"):
        train_data = data[task_i]["train"]
        for example_i in range(len(train_data)):

            input_data = train_data[example_i]["input"]
            img = visual_utils.patch2img(input_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"input")
            input_g_img = [img]

            input_groups = groups[task_i][example_i][0]
            for ig_i in range(len(input_groups)):
                patch = src.utils.data_utils.group2patch(input_data, input_groups[ig_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"in_group_{ig_i}")
                input_g_img.append(img)

            output_data = train_data[example_i]["output"]
            img = visual_utils.patch2img(output_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text="output")
            output_g_img = [img]
            output_groups = groups[task_i][example_i][1]
            for og_i in range(len(output_groups)):
                patch = src.utils.data_utils.group2patch(output_data, output_groups[og_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"out_group_{og_i}")
                output_g_img.append(img)

            input_g_img, output_g_img = visual_utils.align_white_imgs(input_g_img, output_g_img)
            input_g_img = visual_utils.hconcat_resize(input_g_img)
            output_g_img = visual_utils.hconcat_resize(output_g_img)

            g_img = visual_utils.vconcat_resize([input_g_img, output_g_img])

            # save the images

            img_file = output_path / f"{task_i:03d}_{example_i}_{group_type}.png"
            visual_utils.save_image(g_img, str(img_file))


if __name__ == "__main__":
    raw_data = file_utils.get_raw_data()
    data_type = "train"
    color_groups_eval_cha = grouping.group_by_color(raw_data[data_type])

    # line_indices = [5, 9, 12, 20, 23, 24, 25, 27, 39, 44, 46, 49, 53, 54, 59, 63, 64, 65, 71, 74, 75, 83, 90, 91, 93,
    #                 108, 121, 126, 130, 137, 140, 150, 160, 164, 167, 174, 186, 188, 189, 196, 198, 199, 201, 211, 212,
    #                 213, 218, 235, 256, 259, 251, 253, 292, 296, 302]
    # line_indices = [f"{i:03d}" for i in line_indices]
    # export_line_groups(raw_data[data_type], color_groups_eval_cha, f"{data_type}_cha", line_indices)

    rect_indices = [28, 93, 158, 170, 181, 202, 207, 219, 223, 227, 239, 280, 289, 293, 297, 301, 309, 335, 337, 351,
                    366, 395]
    rect_indices = [f"{i:03d}" for i in rect_indices]
    export_obj_groups(raw_data[data_type], color_groups_eval_cha, f"{data_type}_cha_rect", rect_indices)
