# Created by jing at 16.06.24
import os
import numpy as np
from tqdm import tqdm
from utils import visual_utils, file_utils, data_utils
import config


def export_data_as_images(raw_data, data_type):
    # export data as images
    output_folder = config.output / data_type
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visual_utils.export_task_img(raw_data[data_type], output_folder)

def export_groups_as_images(data, groups):
    output_path = config.output / 'groups'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for task_i in tqdm(range(len(data)), desc="Exporting group images" ):
        train_data = data[task_i]["train"]
        test_data = data[task_i]["test"]
        for example_i in range(len(train_data)):
            input_data = train_data[example_i]["input"]
            img = visual_utils.patch2img(input_data)
            input_g_img = [img]

            input_groups = groups[task_i][example_i][0]
            for input_group in input_groups:
                patch = visual_utils.group2patch(input_data, input_group)
                img = visual_utils.patch2img(patch)
                group_img = visual_utils.padding_img(img, lbw=50, rbw=200, tbw=50, bbw=200)
                visual_utils.addText(group_img, f"group", color=(255, 0, 0), pos=(100, 700))
                input_g_img.append(group_img)
            input_g_img = visual_utils.hconcat_resize(input_g_img)

            output_data = train_data[example_i]["output"]
            img = visual_utils.patch2img(output_data)
            output_g_img = [img]
            output_groups = groups[task_i][example_i][1]
            for output_group in output_groups:
                patch = visual_utils.group2patch(output_data, output_group)
                img = visual_utils.patch2img(patch)
                group_img = visual_utils.padding_img(img, lbw=50, rbw=200, tbw=50, bbw=200)
                visual_utils.addText(group_img, f"group", color=(255, 0, 0), pos=(100, 700))
                output_g_img.append(group_img)
            output_g_img = visual_utils.hconcat_resize(output_g_img)

            # save the images
            input_img_file = output_path / f'task_{task_i:03d}_ex_{example_i}_input.png'
            output_img_file = output_path /  f'task_{task_i:03d}_ex_{example_i}_output.png'
            visual_utils.save_image(input_g_img, str(input_img_file))
            visual_utils.save_image(output_g_img, str(output_img_file))





# export_data_as_images(file_utils.get_raw_data(), "train_cha")
# export_data_as_images(file_utils.get_raw_data(), "eval_cha")
# export_data_as_images(file_utils.get_raw_data(), "test_cha")
# export_data_as_images(file_utils.get_raw_data(), "train_sol")
# export_data_as_images(file_utils.get_raw_data(), "eval_sol")
