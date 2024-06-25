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


def export_groups_as_images(data, train_gs, group_type):
    output_path = config.output / f'{group_type}_groups'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for task_i in tqdm(range(len(data)), desc="Exporting group images"):
        train_data = data[task_i]["train"]
        task_img = []
        for example_i in range(len(train_data)):
            input_data = train_data[example_i]["input"]
            img = visual_utils.patch2img(input_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"input")
            input_g_img = [img]
            input_groups = train_gs[task_i][example_i][0]
            for ig_i in range(len(input_groups)):
                patch = visual_utils.group2patch(input_data, input_groups[ig_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"in_group_{ig_i}")
                input_g_img.append(img)

            output_data = train_data[example_i]["output"]
            img = visual_utils.patch2img(output_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text="output")
            output_g_img = [img]
            output_groups = train_gs[task_i][example_i][1]
            for og_i in range(len(output_groups)):
                patch = visual_utils.group2patch(output_data, output_groups[og_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"out_group_{og_i}")
                output_g_img.append(img)

            input_g_img, output_g_img = visual_utils.align_white_imgs(input_g_img, output_g_img)
            input_g_img = visual_utils.hconcat_resize(input_g_img)
            output_g_img = visual_utils.hconcat_resize(output_g_img)

            g_img = visual_utils.vconcat_resize([input_g_img, output_g_img])
            task_img.append(g_img)
            # save the images

        # test image
        test_data = data[task_i]["test"]
        if len(test_data) != 1:
            raise ValueError
        test_data = test_data[0]["input"]
        img = visual_utils.patch2img(test_data)
        img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"test input")
        test_g_img = [img]
        test_groups = test_gs[task_i][0]
        for g_i in range(len(test_groups)):
            patch = visual_utils.group2patch(test_data, test_groups[g_i])
            img = visual_utils.patch2img(patch)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"in_group_{g_i}")
            test_g_img.append(img)

        task_img.append(test_g_img)
        task_img = visual_utils.vconcat_resize(task_img)
        img_file = output_path / f"{task_i:03d}_{group_type}.png"
        visual_utils.save_image(task_img, str(img_file))


# export_groups_as_images()
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
                patch = visual_utils.group2patch(input_data, input_groups[ig_i])
                img = visual_utils.patch2img(patch)
                img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text=f"in_group_{ig_i}")
                input_g_img.append(img)

            output_data = train_data[example_i]["output"]
            img = visual_utils.patch2img(output_data)
            img = visual_utils.img_processing(img, lbw=50, rbw=200, tbw=50, bbw=200, text="output")
            output_g_img = [img]
            output_groups = groups[task_i][example_i][1]
            for og_i in range(len(output_groups)):
                patch = visual_utils.group2patch(output_data, output_groups[og_i])
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
