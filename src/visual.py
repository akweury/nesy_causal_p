# Created by jing at 16.06.24
import os

from utils import visual_utils, file_utils
import config


def export_data_as_images(raw_data, data_type):
    # export data as images
    output_folder = config.output / data_type
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    visual_utils.export_task_img(raw_data[data_type], output_folder)


# export_data_as_images(file_utils.get_raw_data(), "train_cha")
export_data_as_images(file_utils.get_raw_data(), "eval_cha")
export_data_as_images(file_utils.get_raw_data(), "test_cha")
# export_data_as_images(file_utils.get_raw_data(), "train_sol")
# export_data_as_images(file_utils.get_raw_data(), "eval_sol")
