# Created by X at 16.06.24

import os
import cv2 as cv
from pathlib import Path
import shutil


root = Path(__file__).parents[0]


def clear_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)  # Removes a folder even if it's not empty
        print(f"Folder '{folder_path}' has been removed.")
    os.makedirs(folder_path, exist_ok=True)


storage = root / 'storage'
output = storage / 'output'
lark_file = root / "src" / "alpha" / "exp.lark"
models = storage / "models"
# clear_folder(models)
model_visual = models / "visual"
if not os.path.exists(storage):
    os.mkdir(storage)
if not os.path.exists(models):
    os.mkdir(models)
if not os.path.exists(model_visual):
    os.mkdir(model_visual)
if not os.path.exists(storage):
    os.mkdir(storage)
if not os.path.exists(output):
    os.mkdir(output)


code_group_relation = {
    "a_eq_b": 0,
    "a_inc_b": 1,
    "b_inc_a": 2,
    "else": 3
}

################# labels ########################
obj_true = [1, 0]
obj_false = [0, 1]
# --------------------------------------------------
pixel_size = 64
tile_pad_width = 1
index_font_scale = 80
index_x_pos = 2.3
index_y_pos = 1.7
index_font = cv.FONT_HERSHEY_SIMPLEX
index_font_color = (0, 0, 255)
index_font_thickness = 1

color_tiles = [
    (255, 67, 199),  # 0
    (134, 33, 51),  # 1
    (229, 78, 62),  # 2
    (239, 139, 59),  # 3
    (249, 221, 74),  # 4
    (115, 201, 76),  # 5
    (71, 145, 247),  # 6
    (154, 214, 238),  # 7
    (153, 0, 153),  # 8
    (211, 73, 160),  # 9
    (255, 255, 255),  # 10
]
kp_base_dataset = storage / "dataset" / "basic"
kp_challenge_dataset = storage / "dataset" / "challenge"
kp_gestalt_dataset = storage / "dataset" / "gestalt"
grb_base = storage / "dataset" / "grb"
grb_prox = grb_base / "proximity"


# clear_folder(kp_gestalt_dataset)
kp_gestalt_dataset_all = storage / "dataset" / "gestalt_all"
data_file_train_cha = storage / "dataset" / "arc-prize-2024" / "arc-agi_training_challenges.json"
data_file_train_sol = storage / "dataset" / "arc-prize-2024" / "arc-agi_training_solutions.json"
data_file_test_cha = storage / "dataset" / "arc-prize-2024" / "arc-agi_test_challenges.json"
data_file_eval_cha = storage / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_challenges.json"
data_file_eval_sol = storage / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_solutions.json"

alpha_mode = {
    'inter_input_group': 0,
    'inter_output_group': 1,
    'inter_io_group': 2,
    'ig': 3,
    'og': 4
}

#### output dirs
mb_outlines = storage / "dataset" / "mb_outlines"
os.makedirs(mb_outlines, exist_ok=True)

##############
model_gestalt = models / "gestalt_rl"

model_group_kp_line = storage / "output" / "kp_sy_line" / "line_detector_model.pth"
model_group_kp_square = storage / "output" / "kp_sy_square" / "square_detector_model.pth"
model_group_kp_circle = storage / "output" / "kp_sy_circle" / "circle_detector_model.pth"
model_group_kp_triangle = storage / "output" / "kp_sy_triangle" / "triangle_detector_model.pth"

model_group_kp_triangle_only = storage / "output" / "kp_sy_triangle_only" / "triangle_only_detector_model.pth"

# nn settings
kernel_size = 3

gestalt_action = ["proximity", "color_similarity", "shape_similarity", "closure"]

group_index = 10
