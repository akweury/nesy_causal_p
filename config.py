# Created by jing at 16.06.24

import os
import cv2 as cv
from pathlib import Path

root = Path(__file__).parents[0]
output = root / 'output'
lark_file = root / "src" / "alpha" / "exp.lark"

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

tile_pad_width = 1
index_font_scale = 80
index_x_pos = 2.3
index_y_pos = 1.7
index_font = cv.FONT_HERSHEY_SIMPLEX
index_font_color = (0, 0, 255)
index_font_thickness = 2

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

data_file_train_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_training_challenges.json"
data_file_train_sol = root / "dataset" / "arc-prize-2024" / "arc-agi_training_solutions.json"
data_file_test_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_test_challenges.json"
data_file_eval_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_challenges.json"
data_file_eval_sol = root / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_solutions.json"

alpha_mode = {
    'inter_input_group': 0,
    'inter_output_group': 1,
    'inter_io_group': 2,
    'ig': 3,
    'og': 4
}
