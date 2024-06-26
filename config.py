# Created by jing at 16.06.24

import os
from pathlib import Path

root = Path(__file__).parents[0]
output = root / 'output'
lark_file = root/ "src" / "alpha" / "exp.lark"

if not os.path.exists(output):
    os.mkdir(output)

code_group_relation = {
    "a_eq_b": 0,
    "a_inc_b": 1,
    "b_inc_a": 2,
    "else": 3
}

tile_pad_width = 1
color_tiles = [
    (0, 0, 0),  # 0
    (134, 33, 51),  # 1
    (229, 78, 62),  # 2
    (239, 139, 59),  # 3
    (249, 221, 74),  # 4
    (115, 201, 76),  # 5
    (71, 145, 247),  # 6
    (154, 214, 238),  # 7
    (153, 153, 153),  # 8
    (211, 73, 160),  # 9
    (192, 192, 192),  # 10
]

data_file_train_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_training_challenges.json"
data_file_train_sol = root / "dataset" / "arc-prize-2024" / "arc-agi_training_solutions.json"
data_file_test_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_test_challenges.json"
data_file_eval_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_challenges.json"
data_file_eval_sol = root / "dataset" / "arc-prize-2024" / "arc-agi_evaluation_solutions.json"
