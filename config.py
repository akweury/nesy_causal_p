# Created by jing at 16.06.24

import os
from pathlib import Path

root = Path(__file__).parents[0]
output = root / 'output'
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
    (211, 73, 160)  # 9
]

data_file_train_cha = root / "dataset" / "arc-prize-2024" / "arc-agi_training_challenges.json"
data_file_train_sol = root / "dataset" / "arc-prize-2024" / "arc-agi_training_solutions.json"
