# Created by jing at 16.06.24

import json


def get_raw_data(f_train_cha):
    with open(f_train_cha, 'r') as f:
        raw_data = json.load(f)
    return raw_data
