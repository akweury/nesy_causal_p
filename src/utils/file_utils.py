# Created by jing at 16.06.24

import json
import os
import config


def get_raw_data():
    f_train_cha = config.data_file_train_cha
    f_train_sol = config.data_file_train_sol
    f_eval_cha = config.data_file_eval_cha
    f_eval_sol = config.data_file_eval_sol
    f_test_cha = config.data_file_test_cha

    with open(f_train_cha, 'r') as f:
        train_data_cha = list(json.load(f).values())
    with open(f_train_sol, 'r') as f:
        train_data_sol = list(json.load(f).values())
    with open(f_eval_cha, 'r') as f:
        eval_data_cha = list(json.load(f).values())
    with open(f_eval_sol, 'r') as f:
        eval_data_sol = list(json.load(f).values())
    with open(f_test_cha, 'r') as f:
        test_data_cha = list(json.load(f).values())
    data = {
        "train": {"cha": train_data_cha, "sol": train_data_sol},
        "eval": {"cha": eval_data_cha, "sol": eval_data_sol},
        "test": {"cha": test_data_cha}
    }
    return data


def save_json(json_data, path):
    with open(path, 'w') as f:
        json.dump(json_data, f)


def get_all_files(path, file_extension, name_only=False):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(file_extension):
                if name_only:
                    all_files.append(file)
                else:
                    all_files.append(os.path.join(root, file))
    return all_files


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
