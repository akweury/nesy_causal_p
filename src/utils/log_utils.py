# Created by shaji at 24/06/2024
import os
import datetime
import wandb

def create_log_file(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = str(log_path / f"log_{time_now}.txt")
    with open(file_name, "w") as f:
        f.write(f"Log ({time_now})")
    print(f"- log_file_path:{file_name}")
    return str(file_name)


def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(str(line_str) + "\n")


def init_wandb(pj_name, archi):
    wandb.init(
        # set the wandb project where this run will be logged
        project=pj_name,
        # track hyperparameters and run metadata
        config={
            "architecture": archi,
        }
    )