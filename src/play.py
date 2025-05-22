# Created by X at 22.10.24
import os
import config
import psutil
import pynvml
import time
import threading
import wandb

from src.utils import args_utils
from src import dataset
from kandinsky_generator import generate_training_patterns
from mbg.object import eval_patch_classifier
from mbg.training import training
from src.dataset import GrbDataset
from mbg.evaluation import evaluation

def init_io_folders(args, data_folder):
    args.train_folder = data_folder / "train" / "task_true_pattern"
    os.makedirs(args.train_folder, exist_ok=True)
    args.test_true_folder = data_folder / "test" / "task_true_pattern"
    os.makedirs(args.test_true_folder, exist_ok=True)
    args.test_random_folder = data_folder / "test" / "task_random_pattern"
    os.makedirs(args.test_random_folder, exist_ok=True)
    args.test_cf_folder = data_folder / "test" / "task_cf_pattern"
    os.makedirs(args.test_cf_folder, exist_ok=True)

    exp_name = args.exp_setting["task_name"]
    args.out_train_folder = config.output / exp_name / "train" / "task_true_pattern"
    os.makedirs(args.out_train_folder, exist_ok=True)
    args.out_positive_folder = config.output / exp_name / "test" / "task_true_pattern"
    os.makedirs(args.out_positive_folder, exist_ok=True)
    args.out_random_folder = config.output / exp_name / "test" / "task_random_pattern"
    os.makedirs(args.out_random_folder, exist_ok=True)
    args.out_cf_folder = config.output / exp_name / "test" / "task_cf_pattern"
    os.makedirs(args.out_cf_folder, exist_ok=True)


def main():
    # load exp arguments
    args = args_utils.get_args()
    combined_loader = dataset.load_combined_dataset()
    obj_model = eval_patch_classifier.load_model(args.device)

    # initialize wandb
    wandb.init(project="grb-pipeline", config=args.__dict__, name=args.exp_name)

    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        task_name = train_data["task"]

        # 1) hyperparam search
        # (best_prox, best_sim, best_topk), train_metrics = training.grid_search(args, train_data, val_data, obj_model)

        # log best hyperparams
        # wandb.log({
        #     "train_acc": train_metrics.get("acc",0),
        # })

        # 2) merge train+val into a single loader
        train_val_data = {
            "task": task_name,
            "positive": train_data["positive"] + val_data["positive"],
            "negative": train_data["negative"] + val_data["negative"]
        }
        # 3) re-learn your final rules on the combined set
        hyp_params = {
            "prox": 0.9,
            "sim": 0.5,
            "top_k": 5,
            "conf_th": 0.5
        }
        final_rules = training.train_rules(train_val_data, obj_model, hyp_params)

        # 4) evaluate on the held-out test set
        test_metrics = evaluation.eval_rules(test_data, obj_model, final_rules, hyp_params)

        # log test results
        wandb.log({
            "test_accuracy": test_metrics.get("acc", 0),
            "test_auc": test_metrics.get("auc", 0),
            "test_f1": test_metrics.get("f1", 0),
        })
        print(f"[{task_name}] Test results:", test_metrics)
    wandb.finish()
    return


def monitor_cpu_memory(interval=0.1):
    process = psutil.Process()
    peak_cpu = 0
    running = True

    def monitor():
        nonlocal peak_cpu
        while running:
            mem = process.memory_info().rss / 1024 ** 2  # Convert to MB
            peak_cpu = max(peak_cpu, mem)
            time.sleep(interval)

    thread = threading.Thread(target=monitor)
    thread.start()

    def stop():
        nonlocal running
        running = False
        thread.join()
        return peak_cpu

    return stop


def monitor_memory(interval=0.1):
    process = psutil.Process()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    peak_cpu, peak_gpu = 0, 0
    running = True

    def monitor():
        nonlocal peak_cpu, peak_gpu
        while running:
            cpu_mem = process.memory_info().rss / 1024 ** 2  # MB
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 ** 2  # MB
            peak_cpu = max(peak_cpu, cpu_mem)
            peak_gpu = max(peak_gpu, gpu_mem)
            time.sleep(interval)

    thread = threading.Thread(target=monitor)
    thread.start()
    return thread, lambda: setattr(thread, 'running', False), lambda: (peak_cpu, peak_gpu), lambda: setattr(globals(),
                                                                                                            'running',
                                                                                                            False)


if __name__ == '__main__':
    main()
