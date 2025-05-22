# Created by MacBook Pro at 18.04.25
import os
import config

# ==== 数据路径 ====
ROOT_DATASET_DIR = config.kp_gestalt_dataset / "train"
CONTOUR_DATASET_DIR = config.mb_outlines
GT_EXTENSION = "gt.json"

# ==== 输出路径 ====
MODEL_SAVE_PATH = config.mb_outlines / "patch_set_classifier.pt"
OBJ_MODEL_SAVE_PATH = config.mb_outlines / "patch_set_obj_classifier.pt"
EVAL_SAVE_DIR = config.mb_outlines / "eval_save"

# ==== 模型结构 ====
NUM_CLASSES = 3  # circle, triangle, rectangle, ellipse
PATCHES_PER_SET = 6
POINTS_PER_PATCH = 16

# ==== 数据参数 ====
NUM_CONTOUR_POINTS = 100  # total sampled per object
RANDOM_SEED = 42

# ==== 训练参数 ====
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 200


# ==== 可视化 ====
LABEL_NAMES = {0: "triangle", 1: "rectangle", 2: "ellipse"}
LABEL_COLORS = {
    0: (0, 255, 0),  # green
    1: (255, 0, 0),  # red
    2: (0, 0, 255),  # blue
    3: (255, 165, 0)  # orange
}

# ==== patch 生成 ====
PATCH_STRATEGY = "random"  # 可扩展为：'grid', 'semantic', etc.
