# Created by jing at 11.01.25


import os
import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

import config
from kandinsky_generator import generate_training_patterns
from utils import args_utils, chart_utils
from utils.chart_utils import van
from src import dataset
from percept import perception
from alpha import alpha
import eval_nsfr



def gestalt_perception_main():
    args = args_utils.get_args()
    os.makedirs(config.model_gestalt, exist_ok=True)
    # generate dataset
    generate_training_patterns.genGestaltTraining()

if __name__ == '__main__':
    gestalt_perception_main()