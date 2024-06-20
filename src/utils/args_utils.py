# Created by jing at 19.06.24

import argparse
import json
import os
import random
import numpy as np
import torch

import config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument("--is_visual", action="store_true")
    args = parser.parse_args()

    if args.device != "cpu":
        args.device = int(args.device)

    return args
