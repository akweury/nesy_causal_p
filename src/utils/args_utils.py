# Created by jing at 19.06.24

import argparse
import json
import os
import random
import numpy as np
import torch

import config
from . import log_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--th_group", type=float, default=0.01)
    parser.add_argument("--th_inv_nc", type=float, default=0.01)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--top_data", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--kernel", type=int, default=10)
    parser.add_argument("--is_visual", action="store_true")
    parser.add_argument("--is_done", action="store_true")
    parser.add_argument("--show_process", action="store_true")
    parser.add_argument("--number_num", type=int, default=10)
    parser.add_argument("--phi_num", type=int, default=2)
    parser.add_argument("--rho_num", type=int, default=2)
    parser.add_argument("--variable_symbol", type=str, default="O")
    parser.add_argument("--im_step", type=int, default=5)
    parser.add_argument("--cim_step", type=int, default=5)
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--bs_clause_eval", type=int, default=5)
    parser.add_argument("--top_fm_k", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_bs_step", type=int, default=3)
    parser.add_argument("--max_obj_num", type=int, default=5)


    args = parser.parse_args()

    if args.device != "cpu":
        args.device = int(args.device)
    args.log_file = log_utils.create_log_file(config.output / "logs")
    args.lark_path = str(config.lark_file)

    os.makedirs(config.output / f"{args.exp_name}", exist_ok=True)
    args.batch_size = 1
    return args
