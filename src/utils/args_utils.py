# Created by jing at 19.06.24

import argparse
import os
import colorlog

import config
from . import log_utils
from src import bk


def init_logger():
    # Create a color handler
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    # Add the color handler to the logger
    logger = colorlog.getLogger("colorLogger")
    logger.addHandler(handler)
    # Prevent logs from propagating to the root logger
    logger.propagate = False
    logger.setLevel(colorlog.DEBUG)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="cpu or cuda", default="cpu", type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--th_group", type=float, default=0.01)
    parser.add_argument("--th_inv_nc", type=float, default=0.01)
    parser.add_argument("--obj_n", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--top_data", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--kernel", type=int, default=10)
    parser.add_argument("--is_visual", action="store_true")
    parser.add_argument("--is_done", action="store_true")
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--solid_pattern", action="store_true")
    parser.add_argument("--rewrite_data", action="store_true")
    parser.add_argument("--show_process", action="store_true")
    parser.add_argument("--number_num", type=int, default=10)
    parser.add_argument("--phi_num", type=int, default=2)
    parser.add_argument("--rho_num", type=int, default=2)
    parser.add_argument("--group_num", type=int, default=3)
    parser.add_argument("--variable_symbol", type=str, default="O")
    parser.add_argument("--im_step", type=int, default=5)
    parser.add_argument("--cim_step", type=int, default=5)
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--bs_clause_eval", type=int, default=5)
    parser.add_argument("--top_fm_k", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_bs_step", type=int, default=12)
    parser.add_argument("--max_obj_num", type=int, default=5)
    parser.add_argument("--group_count_conf_th", type=float, default=0.6)
    parser.add_argument("--fm_th", type=float, default=1.5)
    parser.add_argument("--valid_rule_th", type=float, default=0.8)

    args = parser.parse_args()
    args.logger = init_logger()
    if args.device != "cpu":
        args.device = int(args.device)
    args.log_file = log_utils.create_log_file(args.logger, config.output / "logs")
    args.lark_path = str(config.lark_file)

    os.makedirs(config.output / f"{args.exp_name}", exist_ok=True)
    args.batch_size = 1


    # args.exp_setting = eval(f"bk.{args.exp_name}")
    return args
