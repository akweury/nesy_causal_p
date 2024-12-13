# Created by jing at 10.12.24
import numpy as np
import torch
import time
from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models


def recall_fms(args, bk_shape, bw_img):
    # shift fm and images
    shifted_imgs = get_shifted_matrics(bw_img)

    # convolutional layer
    shifted_fms = models.one_layer_conv(shifted_imgs, bk_shape["kernels"].float())


    # visual input fms
    fm_img = shifted_fms.sum(
        dim=1, keepdim=True).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    chart_utils.visual_batch_imgs(fm_img, args.save_path, "input_fms.png")


    # edge similarity
    img_edge = detect_edge(shifted_imgs.float())
    repo_edge = detect_edge(bk_shape["fm_img"].float())
    sim_edge = data_utils.matrix_equality(repo_edge, img_edge)

    # fm similarity
    sim_fm = data_utils.matrix_equality(bk_shape["fm_repo"], shifted_fms)

    # total similarity
    sim_total = (sim_edge * 0.5 + sim_fm * sim_edge.max() * 0.5).permute(1, 0)

    # find the best shift
    best_shift = find_best_shift(sim_total, shifted_fms.shape[-2:])



    # log
    args.logger.debug(f"recall {len(best_shift[-1])} possible fms, "
                      f"highest conf: {best_shift[-1][0]:.2f}")

    return shifted_fms, best_shift
