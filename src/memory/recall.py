# Created by jing at 10.12.24
import torch
import time
from src.memory.recall_utils import *
from src.utils import data_utils

def recall_fms(args, bk_shape, bw_img):

    # shift fm and images
    shifted_imgs = get_shifted_matrics(bw_img)

    # convolutional layer
    start = time.time()
    shifted_fms = one_layer_conv(shifted_imgs, bk_shape["kernels"].float())
    end = time.time()
    print("Conv took {} seconds".format(end - start))

    # edge similarity
    img_edge = detect_edge(shifted_imgs.float())
    repo_edge = detect_edge(bk_shape["fm_img"].float())
    start = time.time()
    sim_edge = data_utils.matrix_equality(repo_edge, img_edge)
    end = time.time()
    print("Edge Sim took {} seconds".format(end - start))

    # fm similarity
    start = time.time()
    sim_fm = data_utils.matrix_equality(bk_shape["fm_repo"], shifted_fms)
    end = time.time()
    print("FM Sim took {} seconds".format(end - start))


    # total similarity
    sim_total = (sim_edge + sim_fm * sim_edge.max()).permute(1, 0)

    # find the best shift
    best_shift = find_best_shift(sim_total, shifted_fms.shape[-2:])

    # log
    args.logger.debug(
        f"recall {len(best_shift[-1])} possible fms, "
        f"highest conf: {best_shift[-1][0]:.2f}")

    return best_shift
