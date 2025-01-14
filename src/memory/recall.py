# Created by jing at 10.12.24
import numpy as np
from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models
from src.reasoning import reason
from src.utils.chart_utils import van


def recall_fms(args, bk_shapes, bw_img, reshape=None):
    scores = []
    recall_fm = []
    for bk_shape in bk_shapes:
        # convolutional layer
        input_fms = models.one_layer_conv(bw_img, bk_shape["kernels"].float())
        bk_fms = bk_shape["fm_repo"]
        # fm similarity
        sim_total = data_utils.matrix_equality(bk_fms, input_fms.unsqueeze(0))
        scores.append(sim_total.max())
        recall_fm.append(bk_fms[sim_total.argmax()])
    best_shape = np.argmax(scores)
    best_recall_fm = recall_fm[best_shape]
    return best_recall_fm, best_shape


def recall_match(args, bk_shapes, bw_img):
    onside_shapes = []
    onside_percents = torch.zeros(len(bk_shapes))
    for b_i, bk_shape in enumerate(bk_shapes):
        shifted_fms, rc_data = recall_fms(args, bk_shape, bw_img, reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(rc_data, bk_shape, bw_img, reshape=args.obj_fm_size)
        onside_shapes.append(group_data["recalled_bw_img"])
        onside_percents[b_i] = group_data["onside_percent"]
    best_shape = onside_percents.argmax()
    onside_pixels = onside_shapes[best_shape][0].squeeze()
    best_shape = best_shape + 1
    return onside_pixels, best_shape
