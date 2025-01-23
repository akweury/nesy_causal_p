# Created by jing at 10.12.24
import numpy as np
from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models
from src.reasoning import reason
from src.utils.chart_utils import van


def recall_fms(input_fms, bk_shapes, reshape=None):
    # scores = []
    # recall_fm = []
    # convolutional layer
    # input_fms = models.img2fm(img, bk_shapes["kernels"].float())
    bk_fms = bk_shapes["fm_repo"]
    # fm similarity
    sim_total = data_utils.matrix_equality(bk_fms, input_fms.unsqueeze(0))
    scores = sim_total.max()
    recall_fm = bk_fms[sim_total.argmax()]
    # best_shape = np.argmax(scores)
    # best_recall_fm = recall_fm[best_shape]
    return recall_fm


def recall_match(args, bk_shapes, img):
    onside_shapes = []
    onside_percents = torch.zeros(len(bk_shapes))
    cropped_data_all = []
    kernels_all = []
    for b_i, bk_shape in enumerate(bk_shapes):
        input_fms, cropped_data = models.img2fm(img, bk_shape["kernels"].float())
        mem_fms = recall_fms(input_fms, bk_shape, reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(input_fms, mem_fms, reshape=args.obj_fm_size)
        onside_shapes.append(group_data["onside"])
        onside_percents[b_i] = group_data["onside_percent"]
        cropped_data_all.append(cropped_data)
        kernels_all.append(bk_shape["kernels"])
    best_shape = onside_percents.argmax()
    best_cropped_data = cropped_data_all[onside_percents.argmax()]
    onside_pixels = onside_shapes[best_shape].squeeze()
    best_shape = best_shape + 1
    best_kernel = kernels_all[onside_percents.argmax()]
    return onside_pixels, best_shape, best_cropped_data, best_kernel
