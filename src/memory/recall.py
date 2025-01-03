# Created by jing at 10.12.24
import numpy as np
from src.memory.recall_utils import *
from src.utils import data_utils, chart_utils
from src.neural import models
from src.reasoning import reason
from src.utils.chart_utils import van

def recall_fms(args, bk_shape, bw_img, reshape=None):
    # shift fm and images
    # shifted_imgs = get_shifted_matrics(bw_img)

    # convolutional layer
    shifted_fms = models.one_layer_conv(bw_img, bk_shape["kernels"].float())

    # visual input fms
    shifted_fm_img = shifted_fms.sum(dim=1, keepdim=True).permute(0, 2, 3, 1)
    shifted_fm_img = (shifted_fm_img - shifted_fm_img.min()) / (
            shifted_fm_img.max() - shifted_fm_img.min())
    shifted_fm_img = shifted_fm_img.numpy()

    # chart_utils.visual_batch_imgs(shifted_fm_img[:30], args.save_path,
    #                               "in_fm_shifts.png")
    if reshape is not None:
        bk_fm_img = F.interpolate(bk_shape["fm_img"], size=(reshape, reshape),
                                  mode='bilinear', align_corners=False)
        bk_fms = F.interpolate(bk_shape["fm_repo"], size=(reshape, reshape),
                               mode='bilinear', align_corners=False)
    else:
        bk_fm_img = bk_shape["fm_img"]
        bk_fms = bk_shape["fm_repo"]
    # edge similarity
    img_edge = detect_edge(bw_img.float())
    repo_edge = detect_edge(bk_fm_img)
    sim_edge = data_utils.matrix_equality(repo_edge, img_edge)

    # fm similarity
    sim_fm = data_utils.matrix_equality(bk_fms, shifted_fms)

    # total similarity
    sim_total = (sim_edge * 0.5 + sim_fm * sim_edge.max() * 0.5).permute(1, 0)

    # find the best shift
    best_shift = find_best_shift(sim_total, shifted_fms.shape[-2:])

    # log
    # args.logger.debug(f"recall {len(best_shift[-1])} possible fms, "
    #                   f"highest conf: {best_shift[-1][0]:.2f}")

    return shifted_fms, best_shift


def recall_match(args, bk_shapes, bw_img):
    onside_shapes = []
    onside_percents = torch.zeros(len(bk_shapes))
    for b_i, bk_shape in enumerate(bk_shapes):
        shifted_fms, rc_data = recall_fms(args, bk_shape, bw_img,
                                          reshape=args.obj_fm_size)
        # reasoning recalled fms to group
        group_data = reason.reason_fms(rc_data, bk_shape, bw_img,
                                       reshape=args.obj_fm_size)
        onside_shapes.append(group_data["recalled_bw_img"])
        onside_percents[b_i] = group_data["onside_percent"]
    best_shape = onside_percents.argmax()
    onside_pixels = onside_shapes[best_shape][0].squeeze()

    return onside_pixels, best_shape
