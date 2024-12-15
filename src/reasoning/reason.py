# Created by shaji at 25/07/2024
from src.reasoning.reason_utils import *
import torch.nn.functional as F  # Import F for functional operations

from src.neural import models


def fm_registration(fm_mem, bw_img_mem, fms_rc):
    """ fit recalled fm/bw_img to input fm/bw_img """
    mem_fm_idx, mem_fm_shift, mem_fm_conf = fms_rc

    mem_fm = torch.stack(
        [torch.roll(fm_mem[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1))
         for i in range(len(mem_fm_shift))])
    mem_bw_img = torch.stack(
        [torch.roll(bw_img_mem[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1))
         for i in range(len(mem_fm_conf))])
    return mem_fm, mem_bw_img


def reason_fms(args, segment, rc_fms, bk_shape, img, bw_img, reshape=None):
    bw_img = bw_img.squeeze()
    mem_fm_idx, mem_fm_shift, mem_fm_conf = rc_fms

    if reshape is not None:
        mem_fm = F.interpolate(bk_shape["fm_repo"][mem_fm_idx],
                               size=(reshape, reshape), mode='bilinear',
                               align_corners=False)
        mem_bw_img = F.interpolate(bk_shape["fm_img"][mem_fm_idx],
                                   size=(reshape, reshape),
                                   mode='bilinear', align_corners=False)
    else:
        mem_fm = bk_shape["fm_repo"][mem_fm_idx]
        mem_bw_img = bk_shape["fm_img"][mem_fm_idx]

    # image registration
    # mem_fm, mem_bw_img = fm_registration(mem_fm, mem_bw_img, rc_fms)

    # image matching
    onside, offside = img_matching(mem_bw_img, bw_img)

    # visualization
    # fms = models.one_layer_conv(shifted_imgs, bk_shape["kernels"].float())
    # match_same, match_diff, same_percent = get_match_detail(mem_fm, fms.squeeze())

    visual_all(args, rc_fms, segment, mem_bw_img, bk_shape, img, bw_img, onside,
               offside)
    onside = (onside.sum(dim=0) > 0).squeeze()
    # recall confidence
    onside_coverage = bw_img[onside].bool().sum() / bw_img.bool().sum()
    group_data = {
        "onside": onside,
        "recalled_bw_img": mem_bw_img,
        "parents": None,
        "onside_percent": onside_coverage,
    }
    return group_data
