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

    in_fms = models.one_layer_conv(bw_img.unsqueeze(0).unsqueeze(0),
                                   bk_shape["kernels"].float())
    in_fms = in_fms.sum(dim=1).squeeze()
    in_fms = (in_fms - in_fms.min()) / (in_fms.max() - in_fms.min())
    mem_fm = mem_fm.sum(dim=1, keepdim=True)

    mem_fm_min = torch.tensor([fm.min() for fm in mem_fm]).view(-1, 1, 1, 1)
    mem_fm_max = torch.tensor([fm.max() for fm in mem_fm]).view(-1, 1, 1, 1)
    mem_fm = (mem_fm - mem_fm_min) / (mem_fm_max - mem_fm_min)

    # image registration
    # mem_fm, mem_bw_img = fm_registration(mem_fm, mem_bw_img, rc_fms)

    # image matching
    onside, offside = img_matching(mem_fm, in_fms)
    # return onside.max(dim=0)[0]
    # visualization
    # fms = models.one_layer_conv(shifted_imgs, bk_shape["kernels"].float())
    # match_same, match_diff, same_percent = get_match_detail(mem_fm, fms.squeeze())
    visual_all(args, in_fms, rc_fms, segment, mem_bw_img, bk_shape, img, bw_img,
               onside, offside)


    # recall confidence
    fm_diff = torch.abs(mem_fm - in_fms)
    fm_diff[fm_diff > 1] = 0
    onside_percent = 1 - fm_diff
    onside_mask = mem_fm > 0
    onside_percent = onside_percent[onside_mask].mean()
    onside_mask = onside.mean(dim=0) > 0.8
    group_data = {
        "onside": onside_mask,
        "recalled_bw_img": mem_bw_img,
        "parents": None,
        "onside_percent": onside_percent,
    }
    return group_data
