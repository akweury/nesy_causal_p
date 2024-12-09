# Created by shaji at 25/07/2024
from src.reasoning.reason_utils import *


def fm_registration(fm_mem, bw_img_mem, fms_rc):
    """ fit recalled fm/bw_img to input fm/bw_img """
    mem_fm_idx, mem_fm_shift, mem_fm_conf = fms_rc

    mem_fm = torch.stack(
        [torch.roll(fm_mem[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1))
         for i in range(len(mem_fm_conf))])
    mem_bw_img = torch.stack(
        [torch.roll(bw_img_mem[i], shifts=tuple(mem_fm_shift[i]), dims=(-2, -1))
         for i in range(len(mem_fm_conf))])
    return mem_fm, mem_bw_img


def reason_fms(args, rc_fms, bk_shape, img, fms, bw_img):
    bw_img = bw_img.squeeze()
    mem_fm_idx, mem_fm_shift, mem_fm_conf = rc_fms
    mem_fm = bk_shape["fm_repo"][mem_fm_idx]
    mem_bw_img = bk_shape["fm_img"][mem_fm_idx]

    # image registration
    mem_fm, mem_bw_img = fm_registration(mem_fm, mem_bw_img, rc_fms)

    # image matching
    onside, offside = img_matching(mem_bw_img)



    # visualization
    match_same, match_diff, same_percent = get_match_detail(mem_fm, fms.squeeze())
    visual_img_name = args.output_file_prefix + str(f'_group_{bk_shape["name"]}.png')
    visual_all(visual_img_name, img, bw_img, fms, mem_fm, same_percent, match_same,
               match_diff, onside, offside)

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
