# Created by jing at 10.12.24
import torch
import numpy as np
import cv2

from src.utils import data_utils, chart_utils
from src.neural import models


def get_match_detail(mem_fm, visual_fm):
    mask_mem = mem_fm != 0
    same_fm = ((mem_fm == visual_fm) * mask_mem).sum(dim=1).to(torch.float32)
    mask_full_match = torch.all(mem_fm == visual_fm, dim=1) * torch.any(mask_mem,
                                                                        dim=1)
    mask_any_mismatch = torch.any((mem_fm == visual_fm) * mask_mem,
                                  dim=1) * torch.any(
        mask_mem, dim=1) * ~mask_full_match
    all_same_fm = same_fm * mask_full_match
    any_diff_fm = same_fm * mask_any_mismatch
    same_percent = mask_full_match.sum(dim=[1, 2]) / (
            mem_fm.sum(dim=1).bool().sum(dim=[1, 2]) + 1e-20)
    return all_same_fm, any_diff_fm, same_percent


def img_matching(mem_fm, in_fm):
    onside_mask = (in_fm > 0).squeeze() * (mem_fm > 0).squeeze()
    bw_img_reshaped = in_fm.unsqueeze(0).unsqueeze(0)

    data_onside = 1 - torch.abs(mem_fm - bw_img_reshaped).to(
        torch.float32).squeeze()
    data_onside[~onside_mask] = 0

    data_offside = torch.abs(mem_fm - bw_img_reshaped).to(
        torch.float32).squeeze()
    data_offside[~onside_mask] = 0

    return data_onside, data_offside


def eval_onside_conf(args, data, onsides, fm_imgs, bk_shape):
    fm_imgs = fm_imgs.squeeze()
    data_mask = data.squeeze() != 0
    onside_img = (onsides.sum(dim=0).float() * data_mask)
    onside_mask = onside_img > 0
    onside_mask = onside_mask.unsqueeze(0)
    onside_mask = torch.repeat_interleave(onside_mask, len(fm_imgs), dim=0)
    same = onside_mask.to(torch.float) == fm_imgs
    same[~onside_mask] = False
    group_count_conf = torch.zeros(len(onside_mask))
    for i in range(len(onside_mask)):
        onside_matrix = onside_mask.to(torch.float)[i].unsqueeze(0).unsqueeze(0)
        fm_matrix = fm_imgs[i].unsqueeze(0).unsqueeze(0)
        group_count_conf[i] = data_utils.matrix_equality(onside_matrix, fm_matrix)
        # group_count_conf[i] = tensor_similarity(onside_mask.to(torch.float)[i], fm_imgs[i])
    group_count_conf = torch.mean(group_count_conf)

    show_imgs = []
    for i in range(onside_mask.shape[0]):
        show_imgs.append(
            chart_utils.color_mapping(same[i].squeeze(), 1, f"TOP FM {i}"))
    show_imgs.append(chart_utils.color_mapping(onside_img.squeeze(), 1,
                                               f"Conf:{group_count_conf:.2f}"))

    return group_count_conf


def visual_all(args, in_fms, rc_fms, segment, mem_bw_img, bk_shape, img, bw_img,
               onside, offside):
    group_img_name = args.save_path / str(f'group_{bk_shape["name"]}.png')

    # norm_factor = max([bw_img.max(), in_fms.sum(dim=1).max()])
    segment_np = segment.permute(1, 2, 0).numpy().astype(np.uint8)
    segment_np = chart_utils.add_text("SEG", segment_np)

    # in_fms = models.one_layer_conv(bw_img.unsqueeze(0).unsqueeze(0),
    #                                bk_shape["kernels"].float())
    # in_fms = in_fms.sum(dim=1).squeeze()
    # in_fms = (in_fms - in_fms.min()) / (in_fms.max() - in_fms.min())
    seg_fm_img = chart_utils.color_mapping(in_fms, 1, "IN_BW")
    blank_img = chart_utils.color_mapping(torch.zeros_like(bw_img), 1, "")
    compare_imgs = []
    for i in range(min(10, len(onside))):
        # norm_factor = max([in_fm_img.max(), best_fm_img.max()])
        match_percent = f"{rc_fms[2][i] * 100:.2f}%"
        repo_fm_img = chart_utils.color_mapping(mem_bw_img[i].squeeze(), 1,
                                                "RECALL_FM")
        # rc_fm_same = chart_utils.color_mapping(onside[i],
        #                                               1,
        #                                               f"SAME FM {match_percent}")
        # repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff[i],
        #                                               1,
        #                                               "DIFF FM")
        data_onside_img = chart_utils.color_mapping(onside[i], 1,
                                                    f"Onside {match_percent}")
        data_offside_img = chart_utils.color_mapping(offside[i], 1,
                                                     "Offside Objs")

        img_list = [img, segment_np, seg_fm_img, repo_fm_img,
                    data_onside_img, data_offside_img]
        compare_imgs.append(chart_utils.hconcat_imgs(img_list))
    # last row: combined result
    data_mask = ((in_fms > 0) * (mem_bw_img > 0)).squeeze() > 0

    onside_comb = 1 - torch.abs(mem_bw_img - in_fms).to(torch.float32).squeeze()
    onside_comb[~data_mask] = 0
    onside_comb = onside_comb.mean(dim=0)

    match_percent = rc_fms[2].mean() * 100
    onside_comb_img = chart_utils.color_mapping(onside_comb,
                                                1,
                                                f"Onside {match_percent:.2f}%")

    offside_comb = torch.abs(mem_bw_img - in_fms).to(torch.float32).squeeze()
    offside_comb[~data_mask] = 0
    offside_comb = offside_comb.mean(dim=0)
    offside_comb_img = chart_utils.color_mapping(offside_comb, 1,
                                                 "Offside Comb.")

    compare_imgs.append(chart_utils.hconcat_imgs(
        [img, segment_np, seg_fm_img, blank_img, onside_comb_img, offside_comb_img]))
    compare_imgs = chart_utils.vconcat_imgs(compare_imgs)
    compare_imgs = cv2.cvtColor(compare_imgs, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(group_img_name), compare_imgs)
