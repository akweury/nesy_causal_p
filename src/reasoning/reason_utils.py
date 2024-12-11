# Created by jing at 10.12.24
import torch
import cv2

from src.utils import data_utils, chart_utils


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


def img_matching(match_fm_img):
    data_onside = torch.stack(
        [(match_fm_img[i].squeeze()) for i in range(len(match_fm_img))])
    data_offside = torch.stack(
        [((match_fm_img[i].squeeze() == 0)) for i in range(len(match_fm_img))])


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


def visual_all(group_img_name, img, bw_img, data_fm_shifted, fm_best, max_value,
               fm_best_same, fm_best_diff, data_onside, data_offside):

    in_fm_img = data_fm_shifted.squeeze().sum(dim=0)
    mask_img = chart_utils.color_mapping(bw_img, 1, "IN")
    norm_factor = max([in_fm_img.max(), fm_best.sum(dim=1).max()])
    in_fm_norm_img = chart_utils.color_mapping(in_fm_img, norm_factor, "IN_FM")
    blank_img = chart_utils.color_mapping(torch.zeros_like(bw_img), norm_factor, "")
    compare_imgs = []

    for i in range(min(10, len(fm_best))):
        best_fm_img = fm_best[i].sum(dim=0)
        # norm_factor = max([in_fm_img.max(), best_fm_img.max()])
        match_percent = f"{int(max_value[i].item() * 100)}%"

        repo_fm_img = chart_utils.color_mapping(best_fm_img, norm_factor,
                                                "RECALL_FM")
        repo_fm_best_same = chart_utils.color_mapping(fm_best_same[i], norm_factor,
                                                      f"SAME FM {match_percent}")
        repo_fm_best_diff = chart_utils.color_mapping(fm_best_diff[i], norm_factor,
                                                      "DIFF FM")
        data_onside_img = chart_utils.color_mapping(data_onside[i], 1, "Onside Objs")
        data_offside_img = chart_utils.color_mapping(data_offside[i], 1,
                                                     "Offside Objs")

        compare_imgs.append(chart_utils.concat_imgs(
            [img, mask_img, in_fm_norm_img, repo_fm_img, repo_fm_best_same,
             repo_fm_best_diff, data_onside_img,
             data_offside_img]))
    # last row: combined result
    data_mask = bw_img != 0
    onside_comb = (data_onside.sum(dim=0).float() * data_mask)
    onside_comb_img = chart_utils.color_mapping(onside_comb, 1, "Onside Comb.")

    offside_comb = (data_offside.sum(dim=0).float() * data_mask)
    offside_comb_img = chart_utils.color_mapping(offside_comb, 1, "Offside Comb.")

    compare_imgs.append(chart_utils.concat_imgs(
        [img, mask_img, in_fm_norm_img, blank_img, blank_img, blank_img,
         onside_comb_img, offside_comb_img]))
    compare_imgs = chart_utils.vconcat_imgs(compare_imgs)
    compare_imgs = cv2.cvtColor(compare_imgs, cv2.COLOR_BGR2RGB)

    cv2.imwrite(group_img_name, compare_imgs)
