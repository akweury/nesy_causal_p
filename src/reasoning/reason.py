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


def reason_fms(rc_fms, bk_shape, bw_img, reshape=None):
    bw_img = bw_img.squeeze()
    mem_fm_idx, mem_fm_shift, mem_fm_conf = rc_fms

    if reshape is not None:
        mem_fms = F.interpolate(bk_shape["fm_repo"][mem_fm_idx],
                                size=(reshape, reshape), mode='bilinear',
                                align_corners=False)
        mem_bw_img = F.interpolate(bk_shape["fm_img"][mem_fm_idx],
                                   size=(reshape, reshape),
                                   mode='bilinear', align_corners=False)

    else:
        mem_fms = bk_shape["fm_repo"][mem_fm_idx]
        mem_bw_img = bk_shape["fm_img"][mem_fm_idx]

    in_fms = models.one_layer_conv(bw_img.unsqueeze(0).unsqueeze(0),
                                   bk_shape["kernels"].float())
    in_fms = in_fms.sum(dim=1).squeeze()
    in_fms = (in_fms - in_fms.min()) / (in_fms.max() - in_fms.min())
    mem_fms = mem_fms.sum(dim=1, keepdim=True)

    mem_fm_min = torch.tensor([fm.min() for fm in mem_fms]).view(-1, 1, 1, 1)
    mem_fm_max = torch.tensor([fm.max() for fm in mem_fms]).view(-1, 1, 1, 1)
    mem_fms = (mem_fms - mem_fm_min) / (mem_fm_max - mem_fm_min)

    # image matching
    # onside, offside = img_matching(mem_fm, in_fms)

    onside = torch.stack([1 - (mem_fm.squeeze() - in_fms) ** 2 for mem_fm in mem_fms])
    onside = onside.mean(dim=0)
    onside[in_fms == 0] = 0
    onside_mask = in_fms * (onside > 0.9)

    # recall confidence
    onside_percent = 1 - torch.mean((mem_fms[0].squeeze() - in_fms) ** 2).item()
    group_data = {
        "onside": onside_mask,
        "recalled_bw_img": mem_bw_img,
        "parents": None,
        "onside_percent": onside_percent,
    }
    return group_data


def mask_similarity(mask1, mask2):
    similarity = torch.sum(mask1 * mask2) / torch.sum(mask2)
    return similarity


def reason_labels(args, bw_img, objs, crop_data, labels, onside):
    group_objs = torch.zeros_like(labels).bool()
    for o_i, obj in enumerate(objs):
        try:
            if labels[o_i] == 0:
                cropped_img, _ = data_utils.crop_img(obj.input, crop_data)
                seg_mask = data_utils.resize_img(cropped_img,
                                                 resize=args.obj_fm_size).unsqueeze(0) > 0
                simi_conf = mask_similarity(onside, seg_mask)
                if simi_conf > 0.4:
                    group_objs[o_i] = True
                    # find the mask of that object, remove the pixels of that object
                    bw_img[seg_mask] = 0
        except IndexError:
            raise IndexError
    return group_objs

#
# def remove_objs(args, labels, objs):
#     for l_i in range(len(labels)):
#         if labels[l_i] > -1:
#             obj = objs[l_i]
#             seg_mask = data_utils.rgb2bw(obj.input.astype(np.uint8), crop=False,
#                                          resize=args.obj_fm_size).unsqueeze(0) > 0
#             simi_conf = mask_similarity(onside, seg_mask)
#
#     return labels

#     shape_mask = torch.zeros_like(onside_argsmax)
#     for loc_group in input_groups:
#         input_seg = loc_group.input
#         seg_np = input_seg.astype(np.uint8)
#         seg_img = data_utils.rgb2bw(seg_np, crop=False,
#                                     resize=args.obj_fm_size).unsqueeze(0)
#         seg_mask = seg_img > 0
#         shape_mask += onside_mask * seg_mask.squeeze()
#
#     group_data = {
#         "onside": shape_mask,
#         "recalled_bw_img": shape_mask.unsqueeze(0).unsqueeze(0),
#         "parents": None,
#         "onside_percent": 0,
#     }
#
#     # # convert data to group object
#     group = Group(id=b_i,
#                   name=bk_shape["shape"],
#                   input_signal=img,
#                   onside_signal=group_data["onside"],
#                   memory_signal=group_data['recalled_bw_img'],
#                   parents=input_groups,
#                   coverage=group_data["onside_percent"],
#                   color=None)
#     obj_groups.append(group)
#
# best_idx = torch.tensor([g.onside_coverage for g in obj_groups]).argmax()
# best_group = obj_groups[best_idx]
