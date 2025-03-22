# Created by X at 25/07/2024

from src.reasoning.reason_utils import *
import torch.nn.functional as F  # Import F for functional operations

from src.neural import models
from src.alpha.fol.logic import Const


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


def reason_fms(in_fms, mem_fms, reshape=None):
    # in_fms = models.img2fm(img, kernel)
    in_fms = models.fm_merge(in_fms)
    mem_fms = models.fm_merge(mem_fms)

    # onside = 1 - (mem_fms - in_fms) ** 2
    # onside[in_fms == 0] = 0
    # onside_mask = in_fms * (onside > 0.8)
    # memory and input intersection
    onside_mask = (in_fms > 0) * (mem_fms > 0)
    # recall confidence
    onside_percent = torch.bitwise_and((mem_fms > 0), (in_fms > 0)).sum() / (mem_fms > 0).sum()
    group_data = {
        "onside": onside_mask,
        "parents": None,
        "onside_percent": onside_percent,
    }
    return group_data


def mask_similarity(mask1, mask2):
    similarity = torch.sum(mask1 * mask2) / torch.sum(mask2)
    return similarity


def reason_labels(objs, crop_data, labels, onside, kernels):
    group_objs = torch.zeros_like(labels).bool()
    for o_i, obj in enumerate(objs):
        try:
            if labels[o_i] == 0:
                obj_img = obj.input
                obj_fm, _ = models.img2fm(obj_img, kernels, crop_data)
                obj_fm = models.fm_merge(obj_fm)
                obj_mask = obj_fm > 0
                # cropped_img, _ = data_utils.crop_img(obj.input, crop_data)
                # seg_mask = data_utils.resize_img(cropped_img,
                #                                  resize=args.obj_fm_size).unsqueeze(0) > 0
                simi_conf = mask_similarity(onside, obj_mask)
                if simi_conf > 0.5:
                    group_objs[o_i] = True
                    # find the mask of that object, remove the pixels of that object
                    # bw_img[seg_mask] = 0
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


def sementical_same_clause(clause1, clause2):
    sementical_same = True
    for t_i in range(len(clause1.body)):
        if clause1.body[t_i].pred.name in ["in_pattern", "in_group"]:
            continue
        c1_pred = clause1.body[t_i].pred.name
        c2_pred = clause2.body[t_i].pred.name
        c1_terms = [t.name for t in clause1.body[t_i].terms if isinstance(t, Const)]
        c2_terms = [t.name for t in clause2.body[t_i].terms if isinstance(t, Const)]
        same_terms = c1_terms == c2_terms
        same_pred = c1_pred == c2_pred
        same_atom = same_terms and same_pred
        sementical_same = sementical_same and same_atom
    return sementical_same


def remove_sementic_same_clauses(clause_list1, clause_list2, counter):
    list1_only_clauses = []
    list1_only_counter = []
    for c_i, clause in enumerate(clause_list1):
        only_list1 = True
        for clause2 in clause_list2:
            if sementical_same_clause(clause, clause2):
                only_list1 = False
                break
        if only_list1:
            list1_only_clauses.append(clause)
            list1_only_counter.append(counter[c_i])
    sementic_unique_clauses = []
    sementic_unique_counter = []
    for c_i, clause in enumerate(list1_only_clauses):
        sementic_unique = True
        for unique_c in sementic_unique_clauses:
            if sementical_same_clause(clause, unique_c):
                sementic_unique = False
                break
        if sementic_unique:
            sementic_unique_clauses.append(clause)
            sementic_unique_counter.append(list1_only_counter[c_i])
    return sementic_unique_clauses, sementic_unique_counter


def get_all_clauses(img_clauses, level):
    clauses = []
    for ic in img_clauses:
        for g in ic:
            if level == "group":
                for g_c, _ in g["group_clauses"].items():
                    clauses.append(g_c)
            else:
                for obj_clause, _ in g["obj_clauses"].items():
                    clauses.append(obj_clause)
    return clauses


def calc_ctt(image_clauses):
    # list all clauses
    # g_clauses = get_all_clauses(image_clauses, "group")
    o_clauses = get_all_clauses(image_clauses, "object")

    img_num = len(image_clauses)
    g_num = [len(igs) for igs in image_clauses]
    # gc_num = [len(ig["group_clauses"]) for igs in image_clauses for ig in igs]

    oc_num = [len(ig["obj_clauses"]) for igs in image_clauses for ig in igs]

    clause_num = sum(oc_num)  # + sum(gc_num)

    # calculate clause truth table
    ctt = torch.zeros(sum(oc_num), img_num, max(g_num), max(oc_num)).to(torch.bool)
    ctt_count = torch.zeros(sum(oc_num), img_num, max(g_num), max(oc_num))
    # g_ctt = torch.zeros(sum(gc_num), img_num, max(g_num), max(gc_num)).to(torch.bool)
    # g_ctt_count = torch.zeros(sum(gc_num), img_num, max(g_num), max(gc_num))

    # fulfill the ctt
    for c_i in range(sum(oc_num)):
        for i_i, img_c in enumerate(image_clauses):
            for g_i, group in enumerate(img_c):
                for oc_i, (obj_clause, c_count) in enumerate(group["obj_clauses"].items()):
                    ctt_count[c_i, i_i, g_i, oc_i] = c_count
                    if (sementical_same_clause(obj_clause, o_clauses[c_i])):
                        ctt[c_i, i_i, g_i, oc_i] = True
    # # fulfill the g_ctt
    # for c_i in range(sum(gc_num)):
    #     for i_i, img_c in enumerate(image_clauses):
    #         for g_i, group in enumerate(img_c):
    #             # group clause truth
    #             for gc_i, (gc, gc_count) in enumerate(group["group_clauses"].items()):
    #                 g_ctt_count[c_i, i_i, g_i, gc_i] = gc_count
    #                 if (sementical_same_clause(gc, g_clauses[c_i])):
    #                     g_ctt[c_i, i_i, g_i, gc_i] = True
    return ctt, ctt_count


def calc_g_ctt(image_clauses):
    # list all clauses
    g_clauses = get_all_clauses(image_clauses, "group")
    img_num = len(image_clauses)
    g_num = [len(igs) for igs in image_clauses]
    gc_num = [len(ig["group_clauses"]) for igs in image_clauses for ig in igs]

    clause_num = sum(gc_num)

    # calculate clause truth table
    g_ctt = torch.zeros(sum(gc_num), img_num, max(g_num), max(gc_num)).to(torch.bool)
    g_ctt_count = torch.zeros(sum(gc_num), img_num, max(g_num), max(gc_num))

    # fulfill the g_ctt
    for c_i in range(sum(gc_num)):
        for i_i, img_c in enumerate(image_clauses):
            for g_i, group in enumerate(img_c):
                # group clause truth
                for gc_i, (gc, gc_count) in enumerate(group["group_clauses"].items()):
                    g_ctt_count[c_i, i_i, g_i, gc_i] = gc_count
                    if (sementical_same_clause(gc, g_clauses[c_i])):
                        g_ctt[c_i, i_i, g_i, gc_i] = True
    return g_ctt, g_ctt_count


def in_all_image(g_ctt, ctt, ctt_count, clauses, level):
    """
    for each clause, check if it is true in each image

    :param ctt:
    :param ctt_count:
    :param clauses:
    :return: validation, NxI boolean tensor, N is clause number, I is image number;
    counter, NxI integer tensor, N is clause number, I is image number
    """
    # true in group
    # counter for each group
    if level == "object":
        counter_group = ctt.sum(dim=-1)
        counter_img = counter_group.sum(dim=-1)
        true_in_all_image = ctt.any(dim=-1).any(dim=-1).all(dim=-1)
        valid_clauses = [clauses[i] for i in range(len(clauses)) if true_in_all_image[i]]
        counter = [counter_img[i] for i in range(len(clauses)) if true_in_all_image[i]]
    else:
        tt = g_ctt.any(dim=-1).any(dim=-1).all(dim=-1)
        valid_clauses = [clauses[i] for i in range(len(clauses)) if tt[i]]
        counter_img_group_level = g_ctt.sum(dim=-1)
        counter = [counter_img_group_level[i] for i in range(len(clauses)) if tt[i]]

    return valid_clauses, counter


def in_all_group(ctt, ctt_count, clauses):
    # ctt: clause num x image num x group num x obj_clause num
    # true in group

    # counter for each group
    counter_group = ctt.sum(dim=-1)

    true_any_objc_any_obj = ctt.any(dim=-1)
    true_in_all_groups = torch.zeros(ctt.shape[:2]).bool()
    for c_i in range(ctt.shape[0]):
        for img_i in range(ctt.shape[1]):
            g_num = (ctt_count[c_i, img_i].sum(dim=-1) > 0).sum()
            true_in_all_groups[c_i, img_i] = true_any_objc_any_obj[c_i, img_i, :g_num].all()
    true_in_all_image_all_groups = true_in_all_groups.all(dim=-1)
    true_in_all_image_all_groups_cs = [clauses[i] for i in range(len(clauses)) if true_in_all_image_all_groups[i]]

    counter = [counter_group[i] for i in range(len(clauses)) if true_in_all_image_all_groups[i]]
    return true_in_all_image_all_groups, true_in_all_image_all_groups_cs, counter


def in_exact_one_group(ctt, ctt_count, clauses):
    true_any_objc_any_obj = ctt.any(dim=-1).any(dim=-1)
    true_in_exact_one_group = torch.zeros(ctt.shape[:2]).bool()
    for c_i in range(ctt.shape[0]):
        for img_i in range(ctt.shape[1]):
            g_num = (ctt_count[c_i, img_i].sum(dim=-1) > 0).sum()
            true_in_exact_one_group[c_i, img_i] = true_any_objc_any_obj[c_i, img_i, :g_num].sum() == 1
    true_in_all_image_all_groups = true_in_exact_one_group.all(dim=-1)
    true_in_all_image_all_groups_cs = [clauses[i] for i in range(len(clauses)) if true_in_all_image_all_groups[i]]
    return true_in_all_image_all_groups, true_in_all_image_all_groups_cs


def find_common_rules(image_group_clauses_pos, image_obj_clauses_pos,
                      image_group_clauses_neg, image_obj_clauses_neg):
    # for each clause, check if it true in all image, check if it is true with condition.
    # 1: check if it is true in all groups, is_true(C, allG), is_true(C, allI), is_true(C, amoG), is_true
    # give one object's tensor, has_color(O1, red)
    # give clause's tensor, so we need convert clause to tensors
    # clause truth tensor: N * I * G * O
    # N: clause number
    # I: image number
    # G: group number
    # O: object number
    g_clauses_pos = get_all_clauses(image_group_clauses_pos, "group")
    o_clauses_pos = get_all_clauses(image_obj_clauses_pos, "object")
    g_clauses_neg = get_all_clauses(image_group_clauses_neg, "group")
    o_clauses_neg = get_all_clauses(image_obj_clauses_neg, "object")

    # calculate clause truth table

    g_ctt_pos, g_ctt_count_pos = calc_g_ctt(image_group_clauses_pos)
    ctt_pos, ctt_count_pos = calc_ctt(image_obj_clauses_pos)
    g_ctt_neg, g_ctt_count_neg = calc_g_ctt(image_group_clauses_neg)
    ctt_neg, ctt_count_neg = calc_ctt(image_obj_clauses_neg)

    # reason if clauses are truth in all images
    img_c, count_pos_c = in_all_image(g_ctt_pos, ctt_pos, ctt_count_pos, o_clauses_pos, "object")
    img_c_neg, count_neg_c = in_all_image(g_ctt_neg, ctt_neg, ctt_count_neg, o_clauses_neg, "object")
    img_g_c, counter_pos_g_c = in_all_image(g_ctt_pos, ctt_pos, ctt_count_pos, g_clauses_pos, "group")

    # remove clauses that exist in both pos and neg
    img_c_pos_only, counter_img_c_pos_only = remove_sementic_same_clauses(img_c, img_c_neg, count_pos_c)
    img_g_c_pos_only, counter_img_g_c_pos_only = remove_sementic_same_clauses(img_g_c, g_clauses_neg, counter_pos_g_c)

    # reason if clauses are truth in all groups in all images
    true_all_group, true_all_group_clauses, counter_group_pos = in_all_group(ctt_pos, ctt_count_pos, o_clauses_pos)
    true_all_group_neg, true_all_group_clauses_neg, counter_group_neg = in_all_group(ctt_neg, ctt_count_neg,
                                                                                     o_clauses_neg)
    # remove duplicate clauses
    true_all_group_clauses_pos_only, counter_all_group_pos_only = remove_sementic_same_clauses(true_all_group_clauses,
                                                                                               true_all_group_clauses_neg,
                                                                                               counter_group_pos)
    # in all group
    # group level
    common_rues = []
    for c_i, c in enumerate(img_c_pos_only):
        common_rues.append({
            "rule": c,
            "counter": counter_img_c_pos_only[c_i],
            "type": "true_all_image"
        })
    for c_i, c in enumerate(img_g_c_pos_only):
        common_rues.append({
            "rule": c,
            "counter": counter_img_g_c_pos_only[c_i],
            "type": "true_all_image_g"
        })
    for c_i, c in enumerate(true_all_group_clauses_pos_only):
        common_rues.append({
            "rule": c,
            "counter": counter_all_group_pos_only[c_i],
            "type": "true_all_group"
        })
    return common_rues


def check_true_in_image(group_scores, level, th=0.8):
    pred = False
    group_preds = []
    for group in group_scores:
        # if level == "group":
        #     group_pred = max(group) > th
        # else:
        group_pred = False
        for g in group:
            group_pred = torch.bitwise_or(group_pred, max(g) > th)
        group_preds.append(group_pred)

        pred = torch.bitwise_or(pred, group_pred)
    return pred, group_preds


def check_true_all_group(group_scores, th=0.8):
    pred = True
    group_preds = []

    for c_scores in group_scores:
        group_pred = True
        for g in c_scores:
            group_pred = torch.bitwise_and(group_pred, max(g) > th)  # true in all group
        group_preds.append(group_pred)
        pred = torch.bitwise_and(pred, group_pred)
    return pred, group_preds


def check_true_exact_one_group(group_scores, th=0.8):
    group_preds = []
    for group in group_scores:
        group_pred = max(group) > th
        group_preds.append(group_pred)

    pred = sum(group_preds) == 1
    return pred, group_preds


def reason_test_results(clause_scores, label, level="group"):
    preds = torch.zeros(len(clause_scores), dtype=torch.bool)
    pred_details = []
    for example_i in range(len(clause_scores)):
        ex_pred = True
        ex_pred_details = []
        if len(clause_scores[example_i]) == 0:
            ex_pred = False
        if label in ["true_all_image", "true_all_image_g"]:
            clause_pred, group_preds = check_true_in_image(clause_scores[example_i], level)
            ex_pred_details.append(group_preds)
            ex_pred = torch.bitwise_and(clause_pred, ex_pred)
        elif label == "true_all_group":
            clause_pred, group_preds = check_true_all_group(clause_scores[example_i])
            ex_pred_details.append(group_preds)
            ex_pred = torch.bitwise_and(clause_pred, ex_pred)
        elif label == "true_exact_one_group":
            clause_pred, group_preds = check_true_exact_one_group(clause_scores[example_i])
            ex_pred_details.append(group_preds)
            ex_pred = torch.bitwise_and(clause_pred, ex_pred)
        else:
            raise ValueError(f"Unknown rule logic_type: {bk.rule_logic_types[label]}")
        pred_details.append(group_preds)
        preds[example_i] = ex_pred
    return preds.bool(), pred_details
