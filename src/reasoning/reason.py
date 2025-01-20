# Created by shaji at 25/07/2024
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


def reason_fms(mem_fms, kernel, bw_img, reshape=None):
    bw_img = bw_img.squeeze().unsqueeze(0).unsqueeze(0)
    in_fms = models.one_layer_conv(bw_img, kernel)
    in_fms = in_fms.sum(dim=1).squeeze()
    in_fms = (in_fms - in_fms.min()) / ((in_fms.max() - in_fms.min()) + 1e-20)

    # mem fms
    mem_fms = mem_fms.squeeze().sum(dim=0)
    mem_fms = (mem_fms - mem_fms.min()) / ((mem_fms.max() - mem_fms.min()) + 1e-20)

    # onside = 1 - (mem_fms - in_fms) ** 2
    # onside[in_fms == 0] = 0
    # onside_mask = in_fms * (onside > 0.8)
    # memory and input intersection
    onside_mask = (in_fms > 0) * (mem_fms > 0)
    # recall confidence
    onside_percent = 1 - torch.mean((mem_fms - in_fms) ** 2).item()
    group_data = {
        "onside": onside_mask,
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


def sementical_same_clause(clause1, clause2):
    c1_pred = clause1.body[0].pred.name
    c2_pred = clause2.body[0].pred.name
    c1_terms = [t.name for t in clause1.body[0].terms if isinstance(t, Const)]
    c2_terms = [t.name for t in clause2.body[0].terms if isinstance(t, Const)]
    same_terms = c1_terms == c2_terms
    same_pred = c1_pred == c2_pred
    sementical_same = same_terms and same_pred
    return sementical_same


def remove_sementic_same_clauses(clause_list1, clause_list2):
    list1_only_clauses = []
    for clause in clause_list1:
        only_list1 = True
        for clause2 in clause_list2:
            if sementical_same_clause(clause, clause2):
                only_list1 = False
                break
        if only_list1:
            list1_only_clauses.append(clause)
    sementic_unique_clauses = []
    for clause in list1_only_clauses:
        sementic_unique = True
        for unique_c in sementic_unique_clauses:
            if sementical_same_clause(clause, unique_c):
                sementic_unique = False
                break
        if sementic_unique:
            sementic_unique_clauses.append(clause)
    return sementic_unique_clauses


def get_all_clauses(img_clauses):
    clauses = []
    for ic in img_clauses:
        for g in ic:
            clauses += g["group_clauses"]
            for obj_clause in g["obj_clauses"]:
                clauses += obj_clause
    return clauses


def create_ctt(image_clauses):
    # list all clauses
    clauses = get_all_clauses(image_clauses)

    img_num = len(image_clauses)
    group_c_num = [len(igs) for igs in image_clauses]
    obj_num = [len(ig["obj_clauses"]) for igs in image_clauses for ig in igs]
    obj_c_num = [len(oc) for igs in image_clauses for ig in igs for oc in ig["obj_clauses"]]
    c_num = sum(obj_c_num) + sum(group_c_num)

    # calculate clause truth table
    ctt = torch.zeros(c_num, img_num, max(group_c_num), max(obj_num), max(obj_c_num)).to(torch.bool)
    ctt_count = torch.zeros(c_num, img_num, max(group_c_num), max(obj_num))
    for c_i in range(c_num):
        for i_i, img_c in enumerate(image_clauses):
            for g_i, group in enumerate(img_c):
                # group clause truth
                for o_i, obj_clauses in enumerate(group["obj_clauses"]):
                    ctt_count[c_i, i_i, g_i, o_i] = len(obj_clauses)
                    for oc_i, clause in enumerate(obj_clauses):
                        # group_clause = group["group_clauses"][0]
                        # if (sementical_same_clause(group_clause, clauses[c_i])):
                        #     ctt[c_i, i_i, g_i, o_i, 0] = True
                        if (sementical_same_clause(clause, clauses[c_i])):
                            ctt[c_i, i_i, g_i, o_i, oc_i] = True
    return ctt, ctt_count


def in_all_image(ctt, ctt_count, clauses):
    true_in_all_image = ctt.any(dim=-1).any(dim=-1).any(dim=-1).all(dim=-1)
    true_in_all_image_cs = [clauses[i] for i in range(len(clauses)) if true_in_all_image[i]]
    return true_in_all_image_cs


def in_all_group(ctt, ctt_count, clauses):
    true_any_objc_any_obj = ctt.any(dim=-1).any(dim=-1)
    true_in_all_groups = torch.zeros(ctt.shape[:2]).bool()
    for c_i in range(ctt.shape[0]):
        for img_i in range(ctt.shape[1]):
            g_num = (ctt_count[c_i, img_i].sum(dim=-1) > 0).sum()
            true_in_all_groups[c_i, img_i] = true_any_objc_any_obj[c_i, img_i, :g_num].all()
    true_in_all_image_all_groups = true_in_all_groups.all(dim=-1)
    true_in_all_image_all_groups_cs = [clauses[i] for i in range(len(clauses)) if true_in_all_image_all_groups[i]]
    return true_in_all_image_all_groups, true_in_all_image_all_groups_cs


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


def find_common_rules(image_clauses_pos, image_clauses_neg):
    # for each clause, check if it true in all image, check if it is true with condition.
    # 1: check if it is true in all groups, is_true(C, allG), is_true(C, allI), is_true(C, amoG), is_true
    # give one object's tensor, has_color(O1, red)
    # give clause's tensor, so we need convert clause to tensors
    # clause truth tensor: N * I * G * O
    # N: clause number
    # I: image number
    # G: group number
    # O: object number

    clauses_pos = get_all_clauses(image_clauses_pos)
    clauses_neg = get_all_clauses(image_clauses_neg)

    # create clause truth table
    ctt_pos, ctt_count_pos = create_ctt(image_clauses_pos)
    ctt_neg, ctt_count_neg = create_ctt(image_clauses_neg)

    # reason if clauses are truth in all images
    true_all_image_clauses = in_all_image(ctt_pos, ctt_count_pos, clauses_pos)
    true_all_image_clauses_neg = in_all_image(ctt_neg, ctt_count_neg, clauses_neg)
    # remove clauses that exist in both pos and neg
    true_all_image_clauses_pos_only = remove_sementic_same_clauses(true_all_image_clauses, true_all_image_clauses_neg)

    # reason if clauses are truth in all groups in all images
    true_all_group, true_all_group_clauses = in_all_group(ctt_pos, ctt_count_pos, clauses_pos)
    true_all_group_neg, true_all_group_clauses_neg = in_all_group(ctt_neg, ctt_count_neg, clauses_neg)
    true_all_group_clauses_pos_only = remove_sementic_same_clauses(true_all_group_clauses, true_all_group_clauses_neg)

    # reason if clauses are truth in at most one group in all images
    true_exact_one_group, true_exact_one_group_clauses = in_exact_one_group(ctt_pos, ctt_count_pos, clauses_pos)
    true_exact_one_group_neg, true_exact_one_group_clauses_neg = in_exact_one_group(ctt_neg, ctt_count_neg, clauses_neg)
    true_exact_one_group_clauses_pos_only = remove_sementic_same_clauses(true_exact_one_group_clauses,
                                                                         true_exact_one_group_clauses_neg)

    common_rules_dict = {
        "true_all_image": true_all_image_clauses_pos_only,
        "true_all_group": true_all_group_clauses_pos_only,
        "true_exact_one_group": true_exact_one_group_clauses_pos_only
    }
    return common_rules_dict


def check_true_in_image(group_scores):
    raise NotImplementedError


def check_true_all_group(group_scores, th=0.8):
    pred = True
    group_preds = []
    for group in group_scores:
        group_score = max(group) > th
        group_preds.append(group_score)
        pred *= group_score
    return pred, group_preds


def check_true_exact_one_group(group_scores):
    raise NotImplementedError


def reason_test_results(clause_scores, clause_labels):
    preds = torch.zeros(len(clause_scores), dtype=torch.bool)
    pred_details = []
    for example_i in range(len(clause_scores)):
        pred = True
        for c_i, label in enumerate(clause_labels):
            if bk.rule_logic_types[label] == "true_all_image":
                pred *= check_true_in_image(clause_scores[example_i][c_i])
            elif bk.rule_logic_types[label] == "true_all_group":
                clause_preds, group_preds = check_true_all_group(clause_scores[example_i][c_i])
                pred_details.append(group_preds)
                pred *= clause_preds
            elif bk.rule_logic_types[label] == "true_exact_one_group":
                pred *= check_true_exact_one_group(clause_scores[example_i][c_i])
            else:
                raise ValueError(f"Unknown rule logic_type: {bk.rule_logic_types[label]}")
        preds[example_i] = pred
    return preds.float(), pred_details
