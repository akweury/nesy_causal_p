# Created by shaji at 27/10/2024
import cv2
import torch

import config
from utils import file_utils, args_utils, data_utils
from percept import perception
from src.alpha import alpha
from src import bk
from src.reasoning import reason


def load_data(args, image_path):
    file_name, file_extension = image_path.split(".")
    data = file_utils.load_json(f"{file_name}.json")
    patch = data_utils.oco2patch(data).unsqueeze(0).to(args.device)
    img = file_utils.load_img(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, patch


def load_bk(args, bk_shapes):
    # load background knowledge
    bk = []
    for bk_shape in bk_shapes:
        if bk_shape == "none":
            continue
        kernels = torch.load(config.output / bk_shape / f"kernel_patches_3.pt").to(
            args.device)
        fm_data = torch.load(config.output / bk_shape / f"fms_patches_3.pt").to(
            args.device)
        fm_img = fm_data[:, 0:1]
        fm_repo = fm_data[:, 1:]
        bk.append({
            "name": bk_shape,
            "kernels": kernels,
            "fm_img": fm_img,
            "fm_repo": fm_repo
        })

    return bk


def obj2tensor(shape, color, pos, group_name, group_count_conf):
    obj_tensor = torch.zeros(len(bk.obj_ohc))
    i = 0
    obj_tensor[i] = bk.color_large.index(color)  # color
    i += 1
    obj_tensor[i] = bk.shape_extend.index(shape)  # shape
    i += 1
    obj_tensor[i] = pos[0]  # x position
    i += 1
    obj_tensor[i] = pos[1]  # y position
    i += 1
    obj_tensor[i] = bk.group_name_extend.index(group_name)  # group label
    i += 1
    obj_tensor[
        i] = group_count_conf  # group confidence according to the count of objects
    return obj_tensor


def group2ocm(data, groups):
    """ return the object centric matrix of the groups """
    group_max_num = 25
    group_ocms = []
    positions = data_utils.data2positions(data)
    for g_i, group in enumerate(groups):
        # group
        group_ocm = torch.zeros(group_max_num, len(bk.obj_ohc))
        group_name = group["name"]
        group_obj_positions = group["onside"]
        group_count_conf = group["count_conf"]
        pos_count = 0
        for p_i, pos in enumerate(positions):
            if group_obj_positions[pos[1].item(), pos[0].item()] > 0:
                shape = data[p_i]["shape"]
                color = data[p_i]["color_name"]
                obj_tensor = obj2tensor(shape, color, pos, group_name,
                                        group_count_conf)
                group_ocm[pos_count] = obj_tensor
                pos_count += 1
        group_ocms.append(group_ocm)
    group_ocms = torch.stack(group_ocms, dim=0)
    return group_ocms


# def check_clauses(args, lang, ocm_neg, gcm_neg, groups_neg):
#     # load background knowledge
#     example_num = len(ocm_neg)
#     clauses_conf = torch.zeros((example_num, len(lang.clauses)))
#     group_bk = load_bk(args, bk.bk_shapes)
#
#     for e_i in range(example_num):
#         clauses_conf[e_i] = alpha.alpha_test(args, ocm_neg[e_i], gcm_neg[e_i], lang)
#
#         # logger
#         satisfied_clause_indices = torch.nonzero(
#             clauses_conf[idx] >= args.valid_rule_th).reshape(-1)
#         dissatisfied_clause_indices = torch.nonzero(
#             clauses_conf[idx] < args.valid_rule_th).squeeze()
#         satisfied_clauses = [f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n"
#                              for c_i in satisfied_clause_indices]
#         dissatisfied_clauses = [
#             f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n" for c_i in
#             dissatisfied_clause_indices]
#
#         args.logger.debug(f"\n"
#                           f"{image_label} Image {idx} Machine clauses: \n" +
#                           f"".join(satisfied_clauses) +
#                           f"".join(dissatisfied_clauses)
#                           )
#         mean_conf = clauses_conf[idx].mean()
#         if image_label == "POSITIVE" and mean_conf < 0.8:
#             args.logger.warning(
#                 f"\n (FALSE Negative) conf|threshold {mean_conf}|0.8")
#
#     pred_conf = clauses_conf.mean(dim=1)
#
#     args.logger.info(f"\n"
#                      f"Step {args.step_counter}/{args.total_step}: "
#                      f"Test {image_label} Images\n"
#                      f"Confidence for each image: {pred_conf}\n"
#                      f"Average Accuracy: {pred_conf.mean(dim=0):.2f}\n")
#
#     return pred_conf.mean()

def visual_group_on_the_image(img, group_ocms):
    pass


def visual_negative_image(check_results, imgs):
    negative_details = check_results["negative_details"]

    for img_i in range(len(negative_details)):
        img = imgs[img_i]
        for g_i in range(len(negative_details[img_i])):
            valid_group = negative_details[img_i][g_i]
            group_data = check_results["negative_groups"][img_i][g_i]
            if not valid_group:
                visual_group_on_the_image(img, group_data["ocm"])


def check_clause(args, lang, rules, imgs_test):
    # first three images are positive, last three images are negative
    preds = torch.zeros(len(imgs_test))
    image_label = torch.zeros(len(imgs_test))
    image_label[:3] += 1
    all_clauses = rules["true_all_image"] + rules["true_all_group"] + rules["true_exact_one_group"]
    clauses_labels = ([0] * len(rules[bk.rule_logic_types[0]]) +
                      [1] * len(rules[bk.rule_logic_types[1]]) +
                      [2] * len(rules[bk.rule_logic_types[2]]))

    group_bk = load_bk(args, bk.bk_shapes)
    groups = perception.cluster_by_principle(args, imgs_test)
    pos_clause_scores = alpha.alpha_test(args, groups["group_pos"], lang, all_clauses)
    neg_clause_scores = alpha.alpha_test(args, groups["group_neg"], lang, all_clauses)

    preds[:3], pred_details_pos = reason.reason_test_results(pos_clause_scores, clauses_labels)
    preds[3:], pred_details_neg = reason.reason_test_results(neg_clause_scores, clauses_labels)
    acc = (preds == image_label).sum() / len(preds)

    check_results = {
        "acc": acc,
        "negative_details": pred_details_neg,
        "negative_groups": groups["group_neg"]
    }

    visual_negative_image(check_results, imgs_test[3:])
    return check_results
    # logger
    # satisfied_clause_indices = torch.nonzero(clauses_conf[idx] >= args.valid_rule_th).reshape(-1)
    # dissatisfied_clause_indices = torch.nonzero(clauses_conf[idx] < args.valid_rule_th).squeeze()
    # satisfied_clauses = [f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n" for c_i in satisfied_clause_indices]
    # dissatisfied_clauses = [f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n" for c_i in
    #                         dissatisfied_clause_indices]

    # args.logger.debug(f"\n"
    #                   f"{image_label} Image {idx} Machine clauses: \n" +
    #                   f"".join(satisfied_clauses) +
    #                   f"".join(dissatisfied_clauses))
    # mean_conf = clauses_conf[idx].mean()
    # if image_label == "POSITIVE" and mean_conf < 0.8:
    #     args.logger.warning(
    #         f"\n (FALSE Negative) conf|threshold {mean_conf}|0.8")
    #
    # pred_conf = clauses_conf.mean(dim=1)
    #
    # args.logger.info(f"\n"
    #                  f"Step {args.step_counter}/{args.total_step}: "
    #                  f"Test {image_label} Images\n"
    #                  f"Confidence for each image: {pred_conf}\n"
    #                  f"Average Accuracy: {pred_conf.mean(dim=0):.2f}\n")


if __name__ == "__main__":
    args = args_utils.get_args()
