# Created by shaji at 27/10/2024
import cv2
import torch
import random

import config
from utils import file_utils, args_utils, data_utils
from percept import perception
from src.alpha import alpha
from src import bk
from src.reasoning import reason
from src.utils.chart_utils import van


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

def visual_group_on_the_image(img_np, obj_tensors, color):
    """
    there is a set of objects in the image,
    the image is given as pytorch tensor with size 1x512x512x3,
    now give the obj_tensors as pytorch tensor with size Nx10, where N is the number of the objects,
    for each object, the 0,1 indices are x and y position,
    index 2 saves the object size in range [0,1], which is the relative size according to the whole image,
    draw the shadow area on the objects using cv2

    :param img_np:
    :param group_ocms:
    :return:
    """
    transparency = 0.35
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # convert color channels
    for obj in obj_tensors:
        x, y, size = obj[0], obj[1], obj[2]
        # Convert coordinates and size from [0,1] range to actual pixel values
        x = int(x * img_np.shape[1])
        y = int(y * img_np.shape[0])
        size = 40  # int(size * min(img_np.shape[0], img_np.shape[1]) * 1.3)
        # Create an overlay
        overlay = img_np.copy()
        # Draw the circle on the overlay
        cv2.circle(overlay, (x, y), size, color, thickness=-1)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, transparency, img_np, 1 - transparency, 0, img_np)
    return img_np


def visual_negative_image(check_results, imgs):
    negative_details = check_results["negative_details"]

    for img_i in range(len(imgs)):
        img = imgs[img_i]
        labeled_img = img.clone().squeeze().numpy()
        group_colors = random.sample(list(bk.color_matplotlib.values()), 10)
        for c_i in range(len(negative_details[img_i])):
            for g_i in range(len(negative_details[img_i][c_i])):
                valid_group = negative_details[img_i][c_i][g_i]
                group_data = check_results["negative_groups"][img_i][g_i]
                if not valid_group:
                    labeled_img = visual_group_on_the_image(labeled_img, group_data["ocm"], group_colors[g_i])
            # save labeled image
            cv2.imwrite(str(config.models / "visual" / f"neg_{img_i}_c{c_i}.png"), labeled_img)


def check_clause(args, lang_obj, lang_group, rules, imgs_test, principle):
    # first three images are positive, last three images are negative
    preds_pos = []
    preds_neg = []
    image_label = torch.zeros(len(imgs_test))
    image_label[:3] += 1

    # group_bk = load_bk(args, bk.bk_shapes)
    groups = perception.cluster_by_principle(args, imgs_test, "test", principle)
    all_details = []
    for rule in rules:
        c = rule["rule"]
        r_type = rule["type"]
        counter = rule["counter"]
        if r_type == "true_all_image_g":
            c_scores_pos = alpha.alpha_test(args, groups["group_pos"], lang_group, [c], "group")
            c_scores_neg = alpha.alpha_test(args, groups["group_neg"], lang_group, [c], "group")

            pred_pos, _ = reason.reason_test_results(c_scores_pos, r_type)
            pred_neg, details = reason.reason_test_results(c_scores_neg, r_type)
            preds_pos.append(pred_pos)
            preds_neg.append(pred_neg)
            all_details.append(details)
        elif r_type in ["true_all_image", "true_all_group"]:
            c_scores_pos = alpha.alpha_test(args, groups["group_pos"], lang_obj, [c], "object")
            c_scores_neg = alpha.alpha_test(args, groups["group_neg"], lang_obj, [c], "object")
            pred_pos, _ = reason.reason_test_results(c_scores_pos, r_type, "object")
            pred_neg, details = reason.reason_test_results(c_scores_neg, r_type)
            preds_pos.append(pred_pos)
            preds_neg.append(pred_neg)
            all_details.append(details)
    preds_pos = torch.stack(preds_pos).reshape(-1, 3).prod(dim=0)
    preds_neg = torch.stack(preds_neg).reshape(-1, 3).prod(dim=0)
    preds = torch.cat((preds_pos, preds_neg))
    acc = (preds == image_label).sum() / len(preds)

    check_results = {
        "acc": acc,
        "negative_details": all_details,
        "negative_groups": groups["group_neg"],
        "principle": groups["principle"]
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
