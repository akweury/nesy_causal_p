# Created by shaji at 27/10/2024
import cv2
import torch

import config
from utils import file_utils, args_utils, data_utils
import train_common_features
from src.alpha import alpha
from src import bk


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
        kernels = torch.load(config.output / bk_shape / f"kernels.pt").to(args.device)
        fm_data = torch.load(config.output / bk_shape / f"fms.pt").to(args.device)
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
    obj_tensor[i] = group_count_conf  # group confidence according to the count of objects
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
                obj_tensor = obj2tensor(shape, color, pos, group_name, group_count_conf)
                group_ocm[pos_count] = obj_tensor
                pos_count += 1
        group_ocms.append(group_ocm)
    group_ocms = torch.stack(group_ocms, dim=0)
    return group_ocms


def check_clause(args, lang, image_paths, image_label, output_path):
    # load background knowledge
    check_size = 4
    clauses_conf = torch.zeros((min(check_size, len(image_paths)), len(lang.clauses)))
    group_bk = load_bk(args, bk.group_name_extend)

    for idx in range(min(check_size, len(image_paths))):
        file_name, file_extension = image_paths[idx].split(".")
        data = file_utils.load_json(f"{file_name}.json")

        img, obj_pos = load_data(args, image_paths[idx])
        groups = train_common_features.img2groups_flexible(args, group_bk, obj_pos, idx, img, output_path)
        if len(groups) != 0:
            group_tensors = group2ocm(data, groups)
            clauses_conf[idx] = alpha.alpha_test(args, group_tensors, lang)

        # logger
        satisfied_clause_indices = torch.nonzero(clauses_conf[idx] >= args.valid_rule_th).squeeze()
        dissatisfied_clause_indices = torch.nonzero(clauses_conf[idx] < args.valid_rule_th).squeeze()
        satisfied_clauses = [f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n" for c_i in satisfied_clause_indices]
        dissatisfied_clauses = [f"({clauses_conf[idx, c_i]:.2f}) {lang.clauses[c_i]}\n" for c_i in dissatisfied_clause_indices]

        args.logger.debug(f"\n"
                          f"{image_label} Image {idx} Machine clauses: \n" +
                          f"".join(satisfied_clauses) +
                          f"".join(dissatisfied_clauses)
                          )
        mean_conf = clauses_conf[idx].mean()
        if image_label == "POSITIVE" and mean_conf < 0.8:
            args.logger.warning(f"\n (FALSE Negative) conf|threshold {mean_conf}|0.8")

    pred_conf = clauses_conf.mean(dim=1)
    return pred_conf


if __name__ == "__main__":
    args = args_utils.get_args()

    image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name / "train" / "true", "png", False)[:500]
    lang = None
    raise NotImplementedError
    check_clause(args, lang, image_paths, True, output_path=config.kp_dataset / args.exp_name / "train" / "true")
