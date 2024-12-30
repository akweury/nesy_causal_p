# Created by jing at 17.06.24

import torch
import cv2
import os

import config
from utils import file_utils, args_utils, data_utils
from src.alpha import alpha
from src import llama_call, bk
from percept import perception
from src.neural import models

def load_bk(args, bk_shapes):
    # load background knowledge
    bk = []
    kernel_size = config.kernel_size
    for s_i, bk_shape in enumerate(bk_shapes):
        if bk_shape == "none":
            continue
        bk_path = config.output / bk_shape
        kernel_file = bk_path / f"kernel_patches_{kernel_size}.pt"
        kernels = torch.load(kernel_file).to(args.device)

        fm_file = bk_path / f"fms_patches_{kernel_size}.pt"
        fm_data = torch.load(fm_file).to(args.device)
        fm_img = fm_data[:, 0:1]
        fm_repo = fm_data[:, 1:]

        # load pretrained autoencoder
        # ae = models.Autoencoder(fm_repo.shape[1])
        # ae.load_state_dict(torch.load(bk_path / "fm_ae.pth"))
        # # load the dimension reduced feature maps
        # ae_fm = torch.load(bk_path / "ae_fms.pt").to(args.device)

        bk.append({
            "shape": s_i,
            "kernel_size": kernel_size,
            "kernels": kernels,
            "fm_img": fm_img,
            "fm_repo": fm_repo,
            # "ae": ae,
            # "ae_fm": ae_fm,
        })
    return bk


def load_data(args, image_path):
    file_name, file_extension = image_path.split(".")
    data = file_utils.load_json(f"{file_name}.json")
    patch = data_utils.oco2patch(data).unsqueeze(0).to(args.device)
    img = file_utils.load_img(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, patch


def load_lang(args):
    lang_file = config.output / args.exp_setting["task_name"] / f'learned_lang.pkl'
    if os.path.exists(lang_file):
        lang_data = torch.load(lang_file)
        if lang_data is not None:
            lang = alpha.init_ilp(args, 1)
            lang.reset_lang(lang_data["g_num"])
            lang.clauses = lang_data["clauses"]
            lang.predicates = lang_data["preds"]
            lang.consts = lang_data["consts"]
            lang.atoms = lang_data["atoms"]
            lang.attrs = lang_data["attrs"]
            lang.llm_clauses = lang_data["llm_clauses"]
            lang.name_dict = lang_data["name_dict"]
            lang.generate_atoms()
            args.logger.debug(
                f"\n ================= Loaded Pretrained Language ================= " +
                f"\n ==== Machine Clauses: " +
                "".join([f"\n{c_i + 1}/{len(lang.clauses)} {lang.clauses[c_i]}"
                         for c_i in range(len(lang.clauses))]) +
                f"\n ==== LLM Description: " +
                "".join(
                    [f"\n{c_i + 1}/{len(lang.llm_clauses)} {lang.llm_clauses[c_i]}"
                     for c_i in range(len(lang.llm_clauses))]))
            return lang
    return None


def save_lang(args, lang):
    lang_dict = {
        "atoms": lang.atoms,
        "clauses": lang.clauses,
        "consts": lang.consts,
        "preds": lang.predicates,
        "g_num": lang.group_variable_num,
        "attrs": lang.attrs,
        "llm_clauses": lang.llm_clauses,
        "name_dict": lang.name_dict
    }
    torch.save(lang_dict, config.output / args.exp_name / f'learned_lang.pkl')


# def obj2tensor(shape, color, pos, group_name, group_count_conf):
#     obj_tensor = torch.zeros(len(bk.obj_ohc))
#     i = 0
#     obj_tensor[i] = bk.color_large.index(color)  # color
#     i += 1
#     obj_tensor[i] = bk.shape_extend.index(shape)  # shape
#     i += 1
#     obj_tensor[i] = pos[0]  # x position
#     i += 1
#     obj_tensor[i] = pos[1]  # y position
#     i += 1
#     obj_tensor[i] = bk.group_name_extend.index(group_name)  # group label
#     i += 1
#     obj_tensor[i] = group_count_conf  # group confidence according to the count of objects
#     return obj_tensor


# def group2ocm(data, groups):
#     """ return the object centric matrix of the groups """
#     group_max_num = 25
#     group_ocms = []
#     positions = data_utils.data2positions(data)
#     for g_i, group in enumerate(groups):
#         # group
#         group_ocm = torch.zeros(group_max_num, len(bk.obj_ohc))
#         group_name = group["name"]
#         group_obj_positions = group["onside"]
#         group_count_conf = group["count_conf"]
#         pos_count = 0
#         for p_i, pos in enumerate(positions):
#             if group_obj_positions[pos[1].item(), pos[0].item()] > 0:
#                 shape = data[p_i]["shape"]
#                 color = data[p_i]["color_name"]
#                 obj_tensor = obj2tensor(shape, color, pos, group_name, group_count_conf)
#                 group_ocm[pos_count] = obj_tensor
#                 pos_count += 1
#         group_ocms.append(group_ocm)
#     group_ocms = torch.stack(group_ocms, dim=0)
#     return group_ocms

# def has_found_global_group(groups):
#     has_found = False
#     for group in groups:
#         if group.onside_coverage > 0.9:
#             has_found = True
#     return has_found


# def percept_gestalt_groups(args,idx, group_bk, img):
#     output_file_prefix = str(args.out_train_folder / f'img_{idx}')
#
#     group_file = output_file_prefix + f"_feature_groups.pt"
#     if os.path.exists(group_file):
#         groups = torch.load(group_file)
#     else:
#         groups = train_common_features.percept_feature_groups(args, group_bk, img, output_file_prefix)
#         torch.save(groups, group_file)
#
#     global_group_file = output_file_prefix + f"_global_groups.pt"
#     if os.path.exists(global_group_file):
#         groups = torch.load(global_group_file)
#     else:
#         groups = train_common_features.percept_object_groups(args, groups, group_bk, img, output_file_prefix)
#         torch.save(groups, global_group_file)
#
#     groups.ocm[0, -1] = 1
#     return [groups]


def train_clauses(args, data_loader):
    args.step_counter += 1
    lang = load_lang(args)

    if lang is None:
        # load background knowledge
        # lang = None
        all_clauses = []
        group_bk = load_bk(args, bk.bk_shapes)

        for idx, (image, data) in enumerate(data_loader):
            image = image.squeeze()
            args.output_file_prefix = str(args.out_train_folder / f'img_{idx}')
            # percepting groups
            groups = perception.percept_groups(args, idx, group_bk, image)

            # reasoning clauses
            lang = alpha.alpha(args, groups)
            all_clauses += lang.clauses

        # remove infrequent clauses
        lang = alpha.filter_infrequent_clauses(all_clauses, lang)

        # convert machine clause to final clause
        lang = llama_call.convert_to_final_clauses(args, lang)

        # save language
        save_lang(args, lang)
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Reasoned {len(lang.llm_clauses)} LLM Rules, "
                     f"{len(lang.clauses)} Machine Clauses")
    return lang


if __name__ == "__main__":
    logger = args_utils.init_logger()
    args = args_utils.get_args(logger)
    out_path = config.output / args.exp_name
    # image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)[:500]
    # train_clauses(args, image_paths, out_path)
