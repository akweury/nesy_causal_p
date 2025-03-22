# Created by X at 17.06.24

import torch
import cv2
import os

import config
from utils import file_utils, args_utils, data_utils
from src.alpha import alpha
from reasoning import reason
from src.alpha.fol.logic import InvAtom

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


def load_lang(args, mode, level):
    lang_file = str(args.output_file_prefix) + f'learned_lang_{mode}_{level}.pkl'
    if os.path.exists(lang_file):
        lang_data = torch.load(lang_file)
        if lang_data is not None:
            lang = alpha.init_ilp(args, lang_data["g_num"])
            lang.reset_lang(lang_data["g_num"], level)
            lang.clauses = lang_data["clauses"]
            lang.predicates = lang_data["preds"]
            lang.consts = lang_data["consts"]
            lang.atoms = lang_data["atoms"]
            inv_atoms = [atom for atom in lang.atoms if  isinstance(atom, InvAtom)]
            lang.attrs = lang_data["attrs"]
            lang.all_groups = lang_data["all_groups"]
            # lang.llm_clauses = lang_data["llm_clauses"]
            # lang.name_dict = lang_data["name_dict"]
            lang.generate_atoms()
            # lang.generate_inv_atoms(lang.predicates)
            # rules = {
            #     "true_all_image": lang_data["true_all_image"],
            #     "true_all_group": lang_data["true_all_group"],
            #     "true_exact_one_group": lang_data["true_exact_one_group"],
            # }
            # args.logger.debug(
            #     f"\n ================= Loaded Pretrained Language ================= " +
            #     f"\n ==== Machine Clauses: " +
            #     "".join([f"\n{c_i + 1}/{len(lang.clauses)} {lang.clauses[c_i]}"
            #              for c_i in range(len(lang.clauses))]) +
            #     f"\n ==== LLM Description: " +
            #     "".join([f"\n{c_i + 1}/{len(lang.llm_clauses)} {lang.llm_clauses[c_i]}"
            #              for c_i in range(len(lang.llm_clauses))]))
            return lang
    return None


def train_clauses(args, groups):
    args.step_counter += 1
    lang_pos_object = load_lang(args, "positive", "object")
    lang_pos_group = load_lang(args, "positive", "group")
    lang_neg_object = load_lang(args, "negative","object")
    lang_neg_group = load_lang(args, "negative","group")
    if lang_pos_object is None:
        lang_pos_object = alpha.alpha_object(args, groups["group_pos"], "positive")
    if lang_pos_group is None:
        lang_pos_group = alpha.alpha_group(args, groups["group_pos"], "positive")
    if lang_neg_object is None:
        lang_neg_object = alpha.alpha_object(args, groups["group_neg"], "negative")
    if lang_neg_group is None:
        lang_neg_group = alpha.alpha_group(args, groups["group_neg"], "negative")

    rules = reason.find_common_rules(lang_pos_group.all_groups,
                                     lang_pos_object.all_groups,
                                     lang_neg_group.all_groups,
                                     lang_neg_object.all_groups)

    return lang_pos_object, lang_pos_group, rules


if __name__ == "__main__":
    logger = args_utils.init_logger()
    args = args_utils.get_args(logger)
    out_path = config.output / args.exp_name
    # image_paths = file_utils.get_all_files(config.kp_dataset / args.exp_name, "png", False)[:500]
    # train_clauses(args, image_paths, out_path)
