# Created by shaji at 24/06/2024
import random

import matplotlib
import torch
import config

mode_recall = 8
mode_excluded_preds = ["target"]
variable_symbol_pattern = "I"
variable_symbol_group = "G"
variable_symbol_obj = "O"

# ------------------------------------------

var_dtypes = {
    "pattern": "pattern",
    "group": "group_data",
    "object": "object"
}

const_dtypes = {
    "object_color": "object_color",
    "object_shape": "object_shape",
    "group_label": "group_label",
    "object_num": "object_num",
}

# -----------------------------------------

trivial_preds = {
    "target": "target",
    "in_group": "in_group",
    "in_pattern": "in_pattern",
}
pred_names = {
    "target": "target",
    "in_group": "in_group",
    "in_pattern": "in_pattern",
    "has_color": "exist_obj_color",
    "has_shape": "exist_obj_shape",
    "group_shape": "exist_group_shape",
    "object_num": "has_obj_num"
}

var_dtype_obj = var_dtypes["object"]
var_dtype_group = var_dtypes["group"]
var_dtype_pattern = var_dtypes["pattern"]

const_dtype_object_color = const_dtypes["object_color"]
const_dtype_object_shape = const_dtypes["object_shape"]
const_dtype_group = const_dtypes["group_label"]
const_dtype_obj_num = const_dtypes["object_num"]


def create_predicate_config(name, term_list):
    config_str = pred_names[name] + ":" + str(len(term_list)) + ":" + "".join(
        [f"{t};" for t in term_list])[:-1]
    return config_str


predicate_configs = {}

predicate_configs["predicate_target"] = create_predicate_config('target', [
    f"{var_dtype_pattern},+"
])

predicate_configs["predicate_in_pattern"] = create_predicate_config('in_pattern', [
    f"{var_dtype_group},-",
    f"{var_dtype_pattern},+"
])

predicate_configs["predicate_in_group"] = create_predicate_config("in_group", [
    f"{var_dtype_obj},-",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+"
])

predicate_configs["predicate_color"] = create_predicate_config("has_color", [
    f"{const_dtype_object_color},#",
    f"{var_dtype_obj},+",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])

predicate_configs["predicate_shape"] = create_predicate_config("has_shape", [
    f"{const_dtype_object_shape},#",
    f"{var_dtype_obj},+",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])
predicate_configs["predicate_g_shape"] = create_predicate_config("group_shape", [
    f"{const_dtype_group},#",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])

predicate_configs["predicate_obj_num"] = create_predicate_config("object_num", [
    f"{const_dtype_obj_num},#",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])

const_dict = {
    f'{var_dtypes["pattern"]},+': "pattern",
    f'{var_dtypes["group"]},+': 'amount_group',
    f'{var_dtypes["object"]},+': 'amount_object',
    f'{const_dtypes["object_color"]},+': 'enum',
    f'{const_dtypes["object_shape"]},+': 'enum',
    f'{const_dtypes["group_label"]},+': 'enum',
    f'{const_dtypes["object_num"]},+': 'quantity'
}

# dtype:
variable = {
    "feature_map": "FM",
    'pattern': "I"
}

obj_ohc = ["color_name", "shape", "x", "y", "group_name", "group_conf"]
prop_idx_dict = {
    "x": 0,
    "y": 1,
    "size": 2,
    "obj_num": 3,
    "rgb_r": 4,
    "rgb_g": 5,
    "rgb_b": 6,
    "shape_tri": 7,
    "shape_sq": 8,
    "shape_cir": 9,
}
color_small = ["blue", "yellow", "red"]

# shape_small = ["circle", "square", "triangle"]
# group_name_small = ["none", "data_triangle", "data_square", "data_circle"]

attr_names = ['object_color', 'object_shape', 'object_num', "group_label"]

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("darkslategray")
color_matplotlib.pop("lightslategray")
color_matplotlib.pop("black")
color_matplotlib.pop("darkgray")

color_dict_rgb2name = {value: key for key, value in color_matplotlib.items()}
color_large = [k for k, v in list(color_matplotlib.items())]
color_large_exclude_gray = [item for item in color_large if item != "lightgray"]
bk_shapes = ["none", "triangle", "square", "circle"]

gestalt_principles = ["proximity", "similarity_shape", "similarity_color", 'closure', "continuity", "symmetry"]

rule_logic_types = [
    "true_all_image",
    "true_all_group",
    "true_exact_one_group"
]


def tensor2dict(tensor):
    tensor_dict = {}
    tensor_dict['position'] = tensor[:2]
    tensor_dict['size'] = tensor[2]
    tensor_dict['obj_num'] = tensor[3]
    tensor_dict['color'] = tensor[4:7]
    tensor_dict['shape'] = tensor[7:7 + len(bk_shapes)]
    return tensor_dict


def load_bk_fms(args, bk_shapes):
    # load background knowledge
    bk = []
    kernel_size = config.kernel_size
    for s_i, bk_shape in enumerate(bk_shapes):
        if bk_shape == "none":
            continue
        bk_path = config.output / bk_shape
        # kernel_file = bk_path / f"kernel_patches_{kernel_size}.pt"
        # kernels = torch.load(kernel_file).to(args.device)

        fm_file = bk_path / f"fms_patches_{kernel_size}.pt"
        fm_data = torch.load(fm_file)

        fms = fm_data["fms"]
        contours = fm_data["contours"]
        labels = torch.tensor(fm_data["labels"])
        if len(fms) > 50:
            fms_indices = random.sample(range(len(fms)), 50)
            fms = fms[fms_indices]
            labels = labels[fms_indices]
        bk.append({
            "shape": s_i,
            "kernel_size": kernel_size,
            # "kernels": kernels,
            "fm_repo": fms,
            "contour":contours,
            "labels": labels,
        })
    return bk
