# Created by shaji at 24/06/2024
import matplotlib

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
    "group_label": "group_label"
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
    "group_shape": "exist_group_shape"
}

var_dtype_obj = var_dtypes["object"]
var_dtype_group = var_dtypes["group"]
var_dtype_pattern = var_dtypes["pattern"]

const_dtype_object_color = const_dtypes["object_color"]
const_dtype_object_shape = const_dtypes["object_shape"]
const_dtype_group = const_dtypes["group_label"]


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
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])

predicate_configs["predicate_shape"] = create_predicate_config("has_shape", [
    f"{const_dtype_object_shape},#",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])
predicate_configs["predicate_g_shape"] = create_predicate_config("group_shape", [
    f"{const_dtype_group},#",
    f"{var_dtype_group},+",
    f"{var_dtype_pattern},+",
])

const_dict = {
    f'{var_dtypes["pattern"]},+': "pattern",
    f'{var_dtypes["group"]},+': 'amount_group',
    f'{var_dtypes["object"]},+': 'amount_object',
    f'{const_dtypes["object_color"]},+': 'enum',
    f'{const_dtypes["object_shape"]},+': 'enum',
    f'{const_dtypes["group_label"]},+': 'enum'
}

# dtype:
variable = {
    "feature_map": "FM",
    'pattern': "I"
}

obj_ohc = ["color_name", "shape", "x", "y", "group_name", "group_conf"]
prop_idx_dict = {
    "color": 0,
    "shape": 1,
    "x": 2,
    "y": 3,
    "group_name": 4,
    "group_conf": 5,
}
color_small = ["blue", "yellow", "red"]

shape_small = ["circle", "square", "triangle"]
group_name_small = ["none", "data_triangle", "data_square", "data_circle"]

attr_names = ['object_color', 'object_shape', "group_label"]

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("black")
no_color = "none"
color_matplotlib[no_color] = (0, 0, 0)
color_large = [k for k, v in list(color_matplotlib.items())]

bk_shapes = ["none", "circle", "square", "triangle"]

# exp setting
task_pattern_types = ["task_true_pattern", "task_random_pattern", "task_cf_pattern"]



exp_count_group = {
    "bk_groups": ["circle_small", "square_small"],
    "task_name": "circlesquare_count",
    "task_true_pattern": "circlesquare_count",
    "task_random_pattern": "random",
    "task_cf_pattern": "circlesquare_count_cf"
}

exp_gestalt_triangle = {
    "bk_groups": ["circle", "triangle"],
    "task_name": "01_gestalt_triangle",
    "task_true_pattern": "gestalt_triangle",
    "task_random_pattern": "random",
    "task_cf_pattern": "gestalt_triangle"
}

exp_triangle_group = {
    "bk_groups": ["circle", "square", "triangle"],
    "task_name": "02_triangle_group",
    "task_true_pattern": "triangle_group",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}

exp_square_circle_group = {
    "bk_groups": ["circle", "square", "triangle"],
    "task_name": "03_square_circle_group",
    "task_true_pattern": "square_circle_group",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}

exp_circle_group = {
    "bk_groups": ["circle", "square", "triangle"],
    "task_name": "04_circle_group",
    "task_true_pattern": "circle_group",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}

exp_proximity_square = {
    "bk_groups": ["circle", "square", "triangle"],
    "task_name": "05_proximity_square",
    "task_true_pattern": "proximity_square",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}

exp_similarity_triangle_circle = {
    "bk_groups": ["circle", "triangle"],
    "task_name": "06_similarity_triangle_circle",
    "task_true_pattern": "similarity_triangle_circle",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}

exp_closure_circle_triangle = {
    "bk_groups": ["circle", "triangle"],
    "task_name": "07_gestalt_circle_triangle",
    "task_true_pattern": "gestalt_circle_triangle",
    "task_random_pattern": "random",
    "task_cf_pattern": "random"
}