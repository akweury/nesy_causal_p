# Created by shaji at 24/06/2024
import matplotlib

mode_recall = 8
mode_excluded_preds = ["target"]
variable_symbol_pattern = "I"
variable_symbol_group = "G"
variable_symbol_obj = "O"

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

predicate_configs = {
    "predicate_target": f"{pred_names['target']}:1:pattern,+",
    "predicate_in_pattern": f"{pred_names['in_pattern']}:2:group_data,-;pattern,+",
    "predicate_in_group": f"{pred_names['in_group']}:3:object,-;group_data,+;pattern,+",
    "predicate_color": f"{pred_names['has_color']}:3:color,#;group_data,+;pattern,+",
    "predicate_shape": f"{pred_names['has_shape']}:3:shape,#;group_data,+;pattern,+",
    "predicate_g_shape": f"{pred_names['group_shape']}:3:group_label,#;group_data,+;pattern,+",
}

var_dtypes = {
    "pattern": "pattern",
    "group": "group_data",
    "object": "object"
}

const_dict = {
    'pattern,+': "pattern",
    'group_data,+': 'amount_group',
    'object,+': 'amount_object',
    'color,+': 'enum',
    'shape,+': 'enum',
    'group_label,+': 'enum'
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

attr_names = ['color', 'shape', "group_label"]

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in
                    list(matplotlib.colors.cnames.items())}
color_matplotlib.pop("black")
color_large = [k for k, v in list(color_matplotlib.items())]

shape_extend = ["circle", "square", "triangle", "diamond"]
group_name_extend = ["none", "circle_small", "square_small", "triangle_small"]
# group_name_extend = ["none", "circle_flex"]

# exp setting
task_pattern_types = ["task_true_pattern", "task_random_pattern", "task_cf_pattern"]
exp_demo = {
    "bk_groups": ["triangler", "circle_flex"],
    "task_name": "trianglecircle_flex",
    "task_true_pattern": "trianglecircle_flex",
    "task_random_pattern": "random",
    "task_cf_pattern": "trianglecircle_flex_cf"
}

exp_count_group = {
    "bk_groups": ["circle_small", "square_small"],
    "task_name": "circlesquare_count",
    "task_true_pattern": "circlesquare_count",
    "task_random_pattern": "random",
    "task_cf_pattern": "circlesquare_count_cf"
}
