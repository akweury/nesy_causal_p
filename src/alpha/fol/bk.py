# Created by shaji at 24/06/2024
mode_recall = 8
mode_excluded_preds = ["target", "has"]
variable_symbol_pattern = "I"
variable_symbol_group = "G"
variable_symbol_obj = "O"
predicate_configs = {
    "predicate_target": "target:1:pattern,+",
    "predicate_in_pattern": "inp:2:group_data,-;pattern,+",
    "predicate_in_group": "ing:3:object,-;group_data,+;pattern,+",
    "predicate_color": "has_color:3:color,#;group_data,+;pattern,+",
    "predicate_shape": "has_shape:3:shape,#;group_data,+;pattern,+",
    "predicate_g_shape": "gshape:3:group_label,#;group_data,+;pattern,+",
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
color = ["blue", "yellow", "red"]
shape = ["circle", "square", "triangle"]
group_name = ["none", "data_triangle", "data_square", "data_circle"]

attr_names = ['color', 'shape', "group_label"]

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
