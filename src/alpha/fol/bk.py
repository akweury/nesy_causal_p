# Created by shaji at 24/06/2024
predicate_target = "target:1:pattern,+"
predicate_has_gp = "has:2:group_pattern,+;group_label,#"
predicate_color = "color:3:group_pattern,+;color,#;pattern,+"
predicate_shape = "shape:3:group_pattern,+;shape,#;pattern,+"
# predicate_phi = "phi:4:feature_map,+;feature_map,+;phi,#;pattern,+"
# predicate_rho = "rho:4:feature_map,+;feature_map,+;rho,#;pattern,+"

# dtype:
variable = {
    "feature_map": "FM",
    'pattern': "I"
}
neural_p = {
    # 'color': 'color:2:group,+;color,#',
    # 'duplicate': 'duplicate:2:group,+;group,+',
    # 'fulfil': 'fulfil:2',
    # 'surround': 'surround:2',
    'line': 'line:1',
    # 'scale': 'scale:3:group,+;group,+;scale,#',
    # 'repeat': 'repeat:2:group,+;group,+'
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

const_dict = {
    'pattern,+': "pattern",
    'group_pattern,+': 'amount_group',
    'color,+': 'enum',
    'shape,+': 'enum',
    'group_label,+': 'enum'
}

attr_names = ['color', 'shape', "group_label"]

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

mode_excluded_preds = ["target"]
variable_symbol_group = "G"
