# Created by shaji at 24/06/2024
predicate_target = "target:1:pattern"
predicate_exist = "in:1:group,pattern"
group_name = "g"

variable = {
    'example': "X",
    "group": "G"
}
neural_p = {
    # 'shape_counter': 'shape_counter:2:group,number',
    # 'color_counter': 'color_counter:2:group,number',
    'color': 'color:2:group,color',
    'shape': 'shape:2:group,shape',
    # 'phi': 'phi:3:group,group,phi',
    # 'rho': 'rho:3:group,group,rho',
    # 'slope': 'slope:2:group,slope',
}
const_dict = {
    'pattern': 'target',
    'color': 'enum',
    'shape': 'enum',
    'group': 'amount_e',
    # 'phi': 'amount_phi',
    # 'rho': 'amount_rho',
    # 'slope': 'amount_slope',
    'number': 'amount_num',
}
attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope", 'number']
color = ["color_1", "color_2", "color_3", "color_4", "color_5",
         "color_6", "color_7", "color_8", "color_9", "color_10"]
shape = ['line', 'rectangle']

# pred_obj_mapping = {
#     'in': None,
#     'shape_counter': ["sphere", "cube", "cone", "cylinder"],
#     'color_counter': ["red", "green", "blue"],
#     'shape': ['sphere', 'cube', 'cone', 'cylinder', 'line', 'circle', 'conic'],
#     'color:': ['red', 'green', 'blue'],
#     'phi': ['x', 'y', 'z'],
#     'rho': ['x', 'y', 'z'],
#     'slope': ['x', 'y', 'z'],
# }
# pred_pred_mapping = {
#     'shape_counter': ['shape'],
#     'color_counter': ['color'],
#     'in': []
# }`
