# Created by shaji at 24/06/2024
target_predicate = [
    'tp:1:pattern',
    'in:2:group,pattern'
]
neural_p = {
    'shape_counter': 'shape_counter:2:group,number',
    'color_counter': 'color_counter:2:group,number',
    'color': 'color:2:group,color',
    'shape': 'shape:2:group,shape',
    'phi': 'phi:3:group,group,phi',
    'rho': 'rho:3:group,group,rho',
    'slope': 'slope:2:group,slope',
}
const_dict = {
    'image': 'target',
    'color': 'enum',
    'shape': 'enum',
    'group': 'amount_e',
    'phi': 'amount_phi',
    'rho': 'amount_rho',
    'slope': 'amount_slope',
    'number': 'amount_e',
}
attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope", 'number']
color = ['pink', 'green', 'blue']
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