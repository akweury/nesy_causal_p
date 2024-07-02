# Created by shaji at 24/06/2024
predicate_target = "target:1:out_pattern"
predicate_has_ig = "hasIG:2:input_group,in_pattern"
predicate_has_og = "hasOG:2:output_group,out_pattern"

# dtype:
variable = {
    'in_pattern': "I",
    "out_pattern": "O",
    "input_group": ["A", "B", "C", "D", "E", "F"],
    "output_group": "G",
}
neural_p = {
    'color_input': 'color_input:2:input_group,color',
    'color_output': 'color_output:2:output_group,color',
    # 'shape': 'shape:2:input_group,shape'
}
const_dict = {
    'out_pattern': "out_pattern",
    'in_pattern': "in_pattern",
    'color': 'enum',
    'shape': 'enum',
    'input_group': 'amount_e',
    'output_group': 'amount_e',
    'number': 'amount_num',
}

color = ["color_1", "color_2", "color_3", "color_4", "color_5",
         "color_6", "color_7", "color_8", "color_9", "color_10"]
shape = ['line', 'rectangle']

attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope", 'number']
