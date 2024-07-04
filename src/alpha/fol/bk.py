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
    'duplicate': 'duplicate:2:input_group,output_group',
    'scale': 'scale:3:input_group,output_group,scale',
    # 'shape': 'shape:2:input_group,shape'
}
const_dict = {
    'out_pattern': "out_pattern",
    'in_pattern': "in_pattern",
    'color': 'enum',
    'shape': 'enum',
    'scale': 'enum',
    'input_group': 'amount_e',
    'output_group': 'amount_e',
    'number': 'amount_num',

}

color = [f'color_{i}' for i in range(1, 11)]
scale = [f'scale_{i}' for i in range(1, 11)]
shape = ['line', 'rectangle']

attr_names = ['color', 'shape', 'scale', 'rho', 'phi', 'group_shape', "slope", 'number']
