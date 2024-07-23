# Created by shaji at 24/06/2024
predicate_target = "target:1:out_pattern,+"
predicate_has_ig = "hasIG:2:input_group,-;in_pattern,+"
predicate_has_og = "hasOG:2:output_group,-;out_pattern,+"

# dtype:
variable = {
    'in_pattern': "I",
    "out_pattern": "O",
    "input_group": "A",
    "output_group": "B",
}
neural_p = {
    # 'color': 'color:2:group,+;color,#',
    # 'duplicate': 'duplicate:2:group,+;group,+',
    # 'fulfil': 'fulfil:2',
    # 'surround': 'surround:2',
    'drawline': 'drawline:2',
    # 'scale': 'scale:3:group,+;group,+;scale,#',
    # 'repeat': 'repeat:2:group,+;group,+'
}

ig_dtype = "input_group,+"
og_dtype = "output_group,+"

const_dict = {
    'out_pattern,+': "out_pattern",
    'in_pattern,+': "in_pattern",
    'color,#': 'enum',
    'shape,#': 'enum',
    'scale,#': 'enum',
    'input_group,+': 'amount_ie',
    'output_group,+': 'amount_oe',
    'number,#': 'amount_num',

}

color = [f'color_{i}' for i in range(1, 11)]
scale = [f'scale_{i}' for i in range(1, 11)]
shape = ['line', 'rectangle']

attr_names = ['color', 'shape', 'scale', 'rho', 'phi',
              'group_shape', "slope", 'number']

inv_p_head = {
    "input": "inv_i_p",
    "output": "inv_o_p",
    "input_output": "inv_io_p"
}

neighbor_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
neighbor_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
