# Created by jing at 11.03.25
import matplotlib

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

var_dtypes = {
    "pattern": "pattern",
    "group": "group_data",
    "object": "object",
    "group_objects": "group_data"
}

const_dtypes = {
    "object_color": "object_color",
    "object_shape": "object_shape",
    "group_label": "group_label",
    "object_num": "object_num",
}

pred_names = {
    "target": "target",
    "in_group": "in_group",
    "in_pattern": "in_pattern",
    "has_color": "exist_obj_color",
    "has_shape": "exist_obj_shape",
    "has_size": "has_size",
    "same_size":"same_size",
    "same_shape":"same_shape",
    "same_color":"same_color",
    "group_shape": "exist_group_shape",
    "object_num": "has_obj_num",
    "symmetry_color": "symmetry_color",
    "symmetry_shape": "symmetry_shape",

}

attr_names = ['object_color', 'object_shape', 'object_num', "group_label"]

bk_shapes = ["none", "triangle", "square", "circle"]


def get_matplotlib_colors():
    color_matplotlib = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
                        for k, v in list(matplotlib.colors.cnames.items())}
    color_matplotlib.pop("darkslategray")
    color_matplotlib.pop("lightslategray")
    color_matplotlib.pop("black")
    color_matplotlib.pop("darkgray")
    return color_matplotlib

color_matplotlib = get_matplotlib_colors()
const_dtype_object_color = const_dtypes["object_color"]
const_dtype_object_shape = const_dtypes["object_shape"]

var_dtype_group = var_dtypes["group"]
const_dtype_obj_num = const_dtypes["object_num"]
const_dtype_group = const_dtypes["group_label"]
