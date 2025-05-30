# Created by MacBook Pro at 16.05.25


# mbg/predicates.py

import torch
from typing import List, Dict, Any, Tuple, Callable
from src import bk
from mbg.patch_preprocess import shift_obj_patches_to_global_positions

# Head predicates (we learn clauses for these)
HEAD_PREDICATES = {
    "image": "image_target",
    "group": "group_target"
}

# Registries
OBJ_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
GRP_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
SOFT: Dict[str, Callable[..., torch.Tensor]] = {}


def object_predicate(name: str):
    def dec(fn): OBJ_HARD[name] = fn; return fn

    return dec


def group_predicate(name: str):
    def dec(fn): GRP_HARD[name] = fn; return fn

    return dec


def soft_predicate(name: str):
    def dec(fn): SOFT[name] = fn; return fn

    return dec


# --- Object‐level hard predicates ---

@object_predicate("has_shape")
def has_shape(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects],
                        dtype=torch.long, device=device)


@object_predicate("has_color")
def has_color(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor([o["s"]["color"] for o in objects],
                        dtype=torch.long, device=device)


@object_predicate("x")
def pos_x(objects, groups=None, device="cpu"):
    return torch.tensor([o["s"]["x"] for o in objects],
                        dtype=torch.float, device=device)


@object_predicate("y")
def pos_y(objects, groups=None, device="cpu"):
    return torch.tensor([o["s"]["y"] for o in objects],
                        dtype=torch.float, device=device)


@object_predicate("w")
def width(objects, groups=None, device="cpu"):
    return torch.tensor([o["s"].get("w", 0) for o in objects],
                        dtype=torch.float, device=device)


@object_predicate("h")
def height(objects, groups=None, device="cpu"):
    return torch.tensor([o["s"].get("h", 0) for o in objects],
                        dtype=torch.float, device=device)


@object_predicate("in_group")
def in_group(objects: List[Dict], groups: List[Dict], device="cpu"):
    O, G = len(objects), len(groups)
    mat = torch.zeros((O, G), dtype=torch.float, device=device)
    obj2i = {o["id"]: i for i, o in enumerate(objects)}
    grp2i = {g["id"]: i for i, g in enumerate(groups)}
    for g in groups:
        gi = grp2i[g["id"]]
        for m in g["members"]:
            mat[obj2i[m["id"]], gi] = 1.0
    return mat


def make_not_has_shape_pred(shape_name: str, shape_idx: int):
    @object_predicate(f"not_has_shape_{shape_name}")
    def _pred(objects: List[Dict[str, Any]], groups=None, device="cpu"):
        shape_ids = torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects],
                                 dtype=torch.long, device=device)
        return (shape_ids != shape_idx).float()

    return _pred


# Register all three
make_not_has_shape_pred("rectangle", bk.bk_shapes.index("rectangle") - 1)
make_not_has_shape_pred("circle", bk.bk_shapes.index("circle") - 1)
make_not_has_shape_pred("triangle", bk.bk_shapes.index("triangle") - 1)


# --- Group‐level hard predicates ---

@group_predicate("group_size")
def group_size(objects=None, groups=None, device="cpu"):
    return torch.tensor([len(g["members"]) for g in groups],
                        dtype=torch.float, device=device)


@group_predicate("principle")
def principle(objects=None, groups=None, device="cpu"):
    PRINC_MAP = {"proximity": 0, "similarity": 1, "closure": 2, "symmetry": 3}
    return torch.tensor([PRINC_MAP[g["principle"]] for g in groups],
                        dtype=torch.float, device=device)




def make_no_member_shape_pred(shape_name: str, shape_idx: int):
    @group_predicate(f"no_member_{shape_name}")
    def _pred(objects: List[Dict], groups: List[Dict], device="cpu"):
        obj2idx = {o["id"]: i for i, o in enumerate(objects)}
        shape_ids = torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects], dtype=torch.long, device=device)
        results = []
        for g in groups:
            member_ids = [obj2idx[m["id"]] for m in g["members"]]
            member_shapes = shape_ids[member_ids]
            results.append(torch.all(member_shapes != shape_idx))
        return torch.stack(results).to(torch.float)
    return _pred

for i, shape in enumerate(bk.bk_shapes[1:]):
    make_no_member_shape_pred(shape, i)  # adjust if your shape encoding starts at -1


# @group_predicate("no_member_triangle")
# def no_member_triangle(objects: List[Dict], groups: List[Dict], device="cpu"):
#     """
#     Returns True if no member of the group has shape == triangle (index 2).
#     """
#     obj2idx = {o["id"]: i for i, o in enumerate(objects)}
#     shape_ids = torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects], dtype=torch.long, device=device)
#     results = []
#
#     for g in groups:
#         member_ids = [obj2idx[m["id"]] for m in g["members"]]
#         member_shapes = shape_ids[member_ids]
#         results.append(torch.all(member_shapes != (bk.bk_shapes.index("triangle") - 1)))
#
#     return torch.stack(results).to(torch.float)


@group_predicate("diverse_shapes")
def diverse_shapes(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if a group contains at least two different shapes.
    """
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    shape_ids = torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects], dtype=torch.long, device=device)
    results = []

    for g in groups:
        member_ids = [obj2idx[m["id"]] for m in g["members"]]
        unique_shapes = torch.unique(shape_ids[member_ids])
        results.append(len(unique_shapes) > 1)

    return torch.tensor(results, dtype=torch.float, device=device)
