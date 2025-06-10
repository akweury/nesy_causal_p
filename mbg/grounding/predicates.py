# Created by MacBook Pro at 13.05.25


# mbg/predicates.py

from typing import List, Dict, Any, Callable, Tuple
import torch
from src import bk
from mbg.patch_preprocess import shift_obj_patches_to_global_positions

# these are the only two “head” predicates we ever learn clauses for:
HEAD_PREDICATES = {
    "image": "image_target",  # heads of form image_target(X)
    "group": "group_target"  # heads of form group_target(G, X)
}

# Registries
OBJ_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
GRP_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
SOFT: Dict[str, Callable[..., torch.Tensor]] = {}


# Decorators for registration
def object_predicate(name: str):
    def dec(fn: Callable[..., torch.Tensor]):
        OBJ_HARD[name] = fn
        return fn

    return dec


def group_predicate(name: str):
    def dec(fn: Callable[..., torch.Tensor]):
        GRP_HARD[name] = fn
        return fn

    return dec


def soft_predicate(name: str):
    def dec(fn: Callable[..., torch.Tensor]):
        SOFT[name] = fn
        return fn

    return dec


# mbg/language/predicates_eval.py
import torch
from typing import Dict, Any, Tuple


#
# ————————————————
# Object­-level predicates
# ————————————————
#
def has_shape_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: int
) -> torch.Tensor:
    """
    hard['has_shape']: Tensor[O] of ints
    a2: the target shape‐code
    => returns Tensor[O] with 1.0 where shape==a2
    """
    return (hard["has_shape"] == a2).float()


def has_color_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: Tuple[int, int, int]
) -> torch.Tensor:
    """
    hard['has_color']: Tensor[O,3]
    a2: (r,g,b)
    => returns Tensor[O] with 1.0 where all three channels match
    """
    colors = hard["has_color"]  # (O,3)
    target = torch.tensor(a2, device=colors.device)
    try:
        res = (colors == target).all(dim=1).float()
    except RuntimeError:
        raise RuntimeError

    return res


def x_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: float,
        tol: float = 1e-3
) -> torch.Tensor:
    return torch.isclose(hard["x"], torch.tensor(a2, device=hard["x"].device), atol=tol).float()


def y_eval(hard, soft, _, a2: float, tol: float = 1e-3) -> torch.Tensor:
    return torch.isclose(hard["y"], torch.tensor(a2, device=hard["y"].device), atol=tol).float()


def w_eval(hard, soft, _, a2: float, tol: float = 1e-3) -> torch.Tensor:
    return torch.isclose(hard["w"], torch.tensor(a2, device=hard["w"].device), atol=tol).float()


def h_eval(hard, soft, _, a2: float, tol: float = 1e-3) -> torch.Tensor:
    return torch.isclose(hard["h"], torch.tensor(a2, device=hard["h"].device), atol=tol).float()


def in_group_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _,
        __
) -> torch.Tensor:
    """
    Return the full O×G membership matrix.
    downstream evaluators (_eval_image, _body_to_group_mats, etc.)
    will handle existential / universal reduction appropriately.
    """
    return hard["in_group"].float()


def group_num_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: int
) -> torch.Tensor:
    return torch.tensor(len(hard["group_size"]) == a2).float()


#
# ————————————————
# Group­-level predicates
# ————————————————
#
def group_size_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: int
) -> torch.Tensor:
    return (hard["group_size"] == a2).float()


def principle_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _: Any,
        a2: int
) -> torch.Tensor:
    return (hard["principle"] == a2).float()


#
# ————————————————
# Soft (neural) predicates
# ————————————————
#
def prox_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _,
        __
) -> torch.Tensor:
    # returns the full O×O proximity matrix
    return soft["prox"]


def grp_sim_eval(
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        _,
        __
) -> torch.Tensor:
    # returns the full G×G group‐similarity matrix
    return soft["grp_sim"]


#
# ————————————————————————————
# Object-level: not_has_shape_* predicates
# ————————————————————————————
#
def not_has_shape_eval_factory(target_shape_id: int):
    def _eval(hard: Dict[str, torch.Tensor],
              soft: Dict[str, torch.Tensor],
              _: Any,
              __: Any) -> torch.Tensor:
        pred_name = "not_has_shape_" + bk.bk_shapes[target_shape_id + 1]
        return (hard[pred_name].all()).float()

    return _eval


# 注册以下形状的 not_has_shape 评估器
not_has_shape_rectangle_eval = not_has_shape_eval_factory(target_shape_id=bk.rect_index)
not_has_shape_circle_eval = not_has_shape_eval_factory(target_shape_id=bk.cir_index)
not_has_shape_triangle_eval = not_has_shape_eval_factory(target_shape_id=bk.tri_index)


def make_no_member_shape_eval(shape_idx: int):
    def _eval(hard: Dict[str, torch.Tensor],
              soft: Dict[str, torch.Tensor],
              _: Any,
              __: Any) -> torch.Tensor:
        in_group = hard["in_group"]  # (O, G)
        has_shape = hard["has_shape"]  # (O,)
        O, G = in_group.shape
        result = []

        for g in range(G):
            member_mask = in_group[:, g].bool()
            group_shapes = has_shape[member_mask]
            all_not_match = (group_shapes != shape_idx).all()
            result.append(float(all_not_match))

        return torch.tensor(result, dtype=torch.float, device=has_shape.device)

    return _eval


no_member_shape_triangle_eval = make_no_member_shape_eval(shape_idx=bk.tri_index)
no_member_shape_square_eval = make_no_member_shape_eval(shape_idx=bk.rect_index)
no_member_shape_circle_eval = make_no_member_shape_eval(shape_idx=bk.cir_index)


# Symmetry-related predicates
def same_shape_eval(hard, soft, _, __):
    return hard["same_shape"]


def same_color_eval(hard, soft, _, __):
    return hard["same_color"]


def mirror_x_eval(hard, soft, _, __):
    return hard["mirror_x"]


def same_y_eval(hard, soft, _, __):
    return hard["same_y"]


def _group_diversity_eval(hard: Dict[str, torch.Tensor], key: str, tol: float = 1e-4) -> torch.Tensor:
    """
    Generic group diversity evaluation.
    key: one of 'has_shape', 'has_color', 'size'
    Returns: Tensor[G], where 1.0 means ≥2 unique values in group
    """
    in_group = hard["in_group"]  # (O, G)
    values = hard[key]  # (O,) or (O,3) or (O,1)

    O, G = in_group.shape
    results = []

    for g in range(G):
        results.append(values[g] == 1)
        # member_mask = in_group[:, g].bool()
        # group_vals = values[member_mask]
        # if group_vals.ndim == 1:
        #     unique = torch.unique(group_vals)
        # else:
        #     unique = torch.unique(group_vals, dim=0)
        # results.append(float(len(unique) >= 2))

    return torch.tensor(results, device=values.device)


def _group_uniqueness_eval(hard: Dict[str, torch.Tensor], key: str, tol: float = 1e-4) -> torch.Tensor:
    """
    Generic group uniqueness evaluation.
    Returns: Tensor[G], where 1.0 means exactly 1 unique value in group
    """
    in_group = hard["in_group"]
    values = hard[key]

    O, G = in_group.shape
    results = []

    for g in range(G):
        results.append(values[g] == 1)
        # member_mask = in_group[:, g].bool()
        # group_vals = values[member_mask]
        # if group_vals.ndim == 1:
        #     unique = torch.unique(group_vals)
        # else:
        #     unique = torch.unique(group_vals, dim=0)
        # results.append(float(len(unique) == 1))

    return torch.tensor(results, device=values.device)


# Actual evaluation function bindings
def diverse_shapes_eval(hard, soft, _, __): return _group_diversity_eval(hard, "diverse_shapes")


def diverse_colors_eval(hard, soft, _, __): return _group_diversity_eval(hard, "diverse_colors")


def diverse_sizes_eval(hard, soft, _, __):  return _group_diversity_eval(hard, "diverse_sizes")


def unique_shapes_eval(hard, soft, _, __):  return _group_uniqueness_eval(hard, "unique_shapes")


def unique_colors_eval(hard, soft, _, __):  return _group_uniqueness_eval(hard, "unique_colors")


def unique_sizes_eval(hard, soft, _, __):   return _group_uniqueness_eval(hard, "unique_sizes")
