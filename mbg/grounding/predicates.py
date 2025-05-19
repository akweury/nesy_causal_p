# Created by MacBook Pro at 13.05.25


# mbg/predicates.py

from typing import List, Dict, Any, Callable, Tuple
import torch
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


# def in_group_eval(
#     hard: Dict[str, torch.Tensor],
#     soft: Dict[str, torch.Tensor],
#     _: Any,
#     a2: str
# ) -> torch.Tensor:
#     """
#     hard['in_group']: Tensor[O,G]
#     a2: e.g. 'g2' or integer index
#     => returns Tensor[O] = column g2 of in_group
#     """
#     if isinstance(a2, str) and a2.startswith("g"):
#         gi = int(a2[1:])
#     else:
#         gi = int(a2)
#     return hard["in_group"][:, gi].float()
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
