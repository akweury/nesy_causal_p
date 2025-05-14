# Created by MacBook Pro at 13.05.25


# mbg/predicates.py

from typing import List, Dict, Any, Callable, Tuple
import torch
from mbg.patch_preprocess import shift_obj_patches_to_global_positions

# these are the only two “head” predicates we ever learn clauses for:
HEAD_PREDICATES = {
    "image": "image_target",   # heads of form image_target(X)
    "group": "group_target"    # heads of form group_target(G, X)
}

# Registries
OBJ_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
GRP_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
SOFT:    Dict[str, Callable[..., torch.Tensor]] = {}

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


#
# === Object‐level hard predicates ===
#

@object_predicate("has_shape")
def has_shape(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"]["shape"] for o in objects],
        dtype=torch.long, device=device
    )

@object_predicate("has_color")
def has_color(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"]["color"] for o in objects],
        dtype=torch.long, device=device
    )

@object_predicate("x")
def pos_x(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"]["x"] for o in objects],
        dtype=torch.float, device=device
    )

@object_predicate("y")
def pos_y(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"]["y"] for o in objects],
        dtype=torch.float, device=device
    )

@object_predicate("w")
def width(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"].get("w", 0) for o in objects],
        dtype=torch.float, device=device
    )

@object_predicate("h")
def height(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor(
        [o["s"].get("h", 0) for o in objects],
        dtype=torch.float, device=device
    )

@object_predicate("in_group")
def in_group(objects: List[Dict[str, Any]], groups: List[Dict[str, Any]], device="cpu"):
    O, G = len(objects), len(groups)
    mat = torch.zeros((O, G), dtype=torch.bool, device=device)
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    grp2idx = {g["id"]: i for i, g in enumerate(groups)}
    for g in groups:
        gi = grp2idx[g["id"]]
        for member in g["members"]:
            mat[obj2idx[member["id"]], gi] = True
    return mat


#
# === Group‐level hard predicates ===
#

@group_predicate("group_size")
def group_size(objects=None, groups: List[Dict[str, Any]]=None, device="cpu"):
    return torch.tensor(
        [len(g["members"]) for g in groups],
        dtype=torch.float, device=device
    )

@group_predicate("principle")
def principle(objects=None, groups: List[Dict[str, Any]]=None, device="cpu"):
    # caller must set princ_map on their side
    return torch.tensor(
        [PRINC_MAP[g["principle"]] for g in groups],
        dtype=torch.long, device=device
    )


#
# === Soft (neural) predicates ===
#

@soft_predicate("prox")
def object_proximity(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    O = len(objects)
    if O == 0:
        return torch.empty((0, 0), device=device)
    embs = []
    for o in objects:
        contour, origin = o["h"][0]
        pts = torch.tensor(contour, device=device)
        org = torch.tensor(origin, device=device)
        emb = shift_obj_patches_to_global_positions(pts, org).flatten()
        embs.append(emb)
    H = torch.stack(embs, dim=0)
    H = H / (H.norm(dim=1, keepdim=True) + 1e-8)
    return H @ H.t()

@soft_predicate("grp_sim")
def group_similarity(objects=None, groups: List[Dict[str, Any]]=None, device="cpu"):
    G = len(groups)
    if G == 0:
        return torch.empty((0, 0), device=device)
    H = torch.stack([g["h"] for g in groups], dim=0)
    H = H / (H.norm(dim=1, keepdim=True) + 1e-8)
    return H @ H.t()


# constant for principle‐map
PRINC_MAP = {"proximity": 0, "similarity": 1, "closure": 2, "symmetry": 3}