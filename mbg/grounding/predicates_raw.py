# Created by MacBook Pro at 16.05.25


# mbg/predicates.py

import torch
from typing import List, Dict, Any, Tuple, Callable
from mbg.patch_preprocess import shift_obj_patches_to_global_positions

# Head predicates (we learn clauses for these)
HEAD_PREDICATES = {
    "image": "image_target",
    "group": "group_target"
}

# Registries
OBJ_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
GRP_HARD: Dict[str, Callable[..., torch.Tensor]] = {}
SOFT:    Dict[str, Callable[..., torch.Tensor]] = {}

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
    return torch.tensor([ o["s"]["shape"] for o in objects],
                        dtype=torch.long, device=device)

@object_predicate("has_color")
def has_color(objects: List[Dict[str, Any]], groups=None, device="cpu"):
    return torch.tensor([ o["s"]["color"] for o in objects],
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
    return torch.tensor([ o["s"].get("w",0) for o in objects],
                        dtype=torch.float, device=device)

@object_predicate("h")
def height(objects, groups=None, device="cpu"):
    return torch.tensor([ o["s"].get("h",0) for o in objects],
                        dtype=torch.float, device=device)

@object_predicate("in_group")
def in_group(objects: List[Dict], groups: List[Dict], device="cpu"):
    O, G = len(objects), len(groups)
    mat = torch.zeros((O,G), dtype=torch.bool, device=device)
    obj2i = {o["id"]:i for i,o in enumerate(objects)}
    grp2i = {g["id"]:i for i,g in enumerate(groups)}
    for g in groups:
        gi = grp2i[g["id"]]
        for m in g["members"]:
            mat[obj2i[m["id"]], gi] = True
    return mat


# --- Group‐level hard predicates ---

@group_predicate("group_size")
def group_size(objects=None, groups=None, device="cpu"):
    return torch.tensor([len(g["members"]) for g in groups],
                        dtype=torch.float, device=device)

@group_predicate("principle")
def principle(objects=None, groups=None, device="cpu"):
    PRINC_MAP = {"proximity":0, "similarity":1, "closure":2, "symmetry":3}
    return torch.tensor([PRINC_MAP[g["principle"]] for g in groups],
                        dtype=torch.long, device=device)


# --- Neural (soft) predicates ---

@soft_predicate("prox")
def object_proximity(objects: List[Dict], groups=None, device="cpu"):
    if not objects:
        return torch.empty(0,0,device=device)
    embs = []
    for o in objects:
        cont, org = o["h"][0]
        e = shift_obj_patches_to_global_positions(torch.tensor(cont,device=device),
                                                  torch.tensor(org, device=device)
                                                 ).flatten()
        embs.append(e)
    H = torch.stack(embs,0)
    H = H / (H.norm(1,keepdim=True)+1e-8)
    return H @ H.t()

@soft_predicate("grp_sim")
def group_similarity(objects=None, groups=None, device="cpu"):
    if not groups:
        return torch.empty(0,0,device=device)
    H = torch.stack([g["h"] for g in groups],0)
    H = H / (H.norm(1,keepdim=True)+1e-8)
    return H @ H.t()