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


@object_predicate("same_shape")
def same_shape(objects: List[Dict], groups=None, device="cpu"):
    O = len(objects)
    shape_ids = torch.tensor([o["s"]["shape"].argmax() - 1 for o in objects], device=device)
    return (shape_ids.unsqueeze(0) == shape_ids.unsqueeze(1)).float()


@object_predicate("same_color")
def same_color(objects: List[Dict], groups=None, device="cpu"):
    O = len(objects)
    colors = torch.tensor([o["s"]["color"] for o in objects], dtype=torch.float32, device=device)
    diff = colors.unsqueeze(1) - colors.unsqueeze(0)
    return (diff.norm(dim=2) < 1e-3).float()


@object_predicate("same_size")
def same_size(objects: List[Dict], groups=None, device="cpu"):
    O = len(objects)
    widths = torch.tensor([o["s"].get("w", 0.0) for o in objects], dtype=torch.float32, device=device)
    heights = torch.tensor([o["s"].get("h", 0.0) for o in objects], dtype=torch.float32, device=device)
    sizes = widths * heights  # Tensor[O]

    # Compute pairwise difference matrix
    diff = sizes.unsqueeze(1) - sizes.unsqueeze(0)
    return (diff.abs() < 1e-2).float()


@object_predicate("mirror_x")
def mirror_x(objects: List[Dict], groups=None, device="cpu"):
    x = torch.tensor([o["s"]["x"] for o in objects], dtype=torch.float32, device=device)
    mirror = torch.abs(x.unsqueeze(0) + x.unsqueeze(1) - 1.0) < 0.05
    return mirror.float()


@object_predicate("same_y")
def same_y(objects: List[Dict], groups=None, device="cpu"):
    y = torch.tensor([o["s"]["y"] for o in objects], dtype=torch.float32, device=device)
    return (torch.abs(y.unsqueeze(0) - y.unsqueeze(1)) < 0.03).float()


# --- Group‐level hard predicates ---

@group_predicate("group_size")
def group_size(objects=None, groups=None, device="cpu"):
    return torch.tensor([len(g["members"]) for g in groups],
                        dtype=torch.float, device=device)


@group_predicate("principle")
def principle(objects=None, groups=None, device="cpu"):
    PRINC_MAP = {"proximity": 0, "similarity": 1, "closure": 2, "symmetry": 3, "continuity": 4}
    return torch.tensor([PRINC_MAP[g["principle"]] for g in groups],
                        dtype=torch.float, device=device)


def make_no_member_shape_pred(shape_name: str, shape_idx: int):
    @group_predicate(f"no_member_{shape_name}")
    def _pred(objects: List[Dict], groups: List[Dict], device="cpu"):
        if len(groups) == 0:
            return torch.tensor([])
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


@group_predicate("unique_shapes")
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
        results.append(len(unique_shapes) == 1)

    return torch.tensor(results, dtype=torch.float, device=device)


@group_predicate("diverse_colors")
def diverse_colors(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if a group contains at least two different colors.
    """
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    color_ids = torch.tensor([tuple(o["s"]["color"]) for o in objects], dtype=torch.int, device=device)
    results = []

    for g in groups:
        member_ids = [obj2idx[m["id"]] for m in g["members"]]
        group_colors = color_ids[member_ids]
        unique_colors = torch.unique(group_colors, dim=0)
        results.append(len(unique_colors) > 1)

    return torch.tensor(results, dtype=torch.float, device=device)


@group_predicate("unique_colors")
def unique_colors(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if a group contains at least two different colors.
    """
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    color_ids = torch.tensor([tuple(o["s"]["color"]) for o in objects], dtype=torch.int, device=device)
    results = []

    for g in groups:
        member_ids = [obj2idx[m["id"]] for m in g["members"]]
        group_colors = color_ids[member_ids]
        unique_colors = torch.unique(group_colors, dim=0)
        results.append(len(unique_colors) == 1)

    return torch.tensor(results, dtype=torch.float, device=device)


@group_predicate("diverse_sizes")
def diverse_sizes(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if a group contains at least two different sizes.
    Size is defined here as the product of width and height.
    """
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    widths = torch.tensor([o["s"].get("w", 0) for o in objects], dtype=torch.float, device=device)
    heights = torch.tensor([o["s"].get("h", 0) for o in objects], dtype=torch.float, device=device)
    sizes = widths * heights
    results = []

    for g in groups:
        member_ids = [obj2idx[m["id"]] for m in g["members"]]
        group_sizes = sizes[member_ids]
        unique_sizes = torch.unique(group_sizes, dim=0)
        results.append(len(unique_sizes) > 1)

    return torch.tensor(results, dtype=torch.float, device=device)


@group_predicate("unique_sizes")
def unique_sizes(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if a group contains at least two different sizes.
    Size is defined here as the product of width and height.
    """
    obj2idx = {o["id"]: i for i, o in enumerate(objects)}
    widths = torch.tensor([o["s"].get("w", 0) for o in objects], dtype=torch.float, device=device)
    heights = torch.tensor([o["s"].get("h", 0) for o in objects], dtype=torch.float, device=device)
    sizes = widths * heights
    results = []

    for g in groups:
        member_ids = [obj2idx[m["id"]] for m in g["members"]]
        group_sizes = sizes[member_ids]
        unique_sizes = torch.unique(group_sizes, dim=0)
        results.append(len(unique_sizes) == 1)

    return torch.tensor(results, dtype=torch.float, device=device)


@group_predicate("same_group_counts")
def same_group_counts(objects: List[Dict], groups: List[Dict], device="cpu"):
    """
    Returns True if the group has a different number of members than at least one other group in the same image.
    """
    group_sizes = torch.tensor([len(g["members"]) for g in groups], dtype=torch.long, device=device)
    results = len(group_sizes.unique()) == 1
    return torch.tensor(results, dtype=torch.float, device=device)


@soft_predicate("sim_color_soft")
def sim_color_soft(objects=None, groups=None, device="cpu", soft=None, sigma=20.0):
    """
    Compare average RGB color per object.
    Returns [O, O] soft similarity matrix.
    """
    embeddings = torch.stack([o["h"] for o in objects])  # [O, 6, 16, 7]
    color = embeddings[:, :, :, 2:5]  # [O, 6, 16, 3]
    avg_color = color.mean(dim=(1, 2))  # [O, 3]

    diff = avg_color.unsqueeze(1) - avg_color.unsqueeze(0)  # [O, O, 3]
    dist = diff.norm(dim=2)  # [O, O]
    sim = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
    return sim


@soft_predicate("sim_shape_soft")
def sim_shape_soft(objects=None, groups=None, device="cpu", soft=None, sigma=0.1):
    """
    Compare contour shape (normalized).
    Returns [O, O] shape similarity.
    """

    embeddings = torch.stack([o["h"] for o in objects])  # [O, 6, 16, 7]
    shape = embeddings[:, :, :, :2]  # [O, 6, 16, 2]
    shape = shape.reshape(shape.size(0), -1)  # [O, 6*16*2]
    shape = shape - shape.mean(dim=1, keepdim=True)  # mean-centering
    shape = shape / (shape.norm(dim=1, keepdim=True) + 1e-5)

    sim = shape @ shape.T  # cosine similarity
    return sim


@soft_predicate("sim_size_soft")
def sim_size_soft(objects=None, groups=None, device="cpu", soft=None, sigma=0.02):
    """
    Compare object sizes (w*h).
    """
    embeddings = torch.stack([o["h"] for o in objects])  # [O, 6, 16, 7]
    wh = embeddings[:, 0, 0, 5:7]  # Assuming width/height same across contour
    size = wh[:, 0] * wh[:, 1]  # [O]
    diff = size.unsqueeze(1) - size.unsqueeze(0)
    sim = torch.exp(- (diff ** 2) / (2 * sigma ** 2))
    return sim
