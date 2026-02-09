

"""
predicates_real.py

Grounding of symbolic predicates for real images (e.g. COCO).
This file defines object-level and group-level predicates that are
used by GRM symbolic reasoning over perceptual group hypotheses.

Design principles:
- No learning here
- Deterministic, interpretable predicates
- Built from bbox, depth, category, and candidate groups
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


# =========================
# Data structures
# =========================

@dataclass
class ObjectInstance:
    """
    Lightweight object representation for symbolic grounding.
    """
    oid: int
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    depth: float
    category: str


@dataclass
class Group:
    """
    A candidate perceptual group.
    """
    gid: str
    members: List[int]  # list of object ids


# =========================
# Utility functions
# =========================

def bbox_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0])


def bbox_area(bbox):
    _, _, w, h = bbox
    return w * h


def euclidean(p1, p2):
    return float(np.linalg.norm(p1 - p2))


# =========================
# Object-level predicates
# =========================

def depth_bin(obj: ObjectInstance, bins: Optional[Tuple[float, float]] = None):
    """
    Discretize depth into symbolic bins.
    
    Args:
        obj: ObjectInstance object
        bins: Optional tuple of (near_mid_threshold, mid_far_threshold).
              If None, uses default fixed bins (1.5, 4.0).
    """
    if bins is None:
        bins = (1.5, 4.0)
    
    if obj.depth < bins[0]:
        return "near"
    elif obj.depth < bins[1]:
        return "mid"
    else:
        return "far"


# =========================
# Group-level predicates
# =========================

def group_size(group: Group) -> int:
    return len(group.members)


def group_depth(
    group: Group,
    objects: Dict[int, ObjectInstance],
    bins: Optional[Tuple[float, float]] = None
) -> str:
    """
    Group depth = majority depth bin of its members.
    
    Args:
        group: Group object
        objects: Dictionary of ObjectInstance objects
        bins: Optional tuple of (near_mid_threshold, mid_far_threshold)
    """
    depth_bins = [depth_bin(objects[oid], bins) for oid in group.members]
    return max(set(depth_bins), key=depth_bins.count)


def group_mean_depth(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> float:
    return float(np.mean([objects[oid].depth for oid in group.members]))


def compact(
    group: Group,
    objects: Dict[int, ObjectInstance],
    norm: float = 1.0,
    thresh: float = 0.15
) -> bool:
    """
    A group is compact if average pairwise distance is small.
    """
    if len(group.members) <= 1:
        return True

    centers = [
        bbox_center(objects[oid].bbox) / norm
        for oid in group.members
    ]

    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(euclidean(centers[i], centers[j]))

    return float(np.mean(dists)) < thresh


def spread_out(
    group: Group,
    objects: Dict[int, ObjectInstance],
    norm: float = 1.0,
    thresh: float = 0.25
) -> bool:
    return not compact(group, objects, norm=norm, thresh=thresh)


def dominant_category(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> str:
    cats = [objects[oid].category for oid in group.members]
    return max(set(cats), key=cats.count)


def functional_group(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> bool:
    """
    Heuristic: functional if it contains a person
    and at least one non-person object close in space.
    """
    cats = [objects[oid].category for oid in group.members]
    if "person" not in cats:
        return False

    if len(group.members) <= 1:
        return False

    return True


def foreground(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> bool:
    """
    Foreground groups tend to be nearer to the camera.
    """
    return group_depth(group, objects) == "near"


def background(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> bool:
    return group_depth(group, objects) == "far"


# =========================
# Group–group relational predicates
# =========================

def in_front_of(
    g1: Group,
    g2: Group,
    objects: Dict[int, ObjectInstance]
) -> bool:
    return group_mean_depth(g1, objects) < group_mean_depth(g2, objects)


def more_salient(
    g1: Group,
    g2: Group,
    objects: Dict[int, ObjectInstance]
) -> bool:
    """
    Salience heuristic:
    foreground + compact + larger size wins.
    """
    score1 = 0
    score2 = 0

    if foreground(g1, objects):
        score1 += 1
    if foreground(g2, objects):
        score2 += 1

    if compact(g1, objects):
        score1 += 1
    if compact(g2, objects):
        score2 += 1

    if group_size(g1) > group_size(g2):
        score1 += 1
    elif group_size(g2) > group_size(g1):
        score2 += 1

    return score1 > score2


# =========================
# Debug / inspection helper
# =========================

def describe_group(
    group: Group,
    objects: Dict[int, ObjectInstance]
) -> Dict:
    """
    Return a symbolic summary of a group for logging or visualization.
    """
    return {
        "gid": group.gid,
        "size": group_size(group),
        "depth": group_depth(group, objects),
        "dominant_category": dominant_category(group, objects),
        "compact": compact(group, objects),
        "functional": functional_group(group, objects),
    }
    
