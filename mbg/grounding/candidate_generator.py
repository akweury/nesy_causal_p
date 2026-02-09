"""
candidate_generator.py

Generate candidate perceptual groups for real images (e.g. COCO).
This module proposes multiple grouping hypotheses based on
simple perceptual cues (proximity, depth, category, ownership).

Design principles:
- No learning
- No scoring / ranking
- Allow overlapping groups
- Multiple hypotheses per image
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from mbg.grounding.predicates_real import (
    ObjectInstance,
    Group,
    bbox_center,
    euclidean,
    depth_bin,
)


# =========================
# Helper utilities
# =========================

def compute_adaptive_depth_bins(objects: Dict[int, ObjectInstance]) -> Tuple[float, float]:
    """
    Compute adaptive depth bin thresholds based on the actual depth range in the image.
    
    Args:
        objects: Dictionary of ObjectInstance objects
    
    Returns:
        Tuple of (near_mid_threshold, mid_far_threshold)
    """
    if not objects:
        return (1.5, 4.0)  # Default fallback
    
    depths = [obj.depth for obj in objects.values()]
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    depth_range = max_depth - min_depth
    
    # If depth range is very small, use default bins
    if depth_range < 0.1:
        return (1.5, 4.0)
    
    # Split into thirds: near (0-33%), mid (33-67%), far (67-100%)
    near_mid_threshold = min_depth + depth_range / 3.0
    mid_far_threshold = min_depth + 2.0 * depth_range / 3.0
    
    return (near_mid_threshold, mid_far_threshold)


def pairwise_distance(o1: ObjectInstance, o2: ObjectInstance, norm=1.0) -> float:
    c1 = bbox_center(o1.bbox) / norm
    c2 = bbox_center(o2.bbox) / norm
    return euclidean(c1, c2)


def same_depth(o1: ObjectInstance, o2: ObjectInstance, bins: Optional[Tuple[float, float]] = None) -> bool:
    return depth_bin(o1, bins) == depth_bin(o2, bins)


# =========================
# Group generators
# =========================

def proximity_groups(
    objects: Dict[int, ObjectInstance],
    dist_thresh: float = 0.1,
    norm: float = 1.0,
) -> List[Group]:
    """
    Generate groups based on spatial proximity (connected components).
    """
    visited = set()
    groups = []
    gid = 0
    oids = list(objects.keys())

    for oid in oids:
        if oid in visited:
            continue

        stack = [oid]
        component = set()

        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            visited.add(cur)

            for other in oids:
                if other in component:
                    continue
                d = pairwise_distance(objects[cur], objects[other], norm)
                if d < dist_thresh:
                    stack.append(other)

        if len(component) >= 2:
            groups.append(Group(gid=f"prox_{gid}", members=list(component)))
            gid += 1

    return groups


def depth_groups(
    objects: Dict[int, ObjectInstance],
    adaptive_bins: Optional[Tuple[float, float]] = None
) -> List[Group]:
    """
    Generate groups by depth bin (near / mid / far).
    Uses adaptive depth bins based on the actual depth range in the image.
    
    Args:
        objects: Dictionary of ObjectInstance objects
        adaptive_bins: Optional tuple of (near_mid_threshold, mid_far_threshold).
                      If None, will be computed from objects.
    """
    if adaptive_bins is None:
        adaptive_bins = compute_adaptive_depth_bins(objects)
    
    bins: Dict[str, List[int]] = {"near": [], "mid": [], "far": []}

    for oid, obj in objects.items():
        bins[depth_bin(obj, adaptive_bins)].append(oid)

    groups = []
    for k, members in bins.items():
        if len(members) >= 2:
            groups.append(Group(gid=f"depth_{k}", members=members))

    return groups


def person_centered_groups(
    objects: Dict[int, ObjectInstance],
    dist_thresh: float = 0.2,
    norm: float = 1.0,
) -> List[Group]:
    """
    Generate groups centered around each person object
    (ownership / interaction hypothesis).
    """
    groups = []
    gid = 0

    persons = [o for o in objects.values() if o.category == "person"]

    for p in persons:
        members = [p.oid]
        for oid, obj in objects.items():
            if oid == p.oid:
                continue
            d = pairwise_distance(p, obj, norm)
            if d < dist_thresh:
                members.append(oid)

        if len(members) >= 2:
            groups.append(Group(gid=f"person_{gid}", members=members))
            gid += 1

    return groups


def category_groups(
    objects: Dict[int, ObjectInstance]
) -> List[Group]:
    """
    Generate groups by shared category (weak hypothesis).
    """
    cat_bins: Dict[str, List[int]] = {}

    for oid, obj in objects.items():
        cat_bins.setdefault(obj.category, []).append(oid)

    groups = []
    for cat, members in cat_bins.items():
        if len(members) >= 2:
            groups.append(Group(gid=f"cat_{cat}", members=members))

    return groups


# =========================
# Main API
# =========================

def generate_candidate_groups(
    objects: Dict[int, ObjectInstance],
    image_diag_norm: float = 1.0,
) -> List[Group]:
    """
    Generate a diverse set of candidate grouping hypotheses.
    Automatically computes adaptive depth bins based on the depth range in the image.

    Returns:
        List[Group]: possibly overlapping candidate groups
    """
    groups: List[Group] = []
    
    # Compute adaptive depth bins for this image
    adaptive_bins = compute_adaptive_depth_bins(objects)
    
    # Print depth range info for debugging
    if objects:
        depths = [obj.depth for obj in objects.values()]
        print(f"  Depth range: [{np.min(depths):.2f}, {np.max(depths):.2f}]")
        print(f"  Adaptive bins: near<{adaptive_bins[0]:.2f}, mid<{adaptive_bins[1]:.2f}, far>={adaptive_bins[1]:.2f}")

    groups += proximity_groups(objects, norm=image_diag_norm)
    groups += depth_groups(objects, adaptive_bins=adaptive_bins)
    groups += person_centered_groups(objects, norm=image_diag_norm)
    groups += category_groups(objects)

    return groups
