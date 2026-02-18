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
    bbox_area,
    euclidean,
    depth_bin,
)


# =========================
# Helper utilities
# =========================

def compute_adaptive_depth_bins(objects: Dict[int, ObjectInstance]) -> Tuple[float, float, float, float]:
    """
    Compute adaptive depth bin thresholds based on the actual depth range in the image.
    Splits depth into 5 levels: very_near, near, mid, far, very_far
    
    Args:
        objects: Dictionary of ObjectInstance objects
    
    Returns:
        Tuple of (very_near_threshold, near_threshold, mid_threshold, far_threshold)
    """
    if not objects:
        return (1.0, 2.0, 3.0, 4.5)  # Default fallback
    
    depths = [obj.depth for obj in objects.values()]
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    depth_range = max_depth - min_depth
    
    # If depth range is very small, use default bins
    if depth_range < 0.1:
        return (1.0, 2.0, 3.0, 4.5)
    
    # Split into fifths: very_near (0-20%), near (20-40%), mid (40-60%), far (60-80%), very_far (80-100%)
    very_near_threshold = min_depth + depth_range * 0.2
    near_threshold = min_depth + depth_range * 0.4
    mid_threshold = min_depth + depth_range * 0.6
    far_threshold = min_depth + depth_range * 0.8
    
    return (very_near_threshold, near_threshold, mid_threshold, far_threshold)


def pairwise_distance(o1: ObjectInstance, o2: ObjectInstance, norm=1.0) -> float:
    c1 = bbox_center(o1.bbox) / norm
    c2 = bbox_center(o2.bbox) / norm
    return euclidean(c1, c2)


def same_depth(o1: ObjectInstance, o2: ObjectInstance, bins: Optional[Tuple[float, float, float, float]] = None) -> bool:
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
            groups.append(Group(gid=f"prox_{gid}", members=list(component),
                                gtype="prox"))
            gid += 1

    return groups


def depth_groups(
    objects: Dict[int, ObjectInstance],
    adaptive_bins: Optional[Tuple[float, float, float, float]] = None
) -> List[Group]:
    """
    Generate groups by depth bin (very_near / near / mid / far / very_far).
    Uses adaptive depth bins based on the actual depth range in the image.
    
    Args:
        objects: Dictionary of ObjectInstance objects
        adaptive_bins: Optional tuple of (very_near_threshold, near_threshold, mid_threshold, far_threshold).
                      If None, will be computed from objects.
    """
    if adaptive_bins is None:
        adaptive_bins = compute_adaptive_depth_bins(objects)
    
    bins: Dict[str, List[int]] = {"very_near": [], "near": [], "mid": [], "far": [], "very_far": []}

    for oid, obj in objects.items():
        bins[depth_bin(obj, adaptive_bins)].append(oid)

    groups = []
    for k, members in bins.items():
        if len(members) >= 2:
            groups.append(Group(gid=f"depth_{k}", members=members, gtype=f"depth"))

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
            groups.append(Group(gid=f"cat_{cat}", members=members, gtype=f"cat"))

    return groups


def size_similarity_groups(
    objects: Dict[int, ObjectInstance],
    size_tolerance: float = 0.25,
    aspect_ratio_tolerance: float = 0.25,
) -> List[Group]:
    """
    Generate groups based on similar bounding box sizes and aspect ratios.
    Objects with similar sizes AND aspect ratios are grouped together.
    
    Args:
        objects: Dictionary of ObjectInstance objects
        size_tolerance: Size similarity tolerance (default 0.25 means 75%-125% range)
        aspect_ratio_tolerance: Aspect ratio tolerance (default 0.25 means 75%-125% range)
    
    Returns:
        List of size-based groups
    """
    if len(objects) < 2:
        return []
    
    # Calculate sizes and aspect ratios for all objects
    obj_sizes = {oid: bbox_area(obj.bbox) for oid, obj in objects.items()}
    obj_aspect_ratios = {}
    for oid, obj in objects.items():
        _, _, w, h = obj.bbox
        obj_aspect_ratios[oid] = w / (h + 1e-6)  # width/height ratio
    
    # Sort objects by size
    sorted_objs = sorted(obj_sizes.items(), key=lambda x: x[1])
    
    visited = set()
    groups = []
    gid = 0
    
    # Group objects with similar sizes AND aspect ratios using connected components
    for oid, size in sorted_objs:
        if oid in visited:
            continue
        
        # Start a new group
        component = [oid]
        visited.add(oid)
        
        # Find all objects with similar size AND aspect ratio to any in current component
        # Use a queue to allow transitive grouping
        queue = [oid]
        
        while queue:
            current_oid = queue.pop(0)
            current_size = obj_sizes[current_oid]
            current_aspect = obj_aspect_ratios[current_oid]
            
            for other_oid, other_size in obj_sizes.items():
                if other_oid in visited:
                    continue
                
                other_aspect = obj_aspect_ratios[other_oid]
                
                # Check if sizes are within tolerance
                size_ratio = other_size / (current_size + 1e-6)
                size_similar = (1.0 - size_tolerance) <= size_ratio <= (1.0 + size_tolerance)
                
                # Check if aspect ratios are within tolerance
                aspect_ratio = other_aspect / (current_aspect + 1e-6)
                aspect_similar = (1.0 - aspect_ratio_tolerance) <= aspect_ratio <= (1.0 + aspect_ratio_tolerance)
                
                # Both size AND aspect ratio must be similar
                if size_similar and aspect_similar:
                    component.append(other_oid)
                    visited.add(other_oid)
                    queue.append(other_oid)
        
        # Only create group if it has at least 2 members
        if len(component) >= 2:
            groups.append(Group(gid=f"size_{gid}", members=component, gtype="size"))
            gid += 1
    
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
        print(f"  Adaptive bins (5 levels):")
        print(f"    very_near < {adaptive_bins[0]:.2f}")
        print(f"    near: {adaptive_bins[0]:.2f} - {adaptive_bins[1]:.2f}")
        print(f"    mid: {adaptive_bins[1]:.2f} - {adaptive_bins[2]:.2f}")
        print(f"    far: {adaptive_bins[2]:.2f} - {adaptive_bins[3]:.2f}")
        print(f"    very_far >= {adaptive_bins[3]:.2f}")

    groups += proximity_groups(objects, norm=image_diag_norm)
    groups += depth_groups(objects, adaptive_bins=adaptive_bins)
    groups += category_groups(objects)
    groups += size_similarity_groups(objects, size_tolerance=0.25)

    return groups
