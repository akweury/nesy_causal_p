"""
coco_group_reasoner.py

Symbolic reasoning over candidate perceptual groups for real COCO images.

This version uses INTERSECTION-BASED GROUP REFINEMENT as the DEFAULT strategy.

Pipeline:
1) Start from primitive candidate groups
2) Generate conjunctive (intersection) groups under strict constraints
3) Perform symbolic selection over the enriched hypothesis space

This reflects perceptual cue integration (e.g., proximity ∧ depth),
not task-driven or post-hoc reasoning.
"""

from typing import Dict, List, Tuple, Set

from mbg.grounding.predicates_real import (
    Group,
    ObjectInstance,
    describe_group,
)


# ============================================================
# Reasoning result container
# ============================================================

class GroupReasoningResult:
    def __init__(
        self,
        main_group: Group,
        all_groups: List[Group],
        explanation: Dict,
    ):
        self.main_group = main_group
        self.all_groups = all_groups
        self.explanation = explanation


# ============================================================
# Intersection-based group refinement
# ============================================================

def refine_groups_by_intersection(
    groups: List[Group],
    objects: Dict[int, ObjectInstance],
    min_size: int = 2,
) -> List[Group]:
    """
    Generate new candidate groups by intersecting groups of DIFFERENT types.

    Constraints:
    - pairwise intersections only
    - groups must be of different generation types
    - intersection size >= min_size
    - keep only novel groups
    """

    refined: List[Group] = groups.copy()
    existing_sets: Set[frozenset] = set(frozenset(g.members) for g in groups)

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]

            # Constraint 1: different group types
            if g1.gid.split("_")[0] == g2.gtype:
                continue

            inter_members = set(g1.members) & set(g2.members)

            # Constraint 2: minimum size
            if len(inter_members) < min_size:
                continue

            key = frozenset(inter_members)
            if key in existing_sets:
                continue
            else:
                existing_sets.add(key)

            # Create new intersection group
            new_gid = f"{g1.gid}_{g2.gid}"
            new_group = Group(
                gid=new_gid,
                members=sorted(list(inter_members)),
                gtype=f"inter({g1.gtype},{g2.gtype})",
            )

            refined.append(new_group)
            existing_sets.add(key)

    return refined


# ============================================================
# Symbolic scoring (simple, interpretable)
# ============================================================

def score_group(summary: Dict) -> int:
    """
    Generic symbolic scoring over group predicates.
    No mode switching; selection emerges from hypothesis richness.
    """

    score = 0

    # prefer non-trivial groups
    score += summary["size"]

    # prefer foreground / near groups
    if summary["depth"] == "near":
        score += 2

    # prefer compact groups
    if summary["compact"]:
        score += 2

    # slight boost for heterogeneous / functional groups
    if summary["functional"]:
        score += 1

    return score


# ============================================================
# Main reasoning API
# ============================================================

def reason_over_groups(
    groups: List[Group],
    objects: Dict[int, ObjectInstance],
    use_intersection: bool = True,
):
    """
    Perform symbolic reasoning over candidate perceptual groups.

    Default behavior:
    - refine hypothesis space via intersections
    - select the best-supported group

    Args:
        groups: primitive candidate groups
        objects: object instances
        use_intersection: enable intersection-based refinement

    Returns:
        GroupReasoningResult
    """

    # --------------------------------------------------------
    # Step 1: hypothesis refinement
    # --------------------------------------------------------
    all_reasoned_groups = []
    if use_intersection:
        intersected_groups = refine_groups_by_intersection(groups, objects)
    else:
        intersected_groups = []

    # --------------------------------------------------------
    # Step 2: summarize predicates
    # --------------------------------------------------------
    summaries = {g.gid: describe_group(g, objects) for g in intersected_groups}

    # --------------------------------------------------------
    # Step 3: score groups
    # --------------------------------------------------------
    scores = {}
    for g in intersected_groups:
        scores[g.gid] = score_group(summaries[g.gid])


    all_reasoned_groups = intersected_groups
    return all_reasoned_groups
