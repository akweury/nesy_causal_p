# Created by MacBook Pro at 24.04.25

# proximity_grouping.py

import torch
from typing import List
from mbg.scorer.context_proximity_scorer import ContextProximityScorer
from matplotlib import pyplot as plt
from mbg.scorer.context_proximity_dataset import obj2context_pair_data


def compute_pairwise_scores(scorer: ContextProximityScorer,
                            patch_sets,
                            gt_pairs) -> torch.Tensor:
    """
    Args:
        scorer: Trained NeuralProximityScorer
        patch_sets: (N, P, L, 2) tensor

    Returns:
        similarity_scores: (N, N) symmetric matrix
    """
    N = len(patch_sets)
    scores = torch.zeros(N, N)
    for i in range(N):
        for j in range(i + 1, N):
            c_i, c_j, others, label = obj2context_pair_data(i, j, patch_sets, gt_pairs)
            s = scorer(c_i.unsqueeze(0), c_j.unsqueeze(0), others)
            pred = (torch.sigmoid(s) > 0.5).float()
            scores[i, j] = scores[j, i] = pred

    return scores


def proximity_grouping(obj_patches,
                       scorer: ContextProximityScorer,
                       gt_pairs,
                       threshold: float = 0.5) -> List[List[int]]:
    """
    Perform grouping based on pairwise proximity scores.

    Args:
        patch_sets: (N, P, L, 2) tensor
        scorer: Trained NeuralProximityScorer
        threshold: Proximity threshold to consider objects connected

    Returns:
        groups: List of groups (each group is a list of indices)
    """
    N = len(obj_patches)
    scores = compute_pairwise_scores(scorer, obj_patches, gt_pairs)  # (N, N)
    adj_matrix = (scores > threshold).int()

    visited = set()
    groups = []

    def dfs(node, current_group):
        for neighbor in range(N):
            if neighbor not in visited and adj_matrix[node, neighbor]:
                visited.add(neighbor)
                current_group.append(neighbor)
                dfs(neighbor, current_group)

    for i in range(N):
        if i not in visited:
            visited.add(i)
            group = [i]
            dfs(i, group)
            if len(group)>1:
                groups.append(group)

    return groups
