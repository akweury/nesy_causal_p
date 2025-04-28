# Created by MacBook Pro at 23.04.25

import torch

import mbg.mbg_config as param
from mbg.scorer import scorer_config
from mbg.group import proximity_grouping


def extract_patches_from_objs(objs):
    patches = []
    for obj in objs:
        patches.append(obj["patch"])
        # x_new = obj["patch"][0][0][:, :, 0] + obj["patch"][0][1][0]
        # y_new = obj["patch"][0][0][:, :, 1] + obj["patch"][0][1][0]
        # new_patches = torch.stack([x_new, y_new], dim=2)
        # patches.append(new_patches)
    return patches


def eval_groups(objs, gt_pairs):
    obj_patches = extract_patches_from_objs(objs)
    group_prox_model = scorer_config.load_scorer_model("proximity")
    groups = proximity_grouping.proximity_grouping(obj_patches, group_prox_model, gt_pairs)
