# Created by MacBook Pro at 23.04.25

import torch
import torch.nn.functional as F
import mbg.mbg_config as param
from mbg.scorer import scorer_config
from mbg.scorer.context_proximity_dataset import obj2context_pair_data

from mbg.group import proximity_grouping
from mbg.group import grouping_similarity
from mbg.group import symbolic_group_features
from mbg.group.neural_group_features import NeuralGroupEncoder
from src import bk
from mbg.group import closure_grouping

def extract_patches_from_objs(objs):
    patches = []
    for obj in objs:
        patches.append(obj["h"])
    return patches


def extract_patch_tensors_from_objs(objs):
    patches = []
    for obj in objs:
        x_new = obj["h"][0][0][:, :, 0] + obj["h"][0][1][0]
        y_new = obj["h"][0][0][:, :, 1] + obj["h"][0][1][0]
        new_patches = torch.stack([x_new / 1024, y_new / 1024], dim=2)
        patches.append(new_patches)
    patches = torch.stack(patches, dim=0)
    return patches


def embedding_principles(group_principle):
    principles = bk.gestalt_principles
    p_id = principles.index(group_principle)
    p_g = F.one_hot(torch.tensor(p_id), num_classes=len(principles)).float()
    return p_g


def embedding_group_neural_features(group_objs):
    group_patches = extract_patch_tensors_from_objs(group_objs)
    # build the encoder (dims must match your patch-encoder settings):
    group_encoder = NeuralGroupEncoder(
        obj_embed_dim=64,
        hidden_dim=128,
        group_embed_dim=128
    )

    # get one group-embedding h_g of size 128:
    h_g = group_encoder(group_patches)  # → torch.Size([128])
    return h_g

def dict_group_features(group_objs):
    group_feature = symbolic_group_features.compute_symbolic_group_features(group_objs, 1024, 1024)
    s_g = group_feature.to_dict()
    return s_g

def construct_group_representations(objs, group_obj_ids, principle):
    rep_gs = []
    for g_i, group_obj_id in enumerate(group_obj_ids):
        group_objs = [objs[i] for i in group_obj_id]
        s_g = dict_group_features(group_objs)
        h_g = embedding_group_neural_features(group_objs)
        p_g = principle
        grp = {
            "id":g_i,
            "child_obj_ids": group_obj_id,
            "members": group_objs,
            "h": h_g,
            "principle": p_g
        }
        rep_gs.append(grp)
    return rep_gs


def eval_groups(objs, threshold, principle, device):
    symbolic_objs = [o["s"] for o in objs]
    obj_patches = extract_patches_from_objs(objs)


    if principle=="proximity":
        # load grouping model
        group_model = scorer_config.load_scorer_model("proximity").to(device)
        group_ids = proximity_grouping.proximity_grouping(obj_patches, group_model, threshold)
    elif principle == "similarity":
        group_model = scorer_config.load_scorer_model("similarity").to(device)
        group_ids = grouping_similarity.similarity_grouping(symbolic_objs, group_model, threshold)
    elif principle == "closure":
        group_model = scorer_config.load_scorer_model("closure").to(device)
        group_ids = closure_grouping.closure_grouping(obj_patches, group_model, threshold, device)
    else:
        raise ValueError
    # grouping objects

    # encoding the groups
    groups = construct_group_representations(objs, group_ids, principle)
    return groups
