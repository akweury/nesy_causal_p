# Created by MacBook Pro at 23.04.25

import networkx as nx
from itertools import combinations
import torch
import torch.nn.functional as F
import numpy as np

from mbg.scorer import scorer_config
from mbg.group import proximity_grouping
from mbg.group import symbolic_group_features
from mbg.group.neural_group_features import NeuralGroupEncoder
from mbg.group.gd_transformer import GroupingTransformer
from src import bk
from mbg.patch_preprocess import patch2code


def embedding_principles(group_principle):
    principles = bk.gestalt_principles
    p_id = principles.index(group_principle)
    p_g = F.one_hot(torch.tensor(p_id), num_classes=len(principles)).float()
    return p_g


def embedding_group_neural_features(group_objs, device, input_dim=7):
    group_patches = torch.stack([o["h"] for o in group_objs])[
        :, :, :, :input_dim].to(device)  # (G, P, L, D)
    # build the encoder (dims must match your patch-encoder settings):
    group_encoder = NeuralGroupEncoder(
        input_dim=input_dim,
        obj_embed_dim=64,
        hidden_dim=128,
        group_embed_dim=128
    ).to(device)

    # get one group-embedding h_g of size 128:
    h_g = group_encoder(group_patches)  # → torch.Size([128])
    return h_g


def dict_group_features(group_objs):
    group_feature = symbolic_group_features.compute_symbolic_group_features(
        group_objs, 1024, 1024)
    s_g = group_feature.to_dict()
    return s_g


def construct_group_representations(objs, group_obj_ids, principle, input_dim, device):
    rep_gs = []
    for g_i, group_obj_id in enumerate(group_obj_ids):
        group_objs = [objs[i] for i in group_obj_id]
        # s_g = dict_group_features(group_objs)
        h_g = embedding_group_neural_features(group_objs, device, input_dim)
        p_g = principle
        grp = {
            "id": g_i,
            "child_obj_ids": group_obj_id,
            "members": group_objs,
            "h": h_g,
            "principle": p_g
        }
        rep_gs.append(grp)
    return rep_gs


@torch.no_grad()
def group_objects_with_model(model, objects, device, input_type="pos_color_size", threshold=0.5, dim=7):
    """
    Args:
        model: trained ContextContourScorer model
        objects: list of dicts, each with keys like 'position', 'color', 'size' depending on input_type
        input_type: one of 'pos', 'pos_color', 'pos_color_size'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped
    Returns:
        List of groups, each group is a list of object indices
    """
    model = model.to(device).eval()
    n = len(objects)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        ci, cj = objects[i].unsqueeze(0), objects[j].unsqueeze(0)
        context = [x for k, x in enumerate(objects) if k != i and k != j]
        if len(context) == 0:
            ctx_tensor = torch.zeros((1, 1, 6, 16, 7), device=device)
        else:
            ctx_tensor = torch.stack(context).unsqueeze(0).to(device)

        logit = model(ci[:, :, :, :dim], cj[:, :, :, :dim],
                      ctx_tensor[:, :, :, :, :dim])
        prob = torch.sigmoid(logit).item()
        if prob > threshold:
            G.add_edge(i, j)
    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups


def eval_groups(objs, group_model, principle, device, dim, grp_th=0.5):
    # symbolic_objs = [o["s"] for o in objs]
    neural_objs = [o["h"] for o in objs]
    if principle == "similarity":
        dim = 5

    group_ids = eval_groups.get_transformer_group_ids(
        transformer_model=group_model,
        objects=objs,
        device=device,
        threshold=grp_th)

    # group_ids = group_objects_with_model(
    #     group_model, neural_objs, device, dim=dim, threshold=grp_th)
    # encoding the groups
    groups = construct_group_representations(
        objs, group_ids, principle, dim, device)
    return groups


@torch.no_grad()
def get_transformer_group_ids(transformer_model, objects, device, threshold=0.5):
    """
    Use transformer group detector to get group IDs

    Args:
        transformer_model: trained GroupingTransformer model
        objects: list of object dicts with keys 'position', 'color', 'size', 'contour'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped

    Returns:
        List of groups, each group is a list of object indices
    """
    transformer_model = transformer_model.to(device).eval()

    n = len(objects)
    if n <= 1:
        return [[i] for i in range(n)]  # Each object in its own group

    # Extract features from objects
    positions = []
    colors = []
    sizes = []
    shapes = []
    positions = torch.tensor([[obj['s']['x'], obj['s']['y']]
                             for obj in objects]).to(device).unsqueeze(0)
    colors = torch.tensor([obj['s']['color']
                          for obj in objects]).to(device).unsqueeze(0)
    sizes = torch.tensor([[obj['s']['w']]
                         for obj in objects]).to(device).unsqueeze(0)

    obj_labels = [np.array(bk.bk_shapes_2)[
        obj['s']['shape'].bool().numpy()][0] for obj in objects]
    obj_patches = torch.stack([obj['h'].reshape(-1, 2) for obj in objects])
    shape_code = patch2code(
        obj_patches, obj_labels=obj_labels, device=device).unsqueeze(0)
    # (N, 16)

    # Get predictions from transformer
    with torch.no_grad():
        pred = transformer_model(
            positions, colors, sizes, shape_code)  # (1, N, N)
        pred = torch.sigmoid(pred).squeeze(0)  # (N, N)

    # Build graph from predictions
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if pred[i, j].item() > threshold:
                G.add_edge(i, j)

    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups
