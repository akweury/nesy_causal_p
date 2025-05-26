# Created by MacBook Pro at 23.05.25
import torch

from src import bk


def similarity_grouping(objs, model, threshold):
    N = len(objs)
    scores = torch.zeros(N, N)

    for i in range(N):
        for j in range(i + 1, N):
            c_i, c_j, others = obj2pair_data(objs, i, j)
            s = model(others, c_i, c_j)
            pred = (torch.sigmoid(s) > 0.5).float()
            scores[i, j] = scores[j, i] = pred

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
            if len(group) > 1:
                groups.append(group)

    return groups


def shape_onehot(shape_id):
    onehot = torch.zeros(len(bk.bk_shapes), dtype=torch.float32)
    onehot[shape_id] = 1.0
    return onehot


def obj2pair_data(objects, i, j):
    obj_i = objects[i]
    obj_j = objects[j]

    # Extract features
    c_i = {
        "color": torch.tensor(obj_i["color"], dtype=torch.float32) / 255,
        "size": torch.tensor([obj_i["w"]/1024], dtype=torch.float32),
        "shape": obj_j["shape"],
    }

    c_j = {
        "color": torch.tensor(obj_j["color"], dtype=torch.float32) / 255,
        "size": torch.tensor([obj_j["w"]/1024], dtype=torch.float32),
        "shape": obj_j["shape"],
    }

    # Context (exclude i and j)
    others = []
    for k in range(len(objects)):
        if k != i and k != j:
            obj_k = objects[k]
            others.append({
                "color": torch.tensor(obj_k["color"], dtype=torch.float32) / 255,
                "size": torch.tensor([obj_k["w"]/1024], dtype=torch.float32),
                "shape": obj_k["shape"]
            })

    if not others:
        others = [{
            "color": torch.zeros(3),
            "size": torch.zeros(1),
            "shape": torch.zeros(len(bk.bk_shapes))
        }]
    return c_i, c_j, others
