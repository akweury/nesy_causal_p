# Created by MacBook Pro at 29.04.25
# grounding.py
import time

import torch
from typing import List, Dict, Any, Tuple
from mbg.grounding.predicates_raw import OBJ_HARD, GRP_HARD, OBJ_SOFT


# class GroundingModule:
#     """
#     Turns object‐ and group‐level dicts into:
#       hard_facts: Dict[str, Tensor]  # e.g. has_shape: (O,), in_group: (O,G), group_size: (G,), …
#       soft_facts: Dict[str, Tensor]  # e.g. prox: (O,O), grp_sim: (G,G)
#     """
#
#     def __init__(self, device: str = "cpu"):
#         self.device = device
#         # map your grouping principles to integer codes
#         self.princ_map = {"proximity": 0, "similarity": 1, "closure": 2, "symmetry": 3}
#
#     def ground(
#         self,
#         objects: List[Dict[str, Any]],
#         groups:  List[Dict[str, Any]]
#     ) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
#
#         # 1) Build index maps
#         obj_ids = [o["id"] for o in objects]
#         grp_ids = [g["id"] for g in groups]
#         obj2idx = {oid: i for i, oid in enumerate(obj_ids)}
#         grp2idx = {gid: i for i, gid in enumerate(grp_ids)}
#
#         O, G = len(objects), len(groups)
#
#         # 2) Hard facts
#         # -- object‐level features --
#         has_shape = torch.tensor([o["s"]["shape"] for o in objects], dtype=torch.long,  device=self.device)
#         has_color = torch.tensor([o["s"]["color"] for o in objects], dtype=torch.long,  device=self.device)
#         pos_x     = torch.tensor([o["s"]["x"]      for o in objects], dtype=torch.float, device=self.device)
#         pos_y     = torch.tensor([o["s"]["y"]      for o in objects], dtype=torch.float, device=self.device)
#         widths    = torch.tensor([o["s"].get("w",0) for o in objects], dtype=torch.float, device=self.device)
#         heights   = torch.tensor([o["s"].get("h",0) for o in objects], dtype=torch.float, device=self.device)
#
#         # -- in_group: (O, G) binary matrix --
#         in_group = torch.zeros((O, G), dtype=torch.bool, device=self.device)
#         for g in groups:
#             gi = grp2idx[g["id"]]
#             for member in g["members"]:
#                 oi = obj2idx[member["id"]]
#                 in_group[oi, gi] = True
#
#         # -- group‐level symbolic features --
#         group_size   = torch.tensor([len(g["members"])       for g in groups], dtype=torch.float, device=self.device)
#         principle    = torch.tensor([self.princ_map[g["principle"]] for g in groups], dtype=torch.long,  device=self.device)
#
#         hard_facts = {
#             "has_shape":  has_shape,
#             "has_color":  has_color,
#             "x":          pos_x,
#             "y":          pos_y,
#             "w":          widths,
#             "h":          heights,
#             "in_group":   in_group,
#             "group_size": group_size,
#             "principle":  principle,
#         }
#
#         # 3) Soft facts: cosine similarities
#         if O > 0:
#             # flatten each object's patch‐set embedding into a single vector
#             obj_embs = []
#             for o in objects:
#                 contour, origin = o["h"][0]  # assuming o["h"] = [(contour, origin)]
#                 emb = shift_obj_patches_to_global_positions(
#                           torch.tensor(contour, dtype=torch.float, device=self.device),
#                           torch.tensor(origin,  dtype=torch.float, device=self.device)
#                       ).flatten()
#                 obj_embs.append(emb)
#             H_o = torch.stack(obj_embs, dim=0)              # (O, D)
#             H_o = H_o / (H_o.norm(dim=1, keepdim=True) + 1e-8)
#             prox = H_o @ H_o.t()                            # (O, O)
#         else:
#             prox = torch.empty((0, 0), device=self.device)
#
#         if G > 0:
#             G_embs = torch.stack([g["h"] for g in groups], dim=0).to(self.device)  # (G, D)
#             G_embs = G_embs / (G_embs.norm(dim=1, keepdim=True) + 1e-8)
#             grp_sim = G_embs @ G_embs.t()                                           # (G, G)
#         else:
#             grp_sim = torch.empty((0, 0), device=self.device)
#
#         soft_facts = {
#             "prox":    prox,
#             "grp_sim": grp_sim
#         }
#
#         return hard_facts, soft_facts
#

class GroundingModule:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def ground(
            self,
            objects: List[Dict[str, Any]],
            groups: List[Dict[str, Any]],
            disable_hard: bool = False,
            disable_soft: bool = False
    ):

        hard_facts: Dict[str, torch.Tensor] = {}
        soft_facts: Dict[str, torch.Tensor] = {}
        obj_time = 0.0
        group_time = 0.0
        # 1) Hard facts
        if not disable_hard:
            t1 = time.time()
            for pred, fn in OBJ_HARD.items():
                try:
                    hard_facts[pred] = fn(objects=objects, groups=groups, device=self.device)
                except Exception as e:
                    print(f"[Hard-OBJ] Failed predicate {pred}: {e}")
                    raise ValueError
            t2 = time.time()
            obj_time = t2 - t1

            t3 = time.time()
            for pred, fn in GRP_HARD.items():
                try:
                    hard_facts[pred] = fn(objects=objects, groups=groups, device=self.device)
                except Exception as e:
                    print(f"[Hard-GRP] Failed predicate {pred}: {e}")
                    raise ValueError
            t4 = time.time()
            group_time = t4 - t3

        # 2) Soft facts
        if not disable_soft:
            for pred, fn in OBJ_SOFT.items():
                try:
                    soft_facts[pred] = fn(objects=objects, groups=groups, device=self.device)
                except Exception as e:
                    print(f"[Soft] Failed predicate {pred}: {e}")
        return hard_facts, soft_facts, obj_time, group_time


def ground_facts(objs, grps, disable_hard=False, disable_soft=False):
    grounder = GroundingModule()
    hard, soft, obj_time, group_time = grounder.ground(objs, grps, disable_hard=disable_hard, disable_soft=disable_soft)
    return hard, soft
