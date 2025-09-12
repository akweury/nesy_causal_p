# Created by MacBook Pro at 29.04.25
# grounding.py
import time

import torch
from typing import List, Dict, Any, Tuple
from mbg.grounding.predicates_raw import OBJ_HARD, GRP_HARD, OBJ_SOFT

class GroundingModule:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def ground(self, objects: List[Dict[str, Any]], groups: List[Dict[str, Any]], disable_hard: bool = False, disable_soft: bool = False):
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
