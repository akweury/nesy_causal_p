# Created by MacBook Pro at 15.05.25


# evaluation.py

import torch
from collections import defaultdict
from typing import List, Dict, Any
import inspect

from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.grounding import grounding
from mbg.language.clause_generation import Clause, ScoredRule
from mbg.grounding.predicates import OBJ_HARD, GRP_HARD, SOFT


def _evaluate_clause(
        clause: Clause,
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor]
) -> bool:
    """
    Existential check: does there exist a binding of all
    object‐vars and group‐vars in clause.body that makes
    every atom true?
    """
    O = hard["has_shape"].size(0)
    G = hard["group_size"].size(0)

    # map variable names → domains
    domains = {
        "O": list(range(O)),
        "O1": list(range(O)),
        "O2": list(range(O)),
        "G": list(range(G)),
        "G1": list(range(G)),
        "G2": list(range(G)),
    }

    # collect which vars appear
    vars_in_body = set(v for atom in clause.body for v in atom[1:] if isinstance(v, str) and v.isupper())

    # generate all candidate assignments (cartesian product)
    # but keep it small: only up to two vars gives O^2 or G^2 loops
    var_list = list(vars_in_body)

    def rec(idx, assignment):
        if idx == len(var_list):
            # test this grounding
            for pred, *args in clause.body:
                # pick out tensors
                if pred in hard:
                    h = hard[pred]
                    if h.dim() == 1:
                        # unary: h[obj] == value?
                        oi = assignment[args[1]]
                        val = args[2]
                        if h[oi].item() != val:
                            return False
                    else:
                        # membership: h[obj,grp]
                        oi = assignment[args[0]]
                        gi = assignment[args[1]]
                        if not h[oi, gi].item():
                            return False
                else:
                    # soft‐predicate proximity or grp_sim
                    s = soft[pred]
                    # binary
                    a1, a2 = args[0], args[1]
                    i1, i2 = assignment[a1], assignment[a2]
                    # check threshold already baked into clause.weight
                    # weight>0 means it was above threshold
                    if s[i1, i2].item() < 0:
                        return False
            return True

        var = var_list[idx]
        for v in domains[var]:
            assignment[var] = v
            if rec(idx + 1, assignment):
                return True
        return False

    return rec(0, {})


def compute_metrics(
        all_scores: List[Dict[ScoredRule, float]],
        all_labels: List[int],
        threshold: float = 0.5
):
    """
    Given per‐image rule‐match scores (from apply_rules) and true labels,
    aggregate each image’s rule scores into a single y_hat, then compute
    overall accuracy/f1/etc.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_true, y_hat, y_score = [], [], []
    for scores, y in zip(all_scores, all_labels):
        # weighted average across *all* rules
        # if not scores:
        #     pred_score = 0.0
        # else:
        #     vals = np.array(scores, dtype=float)
        #     pred_score = vals.mean()
        y_true.append(y)
        y_score.append(scores)
        y_hat.append(int(scores >= threshold))

    return {
        "acc": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat),
        "auc": roc_auc_score(y_true, y_score)
    }


# def apply_rules(
#         rules: List[ScoredRule],
#         hard: Dict[str, torch.Tensor],
#         soft: Dict[str, torch.Tensor]
# ) -> Dict[ScoredRule, float]:
#     """
#     For each learned ScoredRule, returns a match score in [0,1] on this one image.
#     - image‐level rules (scope=='image'): existential over objects/groups
#     - group‐existential (scope=='group_exist'): existential over groups
#     - group‐universal (scope=='group_univ'): universal over groups
#     """
#     O = hard["has_shape"].shape[0]
#     G = hard["group_size"].shape[0]
#
#     def eval_body_image(body):
#         # for image_target: at least one grounding must satisfy *all* atoms
#         if not body:
#             return 1.0
#         # we will build a score tensor for each possible grounding and then max
#         # distinguish object‐only atoms vs group‐only vs mixed
#         # for simplicity here we only handle the common patterns:
#         #   ("has_shape","O",v), ("has_color","O",rgb), ("in_group","O","G"), ("prox","O1","O2"), ("grp_sim","G1","G2")
#         scores = []
#
#         # unary object‐level
#         for atom in body:
#             pred, a1, a2 = atom
#             if pred == "has_shape":
#                 v = a2
#                 mat = (hard["has_shape"] == v).float()  # (O,)
#                 scores.append(mat)
#             elif pred == "has_color":
#                 r, g, b = a2
#                 mat = ((hard["has_color"][:, 0] == r) &
#                        (hard["has_color"][:, 1] == g) &
#                        (hard["has_color"][:, 2] == b)).float()
#                 scores.append(mat)
#             elif pred == "in_group":
#                 # in_group("O","G") existential means any True in the entire matrix
#                 scores.append(hard["in_group"].any(dim=1).float())
#             elif pred == "group_size":
#                 scores.append((hard["group_size"] == a2).any().float())
#             elif pred == "prox":
#                 # take the max prox over all object pairs
#                 scores.append(soft["prox"].max().unsqueeze(0))
#             elif pred == "grp_sim":
#                 scores.append(soft["grp_sim"].max().unsqueeze(0))
#             else:
#                 raise NotImplementedError(f"Unknown pred {pred}")
#
#         # now we need to AND all atoms: for unary we have per‐object vectors,
#         # for prox/grp_sim we have single‐value tensors.
#         # to mix them, broadcast the single‐value ones to O entries:
#         mats = []
#         for m in scores:
#             if m.numel() == 1:
#                 mats.append(m.expand(O))
#             else:
#                 mats.append(m)
#         # conjunction per object:
#         conj = torch.stack(mats, dim=0).min(dim=0)[0]  # (O,)
#         if len(conj) == 0:
#             return 0
#         else:
#             return float(conj.max().item())
#
#     def eval_body_group_exist(body):
#         # for group_target existential: at least one group satisfies
#         # similar to image but over groups
#         if not body:
#             return 1.0
#         # we'll reuse hard["in_group"] and group_size/principle
#         # build per‐group scores
#         scores = []
#         for atom in body:
#             pred, a1, a2 = atom
#             if pred == "group_size":
#                 sz = a2
#                 mat = (hard["group_size"] == sz).float()  # (G,)
#                 scores.append(mat)
#             elif pred == "principle":
#                 pr = a2
#                 mat = (hard["principle"] == pr).float()
#                 scores.append(mat)
#             elif pred == "in_group":
#                 # in_group("O","G")—we want ANY object in that group
#                 mat = hard["in_group"].any(dim=0).float()  # (G,)
#                 scores.append(mat)
#             elif pred == "has_shape" or pred == "has_color":
#                 # these refer to object‐unary + in_group; we already have matching object rows
#                 # project to group by max over members
#                 if pred == "has_shape":
#                     v = a2
#                     obj_m = (hard["has_shape"] == v).float()  # (O,)
#                 else:
#                     r, g, b = a2
#                     obj_m = ((hard["has_color"][:, 0] == r) &
#                              (hard["has_color"][:, 1] == g) &
#                              (hard["has_color"][:, 2] == b)).float()  # (O,)
#                 # for each group take max over objects in that group
#                 mat = []
#                 for gi in range(G):
#                     members = hard["in_group"][:, gi]  # (O,)
#                     if members.any():
#                         mat.append(obj_m[members].max())
#                     else:
#                         mat.append(torch.tensor(0.0))
#                 if len(mat)>0:
#                     scores.append(torch.stack(mat))  # (G,)
#             else:
#                 raise NotImplementedError(f"Unknown pred {pred}")
#
#         # AND them per‐group, then OR across groups
#         mats = scores
#         conj = torch.stack(mats, dim=0).min(dim=0)[0]  # (G,)
#         if len(conj) == 0:
#             return 0
#         else:
#             return float(conj.max().item())
#
#     def eval_body_group_univ(body):
#         # universal: every group must satisfy ALL atoms
#         if not body:
#             return 1.0
#         # reuse the same per‐group mats from above
#         mats = []
#         for atom in body:
#             pred, a1, a2 = atom
#             if pred == "group_size":
#                 mats.append((hard["group_size"] == a2).float())
#             elif pred == "principle":
#                 mats.append((hard["principle"] == a2).float())
#             elif pred == "in_group":
#                 mats.append(hard["in_group"].any(dim=0).float())
#             elif pred in ("has_shape", "has_color"):
#                 # same as exist but we need the per‐group vector
#                 if pred == "has_shape":
#                     obj_m = (hard["has_shape"] == a2).float()
#                 else:
#                     r, g, b = a2
#                     obj_m = ((hard["has_color"][:, 0] == r) &
#                              (hard["has_color"][:, 1] == g) &
#                              (hard["has_color"][:, 2] == b)).float()
#                 # project to groups
#                 mat = []
#                 for gi in range(G):
#                     members = hard["in_group"][:, gi]
#                     if members.any():
#                         mat.append(obj_m[members].max())
#                     else:
#                         mat.append(torch.tensor(0.0))
#                 mats.append(torch.stack(mat))
#             else:
#                 raise NotImplementedError
#
#         conj = torch.stack(mats, dim=0).min(dim=0)[0]  # (G,)
#         if len(conj) == 0:
#             return 0
#         else:
#             return float(conj.min().item())  # all groups
#
#     scores = {}
#     for sr in rules:
#         b = sr.clause.body
#         if sr.scope == "image":
#             m = eval_body_image(b)
#         elif sr.scope == "existential":
#             m = eval_body_group_exist(b)
#         elif sr.scope == "universal":
#             m = eval_body_group_univ(b)
#         else:
#             raise ValueError(sr.scope)
#         # combine with rule’s learned confidence
#         scores[sr] = m * sr.confidence
#     return scores

from mbg.grounding.predicates import (
    has_shape_eval, has_color_eval, x_eval, y_eval, w_eval, h_eval,
    in_group_eval, group_size_eval, principle_eval,
    prox_eval, grp_sim_eval
)

EVAL_FN = {
    "has_shape": has_shape_eval,
    "has_color": has_color_eval,
    "x": x_eval,
    "y": y_eval,
    "w": w_eval,
    "h": h_eval,
    "in_group": in_group_eval,
    "group_size": group_size_eval,
    "principle": principle_eval,
    "prox": prox_eval,
    "grp_sim": grp_sim_eval,
}


def _eval_atom(pred, hard, soft, objects, groups, a1, a2, device="cpu"):
    if pred not in EVAL_FN:
        raise ValueError(f"Unknown predicate {pred}")
    # call with exactly (hard, soft, a1, a2); ignore objects/groups here
    return EVAL_FN[pred](hard, soft, a1, a2)


#
# # Helper: given a body = list of (pred_name, arg1, arg2) and a predicate registry,
# # call the right function to get a tensor of shape (N,) or (N,N) or ().
# def _eval_atom(
#         pred: str,
#         hard ,
#         soft ,
#         objects,
#         groups,
#         a1: any,
#         a2: any,
#         device: str = "cpu"
# ) -> torch.Tensor:
#     """
#     Look up pred in your registries and call it.
#     Returns a Tensor of shape
#       - (O,) for object‐level predicates,
#       - (G,) for group‐level predicates,
#       - (O,O) or (G,G) for soft ones.
#     """
#     # helper to call fn with only the parameters it expects
#     def call_flexible(fn, /, **all_kwargs):
#         sig = inspect.signature(fn)
#         kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}
#         return fn(**kwargs)
#     # 1) object‐level hard predicates
#     if pred in OBJ_HARD:
#         # raw = either shape (O,) or (O,G)
#         raw = OBJ_HARD[pred](objects=objects, groups=groups, device=device)
#         if raw.ndim == 1:
#             # e.g. has_shape(o) == a2
#             return (raw == a2).float()
#         elif raw.ndim == 2:
#             # only in_group returns an (O,G) mask
#             # a2 should be something like "g2" or an int
#             if isinstance(a2, str) and a2.startswith("g"):
#                 gi = int(a2[1:])
#             else:
#                 gi = int(a2)
#             return raw[:, gi].float()
#         else:
#             raise RuntimeError(f"Unexpected OBJ_HARD output shape {raw.shape} for {pred}")
#
#     # 2) group‐level hard predicates
#     # 2) group‐level hard predicates
#     if pred in GRP_HARD:
#         raw = call_flexible(
#             GRP_HARD[pred],
#             objects=objects, groups=groups, device=device
#         )
#         # raw: (G,)
#         return (raw == a2).float()
#
#     # 3) soft (neural) predicates
#     if pred in SOFT:
#         # e.g. prox or grp_sim → full matrix
#         return SOFT[pred](objects=objects, groups=groups, device=device)
#     raise KeyError(f"Unknown predicate {pred}")


def _body_to_group_mats(
        body: List[tuple],
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        objects: List[dict],
        groups: List[dict]
) -> List[torch.Tensor]:
    """
    Evaluate each atom in body, fold it down to a vector of length G.
    Supports:
      - 0D scalars → broadcast to [G]
      - 1D tensors [O] or [G]
      - 2D tensors:
         - (G,G)  → max over dim=1 → [G]
         - (O,G) → any over dim=0 → [G]
         - (O,O) → global max → scalar broadcast → [G]
         - (O,3) → interpret as RGB, compare to body’s rgb‐tuple, then project to [G]
    """
    O = hard["has_shape"].size(0)
    G = hard["group_size"].size(0)
    in_grp = hard["in_group"]  # (O,G)

    mats: List[torch.Tensor] = []
    for pred, _a1, _a2 in body:
        t = _eval_atom(pred, hard, soft, objects, groups, _a1, _a2)
        # --- 0D scalar → broadcast to all G ---
        if t.ndim == 0:
            mats.append(t.expand(G))

        # --- 2D cases ---
        elif t.ndim == 2:
            # (G,G)
            if t.shape == (G, G):
                try:
                    mats.append(t.max(dim=1)[0])
                except IndexError:
                    raise IndexError()

            # (O,G)
            elif t.shape == (O, G):
                mats.append(t.any(dim=0).float())

            # (O,O)
            elif t.shape == (O, O):
                mats.append(t.max().expand(G))

            # (O,3) RGB‐vector from legacy has_color
            elif t.shape[1] == 3:
                r, g, b = _a2
                vec = []
                # for each group, did any member match exactly that RGB?
                for gi in range(G):
                    members = in_grp[:, gi]
                    if members.any():
                        # pick the rows for this group
                        block = t[members]  # [n_members, 3]
                        mask = ((block[:, 0] == r) &
                                (block[:, 1] == g) &
                                (block[:, 2] == b)).float()
                        vec.append(mask.max())
                    else:
                        vec.append(torch.tensor(0.0, device=t.device))
                mats.append(torch.stack(vec))

            else:
                raise RuntimeError(f"Unexpected 2D shape {t.shape} for predicate {pred!r}")

        # --- 1D: either object‐unary [O] or group‐unary [G] ---
        elif t.ndim == 1:
            if t.shape[0] == G:
                mats.append(t.float())
            elif t.shape[0] == O:
                # project object‐mask [O] → group mask [G]
                vec = []
                for gi in range(G):
                    members = in_grp[:, gi]
                    if members.any():
                        vec.append(t[members].max())
                    else:
                        vec.append(torch.tensor(0.0, device=t.device))
                mats.append(torch.stack(vec))
            else:
                raise RuntimeError(f"1D tensor of unexpected length {t.shape[0]} for {pred!r}")

        else:
            raise RuntimeError(f"Cannot handle tensor dim {t.ndim} for predicate {pred!r}")

    return mats


def _eval_image(
        body,
        hard,
        soft,
        objects,
        groups,
) -> float:
    """
    Evaluate an image‐level clause body:
      – ∧ across atoms (per object), then ∃ across objects.
    """
    if not body:
        return 1.0

    O = hard["has_shape"].size(0)
    mats: List[torch.Tensor] = []

    for pred, _a1, _a2 in body:
        t = _eval_atom(pred, hard, soft, objects, groups, _a1, _a2)

        # 0‐D scalar → broadcast to objects
        if t.ndim == 0:
            mats.append(t.expand(O))

        # object–object soft: (O, O) → ∃ over j for each i
        elif t.ndim == 2 and t.shape == (O, O):
            if O > 0:
                mats.append(t.max(dim=1)[0])
            else:
                mats.append(torch.zeros(O, device=t.device))

        # group–group soft: (G, G) → ∃ anywhere, broadcast to objects
        elif t.ndim == 2 and t.shape == (hard["group_size"].size(0),
                                         hard["group_size"].size(0)):
            val = t.max() if t.numel() > 0 else torch.tensor(0.0, device=t.device)
            mats.append(val.expand(O))

        # object–group: (O, G) → ∃ over groups per object
        elif t.ndim == 2 and t.shape[0] == O and t.shape[1] == hard["group_size"].size(0):
            if t.shape[1] > 0:
                mats.append(t.max(dim=1)[0])
            else:
                mats.append(torch.zeros(O, device=t.device))

        # object–unary: (O,) → direct
        elif t.ndim == 1 and t.shape[0] == O:
            mats.append(t.float())

        # group–unary: (G,) → ∃ any group, broadcast to objects
        elif t.ndim == 1 and t.shape[0] == hard["group_size"].size(0):
            if t.numel() > 0:
                mats.append(t.max().expand(O))
            else:
                mats.append(torch.zeros(O, device=t.device))

        else:
            raise RuntimeError(f"Unexpected tensor shape {t.shape} for atom {pred}")

    # ∧ across atoms (per object)
    conj = torch.stack(mats, dim=0).min(dim=0)[0]  # shape (O,)
    # ∃ across objects
    return float(conj.max().item()) if conj.numel() > 0 else 0.0


def _eval_group_exist(
        body, hard, soft, objects, groups
) -> float:
    if not body:
        return 1.0
    if len(groups) == 0:
        return 0.0
    if len(objects) == 0:
        return 0.0
    mats = _body_to_group_mats(body, hard, soft, objects, groups)
    # ∧ across atoms, then ∃ over groups
    conj = torch.stack(mats, 0).min(dim=0)[0]  # (G,)
    return float(conj.max().item()) if conj.numel() else 0.0


def _eval_group_univ(
        body, hard, soft, objects, groups
) -> float:
    if not body:
        return 1.0
    mats = _body_to_group_mats(body, hard, soft, objects, groups)
    # ∧ across atoms, then ∀ over groups
    conj = torch.stack(mats, 0).min(dim=0)[0]  # (G,)
    return float(conj.min().item()) if conj.numel() else 0.0


# Dispatcher
_SCOPE_FN = {
    "image": _eval_image,
    "existential": _eval_group_exist,
    "universal": _eval_group_univ,
}


def apply_rules(
        rules: List[ScoredRule],
        hard: Dict[str, torch.Tensor],
        soft: Dict[str, torch.Tensor],
        objects: List[dict],
        groups: List[dict],
) -> Dict[ScoredRule, float]:
    out = {}
    for sr in rules:
        fn = _SCOPE_FN.get(sr.scope)
        if fn is None:
            raise ValueError(f"Unknown scope {sr.scope}")
        if len(objects) == 0 or len(groups) == 0:
            match_score = 0
        else:
            match_score = fn(sr.clause.body, hard, soft, objects, groups)
        out[sr] = match_score * sr.confidence
    return out


def compute_image_score(learned_rules, rule_scores, eps=1e-8):
    """
    Combine rule scores into a single validation image score,
    weighted by the confidence of each learned rule.
    """
    weighted_sum = sum(r.confidence * s for r, s in zip(learned_rules, rule_scores.values()))
    total_weight = sum(r.confidence for r in learned_rules)
    return weighted_sum / (total_weight + eps)


def eval_rules(val_data, obj_model, rules_train, hyp_params):
    # we’ll collect per‐task true / pred
    per_task_scores = defaultdict(list)
    per_task_labels = defaultdict(list)
    conf_th = hyp_params["conf_th"]
    prox_th = hyp_params["prox"]
    task_id = val_data["task"][0].split("_")[0]
    # for each rule, we need to know the valuation result, so either 1 or 0
    for data in val_data["positive"] + val_data["negative"]:
        true_label = int(data["img_label"][0])
        # 1) detect objects & groups
        objs = eval_patch_classifier.evaluate_image(obj_model, data)
        groups = eval_groups.eval_groups(objs, prox_th)
        num_groups = len(groups)

        # 2) ground  & generate validation image’s clauses
        hard, soft = grounding.ground_facts(objs, groups)

        # 3) fetch learned rules for this task
        learned_rules = rules_train[task_id]  # List[ScoredRule]

        # 4) filter out very‐low confidence rules
        kept_rules = [r for r in learned_rules if r.confidence >= conf_th]

        # 5) soft‐match each rule on this image
        rule_scores = apply_rules(learned_rules, hard, soft, objs, groups)
        #    rule_scores: Dict[ScoredRule, float] in [0..1]

        img_score = compute_image_score(learned_rules, rule_scores)
        # # 6) confidence‐powered aggregation
        # num = 0.0
        # den = 0.0
        # for r, m in rule_scores.items():
        #     w = r.confidence ** conf_power
        #     num += m * w
        #     den += w
        #
        # img_score = (num / den) if den > 0 else 0.0

        per_task_scores[task_id].append(img_score)
        per_task_labels[task_id].append(true_label)

    # now compute metrics per task
    metrics = {}
    for task_id in per_task_scores:
        # build lists of just the per‐image score dicts and labels
        scores_list = per_task_scores[task_id]
        labels = per_task_labels[task_id]

        # compute raw metrics (acc, f1, auc)
        m = compute_metrics(scores_list, labels, threshold=conf_th)
        print(f"Task {task_id} →  acc={m['acc']:.3f}   f1={m['f1']:.3f}   auc={m['auc']:.3f}")
        metrics[task_id] = m

    return metrics
