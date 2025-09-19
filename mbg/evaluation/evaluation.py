# Created by MacBook Pro at 15.05.25

# evaluation.py
import time
import torch
from collections import defaultdict
from typing import List, Dict

from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.grounding import grounding
from mbg.language.clause_generation import Clause, ScoredRule
from mbg import patch_preprocess
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mbg.scorer import improved_calibrator

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


from mbg.grounding.predicates import *

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
    'not_has_shape_triangle': not_has_shape_triangle_eval,
    'not_has_shape_rectangle': not_has_shape_rectangle_eval,
    'not_has_shape_circle': not_has_shape_circle_eval,
    "no_member_triangle": no_member_shape_triangle_eval,
    "no_member_rectangle": no_member_shape_square_eval,
    "no_member_circle": no_member_shape_circle_eval,
    "mirror_x": mirror_x_eval,
    "same_shape": same_shape_eval,
    "same_color": same_color_eval,
    "same_y": same_y_eval,
    "diverse_sizes": diverse_sizes_eval,
    "diverse_shapes": diverse_shapes_eval,
    "diverse_colors": diverse_colors_eval,
    "unique_colors": unique_colors_eval,
    "unique_shapes": unique_shapes_eval,
    "unique_sizes": unique_sizes_eval,
    "group_num": group_num_eval,
    "sim_shape_soft": sim_shape_soft_eval,
    "sim_color_soft": sim_color_soft_eval,
    "sim_size_soft": sim_size_soft_eval,

}


def _eval_atom(pred, hard, soft, objects, groups, a1, a2, device="cpu"):
    if pred not in EVAL_FN:
        raise ValueError(f"Unknown predicate {pred}")
    # call with exactly (hard, soft, a1, a2); ignore objects/groups here
    return EVAL_FN[pred](hard, soft, a1, a2)


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
                    members = in_grp[:, gi].to(torch.int)
                    if members.any():
                        vec.append((t * members).max())
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
    if hard:
        O = hard["has_shape"].size(0)
    elif soft:
        O = soft["sim_color_soft"].size(0)
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
            match_score = fn(sr.c.body, hard, soft, objects, groups)
        out[sr] = match_score * sr.confidence
    return out


def compute_image_score(learned_rules, rule_scores, high_conf_th=0.99, fallback_weight=0.1, eps=1e-8):
    """
    Compute image score using high-confidence rule prioritization:
    - If any rule has confidence >= high_conf_th and fires, only those are considered.
    - Otherwise, fallback to confidence-weighted average (penalizing low-conf clauses).
    """
    high_conf_scores = []
    for rule in learned_rules:
        score = rule_scores.get(rule, 0.0)
        if rule.confidence >= high_conf_th:
            high_conf_scores.append(score)

    # Use only high-confidence rules if any fired
    if high_conf_scores:
        return sum(high_conf_scores) / (len(high_conf_scores) + eps)

    # Otherwise, fall back to confidence^2 weighted average
    weighted_sum = 0.0
    total_weight = 0.0
    for rule in learned_rules:
        score = rule_scores.get(rule, 0.0)
        weight = rule.confidence ** 2
        weighted_sum += weight * score
        total_weight += weight

    if total_weight > 0:
        return weighted_sum / (total_weight + eps)
    else:
        return fallback_weight


def is_grouping_error(pred_groups, symbolic_data, iou_threshold=0.5):
    """
    Evaluate grouping error by comparing predicted groups to ground truth groups
    based on object-level group_id in symbolic_data.
    """

    # 1. Build GT groups: group_id -> set of obj indices
    gt_group_dict = defaultdict(set)
    for idx, obj in enumerate(symbolic_data):
        group_id = int(obj["group_id"].item())
        gt_group_dict[group_id].add(idx)
    gt_groups = list(gt_group_dict.values())

    # 2. Build predicted groups: list of sets
    pred_groups_sets = [set(group["child_obj_ids"]) for group in pred_groups]

    # 3. Compute alignment between predicted and gt groups
    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred_set in enumerate(pred_groups_sets):
        for gt_idx, gt_set in enumerate(gt_groups):
            iou = len(pred_set & gt_set) / len(pred_set | gt_set)
            if iou >= iou_threshold:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break  # only allow one match per predicted group

    # 4. Compute match rate and error
    total_gt = len(gt_groups)
    total_pred = len(pred_groups_sets)

    if total_gt == 0 and total_pred == 0:
        return False  # perfect (trivially)
    if total_gt == 0 or total_pred == 0:
        return True  # totally mismatched

    match_rate = len(matched_gt) / total_gt
    return match_rate < 1.0  # grouping error if not all GT groups matched


def is_object_error(objs, symbolic_data):
    """
    Return True if there is a shape mismatch between predicted and ground truth objects.
    Assumes objs and symbolic_data are aligned in order by position.
    """

    if len(objs) != len(symbolic_data):
        return True  # mismatch in number of objects

    for pred_obj, gt_obj in zip(objs, symbolic_data):
        pred_shape_idx = torch.argmax(pred_obj["s"]["shape"]).item()
        gt_shape_idx = gt_obj["shape"].item()
        if pred_shape_idx != gt_shape_idx:
            return True  # shape mismatch

    return False  # all shapes match


def eval_rules(val_data, obj_model, group_model, learned_rules, hyp_params, eval_principle, device, calibrator):
    patch_dim = hyp_params["patch_dim"]
    conf_th = hyp_params["conf_th"]
    top_k = hyp_params["top_k"]

    task_id = val_data["task"][0].split("_")[0]
    all_data = val_data["positive"] + val_data["negative"]
    img_paths = [d["image_path"][0] for d in all_data]
    imgs = patch_preprocess.load_images_fast(img_paths, device=device)
    gt_labels = [int(d["img_label"][0]) for d in all_data]

    per_task_scores = defaultdict(list)
    per_task_labels = defaultdict(list)

    error_counters = {
        "grouping_error": 0,
        "object_error": 0,
        "clause_mismatch": 0,
        "total_errors": 0
    }

    analysis = {
        "calibrated_scores": [],
        "vanilla_scores": [],
        "groundtruth_labels": [],
        "rule_pool_has_good_clause": [],  # True if any rule has confidence > 0.95
        "topk_clause_precision": [],
    }

    # top-k clause selection analysis
    topk_hits = 0
    topk_total_valid = 0
    topk_precision_sum = 0
    topk_precision_count = 0

    for i, img in enumerate(imgs):
        true_label = gt_labels[i]

        # 1) detect objects & groups
        objs = eval_patch_classifier.evaluate_image(obj_model, img, device)
        # align the pred objects with gt objects
        gt_objs = all_data[i]["symbolic_data"]
        objs_gt_aligned, objs, _ = patch_preprocess.align_gt_data_and_pred_data(gt_objs, objs)

        groups = eval_groups.eval_groups(objs, group_model, eval_principle, device, patch_dim)

        # 2) ground symbolic facts
        hard, soft = grounding.ground_facts(objs, groups)

        # 3) filter rules
        kept_rules = [r for r in learned_rules if r.confidence >= conf_th]

        # 4) compute rule scores
        rule_score_dict = apply_rules(kept_rules, hard, soft, objs, groups)
        sorted_rules = sorted(rule_score_dict.items(), key=lambda x: x[1], reverse=True)
        topk_rules = [r[0] for r in sorted_rules[:top_k]]
        rule_score = [s for (_, s) in sorted_rules[:top_k]]
        while len(rule_score) < top_k:
            rule_score.append(0.0)

        # 5) compute image-level score
        if calibrator:
            calibrated_score = calibrator.predict_from_scores(rule_score, device)  # 返回校准后的概率
            # calibrated_score = float(calibrator(torch.tensor(rule_score).to(device)).detach().cpu().numpy())
            # calibrated_score = improved_calibrator.calibrate_one_image(calibrator, rule_score, hyp_params, device)
        else:
            calibrated_score = None
        vanilla_score = compute_image_score(kept_rules, rule_score_dict)

        # choose score for main metric
        img_score = calibrated_score if calibrator else vanilla_score

        per_task_scores[task_id].append(img_score)
        per_task_labels[task_id].append(true_label)

        # 6) classification decision
        predicted_label = int(img_score >= conf_th)
        if predicted_label != true_label:
            error_counters["total_errors"] += 1
            if is_grouping_error(groups, gt_objs):
                error_counters["grouping_error"] += 1
            elif is_object_error(objs, gt_objs):
                error_counters["object_error"] += 1
            else:
                error_counters["clause_mismatch"] += 1
        # 7) top-k clause selection analysis
        correct_clauses = [r for r in kept_rules if r.confidence > conf_th]
        if len(correct_clauses) > 0:
            topk_total_valid += 1
            if any(r in topk_rules for r in correct_clauses):
                topk_hits += 1

        if len(topk_rules) > 0:
            correct_in_topk = sum(1 for r in topk_rules if bool(r.confidence>conf_th) == bool(true_label))
            topk_precision_sum += correct_in_topk / top_k
            topk_precision_count += 1
            analysis["topk_clause_precision"].append(correct_in_topk / top_k)
        else:
            analysis["topk_clause_precision"].append(0.0)



        # 8) record additional analysis fields
        analysis["calibrated_scores"].append(calibrated_score)
        analysis["vanilla_scores"].append(vanilla_score)
        analysis["groundtruth_labels"].append(true_label)
        analysis["rule_pool_has_good_clause"].append(any(r.confidence > 0.5 for r in kept_rules))


    # compute metrics per task
    metrics = {}
    for task_id in per_task_scores:
        scores_list = per_task_scores[task_id]
        labels = per_task_labels[task_id]
        metrics = compute_metrics(scores_list, labels, threshold=conf_th)
    metrics["error_stats"] = error_counters

    # 9) top-k clause metrics
    if topk_total_valid > 0:
        metrics["topk_clause_recall"] = topk_hits / topk_total_valid
    if topk_precision_count > 0:
        metrics["topk_clause_precision"] = topk_precision_sum / topk_precision_count

    metrics["analysis"] = analysis
    return metrics


def eval_rules_legacy(val_data, obj_model, group_model, learned_rules, hyp_params, eval_principle, device, calibrator):
    patch_dim = hyp_params["patch_dim"]
    # we’ll collect per‐task true / pred
    per_task_scores = defaultdict(list)
    per_task_labels = defaultdict(list)
    conf_th = hyp_params["conf_th"]
    task_id = val_data["task"][0].split("_")[0]
    all_data = val_data["positive"] + val_data["negative"]
    img_paths = [d["image_path"][0] for d in all_data]
    imgs = patch_preprocess.load_images_fast(img_paths, device=device)
    gt_labels = [int(d["img_label"][0]) for d in all_data]
    # for each rule, we need to know the valuation result, so either 1 or 0
    for i, img in enumerate(imgs):
        true_label = gt_labels[i]
        # 1) detect objects & groups
        objs = eval_patch_classifier.evaluate_image(obj_model, img, device)
        groups = eval_groups.eval_groups(objs, group_model, eval_principle, device, patch_dim)
        # 2) ground  & generate validation image’s clauses
        hard, soft = grounding.ground_facts(objs, groups)
        # 4) filter out very‐low confidence rules
        kept_rules = [r for r in learned_rules if r.confidence >= conf_th]
        # 5) soft‐match each rule on this image
        rule_score_dict = apply_rules(kept_rules, hard, soft, objs, groups)
        rule_score = list(rule_score_dict.values())
        # Pad if fewer than k
        while len(rule_score) < hyp_params["top_k"]:
            rule_score.append(0.0)
        if calibrator:
            img_score = calibrator(torch.tensor(rule_score).to(device)).detach().cpu().numpy()
        else:
            img_score = compute_image_score(kept_rules, rule_score_dict)
        per_task_scores[task_id].append(img_score)
        per_task_labels[task_id].append(true_label)
    # now compute metrics per task
    metrics = {}
    for task_id in per_task_scores:
        # build lists of just the per‐image score dicts and labels
        scores_list = per_task_scores[task_id]
        labels = per_task_labels[task_id]
        # compute raw metrics (acc, f1, auc)
        metrics = compute_metrics(scores_list, labels, threshold=conf_th)
    return metrics
