# Created by MacBook Pro at 16.05.25
from typing import List, Tuple, Set, NamedTuple, Dict, Optional, Any
import time
import torch.nn as nn
import torch.optim as optim
import torch
from itertools import combinations
from copy import deepcopy

from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.grounding import grounding
from mbg.language import clause_generation
from mbg.language.clause_generation import ScoredRule
from mbg.evaluation import evaluation
from mbg.scorer.context_contour_scorer import ContextContourScorer
from mbg.scorer.calibrator import ConfidenceCalibrator
from mbg import patch_preprocess
from src import bk


def train_grouping_model(train_loader, device, epochs=10, LR=1e-3):
    # pair_dataset = build_pairwise_grouping_dataset(train_loader)  # returns (patch_i, patch_j, label) pairs
    model = ContextContourScorer()  # a small convnet or MLP
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0

        for data in train_loader:
            patch_i, patch_j, label = patch_i.to(
                device), patch_j.to(device), label.to(device)
            pred = model(patch_i, patch_j)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


def train_rules(hard_facts, soft_facts, obj_list, group_list, num_groups, train_image_labels, hyp_params,
                ablation_flags, task_times):
    # 1) 收集每张图的频次 & group 数
    pos_per_task_train = []  # task_id -> List[Counter[Clause,int]] (正例)
    neg_per_task_train = []  # task_id -> List[Counter[Clause,int]] (负例)
    # task_id -> List[int] 每张正例图的 group 数
    pos_group_counts_train = num_groups[:len(num_groups) // 2]
    # task_id -> List[int] 每张负例图的 group 数
    neg_group_counts_train = num_groups[len(num_groups) // 2:]
    # base rule learning
    for i in range(len(train_image_labels)):
        t1 = time.time()
        img_label = train_image_labels[i]
        hard, soft = hard_facts[i], soft_facts[i]
        cg = clause_generation.ClauseGenerator(prox_thresh=hyp_params["prox"], sim_thresh=hyp_params["sim"])
        clauses = cg.generate(hard, soft, obj_list[i], group_list[i], ablation_flags)
        # freq = Counter(clauses)
        # --- 4. 存入对应容器 ---
        if img_label == 1:
            pos_per_task_train.append(clauses)
        else:
            neg_per_task_train.append(clauses)
        t2 = time.time()
        task_times[i] += t2 - t1

    rules_img = clause_generation.filter_image_level_rules(pos_per_task_train, neg_per_task_train)
    rules_g_exist = clause_generation.filter_group_existential_rules(pos_per_task_train, neg_per_task_train)
    rules_g_universal = clause_generation.filter_group_universal_rules(pos_per_task_train, neg_per_task_train, pos_group_counts_train, neg_group_counts_train)
    rules_train = clause_generation.assemble_final_rules(rules_img, rules_g_exist, rules_g_universal, top_k=hyp_params["top_k"])

    return rules_train


def extend_rules_legacy(base_rules, hard_facts_list, soft_facts_list, img_labels, objs_list, groups_list, hyp_params,
                 min_conf=0.6):
    # 1. combine rules
    combined_rules = []
    N_pos = len(img_labels) // 2
    N_neg = len(objs_list) // 2
    # --- 1. Generate all pairs of base rules with same head and scope ---
    for r1, r2 in combinations(base_rules, 2):
        if r1.c.head == r2.c.head and r1.scope == r2.scope:
            merged_body = list(set(r1.c.body) | set(r2.c.body))
            if len(merged_body) == len(r1.c.body) + 1:  # avoid duplicates
                new_clause = deepcopy(r1.c)
                new_clause.body = merged_body
                combined_rules.append((new_clause, r1.scope))

    all_scored: List[ScoredRule] = base_rules
    # --- 2. Evaluate confidence for each combined rule ---
    for clause, scope in combined_rules:
        # clause_conf = 0
        pos_count = 0
        neg_count = 0
        # tp, fp = 0, 0
        for hard_facts, soft_facts, objs, groups, label in zip(hard_facts_list, soft_facts_list, objs_list, groups_list,
                                                               img_labels):
            # Prepare evaluation dict
            fn = evaluation._SCOPE_FN.get(scope)
            if fn is None:
                raise ValueError(f"Unknown scope {scope}")
            if len(objs) == 0 or len(groups) == 0:
                clause_image_score = 0
            else:
                clause_image_score = fn(clause.body, hard_facts, soft_facts, objs, groups)

            if label == 1 and clause_image_score > min_conf:
                pos_count += 1
            elif label == 0 and clause_image_score > min_conf:
                neg_count += 1
        # calculate clause confidence
        support = pos_count / N_pos
        fpr = (neg_count / N_neg) if N_neg > 0 else 0.0
        score = support * (1.0 - fpr)
        if support > 0:
            all_scored.append(ScoredRule(clause, score, scope))

    # sort by confidence descending
    all_scored.sort(key=lambda sr: sr.confidence, reverse=True)
    return all_scored[:hyp_params["top_k"]]



def extend_rules(base_rules, hard_facts_list, soft_facts_list, img_labels, objs_list, groups_list, hyp_params,
                 min_conf=0.6, n_iter=3):
    """
    Iteratively extend rules for n_iter times.
    """
    from collections import defaultdict

    def _generate_combined_rules(base, current):
        combined = []
        seen = set()
        for r1 in base:
            for r2 in current:
                if r1.c.head == r2.c.head and r1.scope == r2.scope:
                    merged_body = tuple(sorted(set(r1.c.body) | set(r2.c.body)))
                    if len(merged_body) == len(r1.c.body) + 1 and merged_body not in seen:
                        new_clause = deepcopy(r1.c)
                        new_clause.body = list(merged_body)
                        combined.append((new_clause, r1.scope))
                        seen.add(merged_body)
        return combined

    def _evaluate_rule_confidence(clause, scope, data, min_conf):
        N_pos = data['N_pos']
        N_neg = data['N_neg']
        pos_count, neg_count = 0, 0
        for hard_facts, soft_facts, objs, groups, label in zip(
                data['hard_facts_list'], data['soft_facts_list'],
                data['objs_list'], data['groups_list'], data['img_labels']):
            fn = evaluation._SCOPE_FN.get(scope)
            if fn is None:
                raise ValueError(f"Unknown scope {scope}")
            if len(objs) == 0 or len(groups) == 0:
                clause_image_score = 0
            else:
                clause_image_score = fn(clause.body, hard_facts, soft_facts, objs, groups)
            if label == 1 and clause_image_score > min_conf:
                pos_count += 1
            elif label == 0 and clause_image_score > min_conf:
                neg_count += 1
        support = pos_count / N_pos if N_pos > 0 else 0.0
        fpr = (neg_count / N_neg) if N_neg > 0 else 0.0
        score = support * (1.0 - fpr)
        return support, score

    data = {
        'hard_facts_list': hard_facts_list,
        'soft_facts_list': soft_facts_list,
        'objs_list': objs_list,
        'groups_list': groups_list,
        'img_labels': img_labels,
        'N_pos': len([l for l in img_labels if l == 1]),
        'N_neg': len([l for l in img_labels if l == 0])
    }

    all_scored = list(base_rules)
    current_rules = list(base_rules)
    seen_bodies = set(tuple(sorted(r.c.body)) for r in base_rules)

    for _ in range(2):
        combined_rules = _generate_combined_rules(base_rules, current_rules)
        new_scored = []
        for clause, scope in combined_rules:
            body_tuple = tuple(sorted(clause.body))
            if body_tuple in seen_bodies:
                continue
            support, score = _evaluate_rule_confidence(clause, scope, data, min_conf)
            if support > 0:
                sr = ScoredRule(clause, score, scope)
                new_scored.append(sr)
                seen_bodies.add(body_tuple)
        if not new_scored:
            break  # No new rules generated
        all_scored.extend(new_scored)
        current_rules = new_scored

    all_scored.sort(key=lambda sr: sr.confidence, reverse=True)
    return all_scored[:hyp_params["top_k"]]




def train_calibrator(final_rules, obj_list, group_list, hard_list, soft_list, img_labels, hyp_params, ablation_flags, device):
    calib_inputs = []
    calib_labels = []

    for hard_facts, soft_facts, objs, groups, label in zip(hard_list, soft_list, obj_list, group_list, img_labels):
        rule_score_dict = evaluation.apply_rules(final_rules, hard_facts, soft_facts, objs, groups)
        rule_scores = [v for v in rule_score_dict.values()]

        while len(rule_scores) < hyp_params["top_k"]:
            rule_scores.append(0.0)  # pad to k

        calib_inputs.append(rule_scores)
        calib_labels.append(float(label))
    if ablation_flags["use_calibrator"]:
        calibrator = ConfidenceCalibrator(input_dim=hyp_params["top_k"]).to(device)
        calibrator.train_from_data(calib_inputs, calib_labels, device=device)
    else:
        calibrator = None
    return calibrator


def ground_facts(train_data, obj_model, group_model, hyp_params, train_principle, device, ablation_flags, task_times):
    if ablation_flags is None:
        ablation_flags = {}
    disable_soft = not ablation_flags.get("use_soft", True)
    disable_hard = not ablation_flags.get("use_hard", True)
    disable_group = not ablation_flags.get("use_group", True)
    disable_object = not ablation_flags.get("use_obj", True)

    hard_facts, soft_facts = [], []
    obj_lists, groups_list = [], []
    group_nums = []

    all_data = train_data["positive"] + train_data["negative"]
    img_paths = [d["image_path"][0] for d in all_data]
    imgs = patch_preprocess.load_images_fast(img_paths, device=device)


    for i, img in enumerate(imgs):
        t1 = time.time()
        # --- 1. Object detection ---
        if not disable_object:
            objs = eval_patch_classifier.evaluate_image(obj_model, img, device)
        else:
            objs = []  # no object-level facts
        obj_lists.append(objs)

        # --- 2. Group detection ---
        if not disable_group:
            groups = eval_groups.eval_groups(objs, group_model, train_principle, device, dim=hyp_params["patch_dim"])
        else:
            groups = []
        groups_list.append(groups)
        group_nums.append(len(groups))

        # --- 3. Fact grounding ---
        hard, soft = grounding.ground_facts(objs, groups, disable_hard=disable_hard, disable_soft=disable_soft)
        hard_facts.append(hard)
        soft_facts.append(soft)
        t2 = time.time()
        task_times[i] += t2 - t1

    return hard_facts, soft_facts, group_nums, obj_lists, groups_list



def ground_clevr_facts(train_data, obj_model, group_model, hyp_params, train_principle, device, ablation_flags, task_times, use_gt_groups=False):
    """Ground facts for CLEVR dataset using ground truth object information.
    
    Uses symbolic_data from CLEVR instead of running object detection.
    If use_gt_groups=True, uses ground truth group annotations instead of trained model.
    """
    if ablation_flags is None:
        ablation_flags = {}
    disable_soft = not ablation_flags.get("use_soft", True)
    disable_hard = not ablation_flags.get("use_hard", True)
    disable_group = not ablation_flags.get("use_group", True)
    disable_object = not ablation_flags.get("use_obj", True)

    hard_facts, soft_facts = [], []
    obj_lists, groups_list = [], []
    group_nums = []

    all_data = train_data["positive"] + train_data["negative"]

    for i, data_sample in enumerate(all_data):
        t1 = time.time()
        
        # --- 1. Use ground truth objects from symbolic_data ---
        if not disable_object:
            symbolic_data = data_sample["symbolic_data"]
            objs = []
            for obj_idx, obj_gt in enumerate(symbolic_data):
                # Convert shape ID to one-hot encoding
                shape_one_hot = torch.zeros(len(bk.bk_shapes_clevr), dtype=torch.float32)
                shape_id = obj_gt['shape']
                if 0 <= shape_id < len(bk.bk_shapes_clevr):
                    shape_one_hot[shape_id] = 1.0 
                
                # Create object in expected format
                obj = {
                    "id": obj_idx,
                    "s": {
                        "shape": shape_one_hot,
                        "color": [obj_gt['color_r'], obj_gt['color_g'], obj_gt['color_b']],
                        "x": obj_gt['x'],
                        "y": obj_gt['y'],
                        "w": obj_gt['size'],
                        "h": obj_gt['size'],
                    },
                    "h": torch.zeros((32, 2), dtype=torch.float32)  # Dummy contour patches
                }
                objs.append(obj)
        else:
            objs = []  # no object-level facts
        
        obj_lists.append(objs)

        # --- 2. Group detection ---
        if not disable_group:
            if use_gt_groups:
                # Use ground truth group IDs from symbolic data
                from collections import defaultdict
                gt_groups = defaultdict(list)
                for obj_idx, obj_gt in enumerate(symbolic_data):
                    group_id = obj_gt.get('group_id', -1)
                    if group_id is not None and group_id >= 0:
                        gt_groups[group_id].append(obj_idx)
                
                # Convert to group format expected by the pipeline
                # Only include groups with 2 or more objects
                groups = []
                for g_idx, obj_indices in gt_groups.items():
                    if len(obj_indices) >= 2:  # Ensure group has at least 2 objects
                        group_objs = [objs[i] for i in obj_indices]
                        grp = {
                            "id": g_idx,
                            "child_obj_ids": obj_indices,
                            "members": group_objs,
                            "h": None,
                            "principle": train_principle
                        }
                        groups.append(grp)
            else:
                # Use trained group detector model
                groups = eval_groups.eval_clevr_groups(objs, group_model, train_principle, device, dim=hyp_params["patch_dim"])
            
            # Ground truth grouping code (for reference)
            # from collections import defaultdict
            # gt_groups = defaultdict(list)
            # for obj_idx, obj_gt in enumerate(symbolic_data):
            #     group_id = obj_gt.get('group_id', -1)
            #     if group_id is not None and group_id >= 0:
            #         gt_groups[group_id].append(obj_idx)
            # 
            # # Convert to group format expected by the pipeline
            # # Only include groups with 2 or more objects
            # groups = []
            # for g_idx, obj_indices in gt_groups.items():
            #     if len(obj_indices) >= 2:  # Ensure group has at least 2 objects
            #         group_objs = [objs[i] for i in obj_indices]
            #         grp = {
            #             "id": g_idx,
            #             "child_obj_ids": obj_indices,
            #             "members": group_objs,
            #             "h": None,
            #             "principle": train_principle
            #         }
            #         groups.append(grp)
        else:
            groups = []
        groups_list.append(groups)
        group_nums.append(len(groups))

        # --- 3. Fact grounding ---
        hard, soft = grounding.ground_facts(objs, groups, disable_hard=disable_hard, disable_soft=disable_soft)
        
        # Ensure no empty tensors in hard_facts - replace with appropriate size zero tensor
        obj_num = len(objs)
        group_num = len(groups)
        
        # Define group-level predicates that should have shape [G]
        group_level_predicates = {"group_size", "principle"}
        
        for pred_name, pred_tensor in hard.items():
            if pred_tensor.numel() == 0:  # Check if tensor is empty
                # Special handling for in_group which is 2D [O, G]
                if pred_name == "in_group":
                    hard[pred_name] = torch.zeros(obj_num, group_num, dtype=pred_tensor.dtype, device=pred_tensor.device)
                # Group-level predicates or predicates starting with group-level patterns
                elif (pred_name in group_level_predicates or 
                      pred_name.startswith("no_member_") or 
                      pred_name.startswith("diverse_") or 
                      pred_name.startswith("unique_")):
                    hard[pred_name] = torch.zeros(group_num, dtype=pred_tensor.dtype, device=pred_tensor.device)
                # Object-level predicates
                else:
                    hard[pred_name] = torch.zeros(obj_num, dtype=pred_tensor.dtype, device=pred_tensor.device)
        
        hard_facts.append(hard)
        soft_facts.append(soft)
        t2 = time.time()
        task_times[i] += t2 - t1

    return hard_facts, soft_facts, group_nums, obj_lists, groups_list