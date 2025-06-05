# Created by MacBook Pro at 16.05.25
from typing import List, Tuple, Set, NamedTuple, Dict, Optional, Any

import itertools
from collections import Counter
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
            patch_i, patch_j, label = patch_i.to(device), patch_j.to(device), label.to(device)
            pred = model(patch_i, patch_j)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


def train_rules(hard_facts, soft_facts, num_groups, train_image_labels, hyp_params):
    # 1) 收集每张图的频次 & group 数
    pos_per_task_train = []  # task_id -> List[Counter[Clause,int]] (正例)
    neg_per_task_train = []  # task_id -> List[Counter[Clause,int]] (负例)
    pos_group_counts_train = num_groups[:len(num_groups) // 2]  # task_id -> List[int] 每张正例图的 group 数
    neg_group_counts_train = num_groups[len(num_groups) // 2:]  # task_id -> List[int] 每张负例图的 group 数

    # base rule learning
    for i in range(len(train_image_labels)):
        img_label = train_image_labels[i]
        hard, soft = hard_facts[i], soft_facts[i]
        cg = clause_generation.ClauseGenerator(prox_thresh=hyp_params["prox"], sim_thresh=hyp_params["sim"])
        clauses = cg.generate(hard, soft)
        freq = Counter(clauses)
        # --- 4. 存入对应容器 ---
        if img_label == 1:
            pos_per_task_train.append(freq)

        else:
            neg_per_task_train.append(freq)

    rules_img = clause_generation.filter_image_level_rules(pos_per_task_train, neg_per_task_train)
    rules_g_exist = clause_generation.filter_group_existential_rules(pos_per_task_train, neg_per_task_train)
    rules_g_universal = clause_generation.filter_group_universal_rules(pos_per_task_train, neg_per_task_train,
                                                                       pos_group_counts_train, neg_group_counts_train)
    rules_train = clause_generation.assemble_final_rules(rules_img, rules_g_exist, rules_g_universal,
                                                         top_k=hyp_params["top_k"])

    return rules_train


def extend_rules(base_rules, hard_facts_list, soft_facts_list, img_labels, objs_list, groups_list,hyp_params, min_conf=0.6):
    # 1. combine rules
    combined_rules = []
    N_pos = len(img_labels) // 2
    N_neg = len(objs_list) // 2
    # --- 1. Generate all pairs of base rules with same head and scope ---
    for r1, r2 in combinations(base_rules, 2):
        if r1.clause.head == r2.clause.head and r1.scope == r2.scope:
            merged_body = list(set(r1.clause.body) | set(r2.clause.body))
            if len(merged_body) == len(r1.clause.body) + 1:  # avoid duplicates
                new_clause = deepcopy(r1.clause)
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

def train_calibrator(final_rules, obj_list, group_list, hard_list, soft_list,img_labels, hyp_params):
    calib_inputs = []
    calib_labels = []

    for hard_facts, soft_facts, objs, groups, label in zip(hard_list, soft_list, obj_list, group_list,
                                                           img_labels):

        rule_score_dict = evaluation.apply_rules(final_rules, hard_facts, soft_facts, objs, groups)
        rule_scores = [v for v in rule_score_dict.values()]


        while len(rule_scores) < hyp_params["top_k"]:
            rule_scores.append(0.0)  # pad to k

        calib_inputs.append(rule_scores)
        calib_labels.append(float(label))
    calibrator = ConfidenceCalibrator(input_dim=hyp_params["top_k"])
    calibrator.train_from_data(calib_inputs, calib_labels)
    return calibrator


def ground_facts(train_data, obj_model, hyp_params, train_principle, device):
    hard_facts, soft_facts = [], []
    obj_lists, groups_list = [], []
    group_nums = []
    for data in train_data["positive"] + train_data["negative"]:
        # --- 2. 物体 & 分组检测 ---
        objs = eval_patch_classifier.evaluate_image(obj_model, data["image_path"][0])
        groups = eval_groups.eval_groups(objs, hyp_params["prox"], train_principle, device, dim=hyp_params["patch_dim"])
        # --- 3. Grounding & Clause Generation ---
        hard, soft = grounding.ground_facts(objs, groups)
        hard_facts.append(hard)
        soft_facts.append(soft)
        group_nums.append(len(groups))
        groups_list.append(groups)
        obj_lists.append(objs)
    return hard_facts, soft_facts, group_nums, obj_lists, groups_list
