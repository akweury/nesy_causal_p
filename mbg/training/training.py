# Created by MacBook Pro at 16.05.25
import itertools
from collections import Counter, defaultdict
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbg.object import eval_patch_classifier
from mbg.group import eval_groups
from mbg.grounding import grounding
from mbg.language import clause_generation
from mbg.evaluation import evaluation
from mbg.scorer.context_contour_scorer import ContextContourScorer

def train_grouping_model(train_loader, device, epochs=10, LR = 1e-3):
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
def train_rules(train_data, obj_model, hyp_params, train_principle):
    # 1) 收集每张图的频次 & group 数
    pos_per_task_train = defaultdict(list)  # task_id -> List[Counter[Clause,int]] (正例)
    neg_per_task_train = defaultdict(list)  # task_id -> List[Counter[Clause,int]] (负例)
    pos_group_counts_train = defaultdict(list)  # task_id -> List[int] 每张正例图的 group 数
    neg_group_counts_train = defaultdict(list)  # task_id -> List[int] 每张负例图的 group 数
    task_id = train_data["task"][0].split("_")[0]
    for data in train_data["positive"]+train_data["negative"]:
        # --- 1. 基础信息读取 ---
        img_label = int(data["img_label"])  # 1 or 0

        # --- 2. 物体 & 分组检测 ---
        objs = eval_patch_classifier.evaluate_image(obj_model, data)
        groups = eval_groups.eval_groups(objs, hyp_params["prox"], train_principle)
        num_groups = len(groups)

        # --- 3. Grounding & Clause Generation ---
        hard, soft = grounding.ground_facts(objs, groups)
        cg = clause_generation.ClauseGenerator(prox_thresh=hyp_params["prox"], sim_thresh=hyp_params["sim"])
        clauses = cg.generate(hard, soft)
        freq = Counter(clauses)

        # --- 4. 存入对应容器 ---
        if img_label == 1:
            pos_per_task_train[task_id].append(freq)
            pos_group_counts_train[task_id].append(num_groups)
        else:
            neg_per_task_train[task_id].append(freq)
            neg_group_counts_train[task_id].append(num_groups)
    img_rules_train = clause_generation.filter_image_level_rules(pos_per_task_train, neg_per_task_train)
    exist_rules_train = clause_generation.filter_group_existential_rules(pos_per_task_train, neg_per_task_train)
    univ_rules_train = clause_generation.filter_group_universal_rules(pos_per_task_train, neg_per_task_train,
                                                                      pos_group_counts_train, neg_group_counts_train)
    rules_train = clause_generation.assemble_final_rules(img_rules_train, exist_rules_train, univ_rules_train,
                                                         top_k=hyp_params["top_k"])
    return rules_train


def grid_search(args, train_data, val_data, obj_model):
    best_cfg = None
    best_acc = -1.0

    for prox, sim, top_k in itertools.product(
            [0.9],
            [0.5],
            [5]
    ):
        hyp_params = {"prox": prox, "sim": sim, "top_k": top_k, "conf_th":0.5}

        # Step 0: Train the proximity grouping model
        # group_prox_model = train_grouping_model(train_loader, device=args.device)
        # learn on train
        rules = train_rules(train_data, obj_model, hyp_params)

        # eval on val
        val_metrics = evaluation.eval_rules(val_data, obj_model, rules, hyp_params)
        # average acc across tasks
        # avg_acc = sum(v["acc"] for v in val_metrics.values()) / len(val_metrics)

        # print(f"prox={prox:.2f} sim={sim:.2f} top_k={top_k} → val_avg_acc={avg_acc:.3f}")
        # if avg_acc > best_acc:
        #     best_acc = avg_acc
        #     best_cfg = (prox, sim, top_k)

    # print("BEST:", best_cfg, "with acc=", best_acc)
    val_metrics = {"acc": best_acc}
    return best_cfg, val_metrics
