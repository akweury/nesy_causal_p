# Created by MacBook Pro at 19.09.25
# calibrators.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mbg.evaluation import evaluation
from typing import List, Optional


# ---------------------------
# Utils: metrics & losses
# ---------------------------

def brier_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error between prob and label."""
    return ((p - y) ** 2).mean()


def soft_ece(p: torch.Tensor, y: torch.Tensor, n_bins: int = 15, temp: float = 12.0) -> torch.Tensor:
    """
    Differentiable Expected Calibration Error (scalar).
    p, y: (N, 1) probabilities and {0,1} labels.
    """
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    N = p.shape[0]
    edges = torch.linspace(0, 1, n_bins + 1, device=p.device, dtype=p.dtype)
    mids = 0.5 * (edges[:-1] + edges[1:])  # (B,)
    # soft assignment of each sample to bins
    # higher temp -> sharper assignment
    assign = torch.exp(-temp * (p - mids.view(1, -1)).abs())  # (N,B)
    assign = assign / (assign.sum(dim=1, keepdim=True) + 1e-8)

    # per-bin total weight
    Zb = assign.sum(dim=0).clamp_min(1e-8)  # (B,)
    # per-bin avg confidence & accuracy (soft)
    conf_b = (assign * p).sum(dim=0) / Zb
    acc_b = (assign * (y > 0.5).float()).sum(dim=0) / Zb

    # bin weights normalized by N
    wb = Zb / (Zb.sum() + 1e-8)
    ece = (wb * (conf_b - acc_b).abs()).sum()
    return ece


# ---------------------------
# 1) Temperature Scaling
# ---------------------------

class TempScaler(nn.Module):
    """
    Platt-style temperature scaling for logits: p = sigmoid(logit / T).
    Fit on a held-out validation set to minimize NLL.
    """

    def __init__(self, init_logT: float = 0.0):
        super().__init__()
        self.logT = nn.Parameter(torch.tensor(init_logT))

    @property
    def T(self):
        return self.logT.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.T)

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward(logits)

    def fit_once(self, logits, y, iters=200):
        logits = logits.detach();
        y = y.detach().float()
        bce = nn.BCEWithLogitsLoss()
        opt = torch.optim.LBFGS([self.logT], max_iter=iters, line_search_fn="strong_wolfe")

        def cl(): opt.zero_grad(); loss = bce(logits / self.T, y); loss.backward(); return loss

        opt.step(cl)

    def fit(self, logits: torch.Tensor, y: torch.Tensor, max_iter: int = 500):
        """
        logits: (N,1) raw scores before sigmoid; y: (N,1) in {0,1}
        """
        logits = logits.detach()
        y = y.detach().float()
        loss_fn = nn.BCEWithLogitsLoss()

        # L-BFGS works well for 1-2 params
        opt = torch.optim.LBFGS(self.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_fn(logits / self.T, y)
            loss.backward()
            return loss

        opt.step(closure)
        return self


@torch.no_grad()
def cv_temperature_scaling(logits, y, folds="loo", iters=300):
    """
    logits,y: tensors on same device, shape (N,1)
    返回：一个已训练好的 TempScaler（其 T 为各折几何平均）
    """
    N = logits.size(0)
    Ts = []
    if folds == "loo" or N < 5:
        idx_splits = [([j for j in range(N) if j != i], [i]) for i in range(N)]
    else:
        # k-fold
        k = int(folds)
        perm = torch.randperm(N).tolist()
        chunk = [perm[i::k] for i in range(k)]
        idx_splits = []
        for t in range(k):
            val = chunk[t]
            tr = [i for j, c in enumerate(chunk) if j != t for i in c]
            idx_splits.append((tr, val))

    for tr, _ in idx_splits:
        ts = TempScaler().to(logits.device)
        ts.fit_once(logits[tr], y[tr], iters=iters)
        Ts.append(float(ts.T.detach().cpu()))

    # 几何平均融合
    T_geo = math.exp(sum(math.log(max(1e-6, t)) for t in Ts) / len(Ts))
    final = TempScaler().to(logits.device)
    with torch.no_grad():
        final.logT.copy_(torch.tensor(math.log(T_geo), device=logits.device))
    return final


# ---------------------------
# 2) Beta Calibrator (heteroscedastic)
# ---------------------------

def _softplus(x):  # stable softplus
    return F.softplus(x, beta=1.0, threshold=20.0)


class BetaCalibrator(nn.Module):
    """
    Learn sample-wise Beta(a,b) over Bernoulli probability.
    Output calibrated probability = E[p] = a/(a+b).

    Suggested training loss:
      L = BetaCE + λ_brier * Brier + λ_ece * SoftECE
    where BetaCE uses E[log p] / E[log(1-p)] under Beta.
    """

    def __init__(self, input_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, feats: torch.Tensor):
        """
        feats: (N, D)
        returns:
          p: (N,1) calibrated mean
          a,b: (N,1) positive concentration parameters
        """
        out = self.net(feats)
        a_raw, b_raw = out.chunk(2, dim=-1)
        a = _softplus(a_raw) + 1e-4
        b = _softplus(b_raw) + 1e-4
        p = a / (a + b)
        return p, a, b

    @staticmethod
    def beta_ce(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Beta cross-entropy = - [ y * E(log p) + (1-y) * E(log (1-p)) ].
        E[log p] = ψ(a) - ψ(a+b), E[log(1-p)] = ψ(b) - ψ(a+b).
        """
        y = y.float()
        psi = torch.special.digamma
        elogp = psi(a) - psi(a + b)
        elog1mp = psi(b) - psi(a + b)
        nll = -(y * elogp + (1.0 - y) * elog1mp)
        return nll.mean()

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            device: torch.device,
            epochs: int = 200,
            lr: float = 1e-3,
            lambda_brier: float = 0.2,
            lambda_ece: float = 0.1,
            batch_size: int = 4096,
            verbose: bool = True):

        self.to(device)
        X = X.to(device).float()
        y = y.to(device).float().view(-1, 1)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = X.shape[0]
        idx = torch.arange(n, device=device)

        for ep in range(epochs):
            # mini-batch SGD
            perm = idx[torch.randperm(n)]
            total = 0.0
            for i in range(0, n, batch_size):
                j = perm[i:i + batch_size]
                xb, yb = X[j], y[j]
                p, a, b = self.forward(xb)
                loss = self.beta_ce(a, b, yb)
                loss = loss + lambda_brier * brier_loss(p, yb) + lambda_ece * soft_ece(p, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)

            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"[BetaCal] epoch {ep:03d}  loss={total / n:.5f}")

        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.eval()
        X = X.to(device).float()
        p, _, _ = self.forward(X)
        return p


# ---------------------------
# 3) 可选：规则选择门控 (Sparsemax)
# ---------------------------

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).
    Returns sparse probability simplex projection.
    """
    # sort
    z_sorted, _ = torch.sort(logits, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim) - 1
    # k(z) = max { k | z_sort_k > (cumsum_k)/k }
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    k_shape = [1] * logits.dim()
    k_shape[dim] = -1
    k = k.view(k_shape)
    is_gt = z_sorted > z_cumsum / k
    k_z = is_gt.sum(dim=dim, keepdim=True)
    # tau
    tau = (z_cumsum.gather(dim, k_z - 1)) / k_z.clamp_min(1)
    # projection
    p = torch.clamp(logits - tau, min=0.0)
    # normalize (should already sum to 1 over support)
    Z = p.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return p / Z


class RuleSelectorGate(nn.Module):
    """
    对同一张图像的 K 条候选规则，输出稀疏权重 w (K,).
    输入:
      feats:  (K, D)  —— 每条规则的特征（可拼上统计特征）
    输出:
      w:      (K,)    —— 稀疏概率，sum=1
    """

    def __init__(self, feat_dim: int, hidden: int = 16, use_sparsemax: bool = True, temperature: float = 1.0):
        super().__init__()
        self.use_sparsemax = use_sparsemax
        self.temperature = temperature
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (K, D)
        logits = self.scorer(feats).squeeze(-1) / self.temperature
        if self.use_sparsemax:
            w = sparsemax(logits, dim=0)
        else:
            w = F.softmax(logits, dim=0)
        return w  # (K,)


# ---------------------------
# 组合：Beta 校准 + 规则加权（可选）
# ---------------------------

class BetaCalibratedSelector(nn.Module):
    """
    一个端到端的模块：
      - 对每条规则特征做 Beta 校准 -> 概率 p_i
      - 用门控模块输出权重 w_i
      - 聚合分数 s = Σ w_i * p_i   （也可换成对数域聚合）
    """

    def __init__(self, feat_dim: int, hidden_cal: int = 16, hidden_gate: int = 16, use_sparsemax: bool = True):
        super().__init__()
        self.beta_cal = BetaCalibrator(feat_dim, hidden=hidden_cal)
        self.gate = RuleSelectorGate(feat_dim, hidden=hidden_gate, use_sparsemax=use_sparsemax)

    def forward(self, feats: torch.Tensor):
        """
        feats: (K, D) rules for a single image.
        returns:
          p_i: (K,1) calibrated probs,
          w:   (K,)  weights,
          s:   ()    aggregated score
        """
        p_i, a, b = self.beta_cal(feats)  # (K,1)
        w = self.gate(feats)  # (K,)
        s = (w.view(-1, 1) * p_i).sum()  # scalar
        return s, w, p_i, (a, b)

    def loss(self,
             feats: torch.Tensor,
             y_img: torch.Tensor,
             lambda_sparse: float = 1e-3,
             lambda_brier: float = 0.1,
             lambda_ece: float = 0.05):
        """
        对一张图像的 K 条规则进行训练的损失:
         - 图像级 BCE(Σ w_i p_i, y_img)
         - 规则级校准正则（Brier + softECE）对 p_i 起到校准作用
         - 稀疏正则（L1 on w）
        """
        s, w, p_i, (a, b) = self.forward(feats)  # s: scalar prob in [0,1]
        # 图像级目标
        eps = 1e-8
        s_clamped = s.clamp(eps, 1 - eps)
        bce_img = -(y_img * torch.log(s_clamped) + (1 - y_img) * torch.log(1 - s_clamped))

        # 规则级校准正则
        y_rules = (y_img * torch.ones_like(p_i)).detach()
        beta_ce = self.beta_cal.beta_ce(a, b, y_rules)
        reg_cal = beta_ce + lambda_brier * brier_loss(p_i, y_rules) + lambda_ece * soft_ece(p_i, y_rules)

        # 稀疏/熵正则（鼓励选择少数几条规则）
        reg_sparse = w.abs().sum()

        return bce_img + reg_cal + lambda_sparse * reg_sparse


# --- improved_train_calibrator.py ---

import math
import torch
import torch.nn as nn
import numpy as np


# 如果放在同一文件，直接从前面定义的类中 import；否则：
# from calibrators import BetaCalibrator, TempScaler

def _build_features_from_scores(scores, k):
    """
    把 rule_scores 转成更有判别力的定长特征：
      [ 原始前K分数(降序&pad) ,
        mean, max, min, std,
        l1, l2, softmax_entropy ]
    返回: (k + 7,) 维
    """
    # 排序并截断/补零
    s = sorted(scores, reverse=True)[:k]
    if len(s) < k:
        s = s + [0.0] * (k - len(s))

    s_arr = np.asarray(s, dtype=np.float32)
    mean = float(np.mean(s_arr))
    mx = float(np.max(s_arr))
    mn = float(np.min(s_arr))
    std = float(np.std(s_arr))
    l1 = float(np.linalg.norm(s_arr, ord=1))
    l2 = float(np.linalg.norm(s_arr, ord=2))
    # softmax 熵（避免全0导致 NaN）
    sm = np.exp(s_arr - np.max(s_arr))
    sm = sm / (np.sum(sm) + 1e-8)
    ent = float(-np.sum(sm * (np.log(sm + 1e-8))))

    feats = np.concatenate([s_arr,
                            np.asarray([mean, mx, mn, std, l1, l2, ent], dtype=np.float32)], axis=0)
    return feats  # shape: (k+7,)


class CalibratorWrapper(nn.Module):
    """
    统一接口封装，便于后续替换校准器实现：
      kind in {"beta", "temp", "mlp"}
    """

    def __init__(self, kind, model):
        super().__init__()
        self.kind = kind
        self.model = model

    def forward(self, x, **kwargs):
        if self.kind == "beta":
            p, _, _ = self.model(x)
            return p
        elif self.kind == "mlp":
            return self.model(x)
        else:
            raise RuntimeError("TempScaler在预测时请直接调用 .predict(logits)，此 wrapper 的 forward 不用于 temp。")


def calibrate_one_image(calibrator, rule_scores, hyp_params, device):
    top_k = int(hyp_params.get("top_k", 8))
    feats = _build_features_from_scores(rule_scores, top_k)
    x = torch.tensor(feats, dtype=torch.float32, device=device).view(1, -1)

    if isinstance(calibrator, CalibratorWrapper):
        with torch.no_grad():
            p = calibrator(x)  # (1,1)
            return float(p.item())
    else:
        # TempScaler 分支：x 在此应是 logits
        with torch.no_grad():
            p = calibrator.predict(x)
            return float(p.item())


# ========== 1) 仿射缩放器：p = sigmoid(a * logit + b) + CV 拟合 ==========

class AffineScaler(nn.Module):
    def __init__(self, a_init=1.0, b_init=0.0, l2=1e-2):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.l2 = l2

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)

    def fit_once(self, logits: torch.Tensor, y: torch.Tensor, iters: int = 400):
        logits = logits.detach();
        y = y.detach().float()
        opt = torch.optim.LBFGS(self.parameters(), max_iter=iters, line_search_fn="strong_wolfe")
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(self.a * logits + self.b, y)
            loss = loss + self.l2 * (self.a ** 2 + self.b ** 2)  # 收缩，防小样本发散
            loss.backward()
            return loss

        opt.step(closure)
        return self


@torch.no_grad()
def cv_affine_scaling(logits: torch.Tensor,
                      y: torch.Tensor,
                      folds="loo",
                      iters: int = 400,
                      l2: float = 1e-2) -> AffineScaler:
    """在小样本上更稳的 CV/LOO 拟合；融合时 a 用几何平均，b 用算术平均。"""
    N = logits.size(0)
    if folds == "loo" or N < 6:
        splits = [([j for j in range(N) if j != i], [i]) for i in range(N)]
    else:
        k = int(folds)
        perm = torch.randperm(N).tolist()
        chunks = [perm[i::k] for i in range(k)]
        splits = []
        for t in range(k):
            val = chunks[t]
            tr = [i for j, c in enumerate(chunks) if j != t for i in c]
            splits.append((tr, val))

    A, B = [], []
    for tr, _ in splits:
        m = AffineScaler(l2=l2).to(logits.device)
        m.fit_once(logits[tr], y[tr], iters=iters)
        A.append(float(m.a.detach().cpu()))
        B.append(float(m.b.detach().cpu()))

    a_geo = math.exp(sum(math.log(max(1e-6, a)) for a in A) / len(A))
    b_mean = sum(B) / len(B)

    final = AffineScaler(l2=l2).to(logits.device)
    final.a.data = torch.tensor(a_geo, device=logits.device)
    final.b.data = torch.tensor(b_mean, device=logits.device)
    return final


# ========== 2) 包装器：直接以 rule_scores 推理（与现有调用对齐） ==========

class RuleAffineCalibrator(nn.Module):
    """
    将 top-k rule_scores 聚合为 anchor 概率 p_anchor（默认取 top1），
    转为 logit 后过 AffineScaler 得到校准概率。
    """

    def __init__(self, scaler: AffineScaler, top_k: int = 8, agg: str = "top1"):
        super().__init__()
        self.scaler = scaler
        self.top_k = int(top_k)
        assert agg in {"top1", "mean", "top3_mean"}, "agg must be one of {'top1','mean','top3_mean'}"
        self.agg = agg

    @staticmethod
    def _aggregate(scores: List[float], k: int, agg: str) -> float:
        s = sorted(scores, reverse=True)[:k]
        if not s:
            return 0.0
        if agg == "top1":
            return float(s[0])
        elif agg == "mean":
            return float(np.mean(s, dtype=np.float32))
        else:  # top3_mean
            return float(np.mean(s[:3], dtype=np.float32))

    @torch.no_grad()
    def predict_from_scores(self, rule_scores: List[float], device: torch.device) -> float:
        p_anchor = self._aggregate(rule_scores, self.top_k, self.agg)
        p_anchor = float(np.clip(p_anchor, 1e-6, 1 - 1e-6))
        logit = math.log(p_anchor) - math.log(1 - p_anchor)
        t = torch.tensor([[logit]], dtype=torch.float32, device=device)
        p_cal = self.scaler(t)
        return float(p_cal.item())


# ========== 3) 你的训练函数：最小改动替换 ==========

def train_calibrator(final_rules,
                     obj_list,
                     group_list,
                     hard_list,
                     soft_list,
                     img_labels,
                     hyp_params,
                     ablation_flags,
                     device):
    """
    改进点：
    - 使用 CV/LOO 仿射缩放校准 (a,b)，极小 eval 集也稳定。
    - 仍以 top-k rule_scores -> anchor 概率，再转 logit 拟合。
    - 可通过 hyp_params 指定用于校准的样本索引或数量。
    额外可配：
      hyp_params["top_k"] (int, default 8)
      hyp_params["calib_indices"] (List[int])  # 明确哪些样本用来拟合校准
      hyp_params["calib_n"] (int)              # 若无 indices，就取这么多个样本随机做校准
      hyp_params["cv_folds"] ("loo"或int)      # 交叉验证折数，默认 "loo"
      hyp_params["affine_l2"] (float)          # 仿射参数 L2 收缩强度
      hyp_params["affine_iters"] (int)         # LBFGS 迭代
      hyp_params["anchor_agg"] ("top1"|"mean"|"top3_mean")
    """
    if not ablation_flags.get("use_calibrator", True):
        return None

    top_k = int(hyp_params.get("top_k", 8))
    cv_folds = hyp_params.get("cv_folds", "loo")
    l2 = float(hyp_params.get("affine_l2", 1e-2))
    iters = int(hyp_params.get("affine_iters", 400))
    agg = hyp_params.get("anchor_agg", "top1")

    # 1) 生成每张图像的 rule_scores & anchor 概率
    anchor_probs = []
    labels = []
    for hard_facts, soft_facts, objs, groups, label in zip(hard_list, soft_list, obj_list, group_list, img_labels):
        rule_score_dict = evaluation.apply_rules(final_rules, hard_facts, soft_facts, objs, groups)
        scores = [v for v in rule_score_dict.values()]
        # 聚合成 anchor 概率（注意：要求 scores 已是概率刻度；若是原始分数，请先映射到[0,1]）
        if len(scores) < top_k:
            scores = scores + [0.0] * (top_k - len(scores))
        s_sorted = sorted(scores, reverse=True)[:top_k]
        if agg == "top1":
            p_anchor = float(s_sorted[0] if s_sorted else 0.0)
        elif agg == "mean":
            p_anchor = float(np.mean(s_sorted, dtype=np.float32))
        else:
            p_anchor = float(np.mean(s_sorted[:3], dtype=np.float32))
        p_anchor = float(np.clip(p_anchor, 1e-6, 1 - 1e-6))
        anchor_probs.append(p_anchor)
        labels.append(float(label))

    # 2) 选择用于校准的子集（建议用你准备的 eval 图；否则回退到随机抽样）
    N = len(anchor_probs)
    if "calib_indices" in hyp_params and hyp_params["calib_indices"]:
        calib_idx = list(hyp_params["calib_indices"])
    else:
        calib_n = int(hyp_params.get("calib_n", min(max(8, N // 5), N)))
        perm = np.random.permutation(N).tolist()
        calib_idx = perm[:calib_n]

    logits = []
    ys = []
    for i in calib_idx:
        p = anchor_probs[i]
        logit = math.log(p) - math.log(1 - p)
        logits.append([logit])
        ys.append([labels[i]])

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)  # (M,1)
    y_t = torch.tensor(ys, dtype=torch.float32, device=device)  # (M,1)

    # 3) 训练仿射缩放器（CV/LOO）
    scaler = cv_affine_scaling(logits_t, y_t, folds=cv_folds, iters=iters, l2=l2)

    # 4) 返回包装好的校准器，推理时可直接喂 rule_scores
    rule_calibrator = RuleAffineCalibrator(scaler=scaler, top_k=top_k, agg=agg).to(device)
    return rule_calibrator
