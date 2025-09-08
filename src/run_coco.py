# Created by MacBook Pro at 07.09.25


# pipeline.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
from config import load_config
import config
from gen_data.coco_data_processing import build_labelstudio_subset_with_bboxes
from gen_data.filter_coco_by_objcount import filter_coco


# ---------- 阶段占位符（逐步实现） ----------
def stage_detect(cfg) -> Path:
    """
    目标检测 → 输出 JSONL（每行一张图）：
    {"image_id":123, "file_name":"000000123.jpg",
     "boxes":[[x1,y1,x2,y2],...], "scores":[...], "labels":[...]}
    """
    out = cfg.paths.detections_dir / "detections.jsonl"
    if out.exists():
        print(f"[detect] reuse {out}")
        return out

    import os, json, torch
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import DataLoader
    from glob import glob
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    img_dir = cfg.paths.coco_images
    paths = sorted(glob(str(img_dir / "*.jpg")))
    assert paths, f"No images found in {img_dir}"

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(cfg.device).eval()
    tfm = weights.transforms()

    bs = int(os.getenv("BATCH_SIZE", "2" if cfg.device == "cpu" else "8"))
    max_n = os.getenv("MAX_IMAGES")  # 可选：调试用
    if max_n: paths = paths[:int(max_n)]

    def collate(batch):  # ([PIL...],[meta...])
        imgs = [b[0] for b in batch];
        metas = [b[1] for b in batch]
        return imgs, metas

    class ImgSet:
        def __init__(self, ps): self.ps = ps

        def __len__(self): return len(self.ps)

        def __getitem__(self, i):
            p = Path(self.ps[i])
            img = Image.open(p).convert("RGB")
            return img, {"image_id": int(p.stem), "file_name": p.name, "w": img.width, "h": img.height}

    loader = DataLoader(ImgSet(paths), batch_size=bs, shuffle=False,
                        num_workers=0 if cfg.device == "cpu" else cfg.num_workers,
                        collate_fn=collate)

    with out.open("w") as f, torch.no_grad():
        for imgs, metas in loader:
            tensors = [tfm(im).to(cfg.device) for im in imgs]  # list[Tensor CxHxW]
            preds = model(tensors)
            for pred, meta in zip(preds, metas):
                boxes = pred["boxes"].detach().cpu().tolist()
                scores = pred["scores"].detach().cpu().tolist()
                labels = pred["labels"].detach().cpu().tolist()
                rec = {
                    "image_id": meta["image_id"],
                    "file_name": meta["file_name"],
                    "size": [meta["w"], meta["h"]],
                    "boxes": boxes, "scores": scores, "labels": labels
                }
                f.write(json.dumps(rec) + "\n")

    print(f"[detect] wrote {out}")
    return out


def stage_build_graph(cfg, det_file: Path) -> Path:
    import json, math
    from tqdm import tqdm

    out = cfg.paths.graphs_dir / "graphs.jsonl"
    if out.exists():
        print(f"[graph] reuse {out}")
        return out

    def center(box):
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def iou(b1, b2):
        x1 = max(b1[0], b2[0]);
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]);
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    with open(det_file) as f_in, open(out, "w") as f_out:
        for line in tqdm(f_in, desc="build_graph"):
            rec = json.loads(line)
            W, H = rec["size"]
            boxes = rec["boxes"];
            scores = rec["scores"];
            labels = rec["labels"]

            nodes = [{"idx": i, "box": b, "score": s, "label": c}
                     for i, (b, s, c) in enumerate(zip(boxes, scores, labels))]
            pairs = []
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    b1, b2 = boxes[i], boxes[j]
                    cx1, cy1 = center(b1);
                    cx2, cy2 = center(b2)
                    dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / math.sqrt(W * H)
                    f = [
                        iou(b1, b2),
                        dist,
                        1.0 if labels[i] == labels[j] else 0.0,
                        min(scores[i], scores[j])
                    ]
                    # heuristic: same group if IoU>0.05 or dist<0.05
                    y = 1 if f[0] > 0.05 or f[1] < 0.05 else 0
                    pairs.append({"i": i, "j": j, "feat": f, "label": y})

            out_rec = {"image_id": rec["image_id"], "nodes": nodes, "pairs": pairs}
            f_out.write(json.dumps(out_rec) + "\n")

    print(f"[graph] wrote {out}")
    return out


# ---- put into run_coco.py ----
def stage_train_grm(cfg, graph_file: Path) -> Path:
    import os, json, math, random
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    out = cfg.paths.models_dir / "grm_edge.pt"
    if out.exists():
        print(f"[train] reuse {out}");
        return out

    # ---- hyperparams (env override) ----
    EPOCHS = int(os.getenv("GRM_EPOCHS", "5"))
    BS = int(os.getenv("GRM_BATCH", "2048"))
    LR = float(os.getenv("GRM_LR", "1e-3"))
    NEG_POS = float(os.getenv("GRM_NEGPOS", "3.0"))  # 每张图 负样本:正样本 采样比
    SEED = int(os.getenv("SEED", "123"))

    random.seed(SEED);
    torch.manual_seed(SEED)

    class GraphPairDS(Dataset):
        def __init__(self, path: Path):
            self.X, self.y = [], []
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    pairs = r.get("pairs", [])
                    if not pairs: continue
                    pos = [p for p in pairs if p["label"] == 1]
                    neg = [p for p in pairs if p["label"] == 0]
                    # 负采样，控制类不平衡
                    k = min(len(neg), int(math.ceil(len(pos) * NEG_POS))) if pos else min(len(neg), 512)
                    neg = random.sample(neg, k) if k and len(neg) > k else neg
                    for p in (pos + neg):
                        self.X.append(p["feat"])
                        self.y.append(p["label"])
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)
            # 统计用于 pos_weight
            self.n_pos = int(self.y.sum().item());
            self.n_tot = len(self.y)
            self.in_dim = self.X.shape[1]
            print(f"[train] pairs: {self.n_tot}  pos: {self.n_pos}  in_dim: {self.in_dim}")

        def __len__(self):
            return self.n_tot

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    ds = GraphPairDS(graph_file)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0 if cfg.device == "cpu" else cfg.num_workers)

    class EdgeMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)  # logits
            )

        def forward(self, x): return self.net(x).squeeze(-1)

    model = EdgeMLP(ds.in_dim).to(cfg.device)
    # 不平衡处理：BCEWithLogits + pos_weight
    n_pos = max(ds.n_pos, 1)
    n_neg = max(ds.n_tot - ds.n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=cfg.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_loss = float("inf")
    for ep in range(1, EPOCHS + 1):
        model.train();
        total = 0.0;
        n = 0
        for xb, yb in dl:
            xb = xb.to(cfg.device);
            yb = yb.to(cfg.device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward();
            opt.step()
            total += loss.item() * xb.size(0);
            n += xb.size(0)
        avg = total / max(n, 1)
        print(f"[train] epoch {ep}/{EPOCHS}  loss={avg:.4f}  pos_weight={pos_weight.item():.2f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"state_dict": model.state_dict(),
                        "in_dim": ds.in_dim,
                        "pos_weight": pos_weight.item()}, out)

    print(f"[train] saved {out}  best_loss={best_loss:.4f}")
    return out


def stage_tune(cfg, graph_file: Path, model_file: Path) -> float:
    import json, torch, numpy as np
    from pathlib import Path
    from sklearn.metrics import precision_recall_fscore_support as prf

    out = cfg.paths.outputs_dir / "tune.json"
    # ---- load graphs & split by image_id ----
    recs = [json.loads(l) for l in open(graph_file)]
    recs.sort(key=lambda r: r["image_id"])
    n = len(recs);
    split = max(1, int(0.9 * n))
    val = recs[split:]

    # ---- load model ----
    ckpt = torch.load(model_file, map_location=cfg.device)

    class EdgeMLP(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(32, 1)
            )

        def forward(self, x): return self.net(x).squeeze(-1)

    mdl = EdgeMLP(ckpt["in_dim"]).to(cfg.device)
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()

    # ---- collect probs on val pairs ----
    y_true, y_prob = [], []
    with torch.no_grad():
        for r in val:
            pairs = r.get("pairs") or []
            if not pairs: continue
            X = torch.tensor([p["feat"] for p in pairs], dtype=torch.float32, device=cfg.device)
            prob = torch.sigmoid(mdl(X)).cpu().numpy()
            y_true.extend([p["label"] for p in pairs])
            y_prob.extend(prob)

    y_true = np.array(y_true);
    y_prob = np.array(y_prob)
    # ---- sweep thresholds ----
    best = {"tau": 0.5, "P": 0, "R": 0, "F1": 0}
    for tau in np.linspace(0.1, 0.9, 17):
        pred = (y_prob >= tau).astype(int)
        P, R, F, _ = prf(y_true, pred, average="binary", zero_division=0)
        if F > best["F1"]:
            best = {"tau": float(tau), "P": float(P), "R": float(R), "F1": float(F)}
    # save
    out.write_text(json.dumps(best, indent=2))
    print(f"[tune] best tau={best['tau']:.2f}  F1={best['F1']:.3f}")

    return best["tau"]


def stage_infer_groups(cfg, model_file: Path, det_file: Path, tau: float = None) -> Path:
    """
    用训练好的边分类器对每张图的候选框两两打分，p>=tau 则合并为同组（并查集）。
    输出: outputs/groups.jsonl, 每行:
      {"image_id":..., "groups":[ [idxs...], [idxs...] ], "tau": 0.73}
    """
    import json, math, torch
    from torchvision.ops import box_iou

    out = cfg.paths.outputs_dir / "groups.jsonl"
    if out.exists():
        print(f"[infer] reuse {out}")
        return out

    # ---- 阈值优先级：显式 tau > tune.json > 默认 0.7
    if tau is None:
        tfile = cfg.paths.outputs_dir / "tune.json"
        if tfile.exists():
            try:
                tau = float(json.loads(tfile.read_text())["tau"])
            except Exception:
                tau = 0.7
        else:
            tau = 0.7
    print(f"[infer] using tau={tau:.3f}")

    # ---- 加载模型（结构需与训练一致：含 Dropout）
    ckpt = torch.load(model_file, map_location=cfg.device)

    class EdgeMLP(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(32, 1)
            )

        def forward(self, x): return self.net(x).squeeze(-1)

    mdl = EdgeMLP(ckpt["in_dim"]).to(cfg.device)
    mdl.load_state_dict(ckpt["state_dict"])  # 结构一致即可 strict=True
    mdl.eval()

    def make_pairs(rec):
        """生成成对特征与索引；可按同类先筛选以降复杂度。"""
        boxes = rec["boxes"]
        labels = rec["labels"]
        scores = rec["scores"]
        W, H = rec["size"]
        n = len(boxes)
        feats = []
        pairs = []

        # 可选：仅同类配对；若想全对，改为 range(i+1,n)
        same_class_only = True

        # 预先张量化以便 IoU 批量算（也可逐对算）
        import torch
        B = torch.tensor(boxes, dtype=torch.float32)

        for c in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if (l == c or not same_class_only)]
            m = len(idxs)
            if m <= 1:
                if same_class_only:  # 若仅同类且该类只有1个，仍可与全体配对（按需）
                    continue
            for a in range(m):
                i = idxs[a]
                bi = boxes[i]
                cx1, cy1 = 0.5 * (bi[0] + bi[2]), 0.5 * (bi[1] + bi[3])
                for b in range(a + 1, m):
                    j = idxs[b]
                    bj = boxes[j]
                    cx2, cy2 = 0.5 * (bj[0] + bj[2]), 0.5 * (bj[1] + bj[3])
                    # IoU
                    iou = float(box_iou(
                        B[i].unsqueeze(0), B[j].unsqueeze(0)
                    ).item())
                    # 归一化中心距
                    dist = math.hypot(cx1 - cx2, cy1 - cy2) / math.sqrt(W * H)
                    same = 1.0 if labels[i] == labels[j] else 0.0
                    msc = float(min(scores[i], scores[j]))
                    feats.append([iou, dist, same, msc])
                    pairs.append((i, j))
        return pairs, torch.tensor(feats, dtype=torch.float32, device=cfg.device)

    def uf_make(n):
        return list(range(n))

    def uf_find(p, x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def uf_union(p, a, b):
        ra, rb = uf_find(p, a), uf_find(p, b)
        if ra != rb: p[rb] = ra

    with open(det_file) as f_in, open(out, "w") as f_out, torch.no_grad():
        for line in f_in:
            rec = json.loads(line)
            n = len(rec["boxes"])
            parents = uf_make(n)
            pairs, X = make_pairs(rec)
            if len(pairs):
                probs = torch.sigmoid(mdl(X)).tolist()
                for (i, j), p in zip(pairs, probs):
                    if p >= tau:
                        uf_union(parents, i, j)
            # 收集连通分量为组
            comp = {}
            for i in range(n):
                r = uf_find(parents, i)
                comp.setdefault(r, []).append(i)
            groups = [v for v in comp.values() if len(v) >= 1]
            f_out.write(json.dumps({
                "image_id": rec["image_id"],
                "groups": groups,
                "tau": tau
            }) + "\n")

    print(f"[infer] wrote {out}")
    return out


def stage_std_nms(cfg, det_file: Path, iou_thr: float = 0.5,
                  score_thr: float = 0.0, topk_per_class: int = 300) -> Path:
    """
    标准 NMS（类别内）。输入: detections.jsonl
    输出: outputs/detections_std_nms.jsonl（与输入同结构）
    """
    import json, torch
    from torchvision.ops import nms

    out = cfg.paths.outputs_dir / "detections_std_nms.jsonl"
    if out.exists():
        print(f"[stdnms] reuse {out}")
        return out

    with open(det_file) as fin, open(out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
            scores = torch.tensor(rec["scores"], dtype=torch.float32)
            labels = torch.tensor(rec["labels"], dtype=torch.int64)

            keep_all = []
            for c in labels.unique().tolist():
                m = (labels == c)
                b = boxes[m];
                s = scores[m]
                if score_thr > 0:
                    km = (s >= score_thr)
                    b, s = b[km], s[km]
                    idx = torch.where(m)[0][km]
                else:
                    idx = torch.where(m)[0]

                if b.numel() == 0:
                    continue
                # 可选：先按分数截断 top-k
                if topk_per_class and len(s) > topk_per_class:
                    k_idx = torch.topk(s, topk_per_class).indices
                    b, s, idx = b[k_idx], s[k_idx], idx[k_idx]

                k = nms(b, s, iou_thr)
                keep_all.append(idx[k])

            if keep_all:
                keep = torch.cat(keep_all).tolist()
            else:
                keep = []

            rec2 = {
                **rec,
                "boxes": [rec["boxes"][i] for i in keep],
                "scores": [rec["scores"][i] for i in keep],
                "labels": [rec["labels"][i] for i in keep],
            }
            fout.write(json.dumps(rec2) + "\n")

    print(f"[stdnms] wrote {out} (iou={iou_thr})")
    return out

def stage_group_nms(cfg, det_file: Path, groups_file: Path,
                    t_intra: float = 0.7, t_inter: float = 0.5,
                    score_thr: float = 0.0, topk_per_class: int = 300) -> Path:
    """
    Group-aware NMS（类别内NMS，但抑制阈值随“是否同组”而变）：
      - 同组：用 t_intra（更宽松，减少错抑制）
      - 跨组：用 t_inter（更严格，减少误保留）
    输入:
      det_file: detections.jsonl  (每行: boxes/scores/labels/size/image_id)
      groups_file: groups.jsonl   (每行: {"image_id":..., "groups": [[idxs...], ...]})
    输出:
      outputs/detections_groupnms.jsonl
    """
    import json, torch
    from torchvision.ops import box_iou

    out = cfg.paths.outputs_dir / "detections_groupnms.jsonl"
    if out.exists():
        print(f"[gNMS] reuse {out}"); return out

    # 读 groups → {image_id: group_id array (len=N, -1 表示未分组)}
    gid_map = {}
    with open(groups_file) as fg:
        for line in fg:
            r = json.loads(line)
            # 暂时只存 groups 列表；具体到每张图时再展开成 group_id 向量
            gid_map[r["image_id"]] = r["groups"]

    with open(det_file) as fin, open(out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            boxes  = torch.tensor(rec["boxes"],  dtype=torch.float32)
            scores = torch.tensor(rec["scores"], dtype=torch.float32)
            labels = torch.tensor(rec["labels"], dtype=torch.int64)

            N = len(boxes)
            if N == 0:
                fout.write(line);  # 空候选，原样写回
                continue

            # 将 groups 映射为每个候选的 group_id（未出现为 -1）
            group_id = torch.full((N,), -1, dtype=torch.int64)
            gdef = gid_map.get(rec["image_id"], [])
            for k, members in enumerate(gdef):
                idx = torch.tensor([m for m in members if 0 <= m < N], dtype=torch.long)
                if idx.numel(): group_id[idx] = k

            keep_all = []
            for c in labels.unique().tolist():
                m = (labels == c)
                if not torch.any(m):
                    continue
                b = boxes[m]; s = scores[m]
                idx_orig = torch.where(m)[0]
                grp = group_id[m]

                # 分数阈值与topk截断
                if score_thr > 0:
                    km = (s >= score_thr)
                    b, s, idx_orig, grp = b[km], s[km], idx_orig[km], grp[km]
                if b.numel() == 0:
                    continue
                if topk_per_class and len(s) > topk_per_class:
                    k_idx = torch.topk(s, topk_per_class).indices
                    b, s, idx_orig, grp = b[k_idx], s[k_idx], idx_orig[k_idx], grp[k_idx]

                # group-aware greedy NMS
                order = s.argsort(descending=True)
                while order.numel() > 0:
                    i = order[0].item()
                    keep_all.append(idx_orig[i])
                    rest = order[1:]
                    if rest.numel() == 0:
                        break
                    ious = box_iou(b[i].unsqueeze(0), b[rest]).squeeze(0)
                    same = (grp[rest] == grp[i]) & (grp[i] >= 0)
                    thr  = torch.where(same,
                                       torch.full_like(ious, float(t_intra)),
                                       torch.full_like(ious, float(t_inter)))
                    mask = ious <= thr
                    order = rest[mask]

            keep = torch.stack(keep_all).tolist() if len(keep_all) else []
            rec2 = {
                **rec,
                "boxes":  [rec["boxes"][i]  for i in keep],
                "scores": [rec["scores"][i] for i in keep],
                "labels": [rec["labels"][i] for i in keep],
                "params": {"t_intra": t_intra, "t_inter": t_inter,
                           "score_thr": score_thr, "topk_per_class": topk_per_class}
            }
            fout.write(json.dumps(rec2) + "\n")

    print(f"[gNMS] wrote {out} (t_intra={t_intra}, t_inter={t_inter})")
    return out

def stage_eval_coco(cfg, det_file: Path, ann_file: Path = None, iou_type: str = "bbox") -> dict:
    """
    评测 COCO mAP（bbox）。输入: detections_*.jsonl
    返回: 指标字典，同时保存到 outputs/eval_<stem>.json
    """
    import json, math
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_file = ann_file or cfg.paths.coco_annotations
    out_json = cfg.paths.outputs_dir / f"eval_{det_file.stem}.json"

    # JSONL -> COCO results list
    results, img_ids = [], set()
    with open(det_file) as f:
        for line in f:
            r = json.loads(line)
            img_id = int(r["image_id"])
            img_ids.add(img_id)
            for b, s, c in zip(r["boxes"], r["scores"], r["labels"]):
                x1, y1, x2, y2 = map(float, b)
                results.append({
                    "image_id": img_id,
                    "category_id": int(c),             # 需与GT类别ID一致
                    "bbox": [x1, y1, x2 - x1, y2 - y1],# xywh
                    "score": float(s)
                })

    cocoGT = COCO(str(ann_file))
    cocoDT = cocoGT.loadRes(results) if results else cocoGT.loadRes([])

    E = COCOeval(cocoGT, cocoDT, iouType=iou_type)
    # 只评测出现过的图，避免全量跑慢
    if img_ids:
        E.params.imgIds = sorted(img_ids)
    E.evaluate(); E.accumulate(); E.summarize()

    # COCOeval.stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    stats = E.stats.tolist() if hasattr(E.stats, "tolist") else list(E.stats)
    metrics = {
        "AP": stats[0], "AP50": stats[1], "AP75": stats[2],
        "APs": stats[3], "APm": stats[4], "APl": stats[5],
        "AR1": stats[6], "AR10": stats[7], "AR100": stats[8],
        "ARs": stats[9], "ARm": stats[10], "ARl": stats[11],
        "evaluated_images": len(img_ids)
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {out_json}")
    return metrics


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser("GRM-COCO pipeline")
    parser.add_argument("--steps", type=str, default="detect,graph,train,infer,groupnms,eval",
                        help="逗号分隔：detect,graph,train,infer,groupnms,eval")
    parser.add_argument("--profile", type=str, default=None, help="local|remote (覆盖 CONFIG_PROFILE)")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--t_intra", type=float, default=0.7)
    parser.add_argument("--t_inter", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)

    args = parser.parse_args()

    import os
    if args.profile: os.environ["CONFIG_PROFILE"] = args.profile
    if args.remote:  os.environ["CONFIG_PROFILE"] = "remote"
    cfg = load_config()
    print(f"[cfg] profile={cfg.profile} device={cfg.device}")
    print(f"[cfg] work_dir={cfg.paths.work_dir}")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    artifacts: Dict[str, Path] = {}

    # optional
    if "filter" in steps and "filter_coco" in globals():
        artifacts["filtered"] = filter_coco(args)

    if "detect" in steps:
        artifacts["det"] = stage_detect(cfg)

    if "graph" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["graph"] = stage_build_graph(cfg, det)

    if "train" in steps:
        graph = artifacts.get("graph", cfg.paths.graphs_dir / "graphs.jsonl")
        artifacts["model"] = stage_train_grm(cfg, graph)

    if "tune" in steps:
        graph = artifacts.get("graph", cfg.paths.graphs_dir / "graphs.jsonl")
        model = artifacts.get("model", cfg.paths.models_dir / "grm_edge.pt")
        tau = stage_tune(cfg, graph, model)
        artifacts["tune"] = cfg.paths.outputs_dir / "tune.json"

    if "infer" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        model = artifacts.get("model", cfg.paths.models_dir / "grm_edge.pt")
        artifacts["groups"] = stage_infer_groups(cfg, model, det)

    # baseline NMS
    if "stdnms" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["std"] = stage_std_nms(cfg, det, iou_thr=args.nms_iou)

    # ours: group-aware NMS（支持阈值）
    if "groupnms" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        groups = artifacts.get("groups", cfg.paths.outputs_dir / "groups.jsonl")
        artifacts["gnms"] = stage_group_nms(cfg, det, groups, t_intra=args.t_intra, t_inter=args.t_inter)

    # eval baseline
    if "evalstd" in steps:
        std = artifacts.get("std", cfg.paths.outputs_dir / "detections_std_nms.jsonl")
        print("[evalstd]", stage_eval_coco(cfg, std))

    # eval ours
    if "eval" in steps:
        g = artifacts.get("gnms", cfg.paths.outputs_dir / "detections_groupnms.jsonl")
        print("[eval]", stage_eval_coco(cfg, g))

    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
