# Created by MacBook Pro at 07.09.25


# pipeline.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, List
from config import load_config
import config
from gen_data.coco_data_processing import build_labelstudio_subset_with_bboxes
from gen_data.filter_coco_by_objcount import filter_coco
import numpy as np


def _load_coco(ann_path):
    from pycocotools.coco import COCO
    return COCO(str(ann_path))


def _match_od_to_gt(boxes_od, img_id, coco, iou_thr=0.5):
    import torch
    from torchvision.ops import box_iou
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if not anns or not boxes_od: return [-1] * len(boxes_od)
    gtb = torch.tensor([[a["bbox"][0], a["bbox"][1],
                         a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=torch.float32)
    gt_ids = [a["id"] for a in anns]
    B = torch.tensor(boxes_od, dtype=torch.float32)
    iou = box_iou(B, gtb)  # [N_od, N_gt]
    val, idx = iou.max(dim=1)
    return [gt_ids[j] if float(val[i]) >= iou_thr else -1 for i, j in enumerate(idx)]


def _EdgeMLP_for_eval(in_dim, device):
    import torch, torch.nn as nn
    class EdgeMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x): return self.net(x).squeeze(-1)

    m = EdgeMLP(in_dim).to(device)
    return m


def quick_check_graph(graph_path):
    import json, numpy as np
    bins_iou = [0, .1, .3, .5, .7, 1.01]
    bins_d = [0, .03, .06, .10, .20, 10]
    pos_iou = [0] * 5;
    cnt_iou = [0] * 5
    pos_d = [0] * 5;
    cnt_d = [0] * 5
    n_pairs = n_pos = 0
    with open(graph_path) as f:
        for L in f:
            r = json.loads(L);
            ps = r.get("pairs", [])
            n_pairs += len(ps);
            n_pos += sum(p["label"] for p in ps)
            for p in ps:
                iou, dist = p["feat"][0], p["feat"][1]
                ki = max(i for i, b in enumerate(bins_iou[:-1]) if iou >= b)
                kd = max(i for i, b in enumerate(bins_d[:-1]) if dist >= b)
                pos_iou[ki] += p["label"];
                cnt_iou[ki] += 1
                pos_d[kd] += p["label"];
                cnt_d[kd] += 1
    pr = n_pos / max(n_pairs, 1)
    mono_iou = [pos_iou[i] / cnt_iou[i] if cnt_iou[i] else 0 for i in range(5)]
    mono_d = [pos_d[i] / cnt_d[i] if cnt_d[i] else 0 for i in range(5)]
    print(f"[diag.graph] pairs={n_pairs:,} pos_rate={pr:.3f}")
    print(f"[diag.graph] pos_rate_by_IoU={mono_iou}")
    print(f"[diag.graph] pos_rate_by_Dist={mono_d}")


def eval_pairs_auc(cfg, graph_path, model_path, heur_iou=0.05, heur_dist=0.05):
    import json, numpy as np, torch
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    ckpt = torch.load(model_path, map_location=cfg.device)
    mdl = _EdgeMLP_for_eval(ckpt["in_dim"], cfg.device);
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()

    recs = [json.loads(l) for l in open(graph_path)]
    recs.sort(key=lambda r: r["image_id"])
    split = max(1, int(0.9 * len(recs)))
    val = recs[split:]

    y, prob, heur = [], [], []
    with torch.no_grad():
        for r in val:
            ps = r.get("pairs") or []
            if not ps: continue
            X = torch.tensor([p["feat"] for p in ps], dtype=torch.float32, device=cfg.device)
            pr = torch.sigmoid(mdl(X)).cpu().numpy()
            prob.extend(pr)
            y.extend([p["label"] for p in ps])
            heur.extend([1 if (p["feat"][0] > heur_iou or p["feat"][1] < heur_dist) else 0 for p in ps])

    if not y:
        print("[diag.pairs] no val pairs");
        return
    y = np.array(y);
    prob = np.array(prob);
    heur = np.array(heur)
    auc = roc_auc_score(y, prob) if y.sum() and (1 - y).sum() else float('nan')
    ap = average_precision_score(y, prob) if y.sum() else float('nan')
    f1 = f1_score(y, prob >= 0.5)
    f1h = f1_score(y, heur)
    print(f"[diag.pairs] AUC={auc:.3f} AP={ap:.3f} F1@0.5={f1:.3f} Heur-F1={f1h:.3f}")


def dump_top_pairs(graph_path, model_path, cfg, K=5, samples=3):
    import json, heapq, torch, random
    ckpt = torch.load(model_path, map_location=cfg.device)
    mdl = _EdgeMLP_for_eval(ckpt["in_dim"], cfg.device);
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()
    lines = open(graph_path).read().splitlines()
    for L in random.sample(lines, min(samples, len(lines))):
        r = json.loads(L);
        ps = r.get("pairs") or []
        if not ps: continue
        X = torch.tensor([p["feat"] for p in ps], dtype=torch.float32, device=cfg.device)
        with torch.no_grad():
            pr = torch.sigmoid(mdl(X)).cpu().tolist()
        top = heapq.nlargest(K, enumerate(ps), key=lambda x: pr[x[0]])
        bot = heapq.nsmallest(K, enumerate(ps), key=lambda x: pr[x[0]])
        print(f"[diag.top] image {r['image_id']} TOP:")
        for idx, pair in top: print("  p=%.3f feat=%s y=%d" % (pr[idx], pair["feat"], pair["label"]))
        print(f"[diag.bot] image {r['image_id']} BOT:")
        for idx, pair in bot: print("  p=%.3f feat=%s y=%d" % (pr[idx], pair["feat"], pair["label"]))


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


def stage_viz_frcnn_vs_gt(cfg, det_file: Path,
                          iou_thr: float = 0.5,
                          score_thr: float = 0.05,
                          limit: int = 200) -> Path:
    """
    可视化 FasterRCNN vs GT：
      绿: TP（预测与某 GT IoU>=iou_thr）
      黄: FP（预测>=score_thr 但未与任何 GT 匹配）
      红: FN（剩余未被匹配的 GT）
    结果：/outputs/viz_frcnn_vs_gt/{image_id}.jpg 及 summary.csv/json
    """
    import json, os, math, csv
    from pathlib import Path
    import numpy as np
    import cv2
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_frcnn_vs_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_json = out_dir / "summary.json"
    summ_csv = out_dir / "summary.csv"

    def iou_xyxy(a, b):
        xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
        xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
        w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
        inter = w * h
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # COCO
    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    images_root = Path(cfg.paths.coco_images)

    # 读检测
    recs = [json.loads(l) for l in open(det_file)]
    if limit: recs = recs[:limit]

    summary = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    GREEN = (80, 200, 60)  # TP
    YELLOW = (40, 220, 220)  # FP
    RED = (30, 30, 230)  # FN
    WHITE = (240, 240, 240)

    for rec in recs:
        img_id = rec["image_id"]
        fn = id2fname.get(img_id)
        if not fn: continue
        img_path = images_root / fn
        if not img_path.exists(): continue

        im = cv2.imread(str(img_path))
        if im is None: continue
        H, W = im.shape[:2]

        # 读取 GT
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        gts = []
        gt_cats = []
        for a in anns:
            x1, y1, w, h = a["bbox"]
            gts.append([x1, y1, x1 + w, y1 + h])
            gt_cats.append(a["category_id"])
        gts = np.array(gts, dtype=np.float32) if gts else np.zeros((0, 4), np.float32)
        gt_used = np.zeros(len(gts), dtype=bool)

        # 读取预测（按类别独立匹配）
        boxes = np.array(rec["boxes"], dtype=np.float32) if rec["boxes"] else np.zeros((0, 4), np.float32)
        scores = np.array(rec["scores"], dtype=np.float32) if rec["scores"] else np.zeros((0,), np.float32)
        labels = np.array(rec["labels"], dtype=np.int32) if rec["labels"] else np.zeros((0,), np.int32)

        keep = scores >= score_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        tp_idx, fp_idx = [], []
        matched_gt = set()

        # 按类匹配（与 COCO 评测一致）
        for c in np.unique(labels):
            m = labels == c
            if not np.any(m): continue
            det_b = boxes[m];
            det_s = scores[m];
            det_i = np.where(m)[0]
            order = np.argsort(-det_s)
            det_b, det_i = det_b[order], det_i[order]

            # 找同类 GT
            gt_idx = [i for i, k in enumerate(gt_cats) if k == c]
            if len(gt_idx) == 0:
                # 该类全是 FP
                fp_idx += det_i.tolist()
                continue

            # 贪心匹配
            used_local = set()
            for k, b in enumerate(det_b):
                best_iou, best_j = 0.0, -1
                for j_local, j in enumerate(gt_idx):
                    if j_local in used_local: continue
                    iou = iou_xyxy(b, gts[j])
                    if iou > best_iou:
                        best_iou, best_j = iou, j_local
                if best_iou >= iou_thr:
                    used_local.add(best_j)
                    tp_idx.append(int(det_i[k]))
                    matched_gt.add(gt_idx[best_j])
                else:
                    fp_idx.append(int(det_i[k]))

        # 未匹配 GT = FN
        fn_idx = [i for i in range(len(gts)) if i not in matched_gt]

        # 绘制：TP 绿, FP 黄, FN 红
        # 先画预测（框+score）
        for i in tp_idx:
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            cv2.rectangle(im, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(im, f"TP {scores[i]:.2f}", (x1, max(10, y1 - 5)), font, 0.5, GREEN, 1, cv2.LINE_AA)
        for i in fp_idx:
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            cv2.rectangle(im, (x1, y1), (x2, y2), YELLOW, 2)
            cv2.putText(im, f"FP {scores[i]:.2f}", (x1, max(10, y1 - 5)), font, 0.5, YELLOW, 1, cv2.LINE_AA)
        # 再画 FN（红）
        for j in fn_idx:
            x1, y1, x2, y2 = gts[j].astype(int).tolist()
            cv2.rectangle(im, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(im, "FN", (x1, max(10, y1 - 5)), font, 0.6, RED, 2, cv2.LINE_AA)

        # 角标 legend+统计
        tp, fp, fn = len(tp_idx), len(fp_idx), len(fn_idx)
        txt = f"TP:{tp}  FP:{fp}  FN:{fn}  IoU@{iou_thr}  thr@{score_thr}"
        cv2.rectangle(im, (0, 0), (int(10 + 7.3 * len(txt)), 24), (0, 0, 0), -1)
        cv2.putText(im, txt, (6, 17), font, 0.5, WHITE, 1, cv2.LINE_AA)

        # 保存
        out_path = out_dir / f"{img_id}.jpg"
        cv2.imwrite(str(out_path), im)

        summary.append({
            "image_id": int(img_id),
            "file_name": fn,
            "TP": tp, "FP": fp, "FN": fn,
            "det": int(len(boxes)), "gt": int(len(gts))
        })

    # 保存汇总
    with open(summ_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summ_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file_name", "TP", "FP", "FN", "det", "gt"])
        w.writeheader();
        w.writerows(summary)

    print(f"[viz] wrote {out_dir}  ({len(summary)} images)  "
          f"json={summ_json.name} csv={summ_csv.name}")
    return out_dir


def stage_build_graph(cfg, det_file: Path) -> Path:
    """
    读取 detections.jsonl → 生成 graphs_<mode>.jsonl
    pairs[*].feat = [iou, norm_center_dist, same_cls, min_score]
    pairs[*].label: 来自 distill/heur/gt
    """
    import os, json, math, torch
    from pathlib import Path
    from torchvision.ops import box_iou

    MODE = os.getenv("SUPERVISION", "distill")  # distill | heur | gt
    MATCH_IOU = float(os.getenv("MATCH_IOU", "0.5"))  # distill: OD↔GT 匹配阈值
    HEUR_IOU = float(os.getenv("HEUR_IOU", "0.05"))  # heur:   IoU>阈值算正
    HEUR_DIST = float(os.getenv("HEUR_DIST", "0.05"))  # heur:   距离<阈值算正
    SAME_ONLY = os.getenv("SAME_CLASS_ONLY", "1") == "1"  # 是否只配对同类，提速
    MAX_PAIRS = int(os.getenv("MAX_PAIRS", "0"))  # >0 则每图随机下采样 pair 数

    out = cfg.paths.graphs_dir / f"graphs_{MODE}.jsonl"
    if out.exists():
        print(f"[graph] reuse {out}")
        return out

    # ---------- helpers ----------
    def _center(box):
        x1, y1, x2, y2 = box;
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]);
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]);
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        u = a1 + a2 - inter
        return inter / u if u > 0 else 0.0

    coco = None
    if MODE in ("distill", "gt"):
        from pycocotools.coco import COCO
        coco = COCO(str(cfg.paths.coco_annotations))

    def _match_od_to_gt(boxes_od, img_id):
        """返回每个 OD 框匹配到的 GT ann_id（未匹配为 -1）"""
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if not anns or not boxes_od:
            return [-1] * len(boxes_od)
        gtb = torch.tensor([[a["bbox"][0], a["bbox"][1],
                             a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=torch.float32)
        gt_ids = [a["id"] for a in anns]
        B = torch.tensor(boxes_od, dtype=torch.float32)
        iou = box_iou(B, gtb)  # [Nod, Ngt]
        val, idx = iou.max(dim=1)
        return [gt_ids[j] if float(val[i]) >= MATCH_IOU else -1 for i, j in enumerate(idx)]

    with open(det_file) as f_in, open(out, "w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            W, H = rec["size"]
            img_id = rec["image_id"]

            # --- 节点来源 ---
            if MODE == "gt":
                ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = coco.loadAnns(ann_ids)
                boxes = [[a["bbox"][0], a["bbox"][1],
                          a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
                labels = [a["category_id"] for a in anns]
                scores = [1.0] * len(anns)
            else:
                boxes = rec["boxes"];
                labels = rec["labels"];
                scores = rec["scores"]

            n = len(boxes)
            nodes = [{"idx": i, "box": boxes[i], "score": scores[i], "label": labels[i]} for i in range(n)]

            # --- 蒸馏匹配（distill） ---
            gt_idx = None
            if MODE == "distill":
                gt_idx = _match_od_to_gt(boxes, img_id)

            # --- 生成 pair 特征与标签 ---
            pairs = []
            # 预先张量化用于快速 IoU（亦可逐对算）
            B = torch.tensor(boxes, dtype=torch.float32) if n > 0 else None

            # 按类别分桶以减小复杂度
            class_buckets = {}
            for i, c in enumerate(labels):
                class_buckets.setdefault(c, []).append(i)
            # 决定配对索引列表
            idx_lists = (class_buckets.values() if SAME_ONLY else [list(range(n))])

            for idxs in idx_lists:
                m = len(idxs)
                for a in range(m):
                    i = idxs[a]
                    bi = boxes[i];
                    cx1, cy1 = _center(bi)
                    for b in range(a + 1, m):
                        j = idxs[b]
                        bj = boxes[j];
                        cx2, cy2 = _center(bj)

                        # feats
                        iou_val = float(box_iou(B[i].unsqueeze(0), B[j].unsqueeze(0)).item()) if n > 0 else _iou(bi, bj)
                        dist = math.hypot(cx1 - cx2, cy1 - cy2) / math.sqrt(W * H)
                        same = 1.0 if labels[i] == labels[j] else 0.0
                        msc = float(min(scores[i], scores[j]))
                        feat = [iou_val, dist, same, msc]

                        # label
                        if MODE == "distill":
                            gi = gt_idx[i];
                            gj = gt_idx[j]
                            y = 1 if (gi != -1 and gj != -1 and gi == gj) else 0
                        elif MODE == "heur":
                            y = 1 if (iou_val > HEUR_IOU or dist < HEUR_DIST) else 0
                        else:  # gt
                            # 简化：GT 下仍用几何近邻当正（也可用 GT 掩码生成接触/容器关系当 GT label）
                            y = 1 if (iou_val > HEUR_IOU or dist < HEUR_DIST) else 0

                        pairs.append({"i": i, "j": j, "feat": feat, "label": int(y)})

            # 可选：每图下采样 pair，控制规模
            if MAX_PAIRS > 0 and len(pairs) > MAX_PAIRS:
                import random
                random.shuffle(pairs)
                pairs = pairs[:MAX_PAIRS]

            f_out.write(json.dumps({
                "image_id": img_id,
                "nodes": nodes,
                "pairs": pairs,
                "mode": MODE
            }) + "\n")

    print(f"[graph] wrote {out}")
    return out


# ---- put into run_coco.py ----
def stage_train_grm(cfg, graph_file: Path = None) -> Path:
    import os, json, math, random, torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split

    MODE = os.getenv("SUPERVISION", "distill")  # distill|heur|gt
    if graph_file is None:
        graph_file = cfg.paths.graphs_dir / f"graphs_{MODE}.jsonl"

    out = cfg.paths.models_dir / f"grm_edge_{MODE}.pt"
    if out.exists():
        print(f"[train] reuse {out}");
        return out

    # ---- hyperparams ----
    EPOCHS = int(os.getenv("GRM_EPOCHS", "5"))
    BS = int(os.getenv("GRM_BATCH", "4096"))
    LR = float(os.getenv("GRM_LR", "1e-3"))
    NEG_POS = float(os.getenv("GRM_NEGPOS", "3.0"))  # 负采样比
    W_NEG_WEAK = float(os.getenv("GRM_W_NEG_WEAK", "0.5"))  # 弱负例降权
    USE_FOCAL = os.getenv("GRM_FOCAL", "0") == "1"
    GAMMA = float(os.getenv("GRM_GAMMA", "2.0"))
    ALPHA_FOC = float(os.getenv("GRM_ALPHA", "0.25"))
    VAL_SPLIT = float(os.getenv("GRM_VAL", "0.1"))
    SEED = int(os.getenv("SEED", "123"))
    AMP = os.getenv("GRM_AMP", "1") == "1"

    random.seed(SEED);
    torch.manual_seed(SEED)

    class GraphPairDS(Dataset):
        def __init__(self, path: Path):
            self.X, self.y, self.w = [], [], []
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    pairs = r.get("pairs", [])
                    if not pairs: continue
                    pos = [p for p in pairs if p["label"] == 1]
                    neg = [p for p in pairs if p["label"] == 0]
                    # 负采样控比
                    k = int(math.ceil(len(pos) * NEG_POS)) if pos else min(len(neg), 512)
                    if k and len(neg) > k:
                        neg = random.sample(neg, k)

                    def add(p):
                        self.X.append(p["feat"])
                        self.y.append(p["label"])
                        # 使用外部权重，否则做一个简单的弱负例降权（IoU或距离边界上的负例）
                        if "weight" in p:
                            self.w.append(float(p["weight"]))
                        else:
                            iou, dist = p["feat"][0], p["feat"][1]
                            if p["label"] == 0 and (0.05 < iou < 0.3 or 0.05 < dist < 0.12):
                                self.w.append(W_NEG_WEAK)
                            else:
                                self.w.append(1.0)

                    for p in pos + neg:
                        add(p)

            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)
            self.w = torch.tensor(self.w, dtype=torch.float32)
            self.in_dim = self.X.shape[1]
            self.n_pos = int(self.y.sum().item());
            self.n_tot = len(self.y)
            print(f"[train] file={path.name} pairs={self.n_tot} pos={self.n_pos} in_dim={self.in_dim}")

        def __len__(self):
            return self.n_tot

        def __getitem__(self, i):
            return self.X[i], self.y[i], self.w[i]

    ds_full = GraphPairDS(graph_file)
    n_val = max(1, int(len(ds_full) * VAL_SPLIT))
    n_tr = len(ds_full) - n_val
    ds_tr, ds_val = random_split(ds_full, [n_tr, n_val], generator=torch.Generator().manual_seed(SEED))
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=0 if cfg.device == "cpu" else cfg.num_workers, pin_memory=False)
    dl_va = DataLoader(ds_val, batch_size=BS, shuffle=False, num_workers=0 if cfg.device == "cpu" else cfg.num_workers, pin_memory=False)

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

    model = EdgeMLP(ds_full.in_dim).to(cfg.device)

    # 损失：BCE 或 Focal（带样本权重）
    bce = nn.BCEWithLogitsLoss(reduction='none')

    def focal_loss(logits, targets, alpha=ALPHA_FOC, gamma=GAMMA):
        # logits->prob
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        w = alpha * targets + (1 - alpha) * (1 - targets)
        return -w * (1 - pt).pow(gamma) * torch.log(pt + 1e-8)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP and (cfg.device.startswith("cuda")))

    best_val = float("inf");
    best_ckpt = None
    for ep in range(1, EPOCHS + 1):
        # ---- train ----
        model.train();
        tot = 0.0;
        n = 0
        for xb, yb, wb in dl_tr:
            xb, yb, wb = xb.to(cfg.device), yb.to(cfg.device), wb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            if AMP and cfg.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss_e = focal_loss(logits, yb) if USE_FOCAL else bce(logits, yb)
                    loss = (loss_e * wb).mean()
                scaler.scale(loss).backward()
                scaler.step(opt);
                scaler.update()
            else:
                logits = model(xb)
                loss_e = focal_loss(logits, yb) if USE_FOCAL else bce(logits, yb)
                loss = (loss_e * wb).mean()
                loss.backward();
                opt.step()
            tot += loss.item() * xb.size(0);
            n += xb.size(0)
        tr_loss = tot / max(n, 1)

        # ---- val ----
        model.eval();
        tot = 0.0;
        n = 0
        with torch.no_grad():
            for xb, yb, wb in dl_va:
                xb, yb, wb = xb.to(cfg.device), yb.to(cfg.device), wb.to(cfg.device)
                logits = model(xb)
                loss_e = focal_loss(logits, yb) if USE_FOCAL else bce(logits, yb)
                loss = (loss_e * wb).mean()
                tot += loss.item() * xb.size(0);
                n += xb.size(0)
        va_loss = tot / max(n, 1)

        print(f"[train] epoch {ep}/{EPOCHS}  loss_tr={tr_loss:.4f}  loss_va={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_ckpt = {
                "state_dict": model.state_dict(),
                "in_dim": ds_full.in_dim,
                "arch": "mlp64_32_do0.1",
                "mode": MODE,
                "meta": {"epochs": ep, "val_loss": va_loss}
            }
            torch.save(best_ckpt, out)

    print(f"[train] saved {out}  best_val={best_val:.4f}")
    return out


def stage_tune(cfg, graph_file: Path, model_file: Path) -> float:
    import json, torch, numpy as np
    from sklearn.metrics import precision_recall_fscore_support as prf

    recs = [json.loads(l) for l in open(graph_file)]
    recs.sort(key=lambda r: r["image_id"])
    split = max(1, int(0.9 * len(recs)))
    val = recs[split:]

    ckpt = torch.load(model_file, map_location=cfg.device)

    class EdgeMLP(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(32, 1))

        def forward(self, x): return self.net(x).squeeze(-1)

    mdl = EdgeMLP(ckpt["in_dim"]).to(cfg.device);
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()

    y_true, logits = [], []
    with torch.no_grad():
        for r in val:
            ps = r.get("pairs") or []
            if not ps: continue
            X = torch.tensor([p["feat"] for p in ps], dtype=torch.float32, device=cfg.device)
            lg = mdl(X).cpu().numpy()
            logits.extend(lg)
            y_true.extend([p["label"] for p in ps])

    import numpy as np
    y_true = np.array(y_true)
    logits = np.array(logits)
    probs = 1 / (1 + np.exp(-logits))

    # 打印分位数，确认量级
    qs = np.quantile(probs, [0.5, 0.9, 0.95, 0.99])
    print(f"[tune] prob quantiles p50={qs[0]:.3f} p90={qs[1]:.3f} p95={qs[2]:.3f} p99={qs[3]:.3f}")

    # 方案A：从预测值里取候选阈值（更稳）
    cands = np.unique(np.round(probs, 5))
    # 也可追加一个细扫低区间
    cands = np.unique(np.concatenate([cands, np.linspace(0.005, 0.15, 60)]))

    best = {"tau": 0.5, "P": 0, "R": 0, "F1": 0}
    for tau in cands:
        yp = (probs >= tau).astype(int)
        P, R, F, _ = prf(y_true, yp, average="binary", zero_division=0)
        if F > best["F1"]:
            best = {"tau": float(tau), "P": float(P), "R": float(R), "F1": float(F)}

    out = cfg.paths.outputs_dir / "tune.json"
    out.write_text(json.dumps(best, indent=2))
    print(f"[tune] best tau={best['tau']:.3f}  F1={best['F1']:.3f}  P={best['P']:.3f}  R={best['R']:.3f}")
    return best["tau"]


def stage_infer_groups(cfg, model_file: Path, det_file: Path, tau: float = None) -> Path:
    """
    读 detections.jsonl & 训练好的边分类器 → 并查集合并成组
    写 outputs/groups.jsonl（含 kept 原始索引）+ groups_stats.json
    环境变量：
      SAME_CLASS_ONLY=1|0   MIN_SCORE=0.0..1.0   TOPK_PER_CLASS=300
      K_NEI=50              BATCH=65536
    """
    import os, json, math, torch, numpy as np
    from torchvision.ops import box_iou

    SAME_CLASS_ONLY = os.getenv("SAME_CLASS_ONLY", "1") == "1"
    MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))
    TOPK_PER_CLASS = int(os.getenv("TOPK_PER_CLASS", "300"))
    K_NEI = int(os.getenv("K_NEI", "50"))
    BATCH = int(os.getenv("BATCH", "65536"))

    out = cfg.paths.outputs_dir / "groups.jsonl"
    stats_out = cfg.paths.outputs_dir / "groups_stats.json"
    if out.exists():
        print(f"[infer] reuse {out}")
        return out

    # --- model ---
    ckpt = torch.load(model_file, map_location=cfg.device)

    class EdgeMLP(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(32, 1))

        def forward(self, x): return self.net(x).squeeze(-1)

    mdl = EdgeMLP(ckpt["in_dim"]).to(cfg.device);
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()

    # --- tau / T ---
    calT = None
    if tau is None:
        tfile = cfg.paths.outputs_dir / "tune.json"
        if tfile.exists():
            tj = json.loads(tfile.read_text());
            tau = float(tj.get("tau", 0.7));
            calT = tj.get("T", None)
        else:
            tau = 0.7
    print(f"[infer] tau={tau:.3f} T={calT if calT is not None else 'None'} "
          f"same_only={int(SAME_CLASS_ONLY)} min_score={MIN_SCORE} k_nei={K_NEI}")

    # --- utils ---
    def uf_make(n):
        return list(range(n))

    def uf_find(p, x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x

    def uf_union(p, a, b):
        ra, rb = uf_find(p, a), uf_find(p, b)
        if ra != rb: p[rb] = ra

    def centers(boxes):
        b = np.asarray(boxes, np.float32)
        return (0.5 * (b[:, 0] + b[:, 2]), 0.5 * (b[:, 1] + b[:, 3]))

    n_img = 0;
    total_pairs = 0;
    total_groups = 0;
    multi_groups = 0
    with open(det_file) as f_in, open(out, "w") as f_out, torch.no_grad():
        for line in f_in:
            rec = json.loads(line)
            boxes0, scores0, labels0 = rec["boxes"], rec["scores"], rec["labels"]
            W, H = rec["size"];
            n0 = len(boxes0)
            if n0 == 0:
                f_out.write(json.dumps({"image_id": rec["image_id"], "groups": [], "kept": [], "tau": tau}) + "\n");
                continue

            # 过滤低分（记录原始索引）
            keep_idx = [i for i, s in enumerate(scores0) if s >= MIN_SCORE]
            if not keep_idx:
                f_out.write(json.dumps({"image_id": rec["image_id"], "groups": [], "kept": [], "tau": tau}) + "\n");
                continue
            boxes = [boxes0[i] for i in keep_idx]
            scores = [scores0[i] for i in keep_idx]
            labels = [labels0[i] for i in keep_idx]
            kept = keep_idx[:]  # 原始 det 索引

            # 每类 top-k（继续保持原始索引映射）
            if TOPK_PER_CLASS > 0:
                byc = {}
                for i, c in enumerate(labels): byc.setdefault(c, []).append((scores[i], i))
                sel = []
                for c, lst in byc.items():
                    lst.sort(reverse=True);
                    sel.extend([i for _, i in lst[:TOPK_PER_CLASS]])
                sel = sorted(set(sel))
                boxes = [boxes[i] for i in sel]
                scores = [scores[i] for i in sel]
                labels = [labels[i] for i in sel]
                kept = [kept[i] for i in sel]
            n = len(boxes)
            if n == 0:
                f_out.write(json.dumps({"image_id": rec["image_id"], "groups": [], "kept": [], "tau": tau}) + "\n");
                continue

            parents = uf_make(n)
            B = torch.tensor(boxes, dtype=torch.float32, device=cfg.device)
            cx, cy = centers(boxes)

            # 生成候选对
            if SAME_CLASS_ONLY:
                buckets = {}
                for i, c in enumerate(labels): buckets.setdefault(c, []).append(i)
                families = buckets.values()
            else:
                families = [list(range(n))]

            pair_idx = []
            for idxs in families:
                m = len(idxs)
                if m <= 1: continue
                if K_NEI > 0:
                    pts = np.stack([np.array(cx)[idxs], np.array(cy)[idxs]], 1)
                    for a in range(m):
                        i = idxs[a]
                        d2 = np.sum((pts - pts[a]) ** 2, axis=1)
                        nn = np.argsort(d2)
                        for nb in nn[1:1 + min(K_NEI, m - 1)]:
                            j = idxs[int(nb)]
                            if i < j: pair_idx.append((i, j))
                else:
                    for a in range(m):
                        i = idxs[a]
                        for b in range(a + 1, m):
                            j = idxs[b];
                            pair_idx.append((i, j))
            if not pair_idx:
                f_out.write(json.dumps({"image_id": rec["image_id"], "groups": [list(range(n))], "kept": kept, "tau": tau}) + "\n")
                continue
            total_pairs += len(pair_idx)

            # 批前向
            def make_feats(chunk):
                i_idx = torch.tensor([p[0] for p in chunk], device=cfg.device)
                j_idx = torch.tensor([p[1] for p in chunk], device=cfg.device)
                iou = box_iou(B[i_idx], B[j_idx]).diagonal()
                cx_i = torch.tensor([cx[p[0]] for p in chunk], device=cfg.device)
                cy_i = torch.tensor([cy[p[0]] for p in chunk], device=cfg.device)
                cx_j = torch.tensor([cx[p[1]] for p in chunk], device=cfg.device)
                cy_j = torch.tensor([cy[p[1]] for p in chunk], device=cfg.device)
                dist = torch.hypot(cx_i - cx_j, cy_i - cy_j) / math.sqrt(W * H)
                same = torch.tensor([1.0 if labels[p[0]] == labels[p[1]] else 0.0 for p in chunk],
                                    device=cfg.device)
                msc = torch.tensor([min(scores[p[0]], scores[p[1]]) for p in chunk], device=cfg.device)
                return torch.stack([iou, dist, same, msc], 1)

            ofs = 0
            while ofs < len(pair_idx):
                chunk = pair_idx[ofs:ofs + BATCH]
                X = make_feats(chunk)
                logits = mdl(X)
                probs = torch.sigmoid(logits / float(calT)) if calT else torch.sigmoid(logits)
                sel = (probs >= tau).nonzero(as_tuple=False).squeeze(1).tolist()
                for s in sel:
                    i, j = chunk[s];
                    uf_union(parents, i, j)
                ofs += len(chunk)

            # 收集组
            roots = {}
            for i in range(n):
                r = uf_find(parents, i);
                roots.setdefault(r, []).append(i)
            groups = list(roots.values())

            # 写出（含 kept 原始索引）
            f_out.write(json.dumps({
                "image_id": rec["image_id"],
                "groups": groups,  # 组内成员是“当前局部索引”
                "kept": kept,  # 映回原始 detections.jsonl 的索引
                "tau": tau
            }) + "\n")

            n_img += 1;
            total_groups += len(groups);
            multi_groups += sum(1 for g in groups if len(g) >= 2)

    stats = {
        "images": n_img,
        "total_pairs": int(total_pairs),
        "avg_groups_per_image": total_groups / max(n_img, 1),
        "multi_member_groups": int(multi_groups),
        "multi_group_ratio": multi_groups / max(total_groups, 1) if total_groups else 0.0
    }
    stats_out.write_text(json.dumps(stats, indent=2))
    print(f"[infer] wrote {out}")
    print(f"[infer] stats -> {stats_out}  {stats}")
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


def stage_group_nms(cfg,
                    det_file: Path,
                    groups_file: Path,
                    t_intra: float = 0.9,
                    t_inter: float = 0.5) -> Path:
    """
    基于 groups.jsonl 的组感知 NMS：
      - 同组采用高阈值 t_intra（更宽松，减少错抑制）
      - 跨组采用低阈值 t_inter（更严格，抑冗余）
    要求 groups.jsonl 每条记录含:
      { "image_id": ..., "groups": [[局部idx...], ...], "kept": [原始det索引...], "tau": ... }

    环境变量（可选）：
      WBF=0|1              # 是否对同组且 IoU>t_intra 的被抑制框做组内融合 (Weighted Boxes Fusion-简化)
      DEBUG_GNMS=0|1       # 打印调试信息
      NMS_CLASSWISE=1|0    # 是否按类别分开做NMS（默认1，保持与COCO评测一致）
    """
    import os, json, torch
    from pathlib import Path
    from torchvision.ops import box_iou

    out = cfg.paths.outputs_dir / "detections_groupnms.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    WBF = os.getenv("WBF", "1") == "1"
    DEBUG = os.getenv("DEBUG_GNMS", "0") == "1"
    CLASSWISE = os.getenv("NMS_CLASSWISE", "1") == "1"

    # 读取 groups 映射：image_id -> {"groups":..., "kept":...}
    gid_map = {}
    with open(groups_file) as f:
        for L in f:
            r = json.loads(L)
            gid_map[r["image_id"]] = r

    def _build_group_vector(n_det, rG):
        """返回长度为 n_det 的 group_id 向量（默认-1），把组成员从局部idx映射回原始det索引。"""
        grp = torch.full((n_det,), -1, dtype=torch.long)
        if not rG:
            return grp
        kept = rG.get("kept", list(range(n_det)))
        for g_id, members in enumerate(rG.get("groups", [])):
            for m in members:
                if 0 <= m < len(kept):
                    orig = kept[m]
                    if 0 <= orig < n_det:
                        grp[orig] = g_id
        return grp

    def _wbf_merge(boxes, scores, keep_idx):
        """简单 WBF：将 keep_idx 的第一项视作赢家，其余按分数加权融合到赢家。"""
        if len(keep_idx) <= 1:
            return
        idxs = torch.tensor(keep_idx, device=boxes.device, dtype=torch.long)
        w = scores[idxs]
        w = w / (w.sum() + 1e-8)
        boxes[idxs[0]] = (boxes[idxs] * w[:, None]).sum(dim=0)
        scores[idxs[0]] = scores[idxs].max()

    def _gnms_single_image(rec, rG):
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        scores = torch.tensor(rec["scores"], dtype=torch.float32)
        labels = torch.tensor(rec["labels"], dtype=torch.long)
        n = boxes.size(0)

        grp = _build_group_vector(n, rG)

        kept_all = []  # 所有保留的全局索引
        # 类别分桶
        if CLASSWISE:
            classes = torch.unique(labels)
            class_iters = [(int(c.item()), torch.nonzero(labels == c, as_tuple=False).squeeze(1)) for c in classes]
        else:
            class_iters = [(-1, torch.arange(n, dtype=torch.long))]

        for c, idxs in class_iters:
            if idxs.numel() == 0:
                continue
            b = boxes[idxs].clone()
            s = scores[idxs].clone()
            g = grp[idxs].clone()

            # 按分数降序
            order = torch.argsort(s, descending=True)
            b, s, g, idxs = b[order], s[order], g[order], idxs[order]

            keep_local = []
            while idxs.numel() > 0:
                i = 0
                keep_local.append(int(idxs[i].item()))  # 记录全局索引

                if idxs.numel() == 1:
                    break

                rest = torch.arange(1, idxs.numel(), dtype=torch.long)
                ious = box_iou(b[i].unsqueeze(0), b[rest]).squeeze(0)  # [rest]

                # 是否同组（-1 表示未知组，按“异组”处理）
                same = (g[rest] == g[i]) & (g[i] >= 0)

                # 选择阈值：同组用 t_intra，异组用 t_inter
                thr = torch.where(same,
                                  torch.full_like(ious, t_intra),
                                  torch.full_like(ious, t_inter))
                suppress = ious > thr  # 被抑制者

                # 可选：对“同组且IoU>t_intra”的被抑制框，做一次 WBF 融合到赢家
                if WBF:
                    merge_mask = same & suppress
                    if merge_mask.any():
                        merge_rest_idx = rest[merge_mask].tolist()
                        merge_global_idx = [int(idxs[i].item())] + [int(idxs[j].item()) for j in merge_rest_idx]
                        _wbf_merge(boxes, scores, merge_global_idx)  # 融合到全局赢家
                        # 同时更新本地副本的赢家框（便于后续 IoU 更稳定）
                        local_merge = torch.cat([torch.tensor([0], dtype=torch.long), merge_rest_idx])
                        w = s[local_merge] / (s[local_merge].sum() + 1e-8)
                        b[0] = (b[local_merge] * w[:, None]).sum(dim=0)
                        s[0] = s[local_merge].max()

                # 过滤保留：剩余中，非 suppress 的留下
                keep_mask = ~suppress
                # 拼回本轮赢家
                b = torch.cat([b[0:1], b[rest][keep_mask]], dim=0)
                s = torch.cat([s[0:1], s[rest][keep_mask]], dim=0)
                g = torch.cat([g[0:1], g[rest][keep_mask]], dim=0)
                idxs = torch.cat([idxs[0:1], idxs[rest][keep_mask]], dim=0)

                # 丢弃赢家本身，进入下一轮
                if idxs.numel() <= 1:
                    break
                b = b[1:];
                s = s[1:];
                g = g[1:];
                idxs = idxs[1:]

            kept_all.extend(keep_local)

            if DEBUG:
                same_cnt = int((grp[kept_all] >= 0).sum().item())
                print(f"[gNMS.debug] img={rec['image_id']} cls={c} kept={len(keep_local)} samegrp_in_kept={same_cnt}")

        # 写出该图的结果（按 kept_all 采样）
        kept_all = torch.tensor(sorted(set(kept_all)), dtype=torch.long)
        out_rec = {
            "image_id": rec["image_id"],
            "file_name": rec.get("file_name"),
            "size": rec["size"],
            "boxes": boxes[kept_all].tolist(),
            "scores": scores[kept_all].tolist(),
            "labels": labels[kept_all].tolist(),
        }
        return out_rec, len(kept_all)

    # 主循环
    n_img = 0
    with open(det_file) as f_in, open(out, "w") as f_out:
        for L in f_in:
            rec = json.loads(L)
            rG = gid_map.get(rec["image_id"])
            # 防御：断言 groups 与 detections 对齐
            if rG is not None and "kept" in rG:
                n_det = len(rec["boxes"])
                if any((k >= n_det or k < 0) for k in rG["kept"]):
                    raise ValueError(f"[gNMS] groups.kept 越界：img={rec['image_id']} n_det={n_det} kept_max={max(rG['kept'])}")
            out_rec, k = _gnms_single_image(rec, rG)
            f_out.write(json.dumps(out_rec) + "\n")
            n_img += 1

    print(f"[gNMS] wrote {out} (t_intra={t_intra}, t_inter={t_inter}, WBF={'on' if WBF else 'off'})")
    return out


def stage_eval_coco(cfg, det_file: Path, ann_file: Path = None, iou_type: str = "bbox") -> dict:
    """
    评测 COCO mAP（bbox）。输入: detections_*.jsonl
    返回: 指标字典，同时保存到 outputs/eval_<stem>.json
    """
    # numpy 1.24+ 兼容 old aliases
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool

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
                    "category_id": int(c),  # 需与GT类别ID一致
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                    "score": float(s)
                })

    cocoGT = COCO(str(ann_file))
    cocoDT = cocoGT.loadRes(results) if results else cocoGT.loadRes([])

    E = COCOeval(cocoGT, cocoDT, iouType=iou_type)
    # 只评测出现过的图，避免全量跑慢
    if img_ids:
        E.params.imgIds = sorted(img_ids)
    E.evaluate();
    E.accumulate();
    E.summarize()

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
    parser.add_argument("--steps", type=str, default="detect,graph,train,infer,groupnms,eval")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--t_intra", type=float, default=0.7)
    parser.add_argument("--t_inter", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--supervision", type=str, default="distill",  # ← 改默认为 distill，和实际一致
                        choices=["heur", "gt", "distill"])
    parser.add_argument("--tau", type=float, default=None, help="override tune.json for infer")  # 可选：手动设阈值
    args = parser.parse_args()

    import os
    if args.profile: os.environ["CONFIG_PROFILE"] = args.profile
    if args.remote:  os.environ["CONFIG_PROFILE"] = "remote"
    # —— 统一：把 supervision 写进 ENV，并用一个变量贯穿全程 ——
    os.environ["SUPERVISION"] = args.supervision
    mode = args.supervision

    cfg = load_config()
    print(f"[cfg] profile={cfg.profile} device={cfg.device}")
    print(f"[cfg] work_dir={cfg.paths.work_dir}")
    print(f"[cfg] supervision={mode}")  # 直观看到模式

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    artifacts: Dict[str, Path] = {}

    # 1) detect
    if "detect" in steps:
        artifacts["det"] = stage_detect(cfg)

    if "viz" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["viz"] = stage_viz_frcnn_vs_gt(cfg, det,
                                                 iou_thr=0.5,  # 可改
                                                 score_thr=0.05,  # 可改
                                                 limit=200)  # 可改

    # 2) graph
    if "graph" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        graph_path = stage_build_graph(cfg, det)  # 内部用 ENV=SUPERVISION 写 graphs_{mode}.jsonl
        artifacts["graph"] = graph_path

        # 只做 graph 质量诊断（不依赖模型）
        if os.getenv("DIAG", "1") == "1":
            print("[diag] quick_check_graph …")
            quick_check_graph(graph_path)

    # 3) train
    if "train" in steps:
        graph = artifacts.get("graph", cfg.paths.graphs_dir / f"graphs_{mode}.jsonl")
        artifacts["model"] = stage_train_grm(cfg, graph)

        if os.getenv("DIAG", "1") == "1":
            model_file = cfg.paths.models_dir / f"grm_edge_{mode}.pt"
            eval_pairs_auc(cfg, graph, model_file)
            dump_top_pairs(graph, model_file, cfg)

    # 4) tune
    if "tune" in steps:
        graph = artifacts.get("graph", cfg.paths.graphs_dir / f"graphs_{mode}.jsonl")
        model = artifacts.get("model", cfg.paths.models_dir / f"grm_edge_{mode}.pt")
        tau = stage_tune(cfg, graph, model)
        artifacts["tune"] = cfg.paths.outputs_dir / "tune.json"

    # 5) infer groups（保持与 *原始* detections 对齐）
    if "infer" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        model = artifacts.get("model", cfg.paths.models_dir / f"grm_edge_{mode}.pt")
        # 支持命令行覆盖 tau（否则从 tune.json 里读）
        artifacts["groups"] = stage_infer_groups(cfg, model, det, tau=args.tau)

    # 6) baselines
    if "stdnms" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["std"] = stage_std_nms(cfg, det, iou_thr=args.nms_iou)

    # 7) group-aware NMS
    if "groupnms" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        groups = artifacts.get("groups", cfg.paths.outputs_dir / "groups.jsonl")
        artifacts["gnms"] = stage_group_nms(cfg, det, groups, t_intra=args.t_intra, t_inter=args.t_inter)

    # 8) eval
    if "evalstd" in steps:
        std = artifacts.get("std", cfg.paths.outputs_dir / "detections_std_nms.jsonl")
        print("[evalstd]", stage_eval_coco(cfg, std))
    if "eval" in steps:
        g = artifacts.get("gnms", cfg.paths.outputs_dir / "detections_groupnms.jsonl")
        print("[eval]", stage_eval_coco(cfg, g))

    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
