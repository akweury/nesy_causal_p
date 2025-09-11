# Created by MacBook Pro at 07.09.25


# pipeline.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from config import load_config
import torch


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
def stage_detect(cfg,
                 batch_size: int = 2,
                 score_thr: float = 0.001,
                 print_every: int = 50, split="train") -> Path:
    """
    运行 FasterRCNN-ResNet50-FPN（预训练）在 cfg.paths.coco_images/annotations 上推理：
      - 输出: detections.jsonl（像素级 xyxy，labels=COCO 稀疏 category_id，scores）
      - 在线监控: 每 print_every 批打印滚动 P/R/F1 和近似 AP50（贪心匹配）
    """
    import json, time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.datasets import CocoDetection
    from pycocotools.coco import COCO

    out_dir = cfg.paths.detections_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        out_file = out_dir / "detections.jsonl"
    elif split == "val":
        out_file = out_dir / "detections_val.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}")

    if out_file.exists():
        print(f"[detect] reuse {out_file}")
        return out_file

    # ---- 路径与 split 校验 ----

    if split == "train":
        img_root = Path(cfg.paths.coco_images)
        ann_file = Path(cfg.paths.coco_annotations)
    elif split == "val":
        img_root = Path(cfg.paths.coco_images_val)
        ann_file = Path(cfg.paths.coco_annotations_val)

    else:
        raise ValueError(f"Unknown split: {split}")

    assert img_root.exists(), f"COCO_IMAGES not found: {img_root}"
    assert ann_file.exists(), f"COCO_ANN not found: {ann_file}"
    coco = COCO(str(ann_file))

    # ---- Dataset 包装：返回真实 COCO image_id ----
    tfm = T.Compose([T.ToTensor()])

    class CocoDetWrap(CocoDetection):
        def __getitem__(self, idx):
            img, targets = super().__getitem__(idx)  # targets: list[ann dict]
            image_id = int(self.ids[idx])  # 真实 COCO image_id
            return img, targets, image_id

    ds = CocoDetWrap(str(img_root), str(ann_file), transform=tfm)

    def collate(batch):
        imgs, tgts, ids = zip(*batch)
        return list(imgs), list(tgts), list(ids)

    # ---- DataLoader（低压设置，避免 /dev/shm 问题）----
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(cfg.num_workers, 2),
        pin_memory=False,
        prefetch_factor=2 if getattr(cfg, "num_workers", 0) > 0 else None,
        persistent_workers=False,
        collate_fn=collate
    )

    # ---- 模型 ----
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(cfg.device).eval()

    # ---- 工具 ----
    def _area(b):
        return max(0., b[2] - b[0]) * max(0., b[3] - b[1])

    def _iou(a, b):
        x1 = max(a[0], b[0]);
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]);
        y2 = min(a[3], b[3])
        inter = max(0., x2 - x1) * max(0., y2 - y1)
        return inter / max(1e-8, _area(a) + _area(b) - inter)

    # 同类、按分数贪心匹配
    def match_greedy(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5):
        if len(pred_boxes) == 0:
            return np.zeros(0, bool), np.zeros(0, bool), 0
        order = np.argsort(-np.asarray(pred_scores))
        used = set()
        tp = np.zeros(len(pred_boxes), dtype=bool)
        fp = np.zeros(len(pred_boxes), dtype=bool)
        gt_idx_by_class = {}
        for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            gt_idx_by_class.setdefault(int(gl), []).append(i)
        for pidx in order:
            c = int(pred_labels[pidx])
            candid = gt_idx_by_class.get(c, [])
            best = -1;
            best_iou = 0.0
            for gi in candid:
                if gi in used: continue
                iou = _iou(pred_boxes[pidx], gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou;
                    best = gi
            if best >= 0 and best_iou >= iou_thr:
                tp[pidx] = True;
                used.add(best)
            else:
                fp[pidx] = True
        fn = (len(gt_boxes) - len(used))
        return tp, fp, fn

    # 在线 AP50 估计（全局）
    all_scores, all_is_tp, total_fn = [], [], 0

    def approx_ap50():
        if len(all_scores) == 0:
            return 0.0
        s = np.asarray(all_scores, np.float32)
        y = np.asarray(all_is_tp, np.bool_)
        order = np.argsort(-s);
        y = y[order]
        tp_cum = np.cumsum(y)
        fp_cum = np.cumsum(~y)
        P = tp_cum / np.maximum(1, tp_cum + fp_cum)
        R = tp_cum / max(1, (tp_cum[-1] + total_fn))
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = np.max(P[R >= t]) if np.any(R >= t) else 0.0
            ap += p
        return ap / 11.0

    # ---- 推理 ----
    t0 = time.time()
    written = 0
    with open(out_file, "w") as fout, torch.no_grad():
        for it, (imgs, targets, ids) in enumerate(dl, 1):
            imgs = [im.to(cfg.device) for im in imgs]
            outputs = model(imgs)

            for img, tgt, out, image_id in zip(imgs, targets, outputs, ids):
                image_id = int(image_id)

                # 预测（像素 xyxy；labels 已是 COCO 稀疏 category_id）
                boxes = out["boxes"].detach().float().cpu().numpy().tolist()
                labels = out["labels"].detach().cpu().numpy().tolist()
                scores = out["scores"].detach().float().cpu().numpy().tolist()

                if score_thr > 0:
                    keep = [i for i, s in enumerate(scores) if s >= score_thr]
                    boxes = [boxes[i] for i in keep]
                    labels = [labels[i] for i in keep]
                    scores = [scores[i] for i in keep]

                fout.write(json.dumps({
                    "image_id": image_id,
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores
                }) + "\n")
                written += 1

                # ---- 在线指标（与 GT 同类、IoU>=0.5）----
                gt_boxes = [];
                gt_labels = []
                for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None)):
                    x, y, w, h = a["bbox"]
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(int(a["category_id"]))

                if len(boxes):
                    tp, fp, fn = match_greedy(boxes, labels, scores, gt_boxes, gt_labels, iou_thr=0.5)
                    all_scores.extend(scores)
                    all_is_tp.extend(tp.tolist())
                    total_fn += int(fn)

            if it % print_every == 0:
                tp_sum = int(np.sum(all_is_tp))
                fp_sum = len(all_is_tp) - tp_sum
                P = tp_sum / max(1, tp_sum + fp_sum)
                R = tp_sum / max(1, tp_sum + total_fn)
                F1 = 2 * P * R / max(1e-8, P + R)
                AP50_est = approx_ap50()
                print(f"[detect] {it:5d}/{len(dl)} imgs  "
                      f"P={P:.3f} R={R:.3f} F1={F1:.3f}  AP50~{AP50_est:.3f}  "
                      f"TP={tp_sum} FP={fp_sum} FN={total_fn}  time={time.time() - t0:.1f}s")

    print(f"[detect] wrote {out_file}  images={written}  total_time={time.time() - t0:.1f}s")
    return out_file


def stage_viz_frcnn_vs_gt(cfg, det_file: Path,
                          iou_thr: float = 0.5,
                          score_thr: float = 0.05,
                          limit: int = 200) -> Path:
    """
    为每个预测实例导出单图：显示 Pred(label/score) 与 匹配到的 GT(label)，
    并把未匹配 GT（FN）也各自导出单图。
    目录结构：outputs/viz_instances/{TP,FP,FN}/imageid_*_*.jpg
    """
    import json, os, cv2, math, numpy as np
    from pathlib import Path
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_instances"
    (out_dir / "TP").mkdir(parents=True, exist_ok=True)
    (out_dir / "FP").mkdir(parents=True, exist_ok=True)
    (out_dir / "FN").mkdir(parents=True, exist_ok=True)

    def iou_xyxy(a, b):
        xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
        xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
        w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
        inter = w * h
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        u = area_a + area_b - inter
        return inter / u if u > 0 else 0.0

    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    cid2name = {c: coco.cats[c]["name"] for c in coco.cats}
    images_root = Path(cfg.paths.coco_images)

    recs = [json.loads(l) for l in open(det_file)]
    if limit: recs = recs[:limit]

    # 画带文字的小工具
    def draw_and_save_crop(im, box, text_lines, color, save_path, pad=4):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = im.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        crop = im[y1:y2 + 1, x1:x2 + 1].copy()
        if crop.size == 0: return
        # 顶部黑条+文字
        bar_h = 22 + 14 * (len(text_lines) - 1)
        bar = np.zeros((bar_h, crop.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(bar, (0, 0), (bar.shape[1] - 1, bar.shape[0] - 1), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, t in enumerate(text_lines):
            cv2.putText(bar, t, (6, 16 + 14 * i), font, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        vis = np.vstack([bar, crop])
        # 外框
        cv2.rectangle(vis, (0, bar_h), (vis.shape[1] - 1, vis.shape[0] - 1), color, 2)
        cv2.imwrite(str(save_path), vis)

    for rec in recs:
        img_id = rec["image_id"]
        fn = id2fname.get(img_id)
        if not fn: continue
        img_path = images_root / fn
        im = cv2.imread(str(img_path))
        if im is None: continue

        # GT
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        gtb = []
        gtl = []
        for a in anns:
            x, y, w, h = a["bbox"]
            gtb.append([x, y, x + w, y + h]);
            gtl.append(a["category_id"])
        gtb = np.array(gtb, np.float32) if gtb else np.zeros((0, 4), np.float32)
        gtl = np.array(gtl, np.int32) if gtl else np.zeros((0,), np.int32)
        gt_used = np.zeros(len(gtb), dtype=bool)

        # 预测（阈值过滤）
        pb = np.array(rec["boxes"], np.float32) if rec["boxes"] else np.zeros((0, 4), np.float32)
        ps = np.array(rec["scores"], np.float32) if rec["scores"] else np.zeros((0,), np.float32)
        pl = np.array(rec["labels"], np.int32) if rec["labels"] else np.zeros((0,), np.int32)
        keep = ps >= score_thr
        pb, ps, pl = pb[keep], ps[keep], pl[keep]

        # 按类贪心匹配
        matched_gt = {}  # det_idx -> gt_idx
        for c in np.unique(pl):
            m = pl == c
            if not np.any(m): continue
            di = np.where(m)[0]
            order = np.argsort(-ps[m])
            di = di[order]
            # 同类 GT 列表
            gt_idx = [i for i, k in enumerate(gtl) if k == c]
            used_local = set()
            for d in di:
                best, bj = 0.0, -1
                for j_local, j in enumerate(gt_idx):
                    if j_local in used_local: continue
                    iou = iou_xyxy(pb[d], gtb[j])
                    if iou > best: best, bj = iou, j_local
                if best >= iou_thr:
                    used_local.add(bj)
                    matched_gt[d] = gt_idx[bj]
                    gt_used[gt_idx[bj]] = True

        # 导出 TP/FP 预测实例
        for i in range(len(pb)):
            pred_cls = int(pl[i])
            pred_name = cid2name.get(pred_cls, str(pred_cls))
            box = pb[i].tolist()
            if i in matched_gt:
                j = matched_gt[i]
                gt_name = cid2name.get(int(gtl[j]), str(int(gtl[j])))
                save_p = out_dir / "TP" / f"{img_id}_det{i}_Pred-{pred_name}_{ps[i]:.2f}_GT-{gt_name}.jpg"
                draw_and_save_crop(im, box,
                                   [f"TP  IoU@{iou_thr}",
                                    f"Pred: {pred_name}  s={ps[i]:.2f}",
                                    f"GT:   {gt_name}"],
                                   color=(80, 200, 60), save_path=save_p)
            else:
                save_p = out_dir / "FP" / f"{img_id}_det{i}_Pred-{pred_name}_{ps[i]:.2f}_GT-None.jpg"
                draw_and_save_crop(im, box,
                                   [f"FP  IoU@{iou_thr}",
                                    f"Pred: {pred_name}  s={ps[i]:.2f}",
                                    "GT:   None"],
                                   color=(40, 220, 220), save_path=save_p)

        # 导出未匹配 GT（FN）
        for j in range(len(gtb)):
            if gt_used[j]: continue
            gt_name = cid2name.get(int(gtl[j]), str(int(gtl[j])))
            save_p = out_dir / "FN" / f"{img_id}_gt{j}_GT-{gt_name}.jpg"
            draw_and_save_crop(im, gtb[j].tolist(),
                               [f"FN  IoU@{iou_thr}",
                                f"GT: {gt_name}",
                                "Pred: None"],
                               color=(30, 30, 230), save_path=save_p)

    print(f"[viz] wrote per-instance crops to {out_dir}")
    return out_dir


def stage_group_by_similarity(cfg, det_file: Path,
                              sim_thr: float = 0.90,
                              iou_thr: float = 0.30,
                              iom_thr: float = 0.70,
                              dist_thr: float = 0.15,
                              sub_nms_iou: float = 0.7,
                              hist_bins: int = 16,
                              pad: int = 2) -> Path:
    """
    类相似 + Proximity 分组：
    - 条件：外观相似>=sim_thr 且 (IoU>=iou_thr or IoM>=iom_thr or center_dist<=dist_thr)
    - 分组：并查集合并；组内再跑一次小NMS（sub_nms_iou）保留多个代表
    - 输出 groups_simprox.jsonl
    """
    import json, cv2, numpy as np
    from pathlib import Path

    out = cfg.paths.outputs_dir / "groups_simprox.jsonl"
    if out.exists():
        print(f"[simprox] reuse {out}")
        return out

    # === helpers ===
    def crop_hsv_hist(im, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = im.shape[:2]
        x1 = max(0, x1 - pad);
        y1 = max(0, y1 - pad);
        x2 = min(w - 1, x2 + pad);
        y2 = min(h - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1: return None
        crop = im[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h1 = cv2.calcHist([hsv], [0], None, [hist_bins], [0, 180])
        h2 = cv2.calcHist([hsv], [1], None, [hist_bins], [0, 256])
        h3 = cv2.calcHist([hsv], [2], None, [hist_bins], [0, 256])
        v = np.concatenate([h1.ravel(), h2.ravel(), h3.ravel()]).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        return v

    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def iou_xyxy(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
        ua = area(a) + area(b) - inter
        return inter / (ua + 1e-8)

    def iom_xyxy(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
        return inter / (min(area(a), area(b)) + 1e-8)

    def center_dist_norm(a, b):
        ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        area_max = max((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return d / (area_max ** 0.5 + 1e-8)

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

    def nms(boxes, scores, thr):
        keep = []
        idxs = np.argsort(scores)[::-1]
        while len(idxs) > 0:
            i = idxs[0];
            keep.append(i)
            if len(idxs) == 1: break
            ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in idxs[1:]])
            idxs = idxs[1:][ious < thr]
        return keep

    # === COCO索引 ===
    from pycocotools.coco import COCO
    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    images_root = Path(cfg.paths.coco_images)

    with open(det_file) as f_in, open(out, "w") as f_out:
        for L in f_in:
            rec = json.loads(L)
            img_id = rec["image_id"];
            fn = id2fname.get(img_id)
            if not fn: continue
            im = cv2.imread(str(images_root / fn))
            if im is None: continue

            boxes = rec.get("boxes", [])
            labels = rec.get("labels", [])
            scores = rec.get("scores", [])
            n = len(boxes)
            if n == 0:
                f_out.write(json.dumps({"image_id": img_id, "groups": [], "kept": []}) + "\n")
                continue

            groups_all = [];
            kept_all = []
            for c in sorted(set(labels)):
                idx = [i for i, l in enumerate(labels) if l == c]
                if len(idx) <= 1:
                    kept_all.extend(idx)
                    # 删除：if len(idx)==1: groups_all.append([0])
                    # 改为：单候选不建组
                    if len(idx) == 1:
                        kept_all.extend(idx)
                        continue

                feats = [];
                valid = []
                for i in idx:
                    v = crop_hsv_hist(im, boxes[i])
                    if v is None: continue
                    feats.append(v);
                    valid.append(i)
                if len(valid) == 0: continue
                feats = np.stack(feats, 0)
                m = len(valid)
                uf = uf_make(m)

                for a in range(m):
                    for b in range(a + 1, m):
                        sim = cos_sim(feats[a], feats[b])
                        if sim < sim_thr: continue
                        iou = iou_xyxy(boxes[valid[a]], boxes[valid[b]])
                        iom = iom_xyxy(boxes[valid[a]], boxes[valid[b]])
                        dist = center_dist_norm(boxes[valid[a]], boxes[valid[b]])
                        if (iou >= iou_thr) or (iom >= iom_thr) or (dist <= dist_thr):
                            uf_union(uf, a, b)

                roots = {}
                for i_loc in range(m):
                    r = uf_find(uf, i_loc)
                    roots.setdefault(r, []).append(i_loc)

                kept_all.extend(valid)
                for vs in roots.values():
                    cand = [valid[i_loc] for i_loc in vs]
                    if not cand: continue
                    # 子簇NMS避免全并成1
                    sub_boxes = [boxes[i] for i in cand]
                    sub_scores = [scores[i] for i in cand]
                    keep_idx = nms(sub_boxes, sub_scores, sub_nms_iou)
                    # 保存成局部索引
                    grp = [cand.index(cand[j]) for j in keep_idx]
                    groups_all.append(grp)

            f_out.write(json.dumps({"image_id": img_id, "groups": groups_all, "kept": kept_all}) + "\n")

    print(f"[simprox] wrote {out}")
    return out


def stage_viz_simprox_groups(cfg,
                             groups_file: Path,
                             det_file: Path,
                             pad: int = 8,
                             limit: int = 0) -> Path:
    """
    可视化 sim+proximity 分组：
      - 每组一张：组内预测(绿) + 同类所有GT(红)
      - 裁剪范围 = 组并集 ∪ 同类GT并集 + pad
    产物：outputs/viz_simprox_groups/*.jpg + index.json/csv
    """
    import json, csv, cv2, numpy as np
    from pathlib import Path
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_simprox_groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_json = out_dir / "index.json"
    idx_csv = out_dir / "index.csv"

    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {im["id"]: im["file_name"] for im in coco.dataset["images"]}
    cid2name = {c: coco.cats[c]["name"] for c in coco.cats}
    images_root = Path(cfg.paths.coco_images)

    # det 映射
    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L)
            det_map[str(r["image_id"])] = r

    GREEN = (80, 200, 60);
    RED = (30, 30, 230);
    WHITE = (240, 240, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    index = []
    wrote = 0

    with open(groups_file) as f:
        for L in f:
            g = json.loads(L)
            img_id = g["image_id"]
            fn = id2fname.get(img_id)
            if not fn: continue
            det = det_map.get(str(img_id))
            if not det: continue

            im = cv2.imread(str(images_root / fn))
            if im is None: continue
            H, W = im.shape[:2]

            # GT
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            boxes = det["boxes"];
            labels = det["labels"];
            scores = det["scores"]
            kept = g.get("kept", list(range(len(boxes))))

            for k, members_local in enumerate(g.get("groups", []), start=1):
                if not members_local: continue
                # 局部→全局 det 索引
                mem_idx = [kept[j] for j in members_local if 0 <= j < len(kept)]
                if not mem_idx: continue

                gb = [boxes[i] for i in mem_idx]
                gl = [int(labels[i]) for i in mem_idx]
                gs = [float(scores[i]) for i in mem_idx]
                cls_set = set(gl)

                # 同类全部 GT（不看 IoU）
                same_cls_gts = []
                for a in anns:
                    cid = int(a["category_id"])
                    if cid in cls_set:
                        x, y, w_, h_ = a["bbox"]
                        same_cls_gts.append(([x, y, x + w_, y + h_],
                                             cid,
                                             cid2name.get(cid, str(cid))))

                # 裁剪：组并集 ∪ 同类GT并集
                xs1 = [b[0] for b in gb] + [g_[0][0] for g_ in same_cls_gts] or [0]
                ys1 = [b[1] for b in gb] + [g_[0][1] for g_ in same_cls_gts] or [0]
                xs2 = [b[2] for b in gb] + [g_[0][2] for g_ in same_cls_gts] or [W - 1]
                ys2 = [b[3] for b in gb] + [g_[0][3] for g_ in same_cls_gts] or [H - 1]
                x1 = max(0, int(min(xs1) - pad));
                y1 = max(0, int(min(ys1) - pad))
                x2 = min(W - 1, int(max(xs2) + pad));
                y2 = min(H - 1, int(max(ys2) + pad))
                crop = im[y1:y2 + 1, x1:x2 + 1].copy()
                if crop.size == 0: continue

                # 画 GT（红）
                for (gxyxy, cid, name) in same_cls_gts:
                    X1, Y1, X2, Y2 = map(int, gxyxy)
                    cv2.rectangle(crop, (X1 - x1, Y1 - y1), (X2 - x1, Y2 - y1), RED, 2)
                    cv2.putText(crop, f"GT:{name}", (X1 - x1, max(14, Y1 - y1 - 4)),
                                font, 0.5, RED, 1, cv2.LINE_AA)

                # 画组成员（绿）
                for b, c, s in zip(gb, gl, gs):
                    B1, B2, B3, B4 = map(int, b)
                    name = cid2name.get(c, str(c))
                    cv2.rectangle(crop, (B1 - x1, B2 - y1), (B3 - x1, B4 - y1), GREEN, 2)
                    cv2.putText(crop, f"{name} s={s:.2f}", (B1 - x1, max(14, B2 - y1 - 4)),
                                font, 0.5, GREEN, 1, cv2.LINE_AA)

                # 顶栏
                gt_names = ",".join(sorted({n for *_, n in same_cls_gts})) or "None"
                bar = np.zeros((26, crop.shape[1], 3), np.uint8)
                cv2.putText(bar, f"img {img_id}  G{k}  members={len(mem_idx)}  GT(same-class): {gt_names}",
                            (6, 18), font, 0.55, WHITE, 1, cv2.LINE_AA)
                vis = np.vstack([bar, crop])

                out_p = out_dir / f"{img_id}_G{k}.jpg"
                cv2.imwrite(str(out_p), vis)

                # 索引
                index.append({
                    "image_id": int(img_id),
                    "file_name": fn,
                    "group_id": int(k),
                    "member_count": len(mem_idx),
                    "members": [
                        {"det_idx": int(i),
                         "label_id": int(c),
                         "label": cid2name.get(int(c), str(int(c))),
                         "score": float(s),
                         "box_xyxy": [float(x) for x in boxes[i]]}
                        for i, c, s in zip(mem_idx, gl, gs)
                    ],
                    "gt_same_class": [
                        {"category_id": int(cid),
                         "category": name,
                         "box_xyxy": [float(x) for x in gxyxy]}
                        for (gxyxy, cid, name) in same_cls_gts
                    ],
                    "viz_path": str(out_p)
                })

                wrote += 1
                if limit and wrote >= limit:
                    idx_json.write_text(json.dumps(index, indent=2))
                    with open(idx_csv, "w", newline="") as fcsv:
                        w = csv.DictWriter(fcsv, fieldnames=["image_id", "file_name", "group_id", "member_count", "viz_path"])
                        w.writeheader()
                        for r in index:
                            w.writerow({k: r[k] for k in w.fieldnames})
                    print(f"[viz-simprox] wrote {out_dir} (limited {limit})")
                    return out_dir

    # 写索引
    idx_json.write_text(json.dumps(index, indent=2))
    with open(idx_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["image_id", "file_name", "group_id", "member_count", "viz_path"])
        w.writeheader()
        for r in index:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"[viz-simprox] wrote {out_dir}\n- index: {idx_json.name}, {idx_csv.name}")
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


def stage_viz_graph_groups(cfg,
                           graph_file: Path,
                           det_file: Path,
                           iou_thr: float = 0.5,
                           limit: int = 200) -> Path:
    """
    可视化：对每张图，把 graph 中 label=1 的边做并查集 → 形成“组”。
    为每个组导出两张图：
      - full：整张图，组内检测框高亮、其他变暗；同时叠加 GT 框
      - crop：以组的并集框裁剪，叠加成员与 GT
    输出目录：/outputs/viz_graph_groups/{full,crops}/imageid_groupK.jpg
    """
    import os, json, cv2, numpy as np
    from pathlib import Path
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_graph_groups"
    (out_dir / "full").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    # 载入 COCO
    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    cid2name = {c: coco.cats[c]["name"] for c in coco.cats}
    images_root = Path(cfg.paths.coco_images)

    # 读取 detections：image_id -> rec
    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L)
            det_map[r["image_id"]] = r

    # 工具
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

    def draw_group_full(im, boxes, labels, group_idx, members, gts, title, save_p):
        vis = im.copy()
        # 暗化整图
        vis = (vis * 0.25).astype(np.uint8)
        # 高亮成员
        COLORS = [(52, 194, 73), (52, 148, 240), (239, 188, 28), (226, 86, 86), (178, 102, 255)]
        for t, i in enumerate(members):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            clr = COLORS[t % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), clr, 2)
            name = cid2name.get(int(labels[i]), str(int(labels[i])))
            cv2.putText(vis, f"G{group_idx}:{name}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
        # 叠加 GT（红）
        for (bx, by, bx2, by2, cname) in gts:
            cv2.rectangle(vis, (int(bx), int(by)), (int(bx2), int(by2)), (36, 36, 240), 2)
            cv2.putText(vis, f"GT:{cname}", (int(bx), max(18, int(by) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 36, 240), 1, cv2.LINE_AA)
        # 顶部标题
        pad = 26
        bar = np.zeros((pad, vis.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        out = np.vstack([bar, vis])
        cv2.imwrite(str(save_p), out)

    def draw_group_crop(im, boxes, members, gts, save_p):
        xs1 = [boxes[i][0] for i in members];
        ys1 = [boxes[i][1] for i in members]
        xs2 = [boxes[i][2] for i in members];
        ys2 = [boxes[i][3] for i in members]
        x1, y1, x2, y2 = int(max(0, min(xs1))), int(max(0, min(ys1))), int(max(xs2)), int(max(ys2))
        x1 = max(0, x1 - 4);
        y1 = max(0, y1 - 4);
        x2 = min(im.shape[1] - 1, x2 + 4);
        y2 = min(im.shape[0] - 1, y2 + 4)
        crop = im[y1:y2 + 1, x1:x2 + 1].copy()
        if crop.size == 0: return
        # 绘制成员（绿）
        for i in members:
            bx1, by1, bx2, by2 = [int(v) for v in boxes[i]]
            cv2.rectangle(crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (60, 200, 60), 2)
        # 绘制 GT（红）
        for (gx1, gy1, gx2, gy2, _) in gts:
            cv2.rectangle(crop, (int(gx1) - x1, int(gy1) - y1), (int(gx2) - x1, int(gy2) - y1), (36, 36, 240), 2)
        cv2.imwrite(str(save_p), crop)

    # 主循环：读 graph，按正边并查集成组并可视化
    cnt_img = 0
    with open(graph_file) as f:
        for k, L in enumerate(f):
            r = json.loads(L)
            img_id = r["image_id"]
            if img_id not in det_map: continue
            fn = id2fname.get(img_id)
            if not fn: continue
            img_path = images_root / fn
            im = cv2.imread(str(img_path))
            if im is None: continue

            det = det_map[img_id]
            boxes = det["boxes"];
            labels = det["labels"];
            scores = det["scores"]
            n = len(boxes)
            if n == 0: continue

            # 并查集：只用 label==1 的边
            parents = uf_make(n)
            for p in (r.get("pairs") or []):
                if p.get("label", 0) == 1:
                    i, j = p["i"], p["j"]
                    if 0 <= i < n and 0 <= j < n:
                        uf_union(parents, i, j)
            # 收集组
            roots = {}
            for i in range(n):
                roots.setdefault(uf_find(parents, i), []).append(i)
            groups = [g for g in roots.values() if len(g) >= 2]  # 只看 size>=2 的组更有意义
            if not groups:
                cnt_img += 1
                if limit and cnt_img >= limit: break
                continue

            # 读取 GT
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gts = []
            for a in anns:
                x, y, w, h = a["bbox"]
                gts.append([x, y, x + w, y + h, cid2name.get(a["category_id"], str(a["category_id"]))])

            # —— 逐组可视化（只画“同类且匹配该组”的 GT）——
            for gi, members in enumerate(groups, start=1):
                # 该组涉及的预测类别集合
                member_cls = {int(labels[m]) for m in members}

                # 预筛 GT：类别在组内类别集合里
                cand_gts = [(gx1, gy1, gx2, gy2, gname, gcid)
                            for (gx1, gy1, gx2, gy2, gname, gcid) in
                            [(a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3],
                              cid2name.get(a["category_id"], str(a["category_id"])),
                              a["category_id"]) for a in anns]
                            if int(gcid) in member_cls]

                # 与组内任一成员 IoU≥阈值 的“同类GT”
                matched_gts = []
                for (gx1, gy1, gx2, gy2, gname, gcid) in cand_gts:
                    hit = False
                    for m in members:
                        bx1, by1, bx2, by2 = boxes[m]
                        xx1, yy1 = max(bx1, gx1), max(by1, gy1)
                        xx2, yy2 = min(bx2, gx2), min(by2, gy2)
                        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                        area_m = max(0, bx2 - bx1) * max(0, by2 - by1)
                        area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
                        iou = inter / (area_m + area_g - inter + 1e-8)
                        if iou >= iou_thr:
                            hit = True
                            break
                    if hit:
                        matched_gts.append((gx1, gy1, gx2, gy2, gname))

                title = f"img {img_id}  G{gi}  members={len(members)}  matchGT={','.join(sorted({g for *_, g in matched_gts})) or 'None'}"

                # 全图（组成员高亮 + 仅匹配GT）
                save_full = out_dir / "full" / f"{img_id}_G{gi}.jpg"
                draw_group_full(im, boxes, labels, gi, members, matched_gts, title, save_full)

                # 裁剪（仅匹配GT）
                save_crop = out_dir / "crops" / f"{img_id}_G{gi}.jpg"
                draw_group_crop(im, boxes, members, matched_gts, save_crop)

            cnt_img += 1
            if limit and cnt_img >= limit: break

    print(f"[viz-graph] wrote {out_dir} (full & crops)")
    return out_dir


def stage_viz_pred_groups(cfg, groups_file, det_file, pad=8, limit=200):
    import json, cv2, numpy as np, csv
    from pathlib import Path
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_pred_groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_json = out_dir / "index.json"
    index_csv = out_dir / "index.csv"

    coco = COCO(str(cfg.paths.coco_annotations))
    id2fname = {im["id"]: im["file_name"] for im in coco.dataset["images"]}
    cid2name = {c: coco.cats[c]["name"] for c in coco.cats}
    images_root = Path(cfg.paths.coco_images)

    # detections map
    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L)
            det_map[str(r["image_id"])] = r

    GREEN = (80, 200, 60);
    RED = (30, 30, 230);
    WHITE = (240, 240, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    index = []  # 收集行：每组一行
    wrote = 0

    with open(groups_file) as f:
        for L in f:
            g = json.loads(L)
            img_id = g["image_id"]
            det = det_map.get(str(img_id))
            if not det: continue

            fn = id2fname.get(img_id)
            if not fn: continue
            im = cv2.imread(str(images_root / fn))
            if im is None: continue
            H, W = im.shape[:2]

            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            boxes = det["boxes"];
            labels = det["labels"];
            scores = det["scores"]
            kept = g.get("kept", list(range(len(boxes))))

            for k, members_local in enumerate(g.get("groups", []), start=1):
                if not members_local: continue
                # 局部→原始 det 索引
                mem_idx = [kept[m] for m in members_local if 0 <= m < len(kept)]
                if not mem_idx: continue

                # 组成员
                gb = [boxes[i] for i in mem_idx]
                gl = [int(labels[i]) for i in mem_idx]
                gs = [float(scores[i]) for i in mem_idx]
                cls_set = set(gl)

                # 同类 GT（不看 IoU，全包含）
                same_class_gts = []
                for a in anns:
                    cid = int(a["category_id"])
                    if cid in cls_set:
                        x, y, w, h = a["bbox"]
                        same_class_gts.append(([x, y, x + w, y + h],
                                               cid,
                                               cid2name.get(cid, str(cid))))

                # 裁剪范围 = 组并集 ∪ 同类GT并集 + pad
                xs1 = [b[0] for b in gb] + [g_[0][0] for g_ in same_class_gts]
                ys1 = [b[1] for b in gb] + [g_[0][1] for g_ in same_class_gts]
                xs2 = [b[2] for b in gb] + [g_[0][2] for g_ in same_class_gts]
                ys2 = [b[3] for b in gb] + [g_[0][3] for g_ in same_class_gts]
                x1 = max(0, int(min(xs1) - pad));
                y1 = max(0, int(min(ys1) - pad))
                x2 = min(W - 1, int(max(xs2) + pad));
                y2 = min(H - 1, int(max(ys2) + pad))
                crop = im[y1:y2 + 1, x1:x2 + 1].copy()
                if crop.size == 0: continue

                # 画 GT（红）
                for (gxyxy, cid, gname) in same_class_gts:
                    X1, Y1, X2, Y2 = map(int, gxyxy)
                    cv2.rectangle(crop, (X1 - x1, Y1 - y1), (X2 - x1, Y2 - y1), RED, 2)
                    cv2.putText(crop, f"GT:{gname}", (X1 - x1, max(14, Y1 - y1 - 4)),
                                font, 0.5, RED, 1, cv2.LINE_AA)

                # 画组成员预测（绿）
                for b, c, s in zip(gb, gl, gs):
                    B1, B2, B3, B4 = map(int, b)
                    name = cid2name.get(c, str(c))
                    cv2.rectangle(crop, (B1 - x1, B2 - y1), (B3 - x1, B4 - y1), GREEN, 2)
                    cv2.putText(crop, f"{name} s={s:.2f}", (B1 - x1, max(14, B2 - y1 - 4)),
                                font, 0.5, GREEN, 1, cv2.LINE_AA)

                # 标题 & 保存
                gt_names = ",".join(sorted({n for *_, n in same_class_gts})) or "None"
                bar = np.zeros((26, crop.shape[1], 3), np.uint8)
                cv2.putText(bar, f"img {img_id}  G{k}  members={len(mem_idx)}  GT(same-class): {gt_names}",
                            (6, 18), font, 0.55, WHITE, 1, cv2.LINE_AA)
                vis = np.vstack([bar, crop])
                out_p = out_dir / f"{img_id}_G{k}.jpg"
                cv2.imwrite(str(out_p), vis)

                # —— 索引行（写到 index.json / index.csv）——
                idx_row = {
                    "image_id": int(img_id),
                    "file_name": fn,
                    "group_id": int(k),
                    "member_count": len(mem_idx),
                    "members": [
                        {"det_idx": int(i),
                         "label_id": int(c),
                         "label": cid2name.get(int(c), str(int(c))),
                         "score": float(s),
                         "box_xyxy": [float(x) for x in boxes[i]]}
                        for i, c, s in zip(mem_idx, gl, gs)
                    ],
                    "gt_same_class": [
                        {"category_id": int(cid),
                         "category": name,
                         "box_xyxy": [float(x) for x in gxyxy]}
                        for (gxyxy, cid, name) in same_class_gts
                    ],
                    "viz_path": str(out_p)
                }
                index.append(idx_row)

                wrote += 1
                if limit and wrote >= limit:
                    print(f"[viz-pred-groups] wrote {out_dir} (limited {limit})")
                    # 同时落盘索引
                    index_json.write_text(json.dumps(index, indent=2))
                    with open(index_csv, "w", newline="") as fcsv:
                        w = csv.DictWriter(fcsv,
                                           fieldnames=["image_id", "file_name", "group_id", "member_count", "viz_path"])
                        w.writeheader()
                        for r in index:
                            w.writerow({k: r[k] for k in w.fieldnames})
                    return out_dir

    # 全量落盘
    index_json.write_text(json.dumps(index, indent=2))
    with open(index_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv,
                           fieldnames=["image_id", "file_name", "group_id", "member_count", "viz_path"])
        w.writeheader()
        for r in index:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"[viz-pred-groups] wrote {out_dir}\n- index: {index_json.name}, {index_csv.name}")
    return out_dir


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


from scipy.optimize import linear_sum_assignment
import numpy as np


def _xywh_to_xyxy(b): x, y, w, h = b; return [x, y, x + w, y + h]


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]);
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1);
    A = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    return inter / max(1e-8, A(a) + A(b) - inter)


def _iom(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]);
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1);
    A = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    return inter / max(1e-8, min(A(a), A(b)))


def _cdist(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2;
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5;
    s = max((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return d / (s ** 0.5 + 1e-8)


def stage_match_to_gt_class(cfg, det_file, alpha=1.0, beta=0.5, gamma=0.3, iou_min=0.1):
    """
    输出 /outputs/matches.jsonl
    每行：{"image_id", "matches":[(pi,gj,cls)], "unmatch_pred":[(cls,pi)], "miss_gt":[(cls,gj)]}
    cost = α(1−IoU)+β(1−IoM)+γ*center_dist_norm，IoU<iou_min 置为大成本。
    """
    import json
    from pathlib import Path
    from pycocotools.coco import COCO

    out = cfg.paths.outputs_dir / "matches.jsonl"
    if out.exists(): print(f"[match] reuse {out}"); return out

    coco = COCO(str(cfg.paths.coco_annotations))
    with open(det_file) as fdet, open(out, "w") as fout:
        for L in fdet:
            rec = json.loads(L)
            img_id = rec["image_id"]
            boxes, labels = rec["boxes"], rec["labels"]

            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gt_by_cls = {}
            for a in anns:
                gt_by_cls.setdefault(int(a["category_id"]), []).append(_xywh_to_xyxy(a["bbox"]))

            res = {"image_id": img_id, "matches": [], "unmatch_pred": [], "miss_gt": []}

            classes = sorted(set(labels) | set(gt_by_cls.keys()))
            for cls in classes:
                p_idx = [i for i, l in enumerate(labels) if int(l) == cls]
                g_box = gt_by_cls.get(cls, [])

                if not p_idx and not g_box:
                    continue
                if not p_idx:
                    res["miss_gt"] += [(cls, j) for j in range(len(g_box))]
                    continue
                if not g_box:
                    res["unmatch_pred"] += [(cls, i) for i in p_idx]
                    continue

                P = [boxes[i] for i in p_idx]
                C = np.empty((len(P), len(g_box)), dtype=np.float32)
                C.fill(1e3)
                for i, pb in enumerate(P):
                    for j, gb in enumerate(g_box):
                        iou = _iou(pb, gb)
                        if iou < iou_min:
                            continue
                        iom = _iom(pb, gb);
                        cd = _cdist(pb, gb)
                        C[i, j] = alpha * (1.0 - iou) + beta * (1.0 - iom) + gamma * cd

                rr, cc = linear_sum_assignment(C)
                used_g = set();
                used_p = set()
                for r, c in zip(rr, cc):
                    if C[r, c] >= 999: continue
                    res["matches"].append((int(p_idx[r]), int(c), int(cls)))
                    used_g.add(c);
                    used_p.add(p_idx[r])

                res["unmatch_pred"] += [(cls, i) for i in p_idx if i not in used_p]
                res["miss_gt"] += [(cls, j) for j in range(len(g_box)) if j not in used_g]

            fout.write(json.dumps(res) + "\n")

    print(f"[match] wrote {out}")
    return out


def stage_viz_match(cfg, det_file, match_file, pad=6, limit=0):
    """
    输出 /outputs/viz_match/*.jpg
    文件名：{image_id}_{className}.jpg
    """
    import json, cv2, numpy as np
    from pathlib import Path
    from pycocotools.coco import COCO

    out_dir = cfg.paths.outputs_dir / "viz_match"
    out_dir.mkdir(parents=True, exist_ok=True)

    GREEN = (80, 200, 60);
    RED = (30, 30, 230);
    YEL = (40, 200, 230);
    PUR = (200, 60, 200);
    WHITE = (240, 240, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    coco = COCO(str(cfg.paths.coco_annotations))
    id2fn = {im["id"]: im["file_name"] for im in coco.dataset["images"]}
    cid2name = {c: coco.cats[c]["name"] for c in coco.cats}
    img_root = Path(cfg.paths.coco_images)

    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L);
            det_map[str(r["image_id"])] = r

    wrote = 0
    with open(match_file) as f:
        for L in f:
            m = json.loads(L)
            img_id = m["image_id"];
            det = det_map.get(str(img_id))
            if det is None: continue

            im = cv2.imread(str(img_root / id2fn[img_id]))
            if im is None: continue
            H, W = im.shape[:2]

            boxes, labels, scores = det["boxes"], det["labels"], det["scores"]
            # 反查：每类有哪些 GT
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gt_by_cls = {}
            for a in anns:
                gt_by_cls.setdefault(int(a["category_id"]), []).append(_xywh_to_xyxy(a["bbox"]))

            # 汇总这个图里出现的类（以匹配文件为准）
            cls_set = set([c for *_, c in m["matches"]]) \
                      | set([c for c, _ in m["unmatch_pred"]]) \
                      | set([c for c, _ in m["miss_gt"]])

            for cls in sorted(cls_set):
                vis = im.copy()

                # 画 GT（红）
                gts = gt_by_cls.get(cls, [])
                for gb in gts:
                    X1, Y1, X2, Y2 = map(int, gb)
                    cv2.rectangle(vis, (X1, Y1), (X2, Y2), RED, 2)

                # 画预测（绿）
                p_idx = [i for i, l in enumerate(labels) if int(l) == cls]
                for i in p_idx:
                    b = list(map(int, boxes[i]))
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), GREEN, 2)
                    cv2.putText(vis, f"s={scores[i]:.2f}", (b[0], max(12, b[1] - 4)), font, 0.45, GREEN, 1, cv2.LINE_AA)

                # 匹配对连线（中心）
                def ctr(bb):
                    return (int((bb[0] + bb[2]) / 2), int((bb[1] + bb[3]) / 2))

                # 建 p->g 映射
                for (pi, gj, c) in m["matches"]:
                    if c != cls: continue
                    pb = boxes[pi];
                    gb = gts[gj] if gj < len(gts) else None
                    if gb is None: continue
                    cv2.line(vis, ctr(pb), ctr(gb), (255, 255, 255), 2)  # 白线

                # 未匹配预测（黄）
                for (c, i) in m["unmatch_pred"]:
                    if c != cls: continue
                    b = list(map(int, boxes[i]))
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), YEL, 2)

                # 未覆盖GT（紫）
                for (c, j) in m["miss_gt"]:
                    if c != cls: continue
                    if j < len(gts):
                        gb = list(map(int, gts[j]))
                        cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), PUR, 2)

                # 顶栏
                bar = np.zeros((26, vis.shape[1], 3), np.uint8)
                txt = f"img {img_id}  cls={cid2name.get(cls, str(cls))}  match={sum(1 for x in m['matches'] if x[2] == cls)}  " \
                      f"unmatch_pred={sum(1 for x in m['unmatch_pred'] if x[0] == cls)}  miss_gt={sum(1 for x in m['miss_gt'] if x[0] == cls)}"
                cv2.putText(bar, txt, (6, 18), font, 0.55, WHITE, 1, cv2.LINE_AA)

                out = np.vstack([bar, vis])
                out_p = out_dir / f"{img_id}_{cid2name.get(cls, str(cls))}.jpg"
                cv2.imwrite(str(out_p), out)
                wrote += 1
                if limit and wrote >= limit:
                    print(f"[viz-match] wrote {out_dir} (limited {limit})")
                    return out_dir

    print(f"[viz-match] wrote {out_dir}")
    return out_dir


class EdgeMLP(torch.nn.Module):
    def __init__(self, in_dim=4, hid=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1)
        )

    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)


# class LabelabilityMLP(torch.nn.Module):
#     def __init__(self, in_dim=5, hid=64):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
#             torch.nn.Linear(hid, hid), torch.nn.ReLU(),
#             torch.nn.Linear(hid, 1)
#         )
#
#     def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)
#

def _pair_feats(bi, bj):
    iou = _iou(bi, bj);
    iom = _iom(bi, bj);
    cd = _cdist(bi, bj)
    # 面积比/长宽比差（可选）
    si = (bi[2] - bi[0]) * (bi[3] - bi[1]);
    sj = (bj[2] - bj[0]) * (bj[3] - bj[1])
    ar_i = (bi[2] - bi[0]) / (bi[3] - bi[1] + 1e-8);
    ar_j = (bj[2] - bj[0]) / (bj[3] - bj[1] + 1e-8)
    return [iou, iom, cd, np.log((si + 1e-6) / (sj + 1e-6)), abs(ar_i - ar_j)]


def stage_build_trainsets(cfg, det_file: Path, match_file: Path) -> Tuple[Path, Path]:
    """
    基于匹配生成：pair-level 样本(edge_train.npz) & node-level 样本(node_train.npz)
    边标签：同一GT→1，其他→0； 点标签：被匹配→1，未匹配→0
    """
    import json
    edge_X, edge_y = [], []
    node_X, node_y = [], []

    # 读取 detections 映射
    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L);
            det_map[str(r["image_id"])] = r

    with open(match_file) as f:
        for L in f:
            m = json.loads(L)
            det = det_map.get(str(m["image_id"]));
            if det is None: continue
            boxes, labels, scores = det["boxes"], det["labels"], det["scores"]

            # 构造同一GT的预测索引集合：cls -> gt_j -> [pi...]
            same_gt = {}
            for (pi, gj, cls) in m["matches"]:
                same_gt.setdefault((cls, gj), []).append(pi)

            # 边：正样本（同一GT内的两两组合）
            for key, pis in same_gt.items():
                for a in range(len(pis)):
                    for b in range(a + 1, len(pis)):
                        edge_X.append(_pair_feats(boxes[pis[a]], boxes[pis[b]]))
                        edge_y.append(1.0)

            # 边：负样本（同类但不同GT/或未匹配）
            by_cls = {}
            for i, l in enumerate(labels): by_cls.setdefault(int(l), []).append(i)
            for cls, idxs in by_cls.items():
                # 简单负采样
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        ia, ib = idxs[a], idxs[b]
                        # 如果两者出现在同一GT正集中，则跳过（正已收）
                        is_pos = False
                        for key, pis in same_gt.items():
                            if key[0] == cls and (ia in pis and ib in pis): is_pos = True; break
                        if is_pos: continue
                        edge_X.append(_pair_feats(boxes[ia], boxes[ib]))
                        edge_y.append(0.0)

            # 点：可标注性（被匹配=1，否则0），加简单特征
            # 最近 GT 的 iou/iom/cdist 作为特征（没有GT时置0）
            # 这里用匹配文件里的 GT 覆盖信息近似
            matched_pis = set(pi for (pi, _, _) in m["matches"])
            for i, b in enumerate(boxes):
                lab = 1.0 if i in matched_pis else 0.0
                # 近邻密度（同类）
                same = [j for j, l in enumerate(labels) if l == labels[i] and j != i]
                dens = float(np.mean([_iou(b, boxes[j]) for j in same])) if same else 0.0
                node_X.append([dens, scores[i], b[2] - b[0], b[3] - b[1], (b[2] - b[0]) * (b[3] - b[1]) ** 0.5])
                node_y.append(lab)

    edge_p = cfg.paths.outputs_dir / "edge_train.npz"
    node_p = cfg.paths.outputs_dir / "node_train.npz"
    np.savez_compressed(edge_p, X=np.array(edge_X, np.float32), y=np.array(edge_y, np.float32))
    np.savez_compressed(node_p, X=np.array(node_X, np.float32), y=np.array(node_y, np.float32))
    print(f"[trainset] edge={len(edge_y)} node={len(node_y)}")
    return edge_p, node_p


def stage_train_edge(cfg, edge_npz: Path, epochs=5, bs=8192) -> Path:
    dat = np.load(edge_npz);
    X = torch.from_numpy(dat["X"]);
    y = torch.from_numpy(dat["y"])
    mdl = EdgeMLP(in_dim=X.shape[1]).to(cfg.device);
    opt = torch.optim.Adam(mdl.parameters(), 1e-3)
    crit = torch.nn.BCELoss()
    for ep in range(1, epochs + 1):
        idx = torch.randperm(len(y));
        loss = 0.0
        for st in range(0, len(y), bs):
            sel = idx[st:st + bs];
            xb = X[sel].to(cfg.device);
            yb = y[sel].to(cfg.device)
            p = mdl(xb);
            l = crit(p, yb);
            opt.zero_grad();
            l.backward();
            opt.step();
            loss += l.item() * len(sel)
        print(f"[edge] ep{ep} loss={loss / len(y):.4f}")
    path = cfg.paths.models_dir / "grm_edge_pair.pt";
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": mdl.state_dict(), "in_dim": X.shape[1]}, path)
    return path


import numpy as np, json, torch, math
from pathlib import Path
from sklearn.metrics import roc_auc_score


def _iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]);
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    A = lambda t: max(0, t[2] - t[0]) * max(0, t[3] - t[1])
    return inter / max(1e-8, A(a) + A(b) - inter)


def _center_dist_norm(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2;
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    s = max((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return d / (s ** 0.5 + 1e-8)


def _feat_one(idx, boxes, labels, scores, gts=None):
    """单框特征：自框+同类上下文+最近GT几何(训练期)+NMS结构"""
    import math, numpy as np
    b = boxes[idx];
    w = b[2] - b[0];
    h = b[3] - b[1];
    area = w * h
    cls = labels[idx]
    same = [j for j, l in enumerate(labels) if l == cls and j != idx]
    if same:
        ious = [_iou_xyxy(b, boxes[j]) for j in same]
        dists = [_center_dist_norm(b, boxes[j]) for j in same]
        mean_iou = float(np.mean(ious));
        max_iou = float(np.max(ious))
        min_dist = float(np.min(dists))
        neigh_cnt = int(sum(1 for j in same if _iou_xyxy(b, boxes[j]) > 0.1))
    else:
        mean_iou = max_iou = 0.0;
        min_dist = 1.0;
        neigh_cnt = 0
    aspect = w / (h + 1e-8)

    # 最近GT几何（训练期）——推理期 gts=None → 置0/1
    if gts:
        iou_max = max((_iou_xyxy(b, g) for g in gts), default=0.0)
        dist_min = min((_center_dist_norm(b, g) for g in gts), default=1.0)
    else:
        iou_max = 0.0;
        dist_min = 1.0

    # 类内NMS结构：rank & 被更高分遮挡
    same_cls = [(scores[j], j) for j, l in enumerate(labels) if l == cls]
    same_cls.sort(reverse=True)
    rank = next(r for r, (_, j) in enumerate(same_cls) if j == idx)
    covered = int(any(j != idx and _iou_xyxy(b, boxes[j]) > 0.5 and scores[j] > scores[idx] for _, j in same_cls))

    return [
        float(scores[idx]),  # 0: 原始置信度
        float(w), float(h),  # 1-2 尺寸
        float((area + 1e-8) ** 0.5),  # 3 尺度
        float(aspect),  # 4 长宽比
        float(neigh_cnt),  # 5 同类邻居数
        float(mean_iou), float(max_iou),  # 6-7 与同类IoU统计
        float(min_dist),  # 8 与最近同类中心距
        float(iou_max), float(dist_min),  # 9-10 与最近GT几何（训练期）
        float(rank), float(covered),  # 11-12 NMS结构
    ]


def stage_build_labelability_trainset(
        cfg,
        det_file: Path = None,
        save_npz: str = "labelability_train_semctx.npz",
        C_MAX: int = 10,
        NB_MAX: int = 8,
        use_area_weight: bool = True,
        knn_metric: str = "center"  # "center" or "iou"
) -> Path:
    """
    产出可标注性训练集（含语义上下文）：
    保存字段：
      X_geo: [N, Dg]         # 几何与全局几何（面积比、相对尺度、中心距等）
      y:     [N]             # 标签（是否与任一GT同类且IoU>=0.5）
      img_ids:[N]
      cat_id:[N]             # 当前框类别（COCO稀疏id）
      ctx_cats:[N, C_MAX]    # 图像级类别索引（不足-1）
      ctx_ws:[N, C_MAX]      # 对应权重（频次比例或面积占比）
      nb_cats:[N, NB_MAX]    # K近邻类别（不足-1）
      nb_ws:[N, NB_MAX]      # 近邻权重（按距离/IoU）
    """
    import json, numpy as np
    from pycocotools.coco import COCO

    if det_file is None:
        det_file = cfg.paths.detections_dir / "detections.jsonl"
    out = cfg.paths.outputs_dir / save_npz
    out.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(cfg.paths.coco_annotations))
    ann_by_img = {}
    for img_id in coco.getImgIds():
        ann_by_img[img_id] = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None))

    def area(b):
        return max(0., b[2] - b[0]) * max(0., b[3] - b[1])

    def iou(a, b):
        x1 = max(a[0], b[0]);
        y1 = max(a[1], b[1]);
        x2 = min(a[2], b[2]);
        y2 = min(a[3], b[3])
        inter = max(0., x2 - x1) * max(0., y2 - y1)
        return inter / max(1e-8, area(a) + area(b) - inter)

    def box_center(b):
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

    X_geo = [];
    y = [];
    img_ids = [];
    cat_id = []
    ctx_cats = [];
    ctx_ws = []
    nb_cats = [];
    nb_ws = []

    with open(det_file, "r") as fin:
        for line in fin:
            r = json.loads(line)
            img_id = int(r["image_id"])
            boxes = r.get("boxes", []) or []
            labels = r.get("labels", []) or []
            scores = r.get("scores", []) or []
            if len(boxes) == 0: continue

            # 图像全局统计
            W = coco.imgs[img_id]["width"];
            H = coco.imgs[img_id]["height"];
            Aimg = W * H
            areas = [area(b) for b in boxes]
            max_area = max(areas) if areas else 1.0

            # ctx（按预测分布或GT都可；这里用“预测分布”，与推理一致）
            # 统计类频次与总面积
            cnt = {}
            mass = {}
            for c, b in zip(labels, boxes):
                c = int(c);
                cnt[c] = cnt.get(c, 0) + 1;
                mass[c] = mass.get(c, 0.0) + area(b)
            # 取权重并归一化
            if use_area_weight:
                # 面积占比
                items = sorted(mass.items(), key=lambda kv: kv[1], reverse=True)[:C_MAX]
                total = sum(v for _, v in items) or 1.0
                cats_sel = [c for c, _ in items]
                ws_sel = [v / total for _, v in items]
            else:
                # 频次占比
                items = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:C_MAX]
                total = sum(v for _, v in items) or 1.0
                cats_sel = [c for c, _ in items]
                ws_sel = [v / total for _, v in items]
            # pad
            cats_sel += [-1] * (C_MAX - len(cats_sel))
            ws_sel += [0.0] * (C_MAX - len(ws_sel))

            # 每框处理
            centers = [box_center(b) for b in boxes]
            for i, (b, c, s) in enumerate(zip(boxes, labels, scores)):
                # y 标签（蒸馏：同类且IoU>=0.5 视为“会被标注”）
                pos = False
                for a in ann_by_img[img_id]:
                    if int(a["category_id"]) == int(c):
                        # GT bbox xywh -> xyxy
                        x, yh, w, h = a["bbox"]
                        g = [x, yh, x + w, yh + h]
                        if iou(b, g) >= 0.5:
                            pos = True;
                            break

                # 几何特征
                cx, cy = centers[i]
                ar = area(b) / max(1.0, Aimg)  # 占图面积比
                rel = area(b) / max(1.0, max_area)  # 相对最大框
                norm_cx, norm_cy = cx / W, cy / H  # 归一中心
                # 距离最近的K个框（用于邻域）
                dists = []
                for j, (bj) in enumerate(boxes):
                    if j == i: continue
                    if knn_metric == "center":
                        cx2, cy2 = centers[j]
                        d = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
                        dists.append((d, j))
                    else:
                        d = 1.0 - iou(b, bj)
                        dists.append((d, j))
                dists.sort(key=lambda x: x[0])
                neigh = [j for _, j in dists[:NB_MAX]]
                # 邻域类别与权重
                nb_c = [int(labels[j]) for j in neigh]
                if knn_metric == "center":
                    ww = []
                    for d, j in dists[:NB_MAX]:
                        wgt = 1.0 / max(1e-3, d / (max(W, H)))  # 近的权重大
                        ww.append(wgt)
                else:
                    ww = []
                    for d, j in dists[:NB_MAX]:
                        iou_ = 1.0 - d
                        ww.append(iou_)
                # 归一化
                ssum = sum(ww) or 1.0
                nb_w = [w / ssum for w in ww]
                # pad
                nb_c += [-1] * (NB_MAX - len(nb_c))
                nb_w += [0.0] * (NB_MAX - len(nb_w))

                # 组合几何（可加你已有的 Dg 特征：与最近同类IoU、密度、topk均值等）
                geo = [float(s), ar, rel, norm_cx, norm_cy]

                X_geo.append(geo)
                y.append(1.0 if pos else 0.0)
                img_ids.append(img_id)
                cat_id.append(int(c))
                ctx_cats.append(cats_sel)
                ctx_ws.append(ws_sel)
                nb_cats.append(nb_c)
                nb_ws.append(nb_w)

    X_geo = np.asarray(X_geo, np.float32)
    y = np.asarray(y, np.float32)
    img_ids = np.asarray(img_ids, np.int64)
    cat_id = np.asarray(cat_id, np.int64)
    ctx_cats = np.asarray(ctx_cats, np.int64)
    ctx_ws = np.asarray(ctx_ws, np.float32)
    nb_cats = np.asarray(nb_cats, np.int64)
    nb_ws = np.asarray(nb_ws, np.float32)

    np.savez(out, X_geo=X_geo, y=y, img_ids=img_ids, cat_id=cat_id,
             ctx_cats=ctx_cats, ctx_ws=ctx_ws, nb_cats=nb_cats, nb_ws=nb_ws)
    print(f"[labset] X={X_geo.shape} pos_rate={float(y.mean()):.3f} saved -> {out}")
    return out


class LabelabilityMLP(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1))

    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)


def stage_train_labelability(cfg, npz_path: Path, epochs=5, lr=1e-3, bs=8192,
                             emb_dim=64, num_classes=91):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    dat = np.load(npz_path)
    print(f"[lab] npz loaded: {npz_path}")
    print(f"[lab] keys: {list(dat.files)}")  # 调试关键：查看可用字段

    # === 读取与你保存一致的字段名 ===
    X_geo   = torch.tensor(dat["X_geo"],   dtype=torch.float32)
    y       = torch.tensor(dat["y"],       dtype=torch.float32)
    cat_id  = torch.tensor(dat["cat_id"],  dtype=torch.int64)
    ctx_ids = torch.tensor(dat["ctx_cats"],dtype=torch.int64)
    ctx_ws  = torch.tensor(dat["ctx_ws"],  dtype=torch.float32)
    nb_ids  = torch.tensor(dat["nb_cats"], dtype=torch.int64)
    nb_ws   = torch.tensor(dat["nb_ws"],   dtype=torch.float32)
    img_ids = torch.tensor(dat["img_ids"], dtype=torch.int64)

    N, Dg = X_geo.shape
    pos_rate = float(y.mean())
    print(f"[lab] X_geo={X_geo.shape}  pos_rate={pos_rate:.3f}  num_imgs={img_ids.unique().numel()}")

    # === 按 image_id 划分 train/val ===
    uniq = img_ids.unique(); perm = torch.randperm(len(uniq))
    n_val = max(1, int(0.1 * len(uniq)))
    val_set = set(uniq[perm[:n_val]].tolist())
    idx_va = [i for i,im in enumerate(img_ids.tolist()) if im in val_set]
    idx_tr = [i for i in range(N) if i not in val_set]

    def take(idxs):
        idx = torch.tensor(idxs, dtype=torch.long)
        return (X_geo[idx], y[idx], cat_id[idx], ctx_ids[idx], ctx_ws[idx], nb_ids[idx], nb_ws[idx])

    tr = take(idx_tr); va = take(idx_va)

    class DS(torch.utils.data.Dataset):
        def __init__(self, pack): self.pack = pack
        def __len__(self): return self.pack[0].shape[0]
        def __getitem__(self, i): return tuple(p[i] for p in self.pack)

    dl_tr = DataLoader(DS(tr), batch_size=bs, shuffle=True,  num_workers=0)
    dl_va = DataLoader(DS(va), batch_size=bs, shuffle=False, num_workers=0)

    # === 模型（与你前面语义版一致）===
    class LabelabilitySem(nn.Module):
        def __init__(self, Dg, K, d=64, hidden=(128,64)):
            super().__init__()
            self.emb = nn.Embedding(K, d)
            self.mlp = nn.Sequential(
                nn.Linear(Dg + d + d + d, hidden[0]), nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                nn.Linear(hidden[1], 1),
            )
        def agg(self, ids, ws):
            mask = (ids>=0).float().unsqueeze(-1)
            e = self.emb(torch.clamp(ids, min=0))
            return (e * (ws.unsqueeze(-1) * mask)).sum(1)
        def forward(self, geo, cid, ctx_i, ctx_w, nb_i, nb_w):
            e_node = self.emb(torch.clamp(cid, min=0))
            e_ctx  = self.agg(ctx_i, ctx_w)
            e_nb   = self.agg(nb_i, nb_w)
            x = torch.cat([geo, e_node, e_ctx, e_nb], dim=-1)
            return self.mlp(x).squeeze(-1)

    device = cfg.device
    mdl = LabelabilitySem(Dg, num_classes, d=emb_dim).to(device)
    pos_w = None
    if 0 < pos_rate < 1:
        pos_w = torch.tensor([(1-pos_rate)/max(pos_rate,1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w) if pos_w is not None else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)

    AMP = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    print(f"[lab] train={len(idx_tr)}  val={len(idx_va)}  in_dim_geo={Dg}  emb_dim={emb_dim}")

    def eval_auc():
        mdl.eval()
        import numpy as np
        all_p=[]; all_y=[]
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
            for geo,yy,cid,ci,cw,ni,nw in dl_va:
                geo=geo.to(device); yy=yy.to(device); cid=cid.to(device)
                ci=ci.to(device); cw=cw.to(device); ni=ni.to(device); nw=nw.to(device)
                logits = mdl(geo,cid,ci,cw,ni,nw).float().cpu().numpy()
                prob = 1.0/(1.0+np.exp(-np.clip(logits,-20,20)))
                all_p.append(prob); all_y.append(yy.cpu().numpy())
        all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
        if len(np.unique(all_y))<2: return 0.5
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(all_y, all_p))

    best_auc = -1.0
    for ep in range(1, epochs+1):
        mdl.train(); tot=0.0; n=0
        for bi,(geo,yy,cid,ci,cw,ni,nw) in enumerate(dl_tr):
            geo=geo.to(device); yy=yy.to(device); cid=cid.to(device)
            ci=ci.to(device); cw=cw.to(device); ni=ni.to(device); nw=nw.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=AMP):
                logits = mdl(geo,cid,ci,cw,ni,nw)
                loss = crit(logits, yy)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.item())*geo.size(0); n += geo.size(0)

            if bi % 50 == 0:
                print(f"[lab][ep{ep}] batch {bi}/{len(dl_tr)}  loss={loss.item():.4f}")

        auc_va = eval_auc()
        print(f"[lab] ep{ep} loss_tr={tot/max(1,n):.4f}  AUC_va={auc_va:.3f}")

        if auc_va > best_auc:
            best_auc = auc_va
            save_p = cfg.paths.models_dir / "grm_node_labelability.pt"
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": mdl.state_dict(),
                        "meta":{"in_dim_geo":Dg,"num_classes":num_classes,"emb_dim":emb_dim,
                                "auc_va":best_auc,"type":"LabelabilitySem"}}, save_p)
            print(f"[lab] checkpoint saved: {save_p}  (best_auc={best_auc:.3f})")

    return cfg.paths.models_dir / "grm_node_labelability.pt"

def _edge_score(mdl, a, b):
    x = np.array([_pair_feats(a, b)], np.float32)
    with torch.no_grad():
        return float(mdl(torch.from_numpy(x).to(next(mdl.parameters()).device))[0].cpu())


def _nms_xyxy(boxes, scores, thr=0.7):
    if not boxes: return []
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1))
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0];
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou < thr]
    return keep


def _logit_clip(p, eps=1e-6):
    p = float(max(min(p, 1.0 - eps), eps))
    return math.log(p / (1.0 - p))


def stage_infer_post(
        cfg,
        det_file: Path = None,
        node_mdl_p: Path = None,
        sub_iou: float = 0.65,
        temp: float = 2.0,
        save_name: str = "detections_post.jsonl"
) -> Path:
    """
    Labelability 后处理（不新增检测，只重排/降权）：
      1) 载入 labelability MLP（自动读取 ckpt.meta，或从 state_dict 反推结构）
      2) 对每个预测框提取几何/上下文特征，得到 p_label
      3) 分数重标定（只降不升）：s' = min( s, s * (p_label)^(1/temp) )
      4) 同类“包含抑制”（Subsumption）：若 j 分数更高且 IoU(i,j)≥sub_iou，则再把 i 的分数乘以 0.1

    输入:  detections.jsonl  (image_id, boxes[xyxy], labels[COCO稀疏], scores)
    输出:  /outputs/detections_post.jsonl （同字段，scores 已更新）
    注：不做硬删除，COCOeval 会按分数截断；这样更稳健。
    """
    import json, numpy as np, torch
    from pathlib import Path
    from pycocotools.coco import COCO

    # ---------- 路径 ----------
    if det_file is None:
        det_file = cfg.paths.detections_dir / "detections.jsonl"
    if node_mdl_p is None:
        node_mdl_p = cfg.paths.models_dir / "grm_node_labelability.pt"
    out = cfg.paths.outputs_dir / save_name
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 工具 ----------
    def _area(b):
        return max(0., b[2] - b[0]) * max(0., b[3] - b[1])

    def _iou(a, b):
        x1 = max(a[0], b[0]);
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]);
        y2 = min(a[3], b[3])
        inter = max(0., x2 - x1) * max(0., y2 - y1)
        return inter / max(1e-8, _area(a) + _area(b) - inter)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- 定义模型（需与训练一致） ----------
    import torch.nn as nn
    class LabelabilityMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: list[int] = [64, 64], act: str = "relu"):
            super().__init__()
            act_layer = nn.ReLU if act == "relu" else nn.GELU
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), act_layer(inplace=True)]
                last = h
            layers += [nn.Linear(last, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # x:[N,D]
            return self.net(x).squeeze(-1)  # logits:[N]

    # ---------- 读取 ckpt（优先 meta；否则从 state_dict 反推结构） ----------
    ckpt = torch.load(node_mdl_p, map_location=cfg.device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    meta = ckpt.get("meta", {})

    def rebuild_from_state_dict(sd):
        in_dim = int(sd["net.0.weight"].shape[1])
        hidden = []
        k = 2
        while f"net.{k}.weight" in sd:
            W = sd[f"net.{k}.weight"]
            out_dim = int(W.shape[0])
            if out_dim == 1:
                break
            hidden.append(out_dim)
            k += 2
        return in_dim, (hidden if hidden else [64])

    if meta and "in_dim" in meta:
        in_dim = int(meta["in_dim"])
        hidden = list(meta.get("hidden", [64, 64]))
    else:
        in_dim, hidden = rebuild_from_state_dict(sd)

    mdl = LabelabilityMLP(in_dim=in_dim, hidden=hidden).to(cfg.device).eval()
    mdl.load_state_dict(sd, strict=False)

    AMP = bool(cfg.device.startswith("cuda"))

    # ---------- COCO（仅用于尺寸/一致性） ----------
    coco = COCO(str(cfg.paths.coco_annotations))
    coco_imgs = coco.imgs  # dict[id] -> rec

    # ---------- 扫描 detections.jsonl 并写出 post 结果 ----------
    total_boxes = 0
    changed = 0
    subsumed_cnt = 0

    with open(det_file, "r") as fin, open(out, "w") as fout, torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
        for line in fin:
            r = json.loads(line)
            img_id = int(r["image_id"])
            boxes = r.get("boxes", []) or []
            labels = r.get("labels", []) or []
            scores = r.get("scores", []) or []
            n = len(boxes)
            total_boxes += n
            if n == 0 or img_id not in coco_imgs:
                fout.write(json.dumps(r) + "\n");
                continue

            # 1) labelability 打分
            #    _feat_one(i, boxes, labels, scores, gts=None) → np.ndarray[D]
            feats = []
            for i in range(n):
                feats.append(_feat_one(i, boxes, labels, scores, gts=None))
            X = torch.tensor(np.asarray(feats, np.float32), device=cfg.device)
            # logits = mdl(X).float().detach().cpu().numpy()

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
                logits = mdl(X)  # torch tensor
            p_lab = torch.sigmoid(logits).cpu().numpy()

            scores_new = []
            for s, pl in zip(scores, p_lab):
                s2 = float(s) * (float(pl) ** (1.0 / max(1e-6, temp)))
                s2 = min(float(s), s2)
                scores_new.append(s2)

            # 3) 同类包含抑制：若存在 j：同类、s_new[j] > s_new[i] 且 IoU≥sub_iou，则 i 乘以 0.1
            #    复杂度 O(n^2)；n 通常较小（已过 NMS），可接受
            if n > 1 and sub_iou > 0:
                by_cls = {}
                for i, c in enumerate(labels):
                    by_cls.setdefault(int(c), []).append(i)
                for c, idxs in by_cls.items():
                    if len(idxs) <= 1:
                        continue
                    # 只与同类比较
                    for a in idxs:
                        for b in idxs:
                            if b == a:
                                continue
                            if scores_new[b] <= scores_new[a]:
                                continue
                            if _iou(boxes[a], boxes[b]) >= sub_iou:
                                scores_new[a] *= 0.1
                                subsumed_cnt += 1
                                break  # a 已被更高分同类覆盖，跳出内层

            if any(abs(s2 - s1) > 1e-8 for s1, s2 in zip(scores, scores_new)):
                changed += 1

            # 写回
            rec = {"image_id": img_id, "boxes": boxes, "labels": labels, "scores": scores_new}
            fout.write(json.dumps(rec) + "\n")

    print(f"[infer_post] wrote {out}  boxes={total_boxes}  changed_imgs={changed}  "
          f"subsumed={subsumed_cnt}  sub_iou={sub_iou}  temp={temp}")
    return out


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


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))]


def stage_eval_post(cfg, det_file: Path = None, save_name: str = "eval_detections_post.json") -> dict:
    """
    评测 detection/post 结果（jsonl: {image_id, boxes[xyxy], labels[cat_id], scores}）
    - 自动按 det_file 名称选择 ann（文件名含“val”→ val2017，否则 train2017）
    - 若仍与 COCO 集不匹配，自动在 train/val 之间切换到匹配的那套 ann
    - 写出 /outputs/{save_name} 并返回 metrics 字典
    """
    import os, json, itertools, numpy as np
    from pathlib import Path
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # numpy 兼容（pycocotools 旧版使用 np.float）
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    # ---- 选择 detection 文件 ----
    if det_file is None:
        det_file = cfg.paths.outputs_dir / "detections_val.jsonl"
    det_file = Path(det_file)
    use_val_hint = ("val" in det_file.stem.lower()) or ("val" in det_file.name.lower())

    # ---- 可用的 ann 路径 ----
    ann_train = Path(cfg.paths.coco_annotations)
    ann_val_env = os.getenv("COCO_VAL_ANN", "")
    ann_val = Path(ann_val_env) if ann_val_env and Path(ann_val_env).exists() else None

    # 先按文件名 hint 选 ann
    ann_file = ann_val if (use_val_hint and ann_val is not None) else ann_train
    coco = COCO(str(ann_file))

    # ---- 先抽样读取 det 的 image_id，判断是否与当前 COCO 集匹配 ----
    def ids_match(sample_img_ids, coco_obj):
        ids = set(sample_img_ids)
        coco_ids = set(coco_obj.getImgIds())
        if not ids: return False
        hit = len(ids & coco_ids) / len(ids)
        return hit >= 0.9  # 90% 以上在集合内

    sample_img_ids = []
    with open(det_file, "r") as fin:
        for line in itertools.islice(fin, 5000):
            r = json.loads(line)
            sample_img_ids.append(int(r["image_id"]))

    # 若不匹配且另一套 ann 存在，则自动切换
    if not ids_match(sample_img_ids, coco) and ann_val is not None:
        alt = COCO(str(ann_val if ann_file == ann_train else ann_train))
        if ids_match(sample_img_ids, alt):
            coco = alt
            ann_file = ann_val if ann_file == ann_train else ann_train
            print(f"[eval_post] auto-switched ann to: {ann_file}")
        else:
            print("[eval_post][warn] sample image_ids match neither train nor val ann; proceeding with current ann.")

    # ---- 读取 det_file 并构造 COCO results ----
    def xyxy_to_xywh(b):
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    results = []
    with open(det_file, "r") as fin:
        for line in fin:
            r = json.loads(line)
            img_id = int(r["image_id"])
            boxes = r.get("boxes", []) or []
            labels = r.get("labels", []) or []
            scores = r.get("scores", []) or []
            for b, c, s in zip(boxes, labels, scores):
                results.append({
                    "image_id": img_id,
                    "category_id": int(c),  # COCO 稀疏 id
                    "bbox": xyxy_to_xywh(b),  # COCO 要求 xywh
                    "score": float(s),
                })

    if not results:
        zeros = {"AP": 0, "AP50": 0, "AP75": 0, "APs": 0, "APm": 0, "APl": 0,
                 "AR1": 0, "AR10": 0, "AR100": 0, "ARs": 0, "ARm": 0, "ARl": 0,
                 "evaluated_images": 0}
        out_p = cfg.paths.outputs_dir / save_name
        out_p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(zeros, open(out_p, "w"))
        print(f"[eval_post] wrote {out_p} (empty results)")
        return zeros

    coco_dt = coco.loadRes(results)

    # ---- 正式评测 ----
    E = COCOeval(coco, coco_dt, iouType="bbox")
    E.evaluate();
    E.accumulate();
    E.summarize()

    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    metrics = {k: float(v) for k, v in zip(keys, E.stats)}
    metrics["evaluated_images"] = int(len(coco.getImgIds()))

    out_p = cfg.paths.outputs_dir / save_name
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out_p, "w"))
    print(f"[eval_post] wrote {out_p}")
    return metrics


def stage_check_detections(cfg, det_file: Path = None, sample_k: int = 24, iou_thr: float = 0.5) -> Path:
    """
    检查 detections.jsonl 的正确性：
      - 自动判断坐标是否归一化并转换到像素
      - 检查 labels 是否为 0..79，需要映射到 COCO 稀疏 id(1..90)
      - 统计任意类/同类 IoU 对齐情况
      - 随机可视化若干张（红=预测，绿=GT）
    输出：
      - /outputs/check_detections_report.json
      - /outputs/viz_check/*.jpg
    """
    import json, random, numpy as np, math, os
    from PIL import Image, ImageDraw
    from collections import Counter, defaultdict
    from pycocotools.coco import COCO

    if det_file is None:
        det_file = cfg.paths.detections_dir / "detections.jsonl"

    out_dir = cfg.paths.outputs_dir / "viz_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_p = cfg.paths.outputs_dir / "check_detections_report.json"

    # ---- utils ----
    def _area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _iou(a, b):
        x1 = max(a[0], b[0]);
        y1 = max(a[1], b[1]);
        x2 = min(a[2], b[2]);
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / max(1e-8, _area(a) + _area(b) - inter)

    # torchvision 0..79 → COCO 稀疏 id
    COCO80_TO_91 = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]

    def map80to91(lbls):
        out = []
        for x in lbls:
            ix = int(x)
            if 0 <= ix < len(COCO80_TO_91):
                out.append(COCO80_TO_91[ix])
            else:
                out.append(-1)
        return out

    coco = COCO(str(cfg.paths.coco_annotations))
    COCO_IDS = sorted(coco.getImgIds())

    def _resolve_img_id(raw_id: int):
        # 1) 真实 COCO id
        if raw_id in coco.imgs:
            return raw_id, "direct"
        # 2) 作为顺序索引的回退映射（0..len-1）
        if 0 <= raw_id < len(COCO_IDS):
            return COCO_IDS[raw_id], "indexed"
        # 3) 失败
        return None, "missing"

    # ---- 首批采样行用于推断坐标/标签空间 ----
    first_lines = []
    with open(det_file, "r") as f:
        for _ in range(128):
            try:
                first_lines.append(json.loads(next(f)))
            except StopIteration:
                break
    if not first_lines:
        raise RuntimeError(f"[checkdet] empty file: {det_file}")

    # 坐标是否归一化
    mx_coord = 0.0
    for r in first_lines:
        for b in r.get("boxes", [])[:10]:
            mx_coord = max(mx_coord, *b)
    norm_boxes = (mx_coord <= 2.0)

    # label 空间检测
    label_counter = Counter()
    for r in first_lines:
        label_counter.update([int(c) for c in r.get("labels", [])])
    lbl_min = min(label_counter) if label_counter else 0
    lbl_max = max(label_counter) if label_counter else 0
    looks_like_80 = (0 <= lbl_min and lbl_max <= 79)

    # ---- 全量遍历统计 ----
    n_images = 0;
    n_images_in_ann = 0
    any_iou_hit = 0;
    same_iou_hit = 0
    pred_total = 0;
    gt_total = 0
    pos_pred_any = 0;
    pos_pred_same = 0

    # 为可视化随机挑 sample_k 张
    random.seed(123)
    sampled_ids = set()
    # 先收集所有 image_id
    all_img_ids = []
    with open(det_file, "r") as f:
        for L in f:
            r = json.loads(L)
            all_img_ids.append(int(r["image_id"]))
    if len(all_img_ids) <= sample_k:
        sampled_ids = set(all_img_ids)
    else:
        sampled_ids = set(random.sample(all_img_ids, sample_k))

    # 再遍历并统计 + 可视化
    with open(det_file, "r") as f:
        for L in f:
            r = json.loads(L)
            img_id = int(r["image_id"])
            boxes = r.get("boxes", [])
            labels = [int(c) for c in r.get("labels", [])]
            scores = r.get("scores", [])
            if not boxes:
                continue
            n_images += 1

            # 图像是否在 ann
            imgs = coco.loadImgs([img_id])
            if not imgs:
                continue
            n_images_in_ann += 1
            W, H = int(imgs[0]["width"]), int(imgs[0]["height"])

            # 归一化→像素坐标
            if norm_boxes:
                boxes = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes]

            # label 映射
            labels_sparse = map80to91(labels) if looks_like_80 else labels

            # GT 装载
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gt_total += len(anns)
            gts_all = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
            gt_by_cls = defaultdict(list)
            for a in anns:
                cid = int(a["category_id"])
                x0, y0, w, h = a["bbox"]
                gt_by_cls[cid].append([x0, y0, x0 + w, y0 + h])

            # IoU 统计
            pred_total += len(boxes)
            # 任意类 IoU≥thr 的图片计数
            hit_any = any(any(_iou(b, g) >= iou_thr for g in gts_all) for b in boxes[:200])
            if hit_any: any_iou_hit += 1

            # 同类 IoU≥thr 的图片计数 + 正样本计数
            hit_same = False
            pos_any_cnt = 0;
            pos_same_cnt = 0
            for i, b in enumerate(boxes):
                # any-class
                if any(_iou(b, g) >= iou_thr for g in gts_all): pos_any_cnt += 1
                # same-class（需要有效类 id）
                gts = gt_by_cls.get(labels_sparse[i], [])
                if gts and any(_iou(b, g) >= iou_thr for g in gts):
                    pos_same_cnt += 1;
                    hit_same = True
            if hit_same: same_iou_hit += 1
            pos_pred_any += pos_any_cnt
            pos_pred_same += pos_same_cnt

            # 可视化
            if img_id in sampled_ids:
                # load image
                img_path = os.path.join(cfg.paths.coco_images, imgs[0]["file_name"])
                try:
                    im = Image.open(img_path).convert("RGB")
                except Exception:
                    # 兜底：若路径不对，跳过该图
                    sampled_ids.remove(img_id)
                    continue
                draw = ImageDraw.Draw(im, "RGBA")
                # GT 绿
                for g in gts_all:
                    draw.rectangle([g[0], g[1], g[2], g[3]], outline=(0, 255, 0, 255), width=3)
                # 预测 红
                for i, b in enumerate(boxes):
                    col = (255, 0, 0, 255)
                    draw.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=2)
                im.save(out_dir / f"check_{img_id}.jpg")

    # ---- 汇总报告 ----
    report = {
        "detections_file": str(det_file),
        "annotations_file": str(cfg.paths.coco_annotations),
        "images_scanned": n_images,
        "images_in_annotations": n_images_in_ann,
        "coord_normalized_input": bool(norm_boxes),
        "label_space_contiguous80": bool(looks_like_80),
        "iou_threshold": iou_thr,
        "images_with_any_IoU>=thr": any_iou_hit,
        "images_with_sameClass_IoU>=thr": same_iou_hit,
        "ratio_any": (any_iou_hit / max(1, n_images_in_ann)),
        "ratio_same": (same_iou_hit / max(1, n_images_in_ann)),
        "pred_total": pred_total,
        "gt_total": gt_total,
        "pred_matched_any": pos_pred_any,
        "pred_matched_same": pos_pred_same,
        "pos_rate_pred_any": (pos_pred_any / max(1, pred_total)),
        "pos_rate_pred_same": (pos_pred_same / max(1, pred_total)),
        "visualizations": str(out_dir),
        "label_mapping_used": "COCO80→91" if looks_like_80 else "as-is",
        "note": "绿色=GT, 红色=预测；若 ratio_same≈0，通常是类ID未映射。若 ratio_any≈0，通常是坐标未转像素或split不一致。"
    }
    with open(report_p, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[checkdet] wrote {report_p}")
    print(f"[checkdet] any/same IoU ratios: {report['ratio_any']:.3f}/{report['ratio_same']:.3f} "
          f"  pos_rate_pred_any/same: {report['pos_rate_pred_any']:.3f}/{report['pos_rate_pred_same']:.3f}")
    return report_p


def stage_quick_sanity(cfg, det_file: Path) -> Path:
    import json, numpy as np
    from collections import defaultdict, Counter
    from pycocotools.coco import COCO

    rep = {
        "images_total": 0, "images_nonempty": 0,
        "ratio_any": 0.0, "ratio_same": 0.0,
        "pred_total": 0, "gt_total": 0,
        "pos_pred_any": 0, "pos_pred_same": 0,
        "coord_normalized_input": None,
        "label_space_contiguous80": None,
        "score_spearman_vs_iou": None
    }

    # 映射 0..79 → 稀疏 id（若需要）
    COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    def map80to91(lbls):
        out = [];
        ok = True
        for x in lbls:
            ix = int(x)
            if 0 <= ix < len(COCO80_TO_91):
                out.append(COCO80_TO_91[ix])
            else:
                out.append(-1);
                ok = False
        return out, ok

    def _area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _iou(a, b):
        x1 = max(a[0], b[0]);
        y1 = max(a[1], b[1]);
        x2 = min(a[2], b[2]);
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / max(1e-8, _area(a) + _area(b) - inter)

    coco = COCO(str(cfg.paths.coco_annotations))
    COCO_IDS = sorted(coco.getImgIds())

    def resolve_id(raw):
        if raw in coco.imgs: return raw
        if 0 <= raw < len(COCO_IDS): return COCO_IDS[raw]
        return None

    # 预判坐标/类别空间
    import itertools
    with open(det_file) as f:
        lines = list(itertools.islice(f, 128))
    import json as _json
    mx = 0.0;
    lbl_ctr = Counter()
    for L in lines:
        r = _json.loads(L)
        for b in r.get("boxes", [])[:10]: mx = max(mx, *b)
        lbl_ctr.update([int(x) for x in r.get("labels", [])])
    norm = (mx <= 2.0)
    rep["coord_normalized_input"] = norm
    looks80 = (lbl_ctr and min(lbl_ctr) >= 0 and max(lbl_ctr) <= 79)
    rep["label_space_contiguous80"] = looks80

    # 主循环统计
    from scipy.stats import spearmanr
    iou_list = [];
    score_list = []
    with open(det_file) as f:
        for L in f:
            r = _json.loads(L)
            raw = int(r["image_id"]);
            img_id = resolve_id(raw)
            if img_id is None: continue
            rep["images_total"] += 1
            boxes = r.get("boxes", []);
            labels = r.get("labels", []);
            scores = r.get("scores", [])
            if boxes: rep["images_nonempty"] += 1
            if not boxes: continue

            img = coco.loadImgs([img_id])[0]
            W, H = int(img["width"]), int(img["height"])
            if norm:
                boxes = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes]
            labels = map80to91(labels)[0] if looks80 else [int(c) for c in labels]

            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gts_all = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
            gt_by_cls = defaultdict(list)
            for a in anns:
                cid = int(a["category_id"]);
                x0, y0, w, h = a["bbox"]
                gt_by_cls[cid].append([x0, y0, x0 + w, y0 + h])

            rep["pred_total"] += len(boxes);
            rep["gt_total"] += len(anns)

            any_hit = False;
            same_hit = False
            for i, b in enumerate(boxes):
                iou_any = max([_iou(b, g) for g in gts_all], default=0.0)
                iou_list.append(iou_any);
                score_list.append(float(scores[i]))
                if iou_any >= 0.5:
                    rep["pos_pred_any"] += 1;
                    any_hit = True
                gts = gt_by_cls.get(labels[i], [])
                if any(_iou(b, g) >= 0.5 for g in gts):
                    rep["pos_pred_same"] += 1;
                    same_hit = True
            if any_hit:  rep["ratio_any"] += 1
            if same_hit: rep["ratio_same"] += 1

    rep["ratio_any"] = rep["ratio_any"] / max(1, rep["images_total"])
    rep["ratio_same"] = rep["ratio_same"] / max(1, rep["images_total"])
    rep["images_nonempty_ratio"] = rep["images_nonempty"] / max(1, rep["images_total"])
    if len(iou_list) > 100 and len(set(score_list)) > 1:
        rep["score_spearman_vs_iou"] = float(spearmanr(iou_list, score_list).correlation)

    out = cfg.paths.outputs_dir / "quick_sanity_report.json"
    import json as _j;
    with open(out, "w") as f:
        _j.dump(rep, f, indent=2)
    print("[sanity]", rep)
    return out


def stage_quick_eval(cfg, det_file: Path, iouType="bbox"):
    import json, numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    def xyxy_to_xywh(b):
        return [float(b[0]), float(b[1]), float(max(0, b[2] - b[0])), float(max(0, b[3] - b[1]))]

    # numpy shim for old cocoeval
    if not hasattr(np, "float"): np.float = float

    # 读 detections
    with open(det_file) as f:
        lines = [json.loads(L) for L in f]
    # 坐标/label 判定（复用 sanity 的逻辑）
    mx = max([max(b) for r in lines for b in r.get("boxes", [])] + [0])
    norm = (mx <= 2.0)
    COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    looks80 = True
    for r in lines[:64]:
        if r.get("labels"):
            looks80 = looks80 and (0 <= min(r["labels"]) and max(r["labels"]) <= 79)

    coco = COCO(str(cfg.paths.coco_annotations))
    COCO_IDS = sorted(coco.getImgIds())

    def resolve_id(raw):
        if raw in coco.imgs: return raw
        if 0 <= raw < len(COCO_IDS): return COCO_IDS[raw]
        return None

    # 转 COCO result
    results = []
    for r in lines:
        img_id = resolve_id(int(r["image_id"]))
        if img_id is None: continue
        W, H = int(coco.imgs[img_id]["width"]), int(coco.imgs[img_id]["height"])
        boxes = r.get("boxes", []);
        labels = r.get("labels", []);
        scores = r.get("scores", [])
        if norm:
            boxes = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes]
        labels = ([COCO80_TO_91[int(c)] for c in labels] if looks80 else [int(c) for c in labels])
        for b, c, s in zip(boxes, labels, scores):
            results.append({"image_id": img_id, "category_id": c, "bbox": xyxy_to_xywh(b), "score": float(s)})

    cocoDt = coco.loadRes(results)
    E = COCOeval(coco, cocoDt, iouType)
    E.params.imgIds = sorted({r["image_id"] for r in results})
    E.evaluate();
    E.accumulate();
    E.summarize()

    summary = {"AP": float(E.stats[0]), "AP50": float(E.stats[1]), "AP75": float(E.stats[2])}
    out = cfg.paths.outputs_dir / "quick_eval_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("[quick_eval]", summary)
    return out


def stage_grid(
        cfg,
        det_file: Path = None,
        node_mdl: Path = None,
        cand_sub=(0.45, 0.50, 0.55, 0.60),
        cand_temp=(1.5, 2.0, 2.5, 3.0),
        ar_drop_pp_max: float = 0.5,
        save_csv: str = "grid_results.csv"
) -> Path:
    """
    网格搜索后处理参数以最大化 mAP (AP@[.50:.95])，并约束 AR100 下降 ≤ ar_drop_pp_max。
    结果保存到 /outputs/grid_results.csv，返回该 CSV 路径。
    """
    import csv, json
    from pathlib import Path

    if det_file is None:
        det_file = cfg.paths.detections_dir / "detections.jsonl"
    if node_mdl is None:
        node_mdl = cfg.paths.models_dir / "grm_node_labelability.pt"

    out_csv = cfg.paths.outputs_dir / save_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # —— 评测 baseline（未后处理）——
    print("[grid] evaluating baseline …")
    base_metrics = stage_eval_post(cfg, det_file)  # 直接复用同一个 eval 入口
    base_AP = float(base_metrics.get("AP", 0.0))
    base_AP75 = float(base_metrics.get("AP75", 0.0))
    base_AR100 = float(base_metrics.get("AR100", 0.0))
    print(f"[grid] baseline  AP={base_AP:.3f}  AP75={base_AP75:.3f}  AR100={base_AR100:.3f}")

    # —— 网格搜索 ——
    rows = []
    best = None  # (AP, AP75), record
    for si in cand_sub:
        for tp in cand_temp:
            save_name = f"detections_post_s{si:.2f}_t{tp:.2f}.jsonl"
            post_path = stage_infer_post(cfg, det_file, node_mdl, sub_iou=float(si), temp=float(tp), save_name=save_name)
            metrics = stage_eval_post(cfg, post_path)

            rec = {
                "sub_iou": float(si),
                "temp": float(tp),
                **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            # 增加相对变化
            rec["dAP"] = rec.get("AP", 0.0) - base_AP
            rec["dAP50"] = rec.get("AP50", 0.0) - base_metrics.get("AP50", 0.0)
            rec["dAP75"] = rec.get("AP75", 0.0) - base_AP75
            rec["dAR100"] = rec.get("AR100", 0.0) - base_AR100
            rows.append(rec)

            # 约束：AR100 不得下降超过 ar_drop_pp_max
            if rec["dAR100"] < -ar_drop_pp_max / 100.0:
                print(f"[grid] skip (AR drop too big): sub_iou={si} temp={tp}  AR100={rec['AR100']:.3f}")
                continue

            key = (rec.get("AP", 0.0), rec.get("AP75", 0.0))
            if best is None or key > best[0]:
                best = (key, rec)

            print(f"[grid] tried sub_iou={si:.2f} temp={tp:.2f}  →  AP={rec['AP']:.3f} (Δ{rec['dAP']:+.3f})  "
                  f"AP75={rec['AP75']:.3f}  AR100={rec['AR100']:.3f} (Δ{rec['dAR100']:+.3f})")

    # —— 写 CSV ——
    if rows:
        # 确定列顺序
        cols = ["sub_iou", "temp", "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl", "dAP", "dAP50", "dAP75", "dAR100"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"[grid] wrote {out_csv}  rows={len(rows)}")
    else:
        print("[grid] no rows written (all filtered?)")

    # —— 打印最优 ——
    if best:
        b = best[1]
        print(f"[grid][BEST] sub_iou={b['sub_iou']:.2f}  temp={b['temp']:.2f}  "
              f"AP={b['AP']:.3f} (Δ{b['dAP']:+.3f})  AP75={b['AP75']:.3f}  "
              f"AR100={b['AR100']:.3f} (Δ{b['dAR100']:+.3f})")
    else:
        print("[grid] no feasible best under AR constraint")

    return out_csv


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser("GRM-COCO pipeline")
    parser.add_argument("--steps", type=str, default="detect,graph,train,infer,groupnms,eval")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--t_intra", type=float, default=0.7)
    parser.add_argument("--t_inter", type=float, default=0.5)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--sub_iou", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=2.0)
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
        artifacts["det"] = stage_detect(cfg, split="val")
    if "checkdet" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["sanity"] = stage_quick_sanity(cfg, det)
    if "evaldet" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["quick_eval"] = stage_quick_eval(cfg, det)

    if "viz" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["viz"] = stage_viz_frcnn_vs_gt(cfg, det,
                                                 iou_thr=0.5,  # 可改
                                                 score_thr=0.05,  # 可改
                                                 limit=200)  # 可改
    if "match" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["match"] = stage_match_to_gt_class(cfg, det)

    if "viz_match" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        mat = artifacts.get("match", cfg.paths.outputs_dir / "matches.jsonl")
        artifacts["viz_match"] = stage_viz_match(cfg, det, mat, limit=200)

    # --- 新管线: labelability ---
    if "labset" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["labset"] = stage_build_labelability_trainset(cfg, det)

    if "train_label" in steps:
        labset = artifacts.get("labset", cfg.paths.outputs_dir / "labelability_train_semctx.npz")
        artifacts["node_mdl"] = stage_train_labelability(cfg, labset)

    if "grid" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        node = artifacts.get("node_mdl", cfg.paths.models_dir / "grm_node_labelability.pt")
        artifacts["grid"] = stage_grid(cfg, det_file=det, node_mdl=node)

    if "infer_post" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        node_m = artifacts.get("lab_mdl", cfg.paths.models_dir / "grm_node_labelability.pt")
        # 可用命令行或环境变量传 sub_iou / temp；这里给默认
        artifacts["det_post"] = stage_infer_post(cfg, det, node_m, sub_iou=0.50, temp=2.0)

    if "eval_post" in steps:
        post = artifacts.get("det_post", cfg.paths.outputs_dir / "detections_grm_post.jsonl")
        print("[eval_post]", stage_eval_post(cfg, post))

    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
