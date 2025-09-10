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


def stage_build_labelability_trainset(cfg, det_file: Path, match_file: Path) -> Path:
    out = cfg.paths.outputs_dir / "labelability_train.npz"
    if out.exists(): print(f"[labset] reuse {out}"); return out

    import json
    from pycocotools.coco import COCO
    coco = COCO(str(cfg.paths.coco_annotations))

    # detections 映射
    det_map = {}
    with open(det_file) as f:
        for L in f:
            r = json.loads(L);
            det_map[str(r["image_id"])] = r

    X = []
    y = []
    img_ids = []
    with open(match_file) as f:
        for L in f:
            m = json.loads(L)
            img_id = m["image_id"]
            det = det_map.get(str(img_id));
            if not det: continue
            boxes, labels, scores = det["boxes"], det["labels"], det["scores"]
            if not boxes: continue

            # 同图同类 GT 集合
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            gt_by_cls = {}
            for a in anns:
                cid = int(a["category_id"])
                x0, y0, w, h = a["bbox"];
                gt_by_cls.setdefault(cid, []).append([x0, y0, x0 + w, y0 + h])

            matched = set(pi for (pi, _, _) in m["matches"])
            for i in range(len(boxes)):
                cls = int(labels[i])
                gts = gt_by_cls.get(cls, [])
                feat = _feat_one(i, boxes, labels, scores, gts=gts)  # 传入GT集合
                X.append(feat)
                y.append(1.0 if i in matched else 0.0)
                img_ids.append(int(img_id))

    X = np.asarray(X, np.float32);
    y = np.asarray(y, np.float32);
    img_ids = np.asarray(img_ids, np.int64)
    np.savez_compressed(out, X=X, y=y, img_ids=img_ids)
    print(f"[labset] X={X.shape} pos_rate={float(y.mean()):.3f}")
    return out


class LabelabilityMLP(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1))

    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)


def stage_train_labelability(cfg, npz_path: Path, epochs=5, lr=1e-3, bs=8192) -> Path:
    dat = np.load(npz_path)
    X, y, img_ids = dat["X"], dat["y"], dat["img_ids"]

    # 按图划分 train/val
    uniq = np.unique(img_ids)
    rng = np.random.default_rng(123)
    rng.shuffle(uniq)
    cut = int(0.8 * len(uniq))
    train_ids = set(uniq[:cut]);
    val_ids = set(uniq[cut:])
    tr_mask = np.array([i in train_ids for i in img_ids])
    va_mask = np.array([i in val_ids for i in img_ids])

    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xva, yva = X[va_mask], y[va_mask]

    # z-score
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    mdl = LabelabilityMLP(in_dim=X.shape[1]).to(cfg.device)
    opt = torch.optim.Adam(mdl.parameters(), lr)
    bce = torch.nn.BCELoss()

    Xtr_t = torch.from_numpy(Xtr).to(cfg.device)
    ytr_t = torch.from_numpy(ytr).to(cfg.device)
    Xva_t = torch.from_numpy(Xva).to(cfg.device)
    yva_t = torch.from_numpy(yva).to(cfg.device)

    for ep in range(1, epochs + 1):
        mdl.train()
        idx = torch.randperm(len(ytr_t))
        total = 0.0
        for st in range(0, len(idx), bs):
            sel = idx[st:st + bs]
            p = mdl(Xtr_t[sel])
            l = bce(p, ytr_t[sel])
            opt.zero_grad();
            l.backward();
            opt.step()
            total += l.item() * len(sel)

        mdl.eval()
        with torch.no_grad():
            pva = mdl(Xva_t).detach().cpu().numpy()
            auc = roc_auc_score(yva, pva) if len(np.unique(yva)) > 1 else float("nan")
        print(f"[lab] ep{ep} loss_tr={total / len(ytr_t):.4f}  AUC_va={auc:.3f}")

    # 保存模型与标准化参数
    out = cfg.paths.models_dir / "grm_node_labelability.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": mdl.state_dict(), "in_dim": X.shape[1], "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}, out)
    print(f"[lab] saved {out}")
    return out


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


def stage_infer_post(cfg,
                     det_file: Path,
                     node_mdl_p: Path = None,
                     sub_iou: float = 0.7,
                     temp: float = 1.0) -> Path:
    """
    无GT推理风格化：
      - 载入 labelability 模型（含 mu/sd）
      - 特征：与训练一致的 _feat_one(..., gts=None)
      - 温度缩放(可选)：q' = sigmoid(logit(q)/T)
      - 重评分 s' = s * q'
      - 类内 NMS(sub_iou)
    """
    import json, torch, numpy as np, math

    if node_mdl_p is None:
        node_mdl_p = cfg.paths.models_dir / "grm_node_labelability.pt"
    ckpt = torch.load(node_mdl_p, map_location=cfg.device, weights_only=False)
    in_dim = int(ckpt["in_dim"])
    mu, sd = ckpt["mu"], ckpt["sd"]  # [1,F]
    mdl = LabelabilityMLP(in_dim=in_dim).to(cfg.device).eval()
    mdl.load_state_dict(ckpt["state_dict"])

    def predict_q(feat_np):
        # feat_np: [N,F] np.float32
        x = (feat_np - mu) / sd
        with torch.no_grad():
            q = mdl(torch.from_numpy(x).to(cfg.device)).cpu().numpy().reshape(-1)
        if temp and abs(temp - 1.0) > 1e-6:
            # 温度缩放在 logit 空间
            q = np.array([1.0 / (1.0 + math.exp(_logit_clip(qq) / float(temp) * -1.0)) for qq in q], dtype=np.float32)
        return q.astype(np.float32)

    out = cfg.paths.outputs_dir / "detections_grm_post.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    num_imgs = 0
    with open(det_file) as fin, open(out, "w") as fout:
        for L in fin:
            r = json.loads(L)
            boxes, labels, scores = r["boxes"], r["labels"], r["scores"]
            if not boxes:
                fout.write(L);
                continue

            # 逐类：重评分 + NMS
            new_boxes = [];
            new_labels = [];
            new_scores = []
            classes = sorted(set(int(c) for c in labels))
            for cls in classes:
                idx = [i for i, l in enumerate(labels) if int(l) == cls]
                if not idx: continue
                # 计算 q(b)
                feats = np.stack([np.array(_feat_one(i, boxes, labels, scores, gts=None), np.float32) for i in idx], axis=0)
                q = predict_q(feats)
                s_prime = (np.array([scores[i] for i in idx], np.float32) * q).tolist()

                # NMS（对 s' 排序）
                B = [boxes[i] for i in idx]
                keep = _nms_xyxy(B, s_prime, thr=sub_iou)
                for k in keep:
                    new_boxes.append(B[k])
                    new_labels.append(cls)
                    new_scores.append(float(s_prime[k]))

            fout.write(json.dumps({
                "image_id": r["image_id"],
                "boxes": new_boxes,
                "labels": new_labels,
                "scores": new_scores
            }) + "\n")
            num_imgs += 1

    print(f"[post] wrote {out}  imgs={num_imgs}  sub_iou={sub_iou}  temp={temp}")
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
    if "match" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        artifacts["match"] = stage_match_to_gt_class(cfg, det)

    if "viz_match" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        mat = artifacts.get("match", cfg.paths.outputs_dir / "matches.jsonl")
        artifacts["viz_match"] = stage_viz_match(cfg, det, mat, limit=200)
    if "labset" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        mat = artifacts.get("match", cfg.paths.outputs_dir / "matches.jsonl")
        artifacts["labset"] = stage_build_labelability_trainset(cfg, det, mat)

    if "train_label" in steps:
        npz = artifacts.get("labset", cfg.paths.outputs_dir / "labelability_train.npz")
        artifacts["lab_mdl"] = stage_train_labelability(cfg, npz)
    if "infer_post" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        node_m = artifacts.get("lab_mdl", cfg.paths.models_dir / "grm_node_labelability.pt")
        # 可用命令行或环境变量传 sub_iou / temp；这里给默认
        artifacts["det_post"] = stage_infer_post(cfg, det, node_m, sub_iou=0.7, temp=1.0)

    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
