# Created by MacBook Pro at 07.09.25


# pipeline.py
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from config import load_config
import torch


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


# ---- put into run_coco.py ----


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



import numpy as np, json, torch, math
from pathlib import Path


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



def stage_train_labelability(cfg, npz_path: Path, epochs=5, lr=1e-3, bs=8192,
                             emb_dim=64, num_classes=91):
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    dat = np.load(npz_path)
    print(f"[lab] npz loaded: {npz_path}")
    print(f"[lab] keys: {list(dat.files)}")  # 调试关键：查看可用字段

    # === 读取与你保存一致的字段名 ===
    X_geo = torch.tensor(dat["X_geo"], dtype=torch.float32)
    y = torch.tensor(dat["y"], dtype=torch.float32)
    cat_id = torch.tensor(dat["cat_id"], dtype=torch.int64)
    ctx_ids = torch.tensor(dat["ctx_cats"], dtype=torch.int64)
    ctx_ws = torch.tensor(dat["ctx_ws"], dtype=torch.float32)
    nb_ids = torch.tensor(dat["nb_cats"], dtype=torch.int64)
    nb_ws = torch.tensor(dat["nb_ws"], dtype=torch.float32)
    img_ids = torch.tensor(dat["img_ids"], dtype=torch.int64)

    N, Dg = X_geo.shape
    pos_rate = float(y.mean())
    print(f"[lab] X_geo={X_geo.shape}  pos_rate={pos_rate:.3f}  num_imgs={img_ids.unique().numel()}")

    # === 按 image_id 划分 train/val ===
    uniq = img_ids.unique();
    perm = torch.randperm(len(uniq))
    n_val = max(1, int(0.1 * len(uniq)))
    val_set = set(uniq[perm[:n_val]].tolist())
    idx_va = [i for i, im in enumerate(img_ids.tolist()) if im in val_set]
    idx_tr = [i for i in range(N) if i not in val_set]

    def take(idxs):
        idx = torch.tensor(idxs, dtype=torch.long)
        return (X_geo[idx], y[idx], cat_id[idx], ctx_ids[idx], ctx_ws[idx], nb_ids[idx], nb_ws[idx])

    tr = take(idx_tr);
    va = take(idx_va)

    class DS(torch.utils.data.Dataset):
        def __init__(self, pack): self.pack = pack

        def __len__(self): return self.pack[0].shape[0]

        def __getitem__(self, i): return tuple(p[i] for p in self.pack)

    dl_tr = DataLoader(DS(tr), batch_size=bs, shuffle=True, num_workers=0)
    dl_va = DataLoader(DS(va), batch_size=bs, shuffle=False, num_workers=0)

    # === 模型（与你前面语义版一致）===
    class LabelabilitySem(nn.Module):
        def __init__(self, Dg, K, d=64, hidden=(128, 64)):
            super().__init__()
            self.emb = nn.Embedding(K, d)
            self.mlp = nn.Sequential(
                nn.Linear(Dg + d + d + d, hidden[0]), nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                nn.Linear(hidden[1], 1),
            )

        def agg(self, ids, ws):
            mask = (ids >= 0).float().unsqueeze(-1)
            e = self.emb(torch.clamp(ids, min=0))
            return (e * (ws.unsqueeze(-1) * mask)).sum(1)

        def forward(self, geo, cid, ctx_i, ctx_w, nb_i, nb_w):
            e_node = self.emb(torch.clamp(cid, min=0))
            e_ctx = self.agg(ctx_i, ctx_w)
            e_nb = self.agg(nb_i, nb_w)
            x = torch.cat([geo, e_node, e_ctx, e_nb], dim=-1)
            return self.mlp(x).squeeze(-1)

    device = cfg.device
    mdl = LabelabilitySem(Dg, num_classes, d=emb_dim).to(device)
    pos_w = None
    if 0 < pos_rate < 1:
        pos_w = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w) if pos_w is not None else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4)

    AMP = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    print(f"[lab] train={len(idx_tr)}  val={len(idx_va)}  in_dim_geo={Dg}  emb_dim={emb_dim}")

    def eval_auc():
        mdl.eval()
        import numpy as np
        all_p = [];
        all_y = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
            for geo, yy, cid, ci, cw, ni, nw in dl_va:
                geo = geo.to(device);
                yy = yy.to(device);
                cid = cid.to(device)
                ci = ci.to(device);
                cw = cw.to(device);
                ni = ni.to(device);
                nw = nw.to(device)
                logits = mdl(geo, cid, ci, cw, ni, nw).float().cpu().numpy()
                prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
                all_p.append(prob);
                all_y.append(yy.cpu().numpy())
        all_p = np.concatenate(all_p);
        all_y = np.concatenate(all_y)
        if len(np.unique(all_y)) < 2: return 0.5
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(all_y, all_p))

    best_auc = -1.0
    for ep in range(1, epochs + 1):
        mdl.train();
        tot = 0.0;
        n = 0
        for bi, (geo, yy, cid, ci, cw, ni, nw) in enumerate(dl_tr):
            geo = geo.to(device);
            yy = yy.to(device);
            cid = cid.to(device)
            ci = ci.to(device);
            cw = cw.to(device);
            ni = ni.to(device);
            nw = nw.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=AMP):
                logits = mdl(geo, cid, ci, cw, ni, nw)
                loss = crit(logits, yy)
            scaler.scale(loss).backward();
            scaler.step(opt);
            scaler.update()
            tot += float(loss.item()) * geo.size(0);
            n += geo.size(0)

            if bi % 50 == 0:
                print(f"[lab][ep{ep}] batch {bi}/{len(dl_tr)}  loss={loss.item():.4f}")

        auc_va = eval_auc()
        print(f"[lab] ep{ep} loss_tr={tot / max(1, n):.4f}  AUC_va={auc_va:.3f}")

        if auc_va > best_auc:
            best_auc = auc_va
            save_p = cfg.paths.models_dir / "grm_node_labelability.pt"
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": mdl.state_dict(),
                        "meta": {"in_dim_geo": Dg, "num_classes": num_classes, "emb_dim": emb_dim,
                                 "auc_va": best_auc, "type": "LabelabilitySem"}}, save_p)
            print(f"[lab] checkpoint saved: {save_p}  (best_auc={best_auc:.3f})")

    return cfg.paths.models_dir / "grm_node_labelability.pt"


# ===================== helpers =====================

import json, math, numpy as np, torch, torch.nn as nn
from pathlib import Path



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


# ===== 快速校准检查（val）=====
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def diag_labelability_calibration(cfg, npz_path: Path, mdl_path: Path,
                                  out_dir_name="lab_viz", bins=15):
    """
    在 labelability 验证集上做校准检查：Reliability、ECE、Brier，并搜索温度T。
    要求：stage_train_labelability 用 image_id 划分的同一份 npz。
    """
    import numpy as np, torch
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.calibration import calibration_curve
    out_dir = (cfg.paths.outputs_dir / out_dir_name);
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 载入数据 ===
    dat = np.load(npz_path)
    X_geo = torch.tensor(dat["X_geo"], dtype=torch.float32)
    y = torch.tensor(dat["y"], dtype=torch.float32)
    img_ids = torch.tensor(dat["img_ids"], dtype=torch.int64)
    cat_id = torch.tensor(dat["cat_id"], dtype=torch.int64)
    ctx_i = torch.tensor(dat["ctx_cats"], dtype=torch.int64)
    ctx_w = torch.tensor(dat["ctx_ws"], dtype=torch.float32)
    nb_i = torch.tensor(dat["nb_cats"], dtype=torch.int64)
    nb_w = torch.tensor(dat["nb_ws"], dtype=torch.float32)

    # 用与训练一致的 val 划分（10%）
    uniq = img_ids.unique();
    perm = torch.randperm(len(uniq))
    n_val = max(1, int(0.1 * len(uniq)))
    val_set = set(uniq[perm[:n_val]].tolist())
    idx_va = torch.tensor([i for i, im in enumerate(img_ids.tolist()) if im in val_set], dtype=torch.long)

    X_geo = X_geo[idx_va];
    y = y[idx_va];
    cat_id = cat_id[idx_va]
    ctx_i = ctx_i[idx_va];
    ctx_w = ctx_w[idx_va];
    nb_i = nb_i[idx_va];
    nb_w = nb_w[idx_va]

    # === 载入模型 ===
    import torch.nn as nn
    ckpt = torch.load(mdl_path, map_location=cfg.device)
    meta = ckpt.get("meta", {})
    Dg = int(meta.get("in_dim_geo", X_geo.shape[1]))
    K = int(meta.get("num_classes", 91))
    d = int(meta.get("emb_dim", 64))

    class LabelabilitySem(nn.Module):
        def __init__(self, Dg, K, d=64, hidden=(128, 64)):
            super().__init__()
            self.emb = nn.Embedding(K, d)
            self.mlp = nn.Sequential(
                nn.Linear(Dg + d + d + d, hidden[0]), nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                nn.Linear(hidden[1], 1),
            )

        def agg(self, ids, ws):
            mask = (ids >= 0).float().unsqueeze(-1)
            e = self.emb(torch.clamp(ids, min=0))
            return (e * (ws.unsqueeze(-1) * mask)).sum(1)

        def forward(self, geo, cid, ctx_i, ctx_w, nb_i, nb_w):
            e_node = self.emb(torch.clamp(cid, min=0))
            e_ctx = self.agg(ctx_i, ctx_w)
            e_nb = self.agg(nb_i, nb_w)
            x = torch.cat([geo, e_node, e_ctx, e_nb], dim=-1)
            return self.mlp(x).squeeze(-1)

    mdl = LabelabilitySem(Dg, K, d).to(cfg.device)
    mdl.load_state_dict(ckpt["state_dict"]);
    mdl.eval()

    # === 前向，得到 logits/prob ===
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=cfg.device.startswith("cuda")):
        logits = mdl(X_geo.to(cfg.device), cat_id.to(cfg.device),
                     ctx_i.to(cfg.device), ctx_w.to(cfg.device),
                     nb_i.to(cfg.device), nb_w.to(cfg.device)).float().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    y_np = y.numpy()

    # === 可靠性图 & ECE ===
    frac_pos, mean_pred = calibration_curve(y_np, prob, n_bins=bins, strategy="quantile")
    # ECE（等频分箱）
    ece = float(np.sum(np.abs(frac_pos - mean_pred) * (len(y_np) / bins) / len(y_np)))

    bs = float(brier_score_loss(y_np, prob))
    nll = float(log_loss(y_np, np.clip(prob, 1e-6, 1 - 1e-6)))

    # plot
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.xlabel("Predicted prob");
    plt.ylabel("Empirical prob")
    plt.title(f"Reliability (ECE={ece:.3f}, Brier={bs:.3f})")
    (out_dir / "reliability.png").unlink(missing_ok=True)
    plt.tight_layout();
    plt.savefig(out_dir / "reliability.png");
    plt.close()

    # === 搜索温度 T（最小化 NLL） ===
    # 快速：在对数域网格搜索
    Ts = np.exp(np.linspace(np.log(0.5), np.log(5.0), 25))

    def nll_T(T):
        p = 1 / (1 + np.exp(-np.clip(logits / float(T), -20, 20)))
        return log_loss(y_np, np.clip(p, 1e-6, 1 - 1e-6))

    nlls = np.array([nll_T(t) for t in Ts])
    T_best = float(Ts[np.argmin(nlls)])

    # 画温度-损失曲线
    plt.figure()
    plt.plot(Ts, nlls);
    plt.axvline(T_best, ls="--")
    plt.xscale("log");
    plt.xlabel("Temperature T");
    plt.ylabel("NLL (val)")
    plt.title(f"Temp scaling: T*={T_best:.2f}")
    (out_dir / "temp_search.png").unlink(missing_ok=True)
    plt.tight_layout();
    plt.savefig(out_dir / "temp_search.png");
    plt.close()

    # 记录
    summary = {"ECE": ece, "Brier": bs, "NLL": nll, "T_star": T_best, "bins": bins,
               "n_val": int(len(y_np))}
    with open(out_dir / "calib_summary.json", "w") as f:
        import json;
        json.dump(summary, f, indent=2)
    print(f"[calib] ECE={ece:.3f}  Brier={bs:.3f}  NLL={nll:.3f}  T*={T_best:.2f}  → {out_dir}")
    return T_best

# =================== infer_post: helpers ===================

import json, numpy as np, torch, torch.nn as nn
from pathlib import Path

def _sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))

def _load_temperature(cfg, override=None):
    if override is not None:
        try: return float(override)
        except: return 1.0
    p = cfg.paths.outputs_dir / "lab_viz" / "calib_summary.json"
    if p.exists():
        try:
            T = float(json.load(open(p))["T_star"])
            return max(T, 1.0)  # 过置信保护
        except Exception:
            pass
    return 1.0

def _iter_dets_file(path):
    """兼容 JSONL（每行一个dict）与 COCO数组（整文件array）。统一输出 {image_id, detections=[...]}。"""
    path = str(path)
    with open(path, "r") as f:
        head = f.read(2)
    if head.startswith('['):
        # COCO results array
        arr = json.load(open(path, "r"))
        from collections import defaultdict
        g = defaultdict(list)
        for d in arr:
            g[d["image_id"]].append(d)
        for img_id, items in g.items():
            yield {"image_id": img_id, "detections": items}
    else:
        with open(path, "r") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if "detections" in r: pass
                elif "boxes" in r:   r["detections"] = r["boxes"]
                elif "preds" in r:   r["detections"] = r["preds"]
                else:                r["detections"] = []
                yield r

def _build_node_model_from_ckpt(ckpt, device):
    sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    meta = ckpt.get("meta", {})

    def is_sem(sd, meta):
        if meta.get("type") == "LabelabilitySem": return True
        if any(k.startswith("emb.") for k in sd): return True
        if any(k.startswith("mlp.") for k in sd): return True
        return False

    if is_sem(sd, meta):
        Dg = int(meta.get("in_dim_geo"))
        K  = int(meta.get("num_classes", 91))
        d  = int(meta.get("emb_dim", 64))

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

        mdl = LabelabilitySem(Dg, K, d).to(device)
        mdl.load_state_dict(sd, strict=True)
        return mdl, {"type":"sem","Dg":Dg,"K":K,"d":d}

    # 兼容旧 MLP（仅几何）
    in_dim = ckpt.get("in_dim")
    if in_dim is None:
        for k in ("net.0.weight","net.2.weight","net.4.weight"):
            if k in sd: in_dim = int(sd[k].shape[1]); break

    class LabelabilityMLP(nn.Module):
        def __init__(self, in_dim, hidden=(128,64)):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden[0]), nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                nn.Linear(hidden[1], 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    mdl = LabelabilityMLP(in_dim).to(device)
    mdl.load_state_dict(sd, strict=True)
    return mdl, {"type":"mlp","in_dim":in_dim}

def _pair_iou_xywh(a, b):
    ax,ay,aw,ah = a[:4]; bx,by,bw,bh = b[:4]
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = aw*ah + bw*bh - inter + 1e-6
    return inter/union

def _bbox_to_geo(b, W, H):
    x,y,w,h = b[:4]; s=b[4]
    cx = (x + w/2)/max(W,1)
    cy = (y + h/2)/max(H,1)
    a  = (w*h)/max(W*H,1)
    r  = w/max(h,1e-6)
    return [cx, cy, a, r, s]

def _build_semctx_features(boxes, W, H, K=91, Kc=10, Kn=10):
    """
    boxes: list of [x,y,w,h,score,cat]（同图，已按score降序且Top-K截断）
    return:
      geo: (N,5), cid: (N,),
      ctx_ids,ctx_ws: (N,Kc),
      nb_ids, nb_ws : (N,Kn)
    """
    N = len(boxes)
    if N == 0:
        return (np.zeros((0,5),np.float32), np.zeros((0,),np.int64),
                np.full((0,Kc),-1,np.int64), np.zeros((0,Kc),np.float32),
                np.full((0,Kn),-1,np.int64), np.zeros((0,Kn),np.float32))

    geo = np.array([_bbox_to_geo(b, W, H) for b in boxes], dtype=np.float32)
    cid = np.array([int(b[5]) for b in boxes], dtype=np.int64)
    scr = np.array([float(b[4]) for b in boxes], dtype=np.float32)

    # 图像级上下文（按类别聚合分数和）
    K = int(K)
    ctx_sum = np.zeros((K,), dtype=np.float32)
    for c in range(K):
        m = (cid == c)
        if m.any(): ctx_sum[c] = float(scr[m].sum())
    ctx_pairs = [(c, ctx_sum[c]) for c in range(K) if ctx_sum[c] > 0]
    ctx_pairs.sort(key=lambda x: x[1], reverse=True)
    ctx_ids = np.full((N, Kc), -1, dtype=np.int64)
    ctx_ws  = np.zeros((N, Kc), dtype=np.float32)
    top_ctx = ctx_pairs[:Kc]
    for i in range(N):
        for j,(c,v) in enumerate(top_ctx):
            ctx_ids[i,j] = c
            ctx_ws[i,j]  = v

    # 邻域（IoU 加权，偏向同类）
    nb_ids = np.full((N, Kn), -1, dtype=np.int64)
    nb_ws  = np.zeros((N, Kn), dtype=np.float32)
    for i in range(N):
        bi = boxes[i]; c0 = cid[i]
        cand=[]
        for j in range(N):
            if j==i: continue
            bj = boxes[j]
            iou = _pair_iou_xywh(bi, bj)
            w   = iou * (1.2 if cid[j]==c0 else 0.6)
            if w>0: cand.append((cid[j], w))
        cand.sort(key=lambda x:x[1], reverse=True)
        for k,(cc,ww) in enumerate(cand[:Kn]):
            nb_ids[i,k] = cc
            nb_ws[i,k]  = ww

    return geo, cid, ctx_ids, ctx_ws, nb_ids, nb_ws


# =================== infer_post: main ===================

def stage_infer_post(cfg, det_file: Path, node_mdl_p: Path,
                     topk=300, sub_iou=0.50, temp=None):
    """
    用 labelability 节点模型对检测结果做后处理：
      - 每图按 score 取 Top-K，构建几何+语义上下文特征；
      - logits / T → sigmoid 得 p_lab；
      - 重打分：score' = score * (p_lab ** (1/T))；
      - 可选移除被大框覆盖的小框（同类且 IoU>=sub_iou 且分数更低）。
    输出：/outputs/detections_post.jsonl
    """
    det_file  = Path(det_file)
    node_mdl_p = Path(node_mdl_p)
    assert det_file.exists(), f"det file not found: {det_file}"
    assert node_mdl_p.exists(), f"node model not found: {node_mdl_p}"

    sizeB = det_file.stat().st_size
    print(f"[infer_post] det_file={det_file}  size={sizeB}B")

    # 温度 & 模型
    T_use = _load_temperature(cfg, override=temp)
    ckpt = torch.load(node_mdl_p, map_location=cfg.device)
    mdl, meta = _build_node_model_from_ckpt(ckpt, cfg.device)
    mdl.eval()
    print(f"[infer_post] loaded node model: {meta}  | temperature={T_use:.2f}  | topk={topk}")

    out_lines = []
    AMP = cfg.device.startswith("cuda")
    total_imgs = total_boxes = changed_imgs = subsumed_cnt = 0

    # 首行预览
    try:
        with open(det_file, "r") as _f:
            _first = _f.readline().strip()
        print(f"[infer_post] first_line={_first[:200]}")
    except Exception:
        pass

    for rec in _iter_dets_file(det_file):
        total_imgs += 1
        img_id = rec.get("image_id")
        W = rec.get("width", 1); H = rec.get("height", 1)

        dets = rec.get("detections", [])
        # 统一成 [x,y,w,h,score,cat]
        # 统一成 [x,y,w,h,score,cat]
        boxes_all = []
        for d in dets:
            if isinstance(d, dict):
                # COCO 风格
                x, y, w, h = d["bbox"]
                s = float(d.get("score", 0.0))
                c = int(d.get("category_id", 0))
            else:
                # 数组/列表风格: [x,y,w,h,score,cat] 或 [x,y,w,h] / [x,y,w,h,score]
                arr = list(d)
                if len(arr) >= 6:
                    x, y, w, h, s, c = arr[:6]
                elif len(arr) == 5:
                    x, y, w, h, s = arr
                    c = 0
                elif len(arr) == 4:
                    x, y, w, h = arr
                    s, c = 0.0, 0
                else:
                    continue
            boxes_all.append([float(x), float(y), float(w), float(h), float(s), int(c)])

        total_boxes += len(boxes_all)
        if total_imgs == 1:
            print(f"[infer_post] first_record: image_id={img_id}  boxes={len(boxes_all)}")

        if len(boxes_all) == 0:
            out_lines.append(json.dumps(rec))
            continue

        # Top-K by score
        boxes_all.sort(key=lambda b:b[4], reverse=True)
        boxes_top = boxes_all[:min(topk, len(boxes_all))]

        # 特征
        geo, cid, ctx_ids, ctx_ws, nb_ids, nb_ws = _build_semctx_features(
            boxes_top, W, H, K=meta.get("K",91)
        )

        # 前向
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
            if meta["type"] == "sem":
                geo_t = torch.tensor(geo, dtype=torch.float32, device=cfg.device)
                cid_t = torch.tensor(cid, dtype=torch.int64, device=cfg.device)
                ctxi_t= torch.tensor(ctx_ids, dtype=torch.int64, device=cfg.device)
                ctxw_t= torch.tensor(ctx_ws, dtype=torch.float32, device=cfg.device)
                nbi_t = torch.tensor(nb_ids, dtype=torch.int64, device=cfg.device)
                nbw_t = torch.tensor(nb_ws, dtype=torch.float32, device=cfg.device)
                logits = mdl(geo_t, cid_t, ctxi_t, ctxw_t, nbi_t, nbw_t).float().cpu().numpy()
            else:
                geo_t = torch.tensor(geo, dtype=torch.float32, device=cfg.device)
                logits = mdl(geo_t).float().cpu().numpy()

        if total_imgs <= 3:
            print(f"[infer_post][debug] img={img_id} logits_head={logits[:5]}")

        logits = logits / float(T_use)
        p_lab  = _sigmoid(logits)
        if total_imgs <= 3:
            print(f"[infer_post][debug] img={img_id} p_head={p_lab[:5]}")

        # 重打分
        for i,b in enumerate(boxes_top):
            b[4] = float(b[4] * (p_lab[i] ** (1.0/float(T_use))))
        changed_imgs += 1

        # 覆盖剔除
        if sub_iou is not None and sub_iou > 0:
            keep = [True]*len(boxes_top)
            for i in range(len(boxes_top)):
                if not keep[i]: continue
                for j in range(len(boxes_top)):
                    if i==j or not keep[j]: continue
                    if boxes_top[i][5] == boxes_top[j][5] and boxes_top[i][4] >= boxes_top[j][4]:
                        if _pair_iou_xywh(boxes_top[i], boxes_top[j]) >= sub_iou:
                            keep[j] = False; subsumed_cnt += 1
            boxes_top = [b for k,b in enumerate(boxes_top) if keep[k]]

        # 合并回非Top-K部分
        new_boxes = boxes_top + boxes_all[len(boxes_top):]
        rec["detections"] = [
            {"bbox":[b[0],b[1],b[2],b[3]], "score":float(b[4]), "category_id":int(b[5])}
            for b in new_boxes
        ]
        out_lines.append(json.dumps(rec))

        if total_imgs % 500 == 0:
            print(f"[infer_post] progress: imgs={total_imgs}  cum_boxes={total_boxes}  changed_imgs={changed_imgs}  subsumed={subsumed_cnt}")

    out_p = cfg.paths.outputs_dir / "detections_post.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for s in out_lines:
            f.write(s if s.endswith("\n") else s+"\n")

    print(f"[infer_post] wrote {out_p}  boxes={total_boxes}  changed_imgs={changed_imgs}  subsumed={subsumed_cnt}  sub_iou={sub_iou}  temp={T_use}")
    return out_p






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
    if "calib" in steps:
        labset = cfg.paths.outputs_dir / "labelability_train_semctx.npz"
        node_m = cfg.paths.models_dir / "grm_node_labelability.pt"
        T_star = diag_labelability_calibration(cfg, labset, node_m)

    if "grid" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        node = artifacts.get("node_mdl", cfg.paths.models_dir / "grm_node_labelability.pt")
        artifacts["grid"] = stage_grid(cfg, det_file=det, node_mdl=node)

    if "infer_post" in steps:
        det = artifacts.get("det", cfg.paths.detections_dir / "detections.jsonl")
        node_m = artifacts.get("lab_mdl", cfg.paths.models_dir / "grm_node_labelability.pt")
        # 可用命令行或环境变量传 sub_iou / temp；这里给默认
        artifacts["det_post"] = stage_infer_post(cfg, det, node_m, sub_iou=0.50, temp=1.2)

    if "eval_post" in steps:
        post = artifacts.get("det_post", cfg.paths.outputs_dir / "detections_grm_post.jsonl")
        print("[eval_post]", stage_eval_post(cfg, post))

    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
