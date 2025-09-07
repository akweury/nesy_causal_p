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


def stage_infer_groups(cfg, model_file: Path, det_file: Path) -> Path:
    """
    推理边分数 -> 阈值/聚类成组
    输出: cfg.paths.outputs_dir / groups.jsonl
    """
    out = cfg.paths.outputs_dir / "groups.jsonl"
    if out.exists():
        print(f"[infer] reuse {out}");
        return out
    # TODO: 载入模型，对每图输出:
    # {"image_id":123, "groups":[ [idxs...], [idxs...] ], "edges":[[i,j,prob],...]}
    out.write_text("")
    print(f"[infer] wrote {out}")
    return out


def stage_std_nms(cfg, det_file: Path, groups_file: Path) -> Path:
    """
    Group-aware NMS：组内/组间不同阈值
    输出: cfg.paths.outputs_dir / detections_groupnms.jsonl
    """
    out = cfg.paths.outputs_dir / "detections_std_nms.jsonl"
    if out.exists():
        print(f"[stdNMS] reuse {out}");
        return out
    # TODO: 读取 det_file + groups_file，执行 group-aware NMS，写 JSONL
    out.write_text("")
    print(f"[stdNMS] wrote {out}")
    return out


def stage_group_nms(cfg, det_file: Path, groups_file: Path) -> Path:
    """
    Group-aware NMS：组内/组间不同阈值
    输出: cfg.paths.outputs_dir / detections_groupnms.jsonl
    """
    out = cfg.paths.outputs_dir / "detections_groupnms.jsonl"
    if out.exists():
        print(f"[gNMS] reuse {out}");
        return out
    # TODO: 读取 det_file + groups_file，执行 group-aware NMS，写 JSONL
    out.write_text("")
    print(f"[gNMS] wrote {out}")
    return out


def stage_eval_coco(cfg, det_file: Path) -> Dict[str, Any]:
    """
    COCO mAP 评测（使用 pycocotools）
    """
    # TODO: 将 JSONL 转为 COCO result 格式，eval -> 返回 dict
    metrics = {"mAP": None}
    print(f"[eval] metrics: {metrics}")
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
