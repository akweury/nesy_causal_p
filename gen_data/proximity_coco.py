# Created by MacBook Pro at 17.09.25


import json, os
from urllib.parse import quote
import random
import numpy as np
import shutil
import argparse
from typing import List, Dict, Tuple, Optional, Iterable
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import config
from src.utils import args_utils


# ---------------- Core proximity utils ----------------

def _bbox_centers_xy(annos: List[Dict]) -> np.ndarray:
    if len(annos) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    b = np.array([a["bbox"] for a in annos], dtype=np.float32)
    cx = b[:, 0] + 0.5 * b[:, 2]
    cy = b[:, 1] + 0.5 * b[:, 3]
    return np.stack([cx, cy], axis=1)


def _connected_components(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        q = [i]
        seen[i] = True
        comp = [i]
        while q:
            u = q.pop(0)
            nbrs = np.nonzero(adj[u])[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def find_proximity_groups(
        annos: List[Dict],
        img_size: Tuple[int, int],
        same_category: bool = True,
        dist_thresh: float = 0.08,
        min_group_size: int = 2,
        valid_category_ids: Optional[Iterable[int]] = None,
) -> List[List[int]]:
    W, H = img_size
    if len(annos) == 0:
        return []
    idxs = list(range(len(annos)))
    if valid_category_ids is not None:
        valid_set = set(valid_category_ids)
        idxs = [i for i in idxs if annos[i]["category_id"] in valid_set]
    if len(idxs) == 0:
        return []

    centers = _bbox_centers_xy([annos[i] for i in idxs]).astype(np.float32)
    scale = float(min(W, H))
    centers_norm = centers / max(scale, 1.0)

    diff = centers_norm[:, None, :] - centers_norm[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    if same_category:
        cats = np.array([annos[i]["category_id"] for i in idxs], dtype=np.int64)
        same_cat = (cats[:, None] == cats[None, :]).astype(np.uint8)
    else:
        same_cat = np.ones_like(dist, dtype=np.uint8)

    adj = ((dist <= dist_thresh).astype(np.uint8) * same_cat)
    np.fill_diagonal(adj, 0)

    comps_all = _connected_components(adj)
    groups = []
    for comp in comps_all:
        if len(comp) >= min_group_size:
            groups.append([idxs[j] for j in comp])
    return groups


# ---------------- QC 统计与可视化 ----------------

def compute_image_stats(coco: COCO, img_id: int, params: Dict) -> Dict:
    info = coco.loadImgs([img_id])[0]
    W, H = info["width"], info["height"]
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    annos = coco.loadAnns(ann_ids)
    annos = [a for a in annos if "bbox" in a and a.get("iscrowd", 0) == 0 and a["bbox"][2] > 0 and a["bbox"][3] > 0]

    groups = find_proximity_groups(
        annos, (W, H),
        same_category=params["same_category"],
        dist_thresh=params["dist_thresh"],
        min_group_size=params["min_group_size"],
        valid_category_ids=params.get("valid_category_ids", None),
    )
    comp_sizes = [len(g) for g in groups] if groups else []
    max_comp = max(comp_sizes) if comp_sizes else 1  # 若无群则设为1方便直方图
    cats = [a["category_id"] for a in annos]
    return {
        "num_objs": len(annos),
        "max_component": max_comp,
        "component_sizes": comp_sizes,
        "categories": cats,
    }

def draw_proximity_on_image(
    img_path: str,
    coco: COCO,
    img_id: int,
    params: Dict,
    save_path: str,
    max_edges_per_node: int = 3,
):
    info = coco.loadImgs([img_id])[0]
    W, H = info["width"], info["height"]
    annos = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
    annos = [a for a in annos if "bbox" in a and a.get("iscrowd", 0) == 0 and a["bbox"][2] > 0 and a["bbox"][3] > 0]

    centers = _bbox_centers_xy(annos)
    groups = find_proximity_groups(
        annos, (W, H),
        same_category=params["same_category"],
        dist_thresh=params["dist_thresh"],
        min_group_size=params["min_group_size"],
        valid_category_ids=params.get("valid_category_ids", None),
    )

    # 载图
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 画 bbox 和中心点
    for a in annos:
        x, y, w, h = a["bbox"]
        draw.rectangle([x, y, x+w, y+h], outline=(255, 215, 0), width=2)
    for (cx, cy) in centers:
        r = 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 255, 0))

    # 给每个群组随机颜色，连线
    rng = random.Random(42 + img_id)
    palette = []
    for _ in range(len(groups)):
        palette.append((rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)))

    for gi, members in enumerate(groups):
        color = palette[gi]
        # 简单：每个节点与其K个最近邻连线（只在组内）
        if len(members) <= 1:
            continue
        pts = centers[members]
        # 距离矩阵
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dmat, 1e9)
        for i in range(len(members)):
            nbr_idx = np.argsort(dmat[i])[:max_edges_per_node]
            x1, y1 = pts[i]
            for j in nbr_idx:
                x2, y2 = pts[j]
                draw.line([x1, y1, x2, y2], fill=color, width=3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def plot_hist(data_pos: List[int], data_neg: List[int], save_path: str, title: str, bins: int = 20):
    plt.figure(figsize=(7,4))
    plt.hist(data_pos, bins=bins, alpha=0.6, label="positive", density=False)
    plt.hist(data_neg, bins=bins, alpha=0.6, label="negative", density=False)
    plt.xlabel("Max connected component size")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def has_proximity_group(
        annos: List[Dict],
        img_size: Tuple[int, int],
        **kwargs
) -> bool:
    groups = find_proximity_groups(annos, img_size, **kwargs)
    return len(groups) > 0


# ---------------- Dataset builder ----------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _copy_or_link(src: str, dst: str, link: bool = False):
    _ensure_dir(os.path.dirname(dst))
    if link:
        # Try symlink; fallback to copy on Windows if needed
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def prox_task_gen(images_root: str, coco_json: str, output_root: str,
                  image_id_subset: Optional[List[int]] = None, max_images: Optional[int] = 2000,
                  min_group_size: int = 2,
                  dist_thresh: float = 0.05,
                  same_category: bool = True,
                  train_ratio: float = 0.8,
                  max_per_split: Optional[int] = 2000,  # 每个 split 下 pos+neg 上限；None 不限

                  seed: int = 0,
                  valid_category_ids: Optional[List[int]] = None,
                  # IO
                  use_symlink: bool = False,

                  ):
    rng = random.Random(seed)
    coco = COCO(coco_json)
    if image_id_subset is None:
        image_ids = [i for i in coco.getImgIds() if os.path.exists(os.path.join(images_root, coco.loadImgs([i])[0]["file_name"]))]
    else:
        image_ids = [i for i in image_id_subset if os.path.exists(os.path.join(images_root, coco.loadImgs([i])[0]["file_name"]))]

    # 计算每张图是否为正样本
    pos_ids, neg_ids = [], []
    for img_id in image_ids:
        info = coco.loadImgs([img_id])[0]
        W, H = info["width"], info["height"]
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        annos = coco.loadAnns(ann_ids)
        annos = [a for a in annos if "bbox" in a and a.get("iscrowd", 0) == 0 and a["bbox"][2] > 0 and a["bbox"][3] > 0]

        if len(annos) < min_group_size:
            neg_ids.append(img_id)
            continue

        ok = has_proximity_group(
            annos, (W, H),
            same_category=same_category,
            dist_thresh=dist_thresh,
            min_group_size=min_group_size,
            valid_category_ids=valid_category_ids,
        )
        if ok:
            pos_ids.append(img_id)
        else:
            neg_ids.append(img_id)
    # 打乱并切分
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def split_ids(ids: List[int], ratio: float):
        n_tr = int(len(ids) * ratio)
        return ids[:n_tr], ids[n_tr:]

    pos_tr, pos_te = split_ids(pos_ids, train_ratio)
    neg_tr, neg_te = split_ids(neg_ids, train_ratio)

    # 可选：限制每个 split 的样本量（保持正负尽量均衡）
    def cap_split(pos_list, neg_list, cap):
        if cap is None:
            return pos_list, neg_list
        # 简单均衡策略：各取 cap//2
        half = max(cap // 2, 1)
        return pos_list[:half], neg_list[:(cap - min(half, len(pos_list[:half])))]

    if max_per_split is not None:
        pos_tr, neg_tr = cap_split(pos_tr, neg_tr, max_per_split)
        pos_te, neg_te = cap_split(pos_te, neg_te, max_per_split)

    # 复制/链接图像到目标结构
    def dump_split(split_name: str, ids_pos: List[int], ids_neg: List[int]):
        out_pos = os.path.join(output_root, split_name, "positive")
        out_neg = os.path.join(output_root, split_name, "negative")
        _ensure_dir(out_pos);
        _ensure_dir(out_neg)

        recs = []
        for is_pos, ids in [(True, ids_pos), (False, ids_neg)]:
            for iid in ids:
                info = coco.loadImgs([iid])[0]
                fn = info["file_name"]
                src = os.path.join(images_root, fn)
                dst = os.path.join(out_pos if is_pos else out_neg, fn)
                if not os.path.exists(src):
                    # 某些 COCO 组织方式可能分子文件夹；此处尝试在 images_root 下递归查找
                    found = None
                    for root, _, files in os.walk(images_root):
                        if fn in files:
                            found = os.path.join(root, fn)
                            break
                    if found is None:
                        # 跳过缺失文件
                        continue
                    src = found
                    dst = os.path.join(out_pos if is_pos else out_neg, os.path.basename(found))
                _copy_or_link(src, dst, link=use_symlink)
                recs.append({
                    "image_id": iid,
                    "file_name": os.path.basename(dst),
                    "split": split_name,
                    "label": "positive" if is_pos else "negative",
                })
        return recs

    records = []
    records += dump_split("train", pos_tr, neg_tr)
    records += dump_split("val", pos_te, neg_te)

    # 保存 manifest
    manifest = {
        "params": {
            "train_ratio": train_ratio,
            "max_per_split": max_per_split,
            "seed": seed,
            "same_category": same_category,
            "dist_thresh": dist_thresh,
            "min_group_size": min_group_size,
            "valid_category_ids": valid_category_ids,
            "use_symlink": use_symlink,
            "coco_json": str(coco_json),
            "images_root": str(images_root),
        },
        "stats": {
            "total_pos": len(pos_ids),
            "total_neg": len(neg_ids),
            "train_pos": len([r for r in records if r["split"] == "train" and r["label"] == "positive"]),
            "train_neg": len([r for r in records if r["split"] == "train" and r["label"] == "negative"]),
            "val_pos": len([r for r in records if r["split"] == "val" and r["label"] == "positive"]),
            "val_neg": len([r for r in records if r["split"] == "val" and r["label"] == "negative"]),
        },
        "records": records
    }
    _ensure_dir(output_root)
    with open(os.path.join(output_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(manifest["stats"], indent=2))




# ---------------- 主流程 ----------------

def main_check():
    args = args_utils.get_args()
    num_samples = 100
    split_names = "train,val"
    dataset_dir = config.get_coco_patterns_path(args.remote) / "proximity_coco"
    seed = 42


    random.seed(seed)
    np.random.seed(seed)

    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    params = manifest["params"]
    coco_json = params["coco_json"]
    images_root = params["images_root"]
    # 兼容 valid_category_ids=None 的情况
    if params.get("valid_category_ids", None) is not None:
        params["valid_category_ids"] = list(params["valid_category_ids"])

    coco = COCO(coco_json)
    splits = split_names.split(",")
    assert len(splits) == 2, "目前只做两个split（train和val/或test）"

    # 统计容器
    stats = {
        "per_split": {},
        "overall": {},
        "params": params
    }

    qc_root = os.path.join(dataset_dir, "qc_vis")
    os.makedirs(qc_root, exist_ok=True)

    for split in splits:
        pos_dir = os.path.join(dataset_dir, split, "positive")
        neg_dir = os.path.join(dataset_dir, split, "negative")

        # 通过 manifest 找到该 split 的 image_id 列表
        recs = [r for r in manifest["records"] if r["split"] == split]
        pos_recs = [r for r in recs if r["label"] == "positive" and os.path.exists(os.path.join(pos_dir, r["file_name"]))]
        neg_recs = [r for r in recs if r["label"] == "negative" and os.path.exists(os.path.join(neg_dir, r["file_name"]))]

        # 基本计数
        split_stats = {
            "num_positive": len(pos_recs),
            "num_negative": len(neg_recs),
            "top_categories_positive": [],
            "top_categories_negative": [],
            "max_component_hist_positive": [],
            "max_component_hist_negative": [],
        }

        # 抽样可视化
        k = num_samples
        pos_vis = random.sample(pos_recs, min(k, len(pos_recs)))
        neg_vis = random.sample(neg_recs, min(k, len(neg_recs)))

        # 统计类别分布 & 最大连通子图
        cat_pos = []
        cat_neg = []
        max_comp_pos = []
        max_comp_neg = []

        # 正样本可视化与统计
        for r in pos_vis:
            img_id = r["image_id"]
            info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(pos_dir, r["file_name"])
            # stats
            st = compute_image_stats(coco, img_id, params)
            max_comp_pos.append(st["max_component"])
            cat_pos.extend(st["categories"])
            # vis
            save_path = os.path.join(qc_root, "images", split, "positive", f"{img_id}.jpg")
            draw_proximity_on_image(img_path, coco, img_id, params, save_path)

        # 负样本可视化与统计
        for r in neg_vis:
            img_id = r["image_id"]
            info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(neg_dir, r["file_name"])
            st = compute_image_stats(coco, img_id, params)
            max_comp_neg.append(st["max_component"])
            cat_neg.extend(st["categories"])
            save_path = os.path.join(qc_root, "images", split, "negative", f"{img_id}.jpg")
            draw_proximity_on_image(img_path, coco, img_id, params, save_path)

        # 全量统计（不止抽样）
        # 为节省时间，可只统计最大连通子图的分布；如果你想更精确，可以遍历 recs
        def accumulate_all_max_comp(records, label):
            arr = []
            # 小心：全量统计可能耗时，按需下采样
            sample_n = min(2000, len(records))  # 采样2000张足够看趋势
            pool = random.sample(records, sample_n) if len(records) > sample_n else records
            for r in pool:
                st = compute_image_stats(coco, r["image_id"], params)
                arr.append(st["max_component"])
            return arr

        split_stats["max_component_hist_positive"] = accumulate_all_max_comp(pos_recs, "positive")
        split_stats["max_component_hist_negative"] = accumulate_all_max_comp(neg_recs, "negative")

        # 类别Top10
        def topk(counter, k=10):
            return [{"category_id": int(cid), "count": int(cnt), "name": coco.loadCats([cid])[0]["name"]} for cid, cnt in counter.most_common(k)]

        split_stats["top_categories_positive"] = topk(Counter(cat_pos), k=10)
        split_stats["top_categories_negative"] = topk(Counter(cat_neg), k=10)

        # 画直方图
        plot_dir = os.path.join(qc_root, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_hist(split_stats["max_component_hist_positive"],
                  split_stats["max_component_hist_negative"],
                  save_path=os.path.join(plot_dir, f"{split}_max_component_hist.png"),
                  title=f"{split}: Max Connected Component Size")

        stats["per_split"][split] = split_stats

    # overall
    stats["overall"]["train_pos"] = stats["per_split"][splits[0]]["num_positive"]
    stats["overall"]["train_neg"] = stats["per_split"][splits[0]]["num_negative"]
    stats["overall"]["val_pos"]   = stats["per_split"][splits[1]]["num_positive"]
    stats["overall"]["val_neg"]   = stats["per_split"][splits[1]]["num_negative"]

    # 输出 stats.json
    out_stats = os.path.join(qc_root, "stats.json")
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("QC done. See:")
    print(" - Images with lines:", os.path.join(qc_root, "images"))
    print(" - Plots:", os.path.join(qc_root, "plots"))
    print(" - Stats JSON:", out_stats)

def main():
    args = args_utils.get_args()
    train_path = config.get_coco_path(args.remote) / "selected" / "train2017"
    val_path = config.get_coco_path(args.remote) / "selected" / "val2017"
    annotation_path = config.get_coco_path(args.remote) / "selected" / "annotations"
    train_ann_file = annotation_path / "instances_train2017.json"
    val_ann_file = annotation_path / "instances_val2017.json"

    out_dir = config.get_coco_patterns_path(args.remote) / "proximity_coco"
    os.makedirs(out_dir, exist_ok=True)
    out_train_dir = out_dir / "train"
    out_val_dir = out_dir / "val"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)

    img_id_subset = [1, 2]
    prox_task_gen(train_path, train_ann_file, out_dir,
                  image_id_subset=None,
                  max_images=200,
                  min_group_size=3)


if __name__ == "__main__":
    main()
    main_check()