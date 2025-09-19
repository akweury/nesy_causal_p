# Created by MacBook Pro at 18.09.25

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
from gen_data import proximity_coco

# ========= New Rule: size-similarity + proximity + min_group_size =========
# Positive iff there exists at least one category C in the image such that:
#   - all objects of C have similar sizes within a relative tolerance
#   - all objects of C are within a distance threshold (pairwise max distance)
#   - count(C) >= min_group_size
#
# Output directory:
#   {output_root}/Rule_SizeProx_C-{Cname}/{train,val}/{positive,negative}/
#
# Recommended defaults: size_tol_rel=0.25, dist_thresh=0.06, min_group_size=3

from collections import defaultdict


def _centers_and_areas(annos: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return centers [N,2] and areas [N] from COCO annos (assume valid bbox)."""
    if len(annos) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    b = np.array([a["bbox"] for a in annos], dtype=np.float32)  # [x,y,w,h]
    cx = b[:, 0] + 0.5 * b[:, 2]
    cy = b[:, 1] + 0.5 * b[:, 3]
    areas = b[:, 2] * b[:, 3]
    return np.stack([cx, cy], axis=1), areas


def _all_within_size_tolerance(areas: np.ndarray, tol_rel: float) -> bool:
    """(max - min) / median <= tol_rel. Require at least 2 objects."""
    if areas.size < 2:
        return False
    med = np.median(areas) if np.median(areas) > 0 else 1.0
    return (areas.max() - areas.min()) / med <= tol_rel


def _max_pairwise_center_dist_norm(centers: np.ndarray, img_size: Tuple[int, int]) -> float:
    """Max pairwise Euclidean distance among centers, normalized by min(W,H)."""
    n = centers.shape[0]
    if n < 2:
        return 0.0
    W, H = img_size
    scale = float(min(W, H)) if min(W, H) > 0 else 1.0
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=-1) / scale
    # Only upper triangle to avoid zeros on diagonal
    iu = np.triu_indices(n, k=1)
    return float(d[iu].max()) if iu[0].size > 0 else 0.0


def _image_annos_by_class(coco: COCO, img_id: int, valid_category_ids: Optional[Iterable[int]] = None) -> Dict[int, List[Dict]]:
    """Return mapping {cat_id: [annos...]} with basic bbox validity filter."""
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    annos = coco.loadAnns(ann_ids)
    annos = [a for a in annos if "bbox" in a and a.get("iscrowd", 0) == 0 and a["bbox"][2] > 0 and a["bbox"][3] > 0]
    cls_map = defaultdict(list)
    vset = set(valid_category_ids) if valid_category_ids is not None else None
    for a in annos:
        if vset is not None and a["category_id"] not in vset:
            continue
        cls_map[a["category_id"]].append(a)
    return cls_map


def _label_sizeprox_for_image(
        coco: COCO,
        img_id: int,
        size_tol_rel: float,
        dist_thresh: float,
        min_group_size: int,
        valid_category_ids: Optional[Iterable[int]] = None,
) -> Tuple[int, Dict]:
    """
    Return (label, aux):
      label: 1 positive / 0 negative
      aux: dict with per-class verdicts and counts for debugging
    """
    info = coco.loadImgs([img_id])[0]
    W, H = info["width"], info["height"]

    cls_map = _image_annos_by_class(coco, img_id, valid_category_ids=valid_category_ids)
    verdicts = {}
    pos_flag = False

    for cid, ann_list in cls_map.items():
        n = len(ann_list)
        if n < min_group_size:
            verdicts[int(cid)] = {"ok": False, "n": n, "reason": "too_few"}
            continue
        centers, areas = _centers_and_areas(ann_list)
        ok_size = _all_within_size_tolerance(areas, size_tol_rel)
        max_d = _max_pairwise_center_dist_norm(centers, (W, H))
        ok_dist = (max_d <= dist_thresh)
        ok = ok_size and ok_dist
        verdicts[int(cid)] = {
            "ok": bool(ok),
            "n": n,
            "ok_size": bool(ok_size),
            "ok_dist": bool(ok_dist),
            "max_pairwise_dist_norm": float(max_d),
            "size_tol_rel": float(size_tol_rel),
        }
        if ok:
            pos_flag = True  # at least one category satisfies all constraints

    return (1 if pos_flag else 0), {
        "verdicts": verdicts,
        "min_group_size": int(min_group_size),
        "dist_thresh": float(dist_thresh),
        "size_tol_rel": float(size_tol_rel),
    }


def build_rule_sizeprox_dataset(
        coco_json: str,
        images_root: str,
        output_root: str,
        # rule params
        size_tol_rel: float = 0.25,
        dist_thresh: float = 0.06,
        min_group_size: int = 3,
        valid_category_ids: Optional[List[int]] = None,  # e.g., [1,3,62]
        # split/sample
        train_ratio: float = 0.8,
        max_per_split: Optional[int] = 2000,
        seed: int = 0,
        use_symlink: bool = False,
        image_id_subset: Optional[List[int]] = None,
):
    rng = random.Random(seed)
    coco = COCO(coco_json)

    if image_id_subset is None:
        image_ids = [i for i in coco.getImgIds() if os.path.exists(os.path.join(images_root, coco.loadImgs([i])[0]["file_name"]))]
    else:
        image_ids = [i for i in image_id_subset if os.path.exists(os.path.join(images_root, coco.loadImgs([i])[0]["file_name"]))]

    # Label all images
    pos_pool, neg_pool, meta = [], [], {}
    for img_id in image_ids:
        # img_id = 3693 # pos
        img_id = 21924 # neg
        y, aux = _label_sizeprox_for_image(
            coco, img_id,
            size_tol_rel=size_tol_rel,
            dist_thresh=dist_thresh,
            min_group_size=min_group_size,
            valid_category_ids=valid_category_ids,
        )
        meta[int(img_id)] = aux
        (pos_pool if y == 1 else neg_pool).append(int(img_id))

    # split
    rng.shuffle(pos_pool);
    rng.shuffle(neg_pool)

    def split_ids(ids, ratio):
        n_tr = int(len(ids) * ratio)
        return ids[:n_tr], ids[n_tr:]

    pos_tr, pos_va = split_ids(pos_pool, train_ratio)
    neg_tr, neg_va = split_ids(neg_pool, train_ratio)

    # cap per split
    if max_per_split is not None:
        half = max_per_split // 2
        pos_tr, neg_tr = pos_tr[:half], neg_tr[:half]
        pos_va, neg_va = pos_va[:half], neg_va[:half]

    # copy/link files
    out_root = os.path.join(output_root, "Rule_SizeProx_AllCats")

    def dump_split(split_name, ids_pos, ids_neg):
        out_pos = os.path.join(out_root, split_name, "positive")
        out_neg = os.path.join(out_root, split_name, "negative")
        os.makedirs(out_pos, exist_ok=True);
        os.makedirs(out_neg, exist_ok=True)
        recs = []

        def _put(iid, is_pos):
            info = coco.loadImgs([iid])[0]
            fn = info["file_name"]
            src = os.path.join(images_root, fn)
            dst = os.path.join(out_pos if is_pos else out_neg, fn)
            if not os.path.exists(src):
                # deep search
                found = None
                for root, _, files in os.walk(images_root):
                    if fn in files:
                        found = os.path.join(root, fn);
                        break
                if not found: return
                src = found
                dst = os.path.join(os.path.dirname(dst), os.path.basename(found))
            try:
                if use_symlink:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)
            except Exception:
                shutil.copy2(src, dst)
            recs.append({
                "image_id": int(iid),
                "file_name": os.path.basename(dst),
                "split": split_name,
                "label": "positive" if is_pos else "negative",
                "aux": meta[int(iid)],
            })

        for iid in ids_pos: _put(iid, True)
        for iid in ids_neg: _put(iid, False)
        return recs

    rec_train = dump_split("train", pos_tr, neg_tr)
    rec_val = dump_split("val", pos_va, neg_va)

    # manifest
    manifest = {
        "task": "Rule_SizeProx_AllCats",
        "params": {
            "train_ratio": train_ratio,
            "max_per_split": max_per_split,
            "seed": seed,
            "size_tol_rel": float(size_tol_rel),
            "dist_thresh": float(dist_thresh),
            "min_group_size": int(min_group_size),
            "valid_category_ids": valid_category_ids,
            "use_symlink": use_symlink,
            "coco_json": str(coco_json),
            "images_root": str(images_root),
        },
        "stats": {
            "total_pos": len(pos_pool),
            "total_neg": len(neg_pool),
            "train_pos": len([r for r in rec_train if r["label"] == "positive"]),
            "train_neg": len([r for r in rec_train if r["label"] == "negative"]),
            "val_pos": len([r for r in rec_val if r["label"] == "positive"]),
            "val_neg": len([r for r in rec_val if r["label"] == "negative"]),
        },
        "records": rec_train + rec_val
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest_rule_sizeprox.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[Rule-SizeProx] Done.")
    print(json.dumps(manifest["stats"], indent=2))


def run_rule_sizeprox_example():
    args = args_utils.get_args()
    base = config.get_coco_path(args.remote) / "selected"
    images_root = str(base / "train2017")  # 也可用 val2017
    coco_json = str(base / "annotations" / "instances_train2017.json")
    output_root = str(config.get_coco_patterns_path(args.remote) / "proximity_coco")

    # 只考虑更容易聚集的类（可选）
    valid_cats = [1, 3, 62, 15, 2]  # person, car, chair, bench, bicycle
    valid_cats = [1]  # person, car, chair, bench, bicycle

    build_rule_sizeprox_dataset(
        coco_json=coco_json,
        images_root=images_root,
        output_root=output_root,
        size_tol_rel=3,  # 尺寸相似相对公差
        dist_thresh=0.7,  # 归一化最大两两距离阈值（相对 min(W,H)）
        min_group_size=3,
        valid_category_ids=valid_cats,
        train_ratio=0.8,
        max_per_split=2000,
        seed=42,
        use_symlink=False,
        image_id_subset=None,  # 或 coco.getImgIds(catIds=valid_cats)
    )


if __name__ == "__main__":
    run_rule_sizeprox_example()