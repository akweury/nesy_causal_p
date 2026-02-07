# Created by MacBook Pro at 27.11.25
import json, os, csv, shutil
from collections import defaultdict
from tqdm import tqdm
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN


import config



def build_proximity_dataset(
    args,
    coco_dict,
    max_images=100,
    eps_rel=0.10,
    min_samples=2,
):
    """
    Take the filtered COCO subset (from filter_coco) and build a
    small proximity-grouping dataset:

    - Sample up to `max_images` images from the filtered subset
    - For each image, cluster object centers with DBSCAN in normalized coords
    - Add `group_id` field to each annotation
    - Save a new JSON: proximity_{split}2017.json
    """
    selected_json_path = Path(coco_dict["selected_json"])
    with open(selected_json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    # Map image_id -> image info
    img_by_id = {img["id"]: img for img in images}

    # Map image_id -> list of annotations
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # Sample up to max_images
    all_img_ids = list(img_by_id.keys())
    if len(all_img_ids) > max_images:
        random.seed(0)
        chosen_img_ids = random.sample(all_img_ids, max_images)
    else:
        chosen_img_ids = all_img_ids

    print(f"[build_proximity_dataset] Using {len(chosen_img_ids)} images "
          f"for proximity grouping (max={max_images})")

    new_images = []
    new_annotations = []

    for img_id in chosen_img_ids:
        img_info = img_by_id[img_id]
        w, h = img_info["width"], img_info["height"]
        anns = img_to_anns[img_id]

        if len(anns) == 0:
            # Should not happen if filter_coco did its job, but guard anyway
            continue

        # Collect normalized centers
        centers = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            centers.append([cx, cy])
        centers = np.asarray(centers, dtype=np.float32)

        # Run DBSCAN on normalized coords
        clustering = DBSCAN(eps=eps_rel, min_samples=min_samples).fit(centers)
        labels = clustering.labels_  # -1 = noise

        # Turn noise into unique singleton groups to avoid -1
        next_gid = int(labels.max()) + 1 if labels.size > 0 else 0
        group_ids = []
        for lab in labels:
            if lab == -1:
                group_ids.append(next_gid)
                next_gid += 1
            else:
                group_ids.append(int(lab))

        # Attach group_id to annotations
        for ann, gid in zip(anns, group_ids):
            # clone to avoid mutating original dicts (in case you reuse them)
            ann_new = dict(ann)
            ann_new["group_id"] = int(gid)
            new_annotations.append(ann_new)

        new_images.append(img_info)

    # Build new COCO-style dict
    prox_coco = {
        "info": info,
        "licenses": licenses,
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }

    split = args.data_split
    if split is None or split == "":
        split = "val"

    prox_json_path = Path(coco_dict["selected_dir"]).parent / "annotations" / f"proximity_{split}2017.json"
    prox_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(prox_json_path, "w") as f:
        json.dump(prox_coco, f)

    print(f"[build_proximity_dataset] Saved proximity JSON to: {prox_json_path}")

    return {
        "proximity_json": str(prox_json_path),
        "num_images": len(new_images),
        "num_annotations": len(new_annotations),
    }

def filter_coco(args, min_objs=5, max_objs=15, data_split="val"):
    """
    Filter COCO images by number of annotated objects and create a reduced subset.

    - Keeps images with min_objs <= #objects <= max_objs
    - Writes:
        * keep_filenames.txt      : list of selected image file names
        * image_object_counts.csv : image_id, file_name, num_objects (for ALL images)
        * instances_{split}2017.json in selected/annotations: pruned COCO JSON
        * Copies selected images into selected/{split}2017/
    """
    # Paths
    coco_root = config.get_coco_path(args.remote)
    coco_json_path = coco_root / "original" / "annotations" / f"instances_{data_split}2017.json"
    orig_img_dir = coco_root / "original" / f"{data_split}2017"
    out_list = coco_root / "original" / "annotations" / "keep_filenames.txt"
    csv_stats = coco_root / "original" / "annotations" / "image_object_counts.csv"
    selected_dir = coco_root / "selected" / f"{data_split}2017"
    selected_ann_dir = coco_root / "selected" / "annotations"
    selected_json_path = selected_ann_dir / f"instances_{data_split}2017.json"

    os.makedirs(selected_dir, exist_ok=True)
    os.makedirs(selected_ann_dir, exist_ok=True)

    # Load original COCO annotations
    with open(coco_json_path, "r") as f:
        coco = json.load(f)


    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    # Build mapping: image_id -> list of annotations
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # 1) Compute object counts and decide which images to keep
    selected_image_ids = set()
    csv_rows = []

    for img in tqdm(images):
        img_id = img["id"]
        file_name = img["file_name"]
        anns = img_to_anns.get(img_id, [])
        num_objs = len(anns)

        # For stats CSV (all images)
        csv_rows.append((img_id, file_name, num_objs))

        # For selection
        if min_objs <= num_objs <= max_objs:
            selected_image_ids.add(img_id)

    # 2) Write CSV stats
    with open(csv_stats, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image_id", "file_name", "num_objects"])
        writer.writerows(csv_rows)

    # 3) Write keep_filenames.txt for selected images
    with open(out_list, "w") as f_txt:
        for img in images:
            if img["id"] in selected_image_ids:
                f_txt.write(f"{img['file_name']}\n")

    # 4) Create pruned COCO JSON (only selected images + anns)
    selected_images = [img for img in images if img["id"] in selected_image_ids]
    selected_annotations = [ann for ann in annotations if ann["image_id"] in selected_image_ids]

    selected_coco = {
        "info": info,
        "licenses": licenses,
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories,
    }

    with open(selected_json_path, "w") as f_out:
        json.dump(selected_coco, f_out)

    # 5) Copy selected image files
    for img in selected_images:
        src = orig_img_dir / img["file_name"]
        dst = selected_dir / img["file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)

    print(
        f"[filter_coco] Selected {len(selected_images)} images "
        f"with {min_objs}–{max_objs} objects. "
        f"Subset JSON: {selected_json_path}"
    )

    return {
        "selected_count": len(selected_images),
        "selected_json": str(selected_json_path),
        "keep_list": str(out_list),
        "csv_stats": str(csv_stats),
        "selected_dir": str(selected_dir),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--data_split", type=str, default="val",
                        help="COCO split prefix, e.g. 'train' or 'val' (default: 'val')")
    parser.add_argument("--min_objs", type=int, default=5,
                        help="Minimum number of objects per image")
    parser.add_argument("--max_objs", type=int, default=15,
                        help="Maximum number of objects per image")
    parser.add_argument("--max_images", type=int, default=5000,
                        help="Maximum number of images in the proximity mini-dataset")
    parser.add_argument("--eps_rel", type=float, default=0.10,
                        help="Relative DBSCAN eps (in normalized coord space)")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="DBSCAN min_samples")
    args = parser.parse_args()
    coco_dict = filter_coco(args, data_split=args.data_split)

    # 2) Build the proximity grouping mini-dataset (<=100 images)
    prox_dict = build_proximity_dataset(
        args,
        coco_dict,
        max_images=args.max_images,
        eps_rel=args.eps_rel,
        min_samples=args.min_samples,
    )

    print(
        f"[main] Done.\n"
        f"  Filtered subset JSON : {coco_dict['selected_json']}\n"
        f"  Selected images dir   : {coco_dict['selected_dir']}\n"
        f"  Proximity JSON        : {prox_dict['proximity_json']}\n"
        f"  #images (prox)        : {prox_dict['num_images']}\n"
        f"  #annotations (prox)   : {prox_dict['num_annotations']}"
    )

    print("Generated COCO subset details:")

