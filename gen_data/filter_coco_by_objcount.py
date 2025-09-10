# Created by MacBook Pro at 06.09.25
import json, os, csv, shutil
from collections import defaultdict
import config


def filter_coco(args, min_objs=5, max_objs=15, data_split=f"val2017"):
    coco_json_path = config.get_coco_path(args.remote) / "original" / "annotations" / f"instances_{data_split}2017.json"
    orig_img_dir = config.get_coco_path(args.remote) / "original" / f"{data_split}2017"
    out_list = config.get_coco_path(args.remote) / "original" / "annotations" / "keep_filenames.txt"
    csv_stats = config.get_coco_path(args.remote) / "original" / "annotations" / "image_object_counts.csv"
    selected_dir = config.get_coco_path(args.remote) / "selected" / f"{data_split}2017"
    selected_ann_dir = config.get_coco_path(args.remote) / "selected" / f"annotations"
    selected_json_path = selected_ann_dir / f"instances_{data_split}2017.json"
    os.makedirs(selected_dir, exist_ok=True)
    os.makedirs(selected_ann_dir, exist_ok=True)
    with open(coco_json_path) as f:
        coco = json.load(f)
    img_id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    print(f"Total images in {data_split}: {len(img_id2name)}")
    counts = defaultdict(int)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        counts[ann["image_id"]] += 1
    kept, rows, kept_img_ids = [], [], []
    # print(orig_img_dir)
    for img_id, fname in img_id2name.items():
        n = counts.get(img_id, 0)
        rows.append((img_id, fname, n))
        if min_objs <= n <= max_objs and os.path.exists(os.path.join(orig_img_dir, fname)):
            src = os.path.join(orig_img_dir, fname)
            dst = os.path.join(selected_dir, fname)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                kept.append(fname)
                kept_img_ids.append(img_id)
    # Extract filtered images and annotations
    filtered_images = [im for im in coco["images"] if im["id"] in kept_img_ids]
    filtered_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in kept_img_ids]
    filtered_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco.get("categories", [])
    }
    with open(selected_json_path, "w") as f:
        json.dump(filtered_coco, f)
    # Optionally, save kept filenames
    with open(out_list, "w") as f:
        for fname in kept:
            f.write(f"{fname}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--data_split", type=str)
    args = parser.parse_args()
    filter_coco(args, data_split=args.data_split)
