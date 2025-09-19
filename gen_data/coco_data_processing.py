# Created by MacBook Pro at 05.09.25


import json, os
from urllib.parse import quote

from typing import Optional, Iterable, Tuple

import config
from src.utils import args_utils


def coco_to_labelstudio(args):
    coco_original_path = config.get_coco_path(args.remote) / "original"
    coco_ann = coco_original_path / "annotations" / "instances_val2017.json"
    img_dir = "val2017"
    out_file = coco_original_path / "annotations" / "labelstudio_import.json"
    coco_original_img_dir = coco_original_path / "val2017"
    with open(coco_ann) as f:
        coco = json.load(f)

    imgid2meta = {im["id"]: im for im in coco["images"]}
    imgid2anns = {im["id"]: [] for im in coco["images"]}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        imgid2anns[ann["image_id"]].append(ann)
    MAX_IMAGES = 2000
    tasks = []
    count = 0
    for img_id, im in imgid2meta.items():
        if MAX_IMAGES and count >= MAX_IMAGES:
            break
        file_name = im["file_name"]
        abs_path = coco_original_img_dir / file_name
        if not os.path.exists(abs_path):
            continue

        W, H = im["width"], im["height"]
        results = []
        for ann in imgid2anns.get(img_id, []):
            x, y, w, h = ann["bbox"]  # COCO像素坐标
            # 转百分比（0-100）
            val = {
                "x": 100.0 * x / W,
                "y": 100.0 * y / H,
                "width": 100.0 * w / W,
                "height": 100.0 * h / H,
                "rotation": 0,
                "rectanglelabels": ["object"]
            }
            results.append({
                "id": f"coco-{ann['id']}",
                "from_name": "obj",
                "to_name": "img",
                "type": "rectanglelabels",
                "value": val,
                "score": 1.0,
                "origin": "preannotation",
                # 你也可以在这里塞入COCO的category_id便于导出后映射
                "meta": {"category_id": ann["category_id"]}
            })

        # Label Studio 本地文件URL（需要URL编码）
        abs_path_url = quote(str(file_name))
        task = {
            "id": int(img_id),
            "data": {
                "image": f"{file_name}"
            },
            "meta": {
                "coco_id": int(img_id),
                "file_name": file_name
            },
            "predictions": [
                {"result": results, "score": 1.0, "model_version": "coco2ls"}
            ]
        }
        tasks.append(task)
        count += 1

    with open(out_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Exported {len(tasks)} tasks to {out_file}")


def build_labelstudio_subset_with_bboxes(
        coco_json_path: str,
        subset_dir: str,
        out_json_path: str,
        *,
        image_field_name: str = "img",  # Label Studio <Image name="...">
        rect_field_name: str = "obj",  # Label Studio <RectangleLabels name="...">
        rect_label_value: str = "object",  # <Label value="...">
        max_images: Optional[int] = None,  # 限制生成前N张（可选）
        skip_crowd: bool = True,  # 跳过 iscrowd=1
        id_from_filename: bool = False  # 任务 id 是否取自文件名（数字）
) -> Tuple[int, int]:
    """
    读取 COCO 标注与子集目录，输出带 bbox 预载的 Label Studio import JSON。
    返回: (生成任务数, 覆盖的对象数)

    约定：Label Studio 模板里使用:
      <Image name="{image_field_name}" .../>
      <RectangleLabels name="{rect_field_name}" toName="{image_field_name}">
        <Label value="{rect_label_value}"/>
      </RectangleLabels>
      <Choices name="group" toName="{rect_field_name}" perRegion="true" .../>
    """
    print (f"Reading COCO from {coco_json_path}, file exists: {os.path.exists(coco_json_path)}")

    print (f"Subset images from {subset_dir}, file exists: {os.path.exists(subset_dir)}")
    print (f"Writing Label Studio tasks to {out_json_path}, file exists: {os.path.exists(out_json_path)}")
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # 子集文件名集合（只处理这个目录中真实存在的图片）
    subset_files = set(os.listdir(subset_dir))
    subset_files = {fn for fn in subset_files if fn.lower().endswith((".jpg", ".jpeg", ".png"))}

    # 映射 image_id -> image meta；只保留文件在子集目录中的图片
    id2im = {im["id"]: im for im in coco["images"] if im.get("file_name") in subset_files}

    # 按 image_id 收集 anns
    ann_by_img = {img_id: [] for img_id in id2im.keys()}
    total_boxes = 0
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in id2im:
            continue
        if skip_crowd and ann.get("iscrowd", 0) == 1:
            continue
        ann_by_img[img_id].append(ann)
        total_boxes += 1

    tasks = []
    n_images = 0
    for img_id, im in id2im.items():
        # 文件必须确实存在于子集目录
        img_path = os.path.join(subset_dir, im["file_name"])
        if not os.path.exists(img_path):
            continue

        W, H = im["width"], im["height"]
        results = []
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]  # COCO像素坐标 [x,y,w,h]
            results.append({
                "id": f"coco-{ann['id']}",
                "from_name": rect_field_name,
                "to_name": image_field_name,
                "type": "rectanglelabels",
                "value": {
                    "x": 100.0 * x / W,
                    "y": 100.0 * y / H,
                    "width": 100.0 * w / W,
                    "height": 100.0 * h / H,
                    "rotation": 0,
                    "rectanglelabels": [rect_label_value],
                },
                "origin": "preannotation",
                "meta": {"category_id": ann.get("category_id")}
            })

        # 任务 id：默认用 COCO image_id；也可改为从文件名取数字
        task_id = int(os.path.splitext(im["file_name"])[0]) if id_from_filename else int(img_id)

        tasks.append({
            "id": task_id,
            "data": {"image": im["file_name"]},  # 只写文件名；由 Local Storage 映射
            "predictions": [{
                "result": results,
                "model_version": "coco2ls",
                "score": 1.0
            }],
            "meta": {
                "coco_image_id": int(img_id),
                "file_name": im["file_name"]
            }
        })

        n_images += 1
        if max_images is not None and n_images >= max_images:
            break

    with open(out_json_path, "w") as f:
        json.dump(tasks, f, indent=2)

    return n_images, total_boxes


if __name__ == "__main__":
    args = args_utils.get_args()
    # coco_to_labelstudio(args)
    n_imgs, n_boxes = build_labelstudio_subset_with_bboxes(
        coco_json_path=config.get_coco_path(args.remote) / "original" / "annotations" / "instances_val2017.json",
        subset_dir=config.get_coco_path(args.remote) / "selected" / "val2017",
        out_json_path=config.get_coco_path(args.remote) / "selected" / "annotations" / "labelstudio_import_subset_with_bbox.json",
        image_field_name="img",
        rect_field_name="obj",
        rect_label_value="object",
        max_images=None,
        skip_crowd=True,
        id_from_filename=False
    )
    print(f"Built {n_imgs} tasks with {n_boxes} boxes.")
