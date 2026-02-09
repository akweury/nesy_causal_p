# Created by MacBook Pro at 05.09.25


import json, os
import shutil
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


def copy_corresponding_depth_files(
        original_depth_dir: str,
        selected_images_dir: str,
        target_depth_dir: str
) -> Tuple[int, int]:
    """
    Copy depth files that correspond to selected images from original depth_maps to target directory.
    
    Args:
        original_depth_dir: Path to original depth_maps directory
        selected_images_dir: Path to directory containing selected images
        target_depth_dir: Path to target directory where depth files should be copied
        depth_file_extension: Extension of depth files (default: .png)
        
    Returns:
        Tuple of (copied_files_count, missing_depth_files_count)
    """
    print(f"Copying depth files from {original_depth_dir} to {target_depth_dir}")
    print(f"Based on selected images in {selected_images_dir}")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_depth_dir, exist_ok=True)
    
    # Get list of selected image files
    if not os.path.exists(selected_images_dir):
        print(f"Selected images directory does not exist: {selected_images_dir}")
        return 0, 0
        
    if not os.path.exists(original_depth_dir):
        print(f"Original depth directory does not exist: {original_depth_dir}")
        return 0, 0
    
    selected_images = [f for f in os.listdir(selected_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    copied_count = 0
    missing_count = 0
    
    for image_file in selected_images:
        # Get base name without extension  
        base_name = os.path.splitext(image_file)[0]
        
        # Three corresponding depth files for each image
        depth_files = [
            base_name + "_depth.npz",
            base_name + "_depth.png", 
            base_name + "_conf.png"
        ]
        
        files_copied_for_image = 0
        files_missing_for_image = 0
        
        for depth_file_name in depth_files:
            original_depth_path = os.path.join(original_depth_dir, depth_file_name)
            target_depth_path = os.path.join(target_depth_dir, depth_file_name)
            
            if os.path.exists(original_depth_path):
                try:
                    shutil.copy2(original_depth_path, target_depth_path)
                    files_copied_for_image += 1
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {original_depth_path} to {target_depth_path}: {e}")
                    files_missing_for_image += 1
                    missing_count += 1
            else:
                files_missing_for_image += 1
                missing_count += 1
        
        # Progress indicator
        if (copied_count + missing_count) % 300 == 0:  # Every 100 images (3 files each)
            print(f"Processed {(copied_count + missing_count) // 3} images, copied {copied_count} depth files...")
        
        # Warn if not all 3 files were found for an image
        if files_missing_for_image > 0:
            print(f"Warning: Only {files_copied_for_image}/3 depth files found for {image_file}")
    
    print(f"Depth file copying completed: {copied_count} copied, {missing_count} missing")
    return copied_count, missing_count


def validate_and_clean_incomplete_samples(
        selected_images_dir: str,
        selected_depth_dir: str
) -> Tuple[int, int]:
    """
    Validate that each sample has all required files (RGB .jpg + 3 depth files).
    Remove any incomplete samples from the selected folders.
    
    Args:
        selected_images_dir: Path to directory containing selected RGB images
        selected_depth_dir: Path to directory containing selected depth files
        
    Returns:
        Tuple of (complete_samples_count, removed_incomplete_samples_count)
    """
    print(f"Validating complete samples in {selected_images_dir} and {selected_depth_dir}")
    
    if not os.path.exists(selected_images_dir):
        print(f"Selected images directory does not exist: {selected_images_dir}")
        return 0, 0
        
    if not os.path.exists(selected_depth_dir):
        print(f"Selected depth directory does not exist: {selected_depth_dir}")
        return 0, 0
    
    # Get all RGB images
    rgb_images = [f for f in os.listdir(selected_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    complete_samples = 0
    removed_samples = 0
    
    for image_file in rgb_images:
        # Get base name without extension  
        base_name = os.path.splitext(image_file)[0]
        
        # Check for all required files
        rgb_path = os.path.join(selected_images_dir, image_file)
        depth_files = [
            os.path.join(selected_depth_dir, base_name + "_depth.npz"),
            os.path.join(selected_depth_dir, base_name + "_depth.png"), 
            os.path.join(selected_depth_dir, base_name + "_conf.png")
        ]
        
        # Check if all files exist
        all_files_exist = os.path.exists(rgb_path) and all(os.path.exists(f) for f in depth_files)
        
        if all_files_exist:
            complete_samples += 1
        else:
            # Remove incomplete sample (all associated files)
            files_to_remove = [rgb_path] + depth_files
            removed_files_count = 0
            
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        removed_files_count += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
            
            print(f"Removed incomplete sample {base_name}: {removed_files_count} files deleted")
            removed_samples += 1
    
    print(f"Validation completed: {complete_samples} complete samples, {removed_samples} incomplete samples removed")
    return complete_samples, removed_samples


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
    
    # Copy corresponding depth files for selected images
    copied, missing = copy_corresponding_depth_files(
        original_depth_dir=config.get_coco_path(args.remote) / "original" / "depth_maps",
        selected_images_dir=config.get_coco_path(args.remote) / "selected" / "val2017",
        target_depth_dir=config.get_coco_path(args.remote) / "selected" / "depth_maps"
    )
    print(f"Depth files: {copied} copied, {missing} missing.")
    
    # Validate and clean incomplete samples
    complete, removed = validate_and_clean_incomplete_samples(
        selected_images_dir=config.get_coco_path(args.remote) / "selected" / "val2017",
        selected_depth_dir=config.get_coco_path(args.remote) / "selected" / "depth_maps"
    )
    print(f"Sample validation: {complete} complete samples, {removed} incomplete samples removed.")
