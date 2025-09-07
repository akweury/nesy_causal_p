# Created by MacBook Pro at 06.09.25

import os, json, requests
from urllib.parse import urlparse, parse_qs
from typing import Dict

import config

LS_URL = "http://localhost:8080"
API_TOKEN = "085ac6a42db23b04fb80eb6a9e99ffbce851e0fd"  # 在 User Menu -> Account -> Access Token 里获取
PROJECT_ID = 5  # 替换为你的项目ID
COCO_JSON = config.get_coco_path(False) / "original" / "annotations" / "instances_val2017.json"
SUBSET_DIR = config.get_coco_path(False) / "selected" / "val2017"

HEADERS = {"Authorization": f"Token {API_TOKEN}"}


def basename_from_data_image(val: str) -> str:
    """从 data.image 提取文件名：既兼容 local-files URL，也兼容纯文件名。"""
    if val.startswith("http") or val.startswith("/data/local-files"):
        # /data/local-files/?d=/abs/path/to/xxx.jpg
        # 或 http://.../data/local-files/?d=/abs/path/to/xxx.jpg
        path = parse_qs(urlparse(val).query).get("d", [""])[0]
        return os.path.basename(path)
    return os.path.basename(val)

def _basename_from_data_image(val: str) -> str:
    """兼容 local-files URL 或纯文件名，抽出 basename."""
    if not val:
        return ""
    if val.startswith("http") or val.startswith("/data/local-files"):
        # /data/local-files/?d=/abs/path/to/xxx.jpg
        qs = parse_qs(urlparse(val).query)
        path = (qs.get("d") or [""])[0]
        return os.path.basename(path)
    return os.path.basename(val)
def fetch_all_tasks(ls_url: str, project_id: int, page_size: int = 100) -> dict:
    """
    拉取项目内所有任务，返回 {filename: task_id} 映射。
    兼容两种响应格式：
      1) 直接返回 list[task]
      2) 返回 {"tasks": list[task], "total":..., "next":...}
    """
    mapping = {}
    page = 1
    while True:
        url = f"{ls_url}/api/projects/{project_id}/tasks"
        r = requests.get(url, params={"page": page, "page_size": page_size},
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        # 统一成列表
        if isinstance(data, list):
            tasks = data
            has_next = len(tasks) == page_size  # 粗略判断是否还有下一页
        else:
            tasks = data.get("tasks") or data.get("results") or []
            # 多数版本会给 next/previous
            has_next = bool(data.get("next"))

        if not tasks:
            break

        for t in tasks:
            img_val = (t.get("data") or {}).get("image", "")
            fn = _basename_from_data_image(img_val)
            if fn:
                mapping[fn] = t["id"]

        if not has_next:
            break
        page += 1

    return mapping


def load_coco_subset_boxes(coco_json: str, subset_dir: str):
    with open(coco_json, "r") as f:
        coco = json.load(f)
    subset_files = {fn for fn in os.listdir(subset_dir)
                    if fn.lower().endswith((".jpg", ".jpeg", ".png"))}
    id2im = {im["id"]: im for im in coco["images"] if im["file_name"] in subset_files}

    ann_by_img = {img_id: [] for img_id in id2im.keys()}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in id2im and ann.get("iscrowd", 0) == 0:
            ann_by_img[img_id].append(ann)
    return id2im, ann_by_img


def post_prediction(task_id: int, result):
    payload = {
        "task": task_id,
        "result": result,
        "score": 1.0,
        "model_version": "coco2ls"
    }
    r = requests.post(f"{LS_URL}/api/predictions", headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()


def main():
    # 1) 任务映射
    fn2task = fetch_all_tasks(LS_URL, PROJECT_ID)
    print(f"Found {len(fn2task)} existing tasks in project {PROJECT_ID}")

    # 2) COCO -> 每图的 bbox
    id2im, ann_by_img = load_coco_subset_boxes(COCO_JSON, SUBSET_DIR)

    attached, skipped = 0, 0
    for img_id, im in id2im.items():
        W, H = im["width"], im["height"]
        file_name = im["file_name"]
        task_id = fn2task.get(file_name)
        if not task_id:
            skipped += 1
            continue

        result = []
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            result.append({
                "id": f"coco-{ann['id']}",
                "from_name": "obj",
                "to_name": "img",
                "type": "rectanglelabels",
                "value": {
                    "x": 100.0 * x / W,
                    "y": 100.0 * y / H,
                    "width": 100.0 * w / W,
                    "height": 100.0 * h / H,
                    "rotation": 0,
                    "rectanglelabels": ["object"]
                },
                "origin": "preannotation",
                "meta": {"category_id": ann.get("category_id")}
            })

        if result:
            post_prediction(task_id, result)
            attached += 1

    print(f"Attached predictions to {attached} tasks; {skipped} skipped (no matching task).")

def convert_to_annotations():
    LS_URL = "http://localhost:8080"
    # API_TOKEN = "YOUR_TOKEN"
    PROJECT_ID = 5
    HEADERS = {"Authorization": f"Token {API_TOKEN}"}

    # 获取所有任务
    tasks = requests.get(f"{LS_URL}/api/projects/{PROJECT_ID}/tasks", headers=HEADERS).json()

    for t in tasks:
        tid = t["id"]
        preds = t.get("predictions") or []
        if not preds:
            continue
        result = preds[0]["result"]
        # 把预测写入 annotations
        r = requests.post(f"{LS_URL}/api/tasks/{tid}/annotations",
                          headers=HEADERS,
                          json={"result": result})
        print(f"Task {tid}: {r.status_code}")
if __name__ == "__main__":
    # main()
    convert_to_annotations()