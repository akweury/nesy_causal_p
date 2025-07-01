import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Each box is in format [x_left, y_top, w, h] where (x_left, y_top) is top-left corner.
    """
    x1_left, y1_top, w1, h1 = box1
    x2_left, y2_top, w2, h2 = box2
    x1_min, y1_min = x1_left, y1_top
    x1_max, y1_max = x1_left + w1, y1_top + h1
    x2_min, y2_min = x2_left, y2_top
    x2_max, y2_max = x2_left + w2, y2_top + h2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compare_attributes(pred_obj, gt_obj):
    scores = {}
    if "shape" in pred_obj.get("s", {}) and "shape" in gt_obj:
        pred_shape = pred_obj["s"]["shape"]
        gt_shape = gt_obj["shape"]
        if isinstance(pred_shape, (list, np.ndarray)):
            pred_shape_idx = np.argmax(pred_shape)
        else:
            pred_shape_idx = pred_shape
        shape_map = {"triangle": 0, "square": 1, "circle": 2}
        if isinstance(gt_shape, str):
            gt_shape_idx = shape_map.get(gt_shape.lower(), -1)
        else:
            gt_shape_idx = gt_shape
        scores["shape"] = 1.0 if pred_shape_idx.argmax() == gt_shape_idx else 0.0
    if "color" in pred_obj.get("s", {}) and all(k in gt_obj for k in ["color_r", "color_g", "color_b"]):
        pred_color = pred_obj["s"]["color"]
        gt_color = [gt_obj["color_r"].item(), gt_obj["color_g"].item(), gt_obj["color_b"].item()]
        if isinstance(pred_color, (list, np.ndarray)):
            pred_color = np.array(pred_color)
            gt_color = np.array(gt_color)
            color_diff = np.linalg.norm(pred_color - gt_color)
            max_diff = np.sqrt(3)
            scores["color"] = max(0.0, 1.0 - (color_diff / max_diff))
        else:
            scores["color"] = 0.0
    return scores


def match_objects(pred_objects, gt_objects, iou_threshold=0.5):
    if not pred_objects or not gt_objects:
        return [], list(range(len(pred_objects))), list(range(len(gt_objects)))
    iou_matrix = np.zeros((len(pred_objects), len(gt_objects)))
    for i, pred_obj in enumerate(pred_objects):
        pred_s = pred_obj["s"]
        pred_box = [float(pred_s["x"]), float(pred_s["y"]), float(pred_s["w"]), float(pred_s["h"])]
        for j, gt_obj in enumerate(gt_objects):
            gt_x = float(gt_obj["x"])
            gt_y = float(gt_obj["y"])
            gt_w = float(gt_obj["w"])
            gt_h = float(gt_obj["h"])
            gt_box = [gt_x, gt_y, gt_w, gt_h]
            iou_matrix[i, j] = compute_iou(pred_box, gt_box)
    from scipy.optimize import linear_sum_assignment
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    matches = []
    matched_pred = set()
    matched_gt = set()
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if iou_matrix[p_idx, g_idx] >= iou_threshold:
            matches.append((p_idx, g_idx, iou_matrix[p_idx, g_idx]))
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
    unmatched_pred = [i for i in range(len(pred_objects)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_objects)) if i not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def compute_ap(precisions, recalls):
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def get_group_bounding_box(object_list, object_indices):
    if not object_indices:
        return None
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
    for idx in object_indices:
        if idx < len(object_list):
            obj = object_list[idx]
            if "s" in obj:
                x, y, w, h = obj["s"]["x"], obj["s"]["y"], obj["s"]["w"], obj["s"]["h"]
                x_mins.append(x)
                y_mins.append(y)
                x_maxs.append(x + w)
                y_maxs.append(y + h)
            else:
                x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
                x_mins.append(x)
                y_mins.append(y)
                x_maxs.append(x + w)
                y_maxs.append(y + h)
    if not x_mins:
        return None
    x_min = min(x_mins)
    y_min = min(y_mins)
    x_max = max(x_maxs)
    y_max = max(y_maxs)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def extract_ground_truth_groups(gt_objects):
    groups = {}
    for i, obj in enumerate(gt_objects):
        group_id = obj["group_id"].item()
        if group_id != -1:
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
    group_boxes = []
    group_info = []
    for group_id, object_indices in groups.items():
        bbox = get_group_bounding_box(gt_objects, object_indices)
        if bbox is not None:
            group_boxes.append(bbox)
            group_info.append({"group_id": group_id, "object_indices": object_indices, "bbox": bbox})
    return group_boxes, group_info


def extract_predicted_groups(pred_groups, pred_objects):
    pred_group_boxes = []
    pred_group_info = []
    for group in pred_groups:
        if "child_obj_ids" in group and group["child_obj_ids"]:
            bbox = get_group_bounding_box(pred_objects, group["child_obj_ids"])
            if bbox is not None:
                pred_group_boxes.append(bbox)
                pred_group_info.append({"child_obj_ids": group["child_obj_ids"], "bbox": bbox, "group_data": group})
    return pred_group_boxes, pred_group_info
