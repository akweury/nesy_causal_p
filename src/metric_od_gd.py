import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import wandb
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
import config
from mbg.scorer import scorer_config
from mbg import patch_preprocess
from mbg.group import eval_groups
from sklearn.metrics import accuracy_score, f1_score
from src.metric_od_gd_utils import compare_attributes, match_objects, compute_ap, extract_ground_truth_groups, \
    extract_predicted_groups


def main_metric():
    args = args_utils.get_args()
    import random
    random.seed(42)
    np.random.seed(42)
    all_principles = ["proximity", "similarity",
                      "closure", "continuity", "symmetry"]
    if hasattr(args, "principle") and args.principle:
        principles = [args.principle]
    else:
        principles = all_principles
    principles = all_principles

    wandb.init(project="gestalt_eval", config=args.__dict__)
    obj_model = eval_patch_classifier.load_model(args.device)
    results = run_all_principles(principles, args, obj_model)
    obj_metrics = {f"{m}_mean": float(np.mean(results["obj_detection"][m])) for m in results["obj_detection"]}
    obj_metrics.update({f"{m}_std": float(np.std(results["obj_detection"][m])) for m in results["obj_detection"]})
    group_metrics = {f"{m}_mean": float(np.mean(results["group_detection"][m])) for m in results["group_detection"]}
    group_metrics.update({f"{m}_std": float(np.std(results["group_detection"][m])) for m in results["group_detection"]})
    wandb.log({"overall/obj": obj_metrics, "overall/group": group_metrics})
    save_results = build_save_results(results, obj_metrics, group_metrics, principles)
    with open(config.output / "od_gd_evaluation_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)
    wandb.save("od_gd_evaluation_results.json")
    return save_results


def run_all_principles(principles, args, obj_model):
    results = {
        "obj_detection": {k: [] for k in ["mAP", "precision", "recall", "f1", "shape_accuracy", "color_accuracy",
                                          "size_accuracy", "count_accuracy"]},
        "group_detection": {k: [] for k in ["mAP", "precision", "recall", "f1", "binary_accuracy", "binary_f1",
                                            "group_count_accuracy", "group_obj_num_accuracy"]},
        "per_principle": {p: {"obj_mAP": [], "obj_precision": [], "obj_recall": [], "obj_f1": [], "obj_shape_acc": [],
                              "obj_color_acc": [], "group_mAP": [], "group_precision": [], "group_recall": [],
                              "group_f1": [], "group_acc": [], "group_binary_f1": []
                              } for p in principles}
    }
    for principle in principles:
        run_one_principle(principle, args, obj_model, results)
    return results


def run_one_principle(principle, args, obj_model, results):
    group_model = scorer_config.load_scorer_model(principle, args.device)
    principle_path = getattr(config, f"grb_{principle}")
    combined_loader = dataset.load_combined_dataset(principle_path, task_num=args.top_data)
    obj_scores = {k: [] for k in ["mAP", "precision",
                                  "recall", "f1", "shape_accuracy", "color_accuracy", "size_accuracy", "count_accuracy"]}
    group_scores = {k: [] for k in ["mAP", "precision", "recall", "f1", "acc", "binary_f1",
                                    "group_count_accuracy", "group_obj_num_accuracy"]}

    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        print(f"\nRunning principle: {principle}, Task {task_idx + 1}/{len(combined_loader)}")
        run_one_task(principle, task_idx, train_data, args, obj_model, group_model, obj_scores, group_scores)
    for metric in obj_scores:
        results["per_principle"][principle][f"obj_{metric}"] = obj_scores[metric]
    results["per_principle"][principle]["group_mAP"] = group_scores["mAP"]
    results["per_principle"][principle]["group_precision"] = group_scores["precision"]
    results["per_principle"][principle]["group_recall"] = group_scores["recall"]
    results["per_principle"][principle]["group_f1"] = group_scores["f1"]
    results["per_principle"][principle]["group_acc"] = group_scores["acc"]
    results["per_principle"][principle]["group_binary_f1"] = group_scores["binary_f1"]
    for metric in obj_scores:
        results["obj_detection"][metric].extend(obj_scores[metric])
    for metric in ["mAP", "precision", "recall", "f1"]:
        results["group_detection"][metric].extend(group_scores[metric])
    results["group_detection"]["binary_accuracy"].extend(group_scores["acc"])
    results["group_detection"]["binary_f1"].extend(group_scores["binary_f1"])
    results["group_detection"]["group_count_accuracy"].extend(group_scores["group_count_accuracy"])
    results["group_detection"]["group_obj_num_accuracy"].extend(group_scores["group_obj_num_accuracy"])

    wandb.log({f"{principle}/summary": {
        **{f"obj_{k}_mean": float(np.mean(obj_scores[k])) for k in obj_scores},
        **{f"group_{k}_mean": float(np.mean(group_scores[k])) for k in group_scores}
    }})


def update_gt_objects(gt_objects):
    for i in range(len(gt_objects)):
        for o_i in range(len(gt_objects[i])):
            gt_objects[i][o_i]['x'] = gt_objects[i][o_i]['x'] - gt_objects[i][o_i]['size'] / 2
            gt_objects[i][o_i]['y'] = gt_objects[i][o_i]['y'] - gt_objects[i][o_i]['size'] / 2
            gt_objects[i][o_i]['w'] = gt_objects[i][o_i]['size']
            gt_objects[i][o_i]['h'] = gt_objects[i][o_i]['size']
    return gt_objects


def run_one_task(principle, task_idx, train_data, args, obj_model, group_model, obj_scores, group_scores):
    hyp_params = {"prox": 0.9, "sim": 0.5, "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    obj_lists, groups_list = [], []
    train_val_data = train_data
    all_data = train_val_data["positive"] + train_val_data["negative"]
    img_paths = [d["image_path"][0] for d in all_data]
    imgs = patch_preprocess.load_images_fast(img_paths, device=args.device)
    gt_objects = [[obj for obj in d["symbolic_data"]] for d in all_data]
    gt_objects = update_gt_objects(gt_objects)
    for img_idx, img in enumerate(imgs):
        objs = eval_patch_classifier.evaluate_image(obj_model, img, args.device)
        obj_lists.append(objs)
        groups = eval_groups.eval_groups(objs, group_model, principle, args.device, dim=hyp_params["patch_dim"])
        groups_list.append(groups)
        # Debug: plot group bounding boxes
        gt_group_boxes, _ = extract_ground_truth_groups(gt_objects[img_idx])
        pred_group_boxes, _ = extract_predicted_groups(groups, objs)
    obj_metrics = evaluate_object_detection(obj_lists, gt_objects)
    for k in obj_scores:
        obj_scores[k].append(obj_metrics[k])
    group_metrics = evaluate_group_detection(groups_list, gt_objects, obj_lists)
    group_scores["acc"].append(group_metrics["binary_accuracy"])
    group_scores["binary_f1"].append(group_metrics["binary_f1"])
    group_scores["group_count_accuracy"].append(group_metrics["group_count_accuracy"])
    group_scores["group_obj_num_accuracy"].append(group_metrics["group_obj_num_accuracy"])
    for k in ["mAP", "precision", "recall", "f1"]:
        if k not in group_scores:
            group_scores[k] = []
        group_scores[k].append(group_metrics[k])
    # wandb.log({f"{principle}/task_{task_idx+1}_obj": obj_metrics, f"{principle}/task_{task_idx+1}_group": group_metrics})


def build_save_results(results, obj_metrics, group_metrics, principles):
    save_results = {
        "overall": {
            "object_detection": {m: {"mean": obj_metrics[f"{m}_mean"], "std": obj_metrics[f"{m}_std"]} for m in
                                 ["mAP", "precision", "recall", "f1", "shape_accuracy", "color_accuracy",
                                  "size_accuracy", "count_accuracy"]},
            "group_detection": {m: {"mean": group_metrics[f"{m}_mean"], "std": group_metrics[f"{m}_std"]} for m in
                                ["mAP", "precision", "recall", "f1", "binary_accuracy", "binary_f1",
                                 "group_count_accuracy", "group_obj_num_accuracy"]}
        },
        "per_principle": {}
    }
    for principle in principles:
        p = results["per_principle"][principle]
        save_results["per_principle"][principle] = {
            "object_detection": {k: {"mean": float(np.mean(p[f"obj_{k}"])), "std": float(np.std(p[f"obj_{k}"]))} for k
                                 in ["mAP", "precision", "recall", "f1", "shape_accuracy", "color_accuracy"]},
            "group_detection": {k: {"mean": float(np.mean(p[f"group_{k}"])), "std": float(np.std(p[f"group_{k}"]))} for
                                k in ["mAP", "precision", "recall", "f1", "acc", "binary_f1"]}
        }
    return save_results


def evaluate_object_detection(obj_lists, ground_truth_objects, iou_threshold=0.5):
    all_matches = []
    all_pred_scores = []
    all_gt_count = 0
    all_pred_count = 0
    attribute_scores = {"shape": [], "color": [], "size": []}
    count_scores = []

    for obj_list, gt_objs in zip(obj_lists, ground_truth_objects):
        if not gt_objs:
            continue
        all_gt_count += len(gt_objs)
        all_pred_count += len(obj_list)
        count_scores.append(1.0 if len(obj_list) == len(gt_objs) else 0.0)
        matches, unmatched_pred, unmatched_gt = match_objects(obj_list, gt_objs, iou_threshold)
        for pred_idx, gt_idx, iou_score in matches:
            attr_scores = compare_attributes(obj_list[pred_idx], gt_objs[gt_idx])
            for attr, score in attr_scores.items():
                attribute_scores[attr].append(score)
        for pred_idx, gt_idx, iou_score in matches:
            all_matches.append(1)
            all_pred_scores.append(1.0)
        for pred_idx in unmatched_pred:
            all_matches.append(0)
            all_pred_scores.append(1.0)
    if not all_matches:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "shape_accuracy": 0.0, "color_accuracy": 0.0, "size_accuracy": 0.0, "count_accuracy": 0.0}
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    sorted_matches = np.array(all_matches)[sorted_indices]
    tp_cumsum = np.cumsum(sorted_matches)
    fp_cumsum = np.cumsum(1 - sorted_matches)
    recalls = tp_cumsum / all_gt_count if all_gt_count > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    mAP = compute_ap(precisions, recalls)
    total_tp = np.sum(sorted_matches)
    precision = total_tp / len(sorted_matches) if len(sorted_matches) > 0 else 0.0
    recall = total_tp / all_gt_count if all_gt_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    shape_acc = np.mean(attribute_scores["shape"]) if attribute_scores["shape"] else 0.0
    color_acc = np.mean(attribute_scores["color"]) if attribute_scores["color"] else 0.0
    size_acc = np.mean(attribute_scores["size"]) if attribute_scores["size"] else 0.0
    count_acc = np.mean(count_scores) if count_scores else 0.0

    return {"mAP": mAP, "precision": precision, "recall": recall, "f1": f1, "shape_accuracy": shape_acc,
            "color_accuracy": color_acc, "size_accuracy": size_acc, "count_accuracy": count_acc}


def evaluate_group_detection(groups_list, gt_objects_list, obj_lists, iou_threshold=0.5):
    all_matches = []
    all_pred_scores = []
    all_gt_count = 0
    all_pred_count = 0
    y_true_binary = []
    y_pred_binary = []
    group_count_accs = []
    group_obj_num_accs = []

    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    for groups, gt_objects, pred_objects in zip(groups_list, gt_objects_list, obj_lists):
        gt_group_boxes, gt_group_info = extract_ground_truth_groups(gt_objects)
        pred_group_boxes, pred_group_info = extract_predicted_groups(groups, pred_objects)
        all_gt_count += len(gt_group_boxes)
        all_pred_count += len(pred_group_boxes)
        has_gt_groups = len(gt_group_boxes) > 0
        has_pred_groups = len(pred_group_boxes) > 0
        y_true_binary.append(int(has_gt_groups))
        y_pred_binary.append(int(has_pred_groups))

        # Group-level count accuracy: match predicted groups to gt groups by IoU
        matched_gt = set()
        correct_count = 0
        total_matched = 0

        for pred_idx, pred_group in enumerate(pred_group_info):
            best_iou = 0
            best_gt_idx = -1
            pred_box = pred_group_boxes[pred_idx]
            for gt_idx, gt_group in enumerate(gt_group_info):
                if gt_idx in matched_gt:
                    continue
                gt_box = gt_group_boxes[gt_idx]
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                pred_count = len(pred_group)
                gt_count = len(gt_group_info[best_gt_idx])
                if pred_count == gt_count:
                    correct_count += 1
                total_matched += 1
                gt_len = len(gt_group_info[best_gt_idx]["object_indices"])
                pred_len = len(pred_group["child_obj_ids"])
                group_obj_num_accs.append(1.0 if gt_len == pred_len else 0.0)
            else:
                group_obj_num_accs.append(0.0)
        group_count_accs.append(correct_count / total_matched if total_matched > 0 else 0.0)

        # Detection metrics
        if gt_group_boxes and pred_group_boxes:
            matches, unmatched_pred, unmatched_gt = match_objects(
                [{"s": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}} for box in pred_group_boxes],
                [{"x": box[0], "y": box[1], "w": box[2], "h": box[3]} for box in gt_group_boxes],
                iou_threshold
            )
            for pred_idx, gt_idx, iou_score in matches:
                all_matches.append(1)
                all_pred_scores.append(iou_score)
            for pred_idx in unmatched_pred:
                all_matches.append(0)
                all_pred_scores.append(1.0)
        elif pred_group_boxes:
            for _ in pred_group_boxes:
                all_matches.append(0)
                all_pred_scores.append(1.0)

    if all_matches:
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        sorted_matches = np.array(all_matches)[sorted_indices]
        tp_cumsum = np.cumsum(sorted_matches)
        fp_cumsum = np.cumsum(1 - sorted_matches)
        recalls = tp_cumsum / all_gt_count if all_gt_count > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        mAP = compute_ap(precisions, recalls)
        total_tp = np.sum(sorted_matches)
        precision = total_tp / len(sorted_matches) if len(sorted_matches) > 0 else 0.0
        recall = total_tp / all_gt_count if all_gt_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        mAP = precision = recall = f1 = 0.0

    binary_acc = accuracy_score(y_true_binary, y_pred_binary) if y_true_binary else 0.0
    binary_f1 = f1_score(y_true_binary, y_pred_binary) if y_true_binary else 0.0
    group_count_acc = np.mean(group_count_accs) if group_count_accs else 0.0

    return {
        "mAP": mAP,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "binary_accuracy": binary_acc,
        "binary_f1": binary_f1,
        "group_count_accuracy": group_count_acc,
        "group_obj_num_accuracy": np.mean(group_obj_num_accs) if group_obj_num_accs else 0.0
    }


def visualize_bounding_boxes(img_tensor, pred_objects, gt_objects, img_name="image",
                             save_path="bbox_visualization.png"):
    """
    Visualize ground truth and predicted bounding boxes on the image.

    Args:
        img_tensor: Image tensor (C, H, W) or (H, W, C)
        pred_objects: List of predicted objects with 's' containing x, y, w, h
        gt_objects: List of ground truth objects with x, y, size
        img_name: Name for the image title
        save_path: Path to save the visualization
    """
    # Convert tensor to numpy and handle different formats
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor

    # Handle different tensor formats (C,H,W) vs (H,W,C)
    if img_np.ndim == 3:
        if img_np.shape[0] == 3 or img_np.shape[0] == 1:  # (C,H,W)
            img_np = np.transpose(img_np, (1, 2, 0))
        # If already (H,W,C), keep as is

    # Normalize to [0,1] if needed
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    # Handle grayscale
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_np)
    ax.set_title(f"{img_name} - GT (Red) vs Predicted (Blue)")

    img_height, img_width = img_np.shape[:2]

    # Draw ground truth boxes in red
    for gt_obj in gt_objects:
        if all(k in gt_obj for k in ["x", "y", "size"]):
            # Convert normalized coordinates to pixel coordinates if needed
            x = float(gt_obj["x"])
            y = float(gt_obj["y"])
            size = float(gt_obj["size"])

            # If coordinates are normalized (0-1), convert to pixels
            if x <= 1.0 and y <= 1.0:
                x *= img_width
                y *= img_height
                size *= min(img_width, img_height)

            # Convert center coordinates to corner coordinates
            x_corner = x - size / 2
            y_corner = y - size / 2

            # Create rectangle patch
            rect = patches.Rectangle(
                (x_corner, y_corner), size, size,
                linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)

            # Add label
            ax.text(x_corner, y_corner - 5, f'GT: {gt_obj.get("shape", "?")}', color='red', fontsize=8,
                    fontweight='bold')

    # Draw predicted boxes in blue
    for pred_obj in pred_objects:
        if "s" in pred_obj and all(k in pred_obj["s"] for k in ["x", "y", "w", "h"]):
            pred_s = pred_obj["s"]
            x = float(pred_s["x"])
            y = float(pred_s["y"])
            w = float(pred_s["w"])
            h = float(pred_s["h"])

            # If coordinates are normalized (0-1), convert to pixels
            if x <= 1.0 and y <= 1.0:
                x *= img_width
                y *= img_height
                w *= img_width
                h *= img_height

            # Convert center coordinates to corner coordinates
            x_corner = x
            y_corner = y

            # Create rectangle patch
            rect = patches.Rectangle((x_corner, y_corner), w, h, linewidth=2, edgecolor='blue', facecolor='none',
                                     alpha=0.8)
            ax.add_patch(rect)

            # Add label with shape if available
            shape_text = "?"
            if "shape" in pred_s:
                shape = pred_s["shape"]
                if isinstance(shape, (list, np.ndarray, torch.Tensor)):
                    shape_idx = np.argmax(shape)
                    shape_names = ["tri", "sq", "cir"]
                    shape_text = shape_names[shape_idx] if shape_idx < len(shape_names) else str(shape_idx)
                else:
                    shape_text = str(shape)

            ax.text(x_corner, y_corner - 5, f'Pred: {shape_text}', color='blue', fontsize=8, fontweight='bold')

    # Add legend
    red_patch = patches.Patch(color='red', label='Ground Truth')
    blue_patch = patches.Patch(color='blue', label='Predicted')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')

    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
    ax.axis('off')

    # Save the visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_group_bboxes(img, gt_group_boxes, pred_group_boxes, img_name, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img
    if img_np.ndim == 3:
        if img_np.shape[0] == 3 or img_np.shape[0] == 1:
            img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_np)
    img_height, img_width = img_np.shape[:2]
    for box in gt_group_boxes:
        x, y, w, h = box
        if x <= 1.0 and y <= 1.0:
            x *= img_width
            y *= img_height
            w *= img_width
            h *= img_height
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    for box in pred_group_boxes:
        x, y, w, h = box
        if x <= 1.0 and y <= 1.0:
            x *= img_width
            y *= img_height
            w *= img_width
            h *= img_height
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    ax.set_title(f'{img_name} - GT(red) vs Pred(blue) group boxes')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main_metric()
