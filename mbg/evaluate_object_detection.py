# Created by MacBook Pro at 16.04.25

import numpy as np
from tqdm import tqdm

import config
from src import bk

import torch

from mbg.patch_set_model import PatchSetClassifier


def match_objects(preds, gts, pos_thresh=0.05):
    matches = []
    used_gt = set()
    for p in preds:
        best_dist = float("inf")
        best_idx = -1
        for i, g in enumerate(gts):
            if i in used_gt:
                continue
            dist = np.linalg.norm([p["x"] - g["x"].numpy(), p["y"] - g["y"].numpy()])
            if dist < pos_thresh and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matches.append((p, gts[best_idx]))
            used_gt.add(best_idx)
    return matches

def compute_color_similarity(c1, c2):
    """
    Compute normalized color similarity between two RGB values.
    Inputs: c1, c2: (r, g, b) tuples or dicts with keys r/g/b
    Output: similarity score in [0, 1]
    """
    if isinstance(c1, dict):
        c1 = np.array([c1["color_r"], c1["color_g"], c1["color_b"]])
        c2 = np.array([int(c2["color_r"]), int(c2["color_g"]), int(c2["color_b"])])
    else:
        c1, c2 = np.array(c1), np.array(c2)

    dist = np.linalg.norm(c1 - c2)
    max_dist = np.linalg.norm([255, 255, 255])  # √(255² + 255² + 255²)
    similarity = 1 - (dist / max_dist)
    return similarity

def compute_size_similarity(pred_obj, gt_obj, shape, img_width):
    pred_area = pred_obj["size"] * (img_width ** 2)  # image area * normalized predicted size

    gt_size = gt_obj["size"]
    if bk.bk_shapes[shape] == "circle" or bk.bk_shapes[shape]  == "rectangle":
        gt_area = 0.36 * (img_width ** 2) * (gt_size ** 2)
    elif bk.bk_shapes[shape]  == "triangle":
        # Roughly estimated as proportional to size^2 like others
        gt_area = 0.35 * (img_width ** 2) * (gt_size ** 2)
    else:
        return 0.0  # unsupported shape

    # Compute similarity based on relative error
    rel_error = abs(pred_area - gt_area) / (gt_area + 1e-6)
    similarity = max(0.0, 1 - rel_error)  # clip to [0,1]
    return similarity


def compute_accuracy(matches, key, image_width=None):
    if key == "color_rgb":
        # Compute average similarity
        sim_scores = [
            compute_color_similarity(p, g)
            for p, g in matches
        ]
        return np.mean(sim_scores) if sim_scores else 0.0
    elif key == "size":
        sim_scores = [
            float(compute_size_similarity(p, g, shape=p["shape"], img_width=image_width))
            for p, g in matches
        ]
        return np.mean(sim_scores) if sim_scores else 0.0
    else:
        correct = sum(1 for p, g in matches if p[key] == g[key])
    return correct / len(matches) if matches else 0.0

def evaluate_symbolic_detection(dataloader, predictor):
    total_shape, total_color, total_size = 0, 0, 0
    total_detected = 0

    model = PatchSetClassifier(input_dim=3 * 5 * 2, hidden_dim=128, num_classes=3)
    model.load_state_dict(torch.load(config.mb_outlines/"patch_set_classifier.pt"))
    model.eval()

    for batch in tqdm(dataloader, desc="Evaluating"):
        for i in range(len(batch["image"])):
            image = batch["image"][i]
            gt_objects = batch["symbolic_data"]
            # Your GRM-based predictor for symbolic object detection
            pred_objects = predictor(image, model)  # List of dicts with same keys as gt_objects
            matches = match_objects(pred_objects, gt_objects)
            if not matches:
                continue
            shape_acc = compute_accuracy(matches, "shape")
            color_acc = compute_accuracy(matches, "color_rgb")
            size_acc = compute_accuracy(matches, "size", image.shape[-1])

            total_shape += shape_acc * len(matches)
            total_color += color_acc * len(matches)
            total_size += size_acc * len(matches)
            total_detected += len(matches)

    print("Symbolic Object Detection Accuracy:")
    print(f"  Shape Accuracy: {total_shape / total_detected:.3f}")
    print(f"  Color Accuracy: {total_color / total_detected:.3f}")
    print(f"  Size Accuracy : {total_size / total_detected:.3f}")


