import torch.nn as nn
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import cv2
import argparse
from torch.utils.data import random_split
import random

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

import matplotlib.pyplot as plt


class CocoGroupAnnotatedDataset(Dataset):
    def __init__(self, coco_json, depth_dir, conf_dir, group_annotations_file, image_limit=None):
        """Dataset that loads images with group annotations."""

        # Load COCO data
        with open(coco_json) as f:
            coco = json.load(f)

        self.all_images = coco["images"]
        self.all_annotations = coco["annotations"]
        self.depth_dir = depth_dir
        self.conf_dir = conf_dir

        # Load group annotations
        if os.path.exists(group_annotations_file):
            with open(group_annotations_file) as f:
                self.group_annotations = json.load(f)
            print(f"Loaded {len(self.group_annotations)} group annotations")
        else:
            print(f"Group annotations file not found: {group_annotations_file}")
            self.group_annotations = {}

        # Filter images to only those with group annotations
        annotated_img_ids = set(int(img_id) for img_id in self.group_annotations.keys())
        self.images = [img for img in self.all_images if img['id'] in annotated_img_ids]

        if image_limit:
            self.images = self.images[:image_limit]

        print(f"Using {len(self.images)} images with group annotations (out of {len(self.all_images)} total)")

        # Group COCO annotations by image ID
        self.imgid_to_anns = {}
        for ann in self.all_annotations:
            if ann.get("iscrowd", 0) == 1:
                continue
            self.imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def _load_depth(self, file_name):
        stem = file_name.replace(".jpg", "")
        depth = np.load(os.path.join(
            self.depth_dir, f"{stem}_depth.npz"))["depth"]
        conf = np.array(
            __import__("cv2").imread(
                os.path.join(self.conf_dir, f"{stem}_conf.png"),
                __import__("cv2").IMREAD_UNCHANGED
            ),
            dtype=np.float32
        )
        return depth, conf

    def _pool_object_depth(self, depth, conf, bbox):
        x, y, w, h = map(int, bbox)
        patch_d = depth[y:y+h, x:x+w]
        patch_c = conf[y:y+h, x:x+w]

        if patch_d.size == 0:
            return 0.0

        weights = patch_c.flatten() + 1e-6
        values = patch_d.flatten()
        return np.median(values)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_id = img["id"]
        anns = self.imgid_to_anns.get(img_id, [])

        # Load group annotation for this image
        group_annotation = self.group_annotations.get(str(img_id))
        if not group_annotation:
            # This shouldn't happen since we filtered, but handle gracefully
            return {"objects": [], "groups": [], "image_id": img_id}

        depth, conf = self._load_depth(img["file_name"])

        objects = []
        img_width, img_height = img["width"], img["height"]
        for i, ann in enumerate(anns):
            d = self._pool_object_depth(depth, conf, ann["bbox"])
            x, y, w, h = ann["bbox"]

            # Normalize x,y,w,h by image dimensions
            x_norm = x / float(img_width)
            y_norm = y / float(img_height)
            w_norm = w / float(img_width)
            h_norm = h / float(img_height)

            objects.append({
                # feature: normalized position and size plus depth
                "feat": torch.tensor([x_norm, y_norm, w_norm, h_norm, d], dtype=torch.float32),
                # keep original pixel bbox for visualization
                "bbox": [int(x), int(y), int(w), int(h)],
                "ann_id": ann["id"],
                "category_id": ann["category_id"]
            })

        # Get ground truth groups
        gt_groups = group_annotation["groups"]

        return {
            "objects": objects,
            "groups": gt_groups,
            "image_id": img_id,
            "file_name": img["file_name"]
        }


def perceptual_distance(o1, o2, lambda_depth=0.2):
    dx = o1[0] - o2[0]
    dy = o1[1] - o2[1]
    d1, d2 = o1[4], o2[4]

    # depth-normalized image distance
    dxy = torch.sqrt(dx**2 + dy**2) / (min(d1, d2) + 1e-6)

    # soft depth difference (acts as gating, not metric)
    dd = torch.abs(d1 - d2) / (max(d1, d2) + 1e-6)

    return dxy + lambda_depth * dd


def create_ground_truth_labels(groups, num_objects):
    """Create pairwise ground truth labels from group annotations."""
    labels = torch.zeros(num_objects, num_objects)
    
    # Set diagonal to 1 (object is always grouped with itself)
    labels.fill_diagonal_(1.0)
    
    # Set pairs in the same group to 1
    for group in groups:
        for i in group:
            for j in group:
                if i < num_objects and j < num_objects:
                    labels[i, j] = 1.0
    
    return labels


class PairwiseGroupScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, o1, o2):
        x = torch.cat([o1, o2], dim=-1)
        return self.net(x)



# --- Training: train_one_epoch helper ---
def train_one_epoch(model, dataset, opt, loss_fn, device="cpu"):
    model.train()
    total_loss, count = 0.0, 0

    for batch in dataset:
        objects = batch["objects"]
        groups = batch["groups"]
        if len(objects) == 0:
            continue

        feats = [o["feat"].to(device) for o in objects]
        num_objects = len(feats)
        gt_labels = create_ground_truth_labels(groups, num_objects).to(device)

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                gt_label = gt_labels[i, j]
                pred_score = model(feats[i], feats[j]).squeeze()
                loss = loss_fn(pred_score, gt_label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                count += 1

    return total_loss / max(count, 1)


# --- Evaluation: loss + accuracy on a dataset ---
def evaluate_loss_and_accuracy(model, dataset, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset:
            objects = batch["objects"]
            groups = batch["groups"]
            if len(objects) == 0:
                continue

            feats = [o["feat"].to(device) for o in objects]
            num_objects = len(feats)
            gt_labels = create_ground_truth_labels(groups, num_objects).to(device)

            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    gt = gt_labels[i, j]
                    pred = model(feats[i], feats[j]).squeeze()
                    total_loss += loss_fn(pred, gt).item()
                    pred_bin = (pred > 0.5).float()
                    correct += (pred_bin == gt).float().item()
                    total += 1

    return total_loss / max(total, 1), correct / max(total, 1)


# --- Training loop: report TEST metrics per epoch ---
def train_with_ground_truth(
    model, train_dataset, test_dataset, epochs=5, device="cpu"
):
    """
    Train on train_dataset, but report loss/accuracy on test_dataset each epoch.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_dataset, opt, loss_fn, device=device
        )

        # --- evaluate on TEST split ---
        test_loss, test_acc = evaluate_loss_and_accuracy(
            model, test_dataset, loss_fn, device=device
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"test-loss={test_loss:.4f} | test-pair-acc={test_acc:.3f}"
        )


def print_sample_predictions(model, dataset, device, max_samples=2):
    """Print sample predictions vs ground truth for debugging."""
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataset:
            if sample_count >= max_samples:
                break
                
            objects = batch["objects"]
            groups = batch["groups"]
            file_name = batch["file_name"]
            
            if len(objects) == 0:
                continue
                
            print(f"\n  Sample: {file_name}")
            print(f"  Ground truth groups: {groups}")
            
            feats = [o["feat"].to(device) for o in objects]
            num_objects = len(feats)
            gt_labels = create_ground_truth_labels(groups, num_objects)
            
            print("  Predictions vs Ground Truth:")
            for i in range(min(num_objects, 4)):  # Limit output
                for j in range(i + 1, min(num_objects, 4)):
                    pred_score = model(feats[i], feats[j]).item()
                    gt_label = gt_labels[i, j].item()
                    print(f"    Pair ({i},{j}): Pred={pred_score:.3f}, GT={gt_label:.0f}")
            
            sample_count += 1
    
    model.train()


def evaluate_model(model, dataset, device="cpu"):
    """Evaluate model performance on the dataset."""
    model.eval()
    total_correct = 0
    total_predictions = 0
    group_accuracies = []
    
    with torch.no_grad():
        for batch in dataset:
            objects = batch["objects"]
            groups = batch["groups"]
            
            if len(objects) == 0:
                continue
                
            feats = [o["feat"].to(device) for o in objects]
            num_objects = len(feats)
            gt_labels = create_ground_truth_labels(groups, num_objects)
            
            # Calculate predictions for all pairs
            correct = 0
            total = 0
            
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    pred_score = model(feats[i], feats[j]).item()
                    gt_label = gt_labels[i, j].item()
                    
                    predicted = 1.0 if pred_score > 0.5 else 0.0
                    if predicted == gt_label:
                        correct += 1
                        total_correct += 1
                    total += 1
                    total_predictions += 1
            
            if total > 0:
                group_accuracies.append(correct / total)
    
    overall_accuracy = total_correct / max(total_predictions, 1)
    avg_per_image_accuracy = np.mean(group_accuracies) if group_accuracies else 0
    print(
        f"[Test ] pair-acc={overall_accuracy:.3f} | "
        f"per-image={avg_per_image_accuracy:.3f} | "
        f"images={len(group_accuracies)}"
    )
    return overall_accuracy, avg_per_image_accuracy


# --- Threshold Estimation ---
def estimate_pairwise_threshold(model, dataset, device="cpu", quantile=0.8, max_images=20):
    """
    Estimate a global threshold from predicted pairwise scores.
    Uses the given quantile over all pairwise scores from a subset of images.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for idx, batch in enumerate(dataset):
            if idx >= max_images:
                break
            objects = batch["objects"]
            if len(objects) < 2:
                continue
            feats = [o["feat"].to(device) for o in objects]
            n = len(feats)
            for i in range(n):
                for j in range(i + 1, n):
                    s = model(feats[i], feats[j]).item()
                    scores.append(s)
    if len(scores) == 0:
        return 0.5
    return float(np.quantile(np.array(scores), quantile))


def visualize_pairwise_heatmap(model, dataset, image_idx=0, device="cpu", threshold=0.5):
    """Visualize pairwise scores vs ground truth for a specific image and then show grouping on the image."""
    model.eval()

    # Get the specific batch
    batch = dataset[image_idx]
    objects = batch["objects"]
    groups = batch["groups"]
    file_name = batch.get("file_name", "")

    feats = [o["feat"].to(device) for o in objects]
    n = len(feats)

    if n == 0:
        print("No objects to visualize.")
        return

    # Get predictions
    pred_scores = torch.zeros((n, n))
    gt_labels = create_ground_truth_labels(groups, n)

    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                if i == j:
                    pred_scores[i, j] = 1.0
                else:
                    pred_scores[i, j] = model(feats[i], feats[j]).item()

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted scores
    im1 = ax1.imshow(pred_scores.cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
    ax1.set_title(f"Predicted Scores\n{file_name}")
    ax1.set_xlabel("Object index")
    ax1.set_ylabel("Object index")
    plt.colorbar(im1, ax=ax1)

    # Ground truth
    im2 = ax2.imshow(gt_labels.cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
    ax2.set_title(f"Ground Truth\nGroups: {groups}")
    ax2.set_xlabel("Object index")
    ax2.set_ylabel("Object index")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    # Additionally visualize groups on the original image
    try:
        visualize_group_graph(model, dataset, image_idx=image_idx, device=device, threshold=threshold)
    except Exception:
        pass


def draw_groups_on_axis(ax, img, objects, clusters, title):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    cmap = plt.get_cmap("tab20")

    # draw object boxes
    for i, o in enumerate(objects):
        bx, by, bw, bh = o.get("bbox", [0, 0, 0, 0])
        rect = plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        cx, cy = int(bx + bw / 2), int(by + bh / 2)
        ax.text(cx, cy, str(i), color="white", fontsize=9, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))

    # draw group bounding boxes
    for gi, members in enumerate(clusters):
        color = cmap((gi % 20) / 20.0)
        xs, ys, xws, yhs = [], [], [], []
        for m in members:
            bx, by, bw, bh = objects[m].get("bbox", [0, 0, 0, 0])
            xs.append(bx); ys.append(by)
            xws.append(bx + bw); yhs.append(by + bh)

        if not xs:
            continue

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xws), max(yhs)

        ax.add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=color,
                linewidth=3,
                alpha=0.9,
            )
        )
        ax.text(
            x_min + 4,
            y_min + 12,
            f"G{gi}",
            color="white",
            fontsize=12,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.8),
        )

# ---- Ground-truth group-only visualization (no object boxes, no indices) ----
def draw_gt_groups_only(ax, img, objects, gt_clusters, title):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    cmap = plt.get_cmap("tab20")

    for gi, members in enumerate(gt_clusters):
        color = cmap((gi % 20) / 20.0)
        xs, ys, xws, yhs = [], [], [], []
        for m in members:
            bx, by, bw, bh = objects[m].get("bbox", [0, 0, 0, 0])
            xs.append(bx); ys.append(by)
            xws.append(bx + bw); yhs.append(by + bh)

        if not xs:
            continue

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xws), max(yhs)

        ax.add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=color,
                linewidth=4,
                alpha=0.95,
            )
        )

        ax.text(
            x_min + 4,
            y_min + 16,
            f"GT-G{gi}",
            color="white",
            fontsize=13,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.85),
        )


def resolve_subset(dataset, image_idx):
    """
    Resolve a torch.utils.data.Subset to its base dataset and original index.
    Returns (base_dataset, base_index).
    """
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return dataset.dataset, dataset.indices[image_idx]
    return dataset, image_idx



def visualize_group_graph(
    model, dataset, image_idx=0, device="cpu", threshold=0.5,
    title_suffix="", save_path=None, show=True
):
    """
    Visualize predicted grouping (left) and ground-truth grouping (right) side by side on the original image.
    """
    model.eval()

    batch = dataset[image_idx]
    objects = batch["objects"]
    gt_groups = batch.get("groups", [])
    file_name = batch.get("file_name", None)

    if len(objects) == 0:
        print("No objects to visualize.")
        return

    n = len(objects)

    # ---------- predicted grouping ----------
    feats = [o["feat"].to(device) for o in objects]
    scores = np.zeros((n, n), dtype=float)
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                s = model(feats[i], feats[j]).item()
                scores[i, j] = s
                scores[j, i] = s

    # Rule: SAME GROUP iff score >= threshold
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if scores[i, j] >= threshold:
                union(i, j)

    pred_groups_dict = {}
    for i in range(n):
        r = find(i)
        pred_groups_dict.setdefault(r, []).append(i)

    pred_clusters = list(pred_groups_dict.values())

    # ---------- ground-truth grouping ----------
    gt_clusters = [g for g in gt_groups if len(g) > 0]

    # ---------- load image ----------
    base_dataset, base_idx = resolve_subset(dataset, image_idx)
    img_info = base_dataset.images[base_idx]
    img_path = Path(config.get_coco_path()) / "selected" / "val2017" / img_info["file_name"]
    if not img_path.exists():
        print(f"Image file not found: {img_path}")
        return

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- side-by-side visualization ----------
    fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(20, 10))

    draw_groups_on_axis(
        ax_pred, img, objects, pred_clusters,
        title=f"Prediction (thr={threshold:.2f}) {title_suffix}"
    )

    draw_gt_groups_only(
        ax_gt, img, objects, gt_clusters,
        title="Ground Truth Grouping"
    )

    print(
        f"[Vis] img={image_idx} | pred-groups={len(pred_clusters)} | gt-groups={len(gt_clusters)}"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    model.train()

# ---- Multi-example grid visualization ----
def visualize_group_graph_grid(
    model, dataset, image_indices, device="cpu", threshold=0.5,
    title_suffix="", save_path=None, show=True
):
    """
    Visualize multiple examples in a grid.
    Each row: [Prediction | Ground Truth]
    """
    model.eval()

    num_rows = len(image_indices)
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 5 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, image_idx in enumerate(image_indices):
        batch = dataset[image_idx]
        objects = batch["objects"]
        gt_groups = batch.get("groups", [])

        if len(objects) == 0:
            continue

        n = len(objects)

        # ---------- predicted grouping ----------
        feats = [o["feat"].to(device) for o in objects]
        scores = np.zeros((n, n), dtype=float)
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    s = model(feats[i], feats[j]).item()
                    scores[i, j] = s
                    scores[j, i] = s

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if scores[i, j] >= threshold:
                    union(i, j)

        pred_groups_dict = {}
        for i in range(n):
            r = find(i)
            pred_groups_dict.setdefault(r, []).append(i)

        pred_clusters = list(pred_groups_dict.values())
        gt_clusters = [g for g in gt_groups if len(g) > 0]

        # ---------- load image ----------
        base_dataset, base_idx = resolve_subset(dataset, image_idx)
        img_info = base_dataset.images[base_idx]
        img_path = Path(config.get_coco_path()) / "selected" / "val2017" / img_info["file_name"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------- draw ----------
        ax_pred = axes[row, 0]
        ax_gt = axes[row, 1]

        draw_groups_on_axis(
            ax_pred, img, objects, pred_clusters,
            title=f"Prediction (thr={threshold:.2f}) {title_suffix}"
        )

        draw_gt_groups_only(
            ax_gt, img, objects, gt_clusters,
            title="Ground Truth Grouping"
        )

        print(
            f"[Vis] img={image_idx} | pred-groups={len(pred_clusters)} | gt-groups={len(gt_clusters)}"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    model.train()


if __name__ == "__main__":
    coco_path = config.get_coco_path()
    ann_path = coco_path / "selected" / "annotations" / "instances_val2017.json"

    parser = argparse.ArgumentParser(
        description="Train COCO depth-aware group detector with ground truth annotations")
    parser.add_argument("--coco_json", type=str, default=ann_path,
                        help="Path to instances_val2017.json")
    parser.add_argument("--depth_dir", type=str, default=coco_path / "selected" / "depth_maps",
                        help="Directory containing *_depth.npz files")
    parser.add_argument("--conf_dir", type=str, default=coco_path / "selected" / "depth_maps",
                        help="Directory containing *_conf.png files")
    parser.add_argument("--group_annotations", type=str, default="grouping_annotations.json",
                        help="Path to group annotations JSON file")
    parser.add_argument("--image_limit", type=int, default=None,
                        help="Number of images to use for training (None for all)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    parser.add_argument("--save_model", type=str, default="group_detector_model.pt",
                        help="Path to save trained model")
    parser.add_argument("--viz", action="store_true",
                        help="Visualize pairwise scores after training")
    parser.add_argument("--viz_image_idx", type=int, default=0,
                        help="Image index to visualize")
    parser.add_argument("--viz_threshold", type=float, default=0.5,
                        help="Threshold for grouping visualization (0-1)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for dataset split (None = random each run)")
    
    args = parser.parse_args()
    args.viz= True 
    args.epochs = 100
    print("Loading dataset with group annotations...")
    dataset = CocoGroupAnnotatedDataset(
        coco_json=args.coco_json,
        depth_dir=args.depth_dir,
        conf_dir=args.conf_dir,
        group_annotations_file=args.group_annotations,
        image_limit=args.image_limit
    )

    if len(dataset) == 0:
        print("No images with group annotations found!")
        print("Please run the interactive annotation tool first to create group annotations.")
        exit(1)

    print(f"Dataset loaded with {len(dataset)} images")

    # ---------------- Train / Test Split ----------------
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    num_total = len(dataset)
    num_train = int(num_total * args.train_ratio)
    num_test = num_total - num_train

    train_dataset, test_dataset = random_split(
        dataset, [num_train, num_test]
    )

    print(f"Train/Test split: {num_train} train / {num_test} test "
          f"(ratio={args.train_ratio}, seed={args.seed})")

    model = PairwiseGroupScorer()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Training on device: {device}")
    train_with_ground_truth(
        model,
        train_dataset,
        test_dataset,
        epochs=args.epochs,
        device=device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    overall_accuracy, avg_per_image_accuracy = evaluate_model(model, test_dataset, device)
    # Save trained model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")

    # Compute automatic grouping threshold from validation statistics
    auto_threshold = estimate_pairwise_threshold(
        model, train_dataset, device=device, quantile=0.8, max_images=20
    )
    print(f"[Thresh] auto={auto_threshold:.3f} (train-set quantile=0.8)")

    if args.viz:
        if len(test_dataset) > 0:
            # Visualize grouping structure
            num_vis = min(4, len(test_dataset))
            vis_indices = np.random.choice(len(test_dataset), num_vis, replace=False).tolist()
            visualize_group_graph_grid(
                model, test_dataset,
                image_indices=vis_indices,
                device=device,
                threshold=auto_threshold,
                title_suffix="[TEST SET]"
            )
        else:
            print(f"Cannot visualize: no images in test set")

    print(
        f"[Done ] model={'OK' if overall_accuracy > 0.6 else 'WEAK'} | "
        f"train={num_train} | test={num_test}"
    )