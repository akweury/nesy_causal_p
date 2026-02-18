
# Created by MacBook Pro at 12.06.25
# ablation_main.py
import json
import time
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import wandb
from collections import defaultdict
from mbg.group.train_gd_transformer import load_group_transformer, train_grouping, GroupDataset
from mbg.group.gd_transformer import GroupingTransformer
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
from mbg.training import training
from mbg.evaluation import evaluation
import config
from mbg.scorer import scorer_config
from mbg.scorer import improved_calibrator
from src import visual_study
from src.dataset import shape_to_id_clevr
from src import bk
from mbg import patch_preprocess
from torch.utils.data import DataLoader
from mbg.group import eval_groups

ABLATED_CONFIGS = {
    "hard_obj": {"use_hard": True, "use_soft": False, "use_obj": True, "use_group": True, "use_calibrator": False},
}

def visualize_group_results(image_path, symbolic_data, gt_groups, pred_groups, output_path):
    """Visualize ground truth and predicted groups side by side.
    
    Args:
        image_path: Path to the image file
        symbolic_data: Symbolic data with object positions and sizes
        gt_groups: Ground truth group assignments (list of group_ids for each object)
        pred_groups: Predicted groups (list of dicts with 'child_obj_ids' key)
        output_path: Path to save the visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Generate distinct colors for groups
    def get_color(group_id, total_groups):
        # Use HSV for better color distinction
        hue = (group_id * (360 / max(total_groups, 1))) % 360
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Ground Truth
    ax1.imshow(img)
    ax1.set_title('Ground Truth Groups', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Extract ground truth groups from symbolic_data
    gt_group_dict = defaultdict(list)
    for obj_idx, obj in enumerate(symbolic_data):
        group_id = obj.get('group_id', 0)
        if group_id is not None:
            gt_group_dict[group_id].append(obj_idx)
    
    num_gt_groups = len(gt_group_dict)
    for group_id, obj_indices in gt_group_dict.items():
        color = get_color(group_id, num_gt_groups)
        for obj_idx in obj_indices:
            obj = symbolic_data[obj_idx]
            x, y, size = obj['x'], obj['y'], obj['size']
            # Convert normalized coordinates to pixel coordinates
            x_pix, y_pix = x * w, y * h
            size_pix = size * min(w, h)
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x_pix - size_pix/2, y_pix - size_pix/2),
                size_pix, size_pix,
                linewidth=3, edgecolor=np.array(color)/255, facecolor='none'
            )
            ax1.add_patch(rect)
            # Add object index
            ax1.text(x_pix, y_pix - size_pix/2 - 10, str(obj_idx),
                    fontsize=10, color='white', backgroundcolor='black',
                    ha='center', va='bottom')
    
    # Add legend for GT
    legend_elements = [patches.Patch(facecolor=np.array(get_color(gid, num_gt_groups))/255,
                                    edgecolor='black', label=f'Group {gid}')
                      for gid in sorted(gt_group_dict.keys())]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Right: Predicted Groups
    ax2.imshow(img)
    ax2.set_title('Predicted Groups', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    num_pred_groups = len(pred_groups)
    for group_id, group in enumerate(pred_groups):
        color = get_color(group_id, num_pred_groups)
        obj_indices = group['child_obj_ids']
        
        for obj_idx in obj_indices:
            if obj_idx < len(symbolic_data):
                obj = symbolic_data[obj_idx]
                x, y, size = obj['x'], obj['y'], obj['size']
                x_pix, y_pix = x * w, y * h
                size_pix = size * min(w, h)
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x_pix - size_pix/2, y_pix - size_pix/2),
                    size_pix, size_pix,
                    linewidth=3, edgecolor=np.array(color)/255, facecolor='none'
                )
                ax2.add_patch(rect)
                # Add object index
                ax2.text(x_pix, y_pix - size_pix/2 - 10, str(obj_idx),
                        fontsize=10, color='white', backgroundcolor='black',
                        ha='center', va='bottom')
    
    # Add legend for predictions
    legend_elements_pred = [patches.Patch(facecolor=np.array(get_color(gid, num_pred_groups))/255,
                                         edgecolor='black', label=f'Group {gid}')
                           for gid in range(num_pred_groups)]
    ax2.legend(handles=legend_elements_pred, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_group_results(data_samples, group_lists, output_dir, task_name, split_name, visualize=False):
    """Save group detection results for each image.
    
    Args:
        data_samples: List of data samples with image paths
        group_lists: List of group lists corresponding to each image
        output_dir: Output directory path
        task_name: Name of the task
        split_name: Name of the split (train/val/test)
        visualize: Whether to create visualizations
    """
    task_output_dir = output_dir / task_name / split_name
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Create visualization directory if needed
    if visualize:
        vis_output_dir = output_dir / task_name / split_name / "visualizations"
        os.makedirs(vis_output_dir, exist_ok=True)
    
    for img_idx, (sample, groups) in enumerate(zip(data_samples, group_lists)):
        img_path = sample["image_path"]
        if isinstance(img_path, list):
            img_path = img_path[0]
        img_name = os.path.basename(img_path).replace(".png", "")
        
        # Convert groups to serializable format
        groups_data = []
        for g in groups:
            group_dict = {
                "group_id": g["id"],
                "child_obj_ids": g["child_obj_ids"],
                "num_members": len(g["members"]),
                "principle": g["principle"]
            }
            groups_data.append(group_dict)
        
        result = {
            "image_path": img_path,
            "image_name": img_name,
            "num_groups": len(groups),
            "groups": groups_data
        }
        
        output_file = task_output_dir / f"{img_name}_groups.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create visualization if enabled
        if visualize:
            symbolic_data = sample["symbolic_data"]
            # Extract ground truth groups
            gt_groups = [obj.get('group_id', 0) for obj in symbolic_data]
            
            vis_output_path = vis_output_dir / f"{img_name}_groups_vis.png"
            visualize_group_results(
                image_path=img_path,
                symbolic_data=symbolic_data,
                gt_groups=gt_groups,
                pred_groups=groups_data,
                output_path=str(vis_output_path)
            )

def run_ablation(train_data, val_data, test_data, obj_model, group_model, train_principle, args, mode_name,
                 ablation_flags):
    task_name = train_data["task"]
    train_val_data = {
        "task": task_name,
        "positive": train_data["positive"] + val_data["positive"],
        "negative": train_data["negative"] + val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5,
                  "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [
        1] * len(train_val_data["positive"]) + [0] * len(train_val_data["negative"])
    # train rule + calibrator
    obj_times = torch.zeros(len(train_img_labels))

    hard, soft, group_nums, obj_list, group_list = training.ground_clevr_facts(train_val_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags,
                                                                         obj_times, use_gt_groups=args.use_gt_groups)
    base_rules = training.train_rules(hard, soft, obj_list, group_list,
                                      group_nums, train_img_labels, hyp_params, ablation_flags, obj_times)
    final_rules = training.extend_rules(
        base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)
    # calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params, ablation_flags, args.device)
    calibrator = improved_calibrator.train_calibrator(
        final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params, ablation_flags, args.device)
    eval_metrics = evaluation.eval_rules_clevr(
        test_data, obj_model, group_model, final_rules, hyp_params, train_principle, args.device, calibrator, use_gt_groups=args.use_gt_groups)
    if eval_metrics["acc"] > 0.9:
        print(
            f"High acc {eval_metrics['acc']} in {task_name} for {mode_name} with rules: {final_rules}")
    return eval_metrics


def run_ablation_train_val(train_data, val_data, test_data, obj_model, group_model, train_principle, args, mode_name,
                           ablation_flags):
    task_name = train_data["task"]
    train_data = {
        "task": task_name,
        "positive": train_data["positive"],
        "negative": train_data["negative"]
    }
    val_data = {
        "task": task_name,
        "positive": val_data["positive"],
        "negative": val_data["negative"]
    }
    hyp_params = {"prox": 0.9, "sim": 0.5,
                  "top_k": 5, "conf_th": 0.5, "patch_dim": 7}
    train_img_labels = [1] * len(train_data["positive"]) + \
        [0] * len(train_data["negative"])
    val_img_labels = [1] * len(val_data["positive"]) + \
        [0] * len(val_data["negative"])
    # train rule + calibrator
    obj_times = torch.zeros(len(train_img_labels))
    val_obj_times = torch.zeros(len(val_img_labels))

    hard, soft, group_nums, obj_list, group_list = training.ground_clevr_facts(
        train_data, obj_model, group_model, hyp_params, train_principle, args.device, ablation_flags, obj_times, use_gt_groups=args.use_gt_groups)
    val_hard, val_soft, val_group_nums, val_obj_list, val_group_list = training.ground_clevr_facts(val_data, obj_model,
                                                                                             group_model, hyp_params,
                                                                                             train_principle,
                                                                                             args.device,
                                                                                             ablation_flags,
                                                                                             val_obj_times,
                                                                                             use_gt_groups=args.use_gt_groups)
    
    # Save group results for train and val data
    if hasattr(args, 'save_groups') and args.save_groups:
        output_dir = config.get_proj_output_path(args.remote) / "group_results"
        visualize = getattr(args, 'visualize', False)
        save_group_results(train_data["positive"] + train_data["negative"], 
                          group_list, output_dir, task_name, "train", visualize=visualize)
        save_group_results(val_data["positive"] + val_data["negative"], 
                          val_group_list, output_dir, task_name, "val", visualize=visualize)

    base_rules = training.train_rules(hard, soft, obj_list, group_list,
                                      group_nums, train_img_labels, hyp_params, ablation_flags, obj_times)
    final_rules = training.extend_rules(
        base_rules, hard, soft, train_img_labels, obj_list, group_list, hyp_params)
    # calibrator = training.train_calibrator(final_rules, obj_list, group_list, hard, soft, train_img_labels, hyp_params, ablation_flags, args.device)
    calibrator = improved_calibrator.train_calibrator(
        final_rules, val_obj_list, val_group_list, val_hard, val_soft, val_img_labels, hyp_params, ablation_flags, args.device)

    test_metrics, test_group_list = evaluation.eval_rules_clevr(
        test_data, obj_model, group_model, final_rules, hyp_params, train_principle, args.device, calibrator, use_gt_groups=args.use_gt_groups)
    
    # Save group results for test data
    if hasattr(args, 'save_groups') and args.save_groups:
        output_dir = config.get_proj_output_path(args.remote) / "group_results"
        visualize = getattr(args, 'visualize', False)
        save_group_results(test_data["positive"] + test_data["negative"], 
                          test_group_list, output_dir, task_name, "test", visualize=visualize)
    
    return test_metrics, final_rules


def load_gd_transformer_model(principle, device, remote):
    # Try to load transformer model
    transformer_model_dir = config.get_proj_output_path(remote) / "models"
    transformer_model_path = transformer_model_dir / \
        f"gd_transformer_{principle}_standalone.pt"
    if transformer_model_path.exists():
        print(
            f"Loading transformer model for {principle} from {transformer_model_path}")
        group_model, _ = load_group_transformer(
            model_path=str(transformer_model_path),
            device=device,
            shape_dim=16,
            app_dim=0,
            d_model=128,
            num_heads=4,
            depth=4,
            rel_dim=64
        )
    # Ensure model is on correct device
    group_model = group_model.to(device)
    return group_model



def load_clevr_task_data(task_path, mode='train', val_split=0.4):
    """Load data for a single CLEVR task.
    
    Args:
        task_path: Path to the task folder (e.g., /path/to/clevr/proximity)
        mode: 'train', 'val', or 'test'
        val_split: Fraction of train data to use for validation
    
    Returns:
        Dictionary with task data in the format expected by the pipeline
    """
    task_name = os.path.basename(task_path)
    task_data = {"task": task_name, "positive": [], "negative": []}
    meta_data_loaded = False
    
    # Determine which folder to use based on mode
    if mode in ['train', 'val']:
        data_folder = os.path.join(task_path, 'train')
    else:  # test
        data_folder = os.path.join(task_path, 'test')
    
    if not os.path.isdir(data_folder):
        return task_data
    
    for class_label, class_name in enumerate(["negative", "positive"]):
        class_folder = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        
        image_files = sorted([f for f in os.listdir(class_folder) if f.endswith(".png")])
        

        selected_files = image_files
        
        for fname in selected_files:
            img_path = os.path.join(class_folder, fname)
            json_path = img_path.replace(".png", ".json")
            
            if not os.path.exists(json_path):
                continue
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Load metadata from logics section (only once per task)
            if not meta_data_loaded:
                logics = json_data.get("logics", {})
                task_data["principle"] = logics.get("principle", "unknown")
                # Set default values for fields that might not be in CLEVR data
                task_data["non_overlap"] = True
                task_data["qualifier_all"] = False
                task_data["qualifier_exist"] = True
                task_data["prop_shape"] = True
                task_data["prop_color"] = True
                task_data["prop_size"] = True
                task_data["prop_count"] = False
                meta_data_loaded = True
            
            # Convert objects to symbolic data format
            sym_data = [{
                'x': od['x'],
                'y': od['y'],
                'size': od['size'],
                'color_r': od['color_r'],
                'color_g': od['color_g'],
                'color_b': od['color_b'],
                'shape': shape_to_id_clevr(od["shape"]),
                "group_id": od.get("group_id", None),
            } for od in json_data["objects"]]
            
            entry = {
                "image_path": [img_path],
                "img_label": class_label,
                "symbolic_data": sym_data,
                "principle": task_data["principle"]
            }
            task_data[class_name].append(entry)
    
    return task_data


def load_clevr_combined_dataset(clevr_path):
    """Load all CLEVR tasks from the clevr folder.
    
    Args:
        clevr_path: Path to the main CLEVR folder containing task subfolders
    
    Returns:
        List of (train_data, val_data, test_data) tuples for each task
    """
    combined_data = []
    
    # Get all task folders (ignore files like test_scene.json)
    task_folders = sorted([f for f in os.listdir(clevr_path) 
                          if os.path.isdir(os.path.join(clevr_path, f)) 
                          and not f.startswith('.')])
    
    print(f"Found task folders: {task_folders}")
    
    for task_folder in task_folders:
        task_path = os.path.join(clevr_path, task_folder)
        
        # Load train, val, and test data for this task
        train_data = load_clevr_task_data(task_path, mode='train')
        val_data = load_clevr_task_data(task_path, mode='val')
        test_data = load_clevr_task_data(task_path, mode='test')
        
        # Only add if we have data
        if (train_data["positive"] or train_data["negative"]) and \
           (test_data["positive"] or test_data["negative"]):
            print(f"  Task {task_folder}: train={len(train_data['positive'])+len(train_data['negative'])}, " +
                  f"val={len(val_data['positive'])+len(val_data['negative'])}, " +
                  f"test={len(test_data['positive'])+len(test_data['negative'])}")
            combined_data.append((train_data, val_data, test_data))
        else:
            print(f"  Skipping {task_folder}: insufficient data")
    
    return combined_data


def train_group_scorer_model(train_data, val_data, train_principle, args, epochs=100, lr=1e-3, mask_dims=None, model_type='nn'):
    """Train a group scorer model on CLEVR data.
    
    Args:
        train_data: Training dataset with positive and negative samples
        val_data: Validation dataset
        train_principle: Gestalt principle (e.g., 'proximity', 'similarity')
        args: Arguments containing device, remote flags
        epochs: Number of training epochs
        lr: Learning rate
        mask_dims: List of dimension types to mask (e.g., ['position'], ['color'], ['shape'])
        model_type: Type of model to use ('nn' for SimplifiedPositionScorer, 'transformer' for TransformerPositionScorer)
    
    Returns:
        Trained scorer model
    """
    from mbg.scorer.simplified_position_scorer import SimplifiedPositionScorer
    from mbg.scorer.transformer_position_scorer import TransformerPositionScorer
    from src import bk
    import torch.nn as nn
    import torch.optim as optim
    
    # Initialize model based on type
    # Shape one-hot has 4 dimensions (bk_shapes_clevr: none, cube, sphere, cylinder)
    if model_type == 'transformer':
        model = TransformerPositionScorer(
            position_dim=9, 
            hidden_dim=64, 
            context_embed_dim=32, 
            mask_dims=mask_dims,
            num_heads=4,
            num_layers=2
        ).to(args.device)
        model_name = "Transformer"
    else:  # default to 'nn'
        model = SimplifiedPositionScorer(
            position_dim=9, 
            hidden_dim=64, 
            context_embed_dim=32, 
            mask_dims=mask_dims
        ).to(args.device)
        model_name = "NN"
    
    if mask_dims:
        print(f"Training {model_name} scorer model for {train_principle} with {epochs} epochs (masking dimensions: {mask_dims})...")
    else:
        print(f"Training {model_name} scorer model for {train_principle} with {epochs} epochs (no dimension masking)...")
    
    # Prepare training data
    train_samples = []
    for samples in [train_data["positive"], train_data["negative"]]:
        for sample in samples:
            symbolic_data = sample["symbolic_data"]
            if len(symbolic_data) >= 2:
                train_samples.append(symbolic_data)
    
    val_samples = []
    for samples in [val_data["positive"], val_data["negative"]]:
        for sample in samples:
            symbolic_data = sample["symbolic_data"]
            if len(symbolic_data) >= 2:
                val_samples.append(symbolic_data)
    
    if len(train_samples) == 0:
        print("WARNING: No training data available, returning untrained model")
        return model
    
    print(f"Training with {len(train_samples)} samples, validating with {len(val_samples)} samples")
    
    # Prepare balanced pairs dataset
    print("Preparing balanced training pairs...")
    positive_pairs = []
    negative_pairs = []
    
    for symbolic_data in train_samples:
        n_objs = len(symbolic_data)
        
        # Create all pairs of objects
        for i in range(n_objs):
            for j in range(i + 1, n_objs):
                # Extract features for objects i and j
                shape_i = torch.zeros(4)
                shape_i[symbolic_data[i]['shape']] = 1.0
                shape_j = torch.zeros(4)
                shape_j[symbolic_data[j]['shape']] = 1.0
                
                pos_i = torch.cat([torch.tensor([symbolic_data[i]['x'], symbolic_data[i]['y'],
                                      symbolic_data[i]['color_r'], symbolic_data[i]['color_g'], symbolic_data[i]['color_b']]), 
                                      shape_i])  # (9,)
                pos_j = torch.cat([torch.tensor([symbolic_data[j]['x'], symbolic_data[j]['y'],
                                      symbolic_data[j]['color_r'], symbolic_data[j]['color_g'], symbolic_data[j]['color_b']]), 
                                      shape_j])  # (9,)
                
                # Context: all other objects' positions, colors, and shapes
                context_features = []
                for k in range(n_objs):
                    if k != i and k != j:
                        shape_k = torch.zeros(4)
                        shape_k[symbolic_data[k]['shape']] = 1.0
                        feat_k = torch.cat([torch.tensor([symbolic_data[k]['x'], symbolic_data[k]['y'],
                                                         symbolic_data[k]['color_r'], symbolic_data[k]['color_g'], symbolic_data[k]['color_b']]),
                                           shape_k])
                        context_features.append(feat_k)
                
                if len(context_features) == 0:
                    ctx_tensor = torch.zeros((0, 9), dtype=torch.float32)
                else:
                    ctx_tensor = torch.stack(context_features)  # (N, 9)
                
                # Label: objects belong to same group?
                group_i = symbolic_data[i].get('group_id', -1)
                group_j = symbolic_data[j].get('group_id', -1)
                label = 1.0 if (group_i == group_j and group_i != -1) else 0.0
                
                pair_data = (pos_i, pos_j, ctx_tensor, label)
                
                if label == 1.0:
                    positive_pairs.append(pair_data)
                else:
                    negative_pairs.append(pair_data)
    
    # Balance the dataset by undersampling the majority class
    num_positive = len(positive_pairs)
    num_negative = len(negative_pairs)
    print(f"Original dataset: {num_positive} positive pairs, {num_negative} negative pairs")
    
    import random
    random.seed(42)
    
    if num_positive > num_negative:
        # Undersample positive pairs
        positive_pairs = random.sample(positive_pairs, num_negative)
        print(f"Undersampled positive pairs to {len(positive_pairs)}")
    elif num_negative > num_positive:
        # Undersample negative pairs
        negative_pairs = random.sample(negative_pairs, num_positive)
        print(f"Undersampled negative pairs to {len(negative_pairs)}")
    
    # Combine and shuffle
    balanced_pairs = positive_pairs + negative_pairs
    random.shuffle(balanced_pairs)
    print(f"Balanced dataset: {len(balanced_pairs)} total pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        # Shuffle pairs each epoch
        random.shuffle(balanced_pairs)
        
        # Training loop
        for pos_i, pos_j, ctx_tensor, label in balanced_pairs:
            # Move to device
            pos_i = pos_i.unsqueeze(0).to(args.device)  # (1, 9)
            pos_j = pos_j.unsqueeze(0).to(args.device)  # (1, 9)
            ctx_tensor = ctx_tensor.unsqueeze(0).to(args.device)  # (1, N, 9)
            label_tensor = torch.tensor([label], dtype=torch.float32).to(args.device)
            
            # Forward pass
            logits = model(pos_i, pos_j, ctx_tensor)
            loss = criterion(logits, label_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1
        
        train_acc = correct / total if total > 0 else 0
        train_loss = total_loss / total if total > 0 else 0
        
        # Validation
        if len(val_samples) > 0:
            model.eval()
            val_correct, val_total = 0, 0
            
            with torch.no_grad():
                for symbolic_data in val_samples:
                    n_objs = len(symbolic_data)
                    
                    for i in range(n_objs):
                        for j in range(i + 1, n_objs):
                            # Extract x, y positions, RGB colors, and shape one-hot from symbolic data
                            # Create shape one-hot encoding (4 dimensions for bk_shapes_clevr)
                            shape_i = torch.zeros(4)
                            shape_i[symbolic_data[i]['shape']] = 1.0
                            shape_j = torch.zeros(4)
                            shape_j[symbolic_data[j]['shape']] = 1.0
                            
                            pos_i = torch.cat([torch.tensor([symbolic_data[i]['x'], symbolic_data[i]['y'],
                                                  symbolic_data[i]['color_r'], symbolic_data[i]['color_g'], symbolic_data[i]['color_b']]), 
                                                  shape_i]).unsqueeze(0).to(args.device)  # (1, 9)
                            pos_j = torch.cat([torch.tensor([symbolic_data[j]['x'], symbolic_data[j]['y'],
                                                  symbolic_data[j]['color_r'], symbolic_data[j]['color_g'], symbolic_data[j]['color_b']]), 
                                                  shape_j]).unsqueeze(0).to(args.device)  # (1, 9)
                            
                            # Context: all other objects' positions, colors, and shapes
                            context_features = []
                            for k in range(n_objs):
                                if k != i and k != j:
                                    shape_k = torch.zeros(4)
                                    shape_k[symbolic_data[k]['shape']] = 1.0
                                    feat_k = torch.cat([torch.tensor([symbolic_data[k]['x'], symbolic_data[k]['y'],
                                                                     symbolic_data[k]['color_r'], symbolic_data[k]['color_g'], symbolic_data[k]['color_b']]),
                                                       shape_k])
                                    context_features.append(feat_k)
                            
                            if len(context_features) == 0:
                                ctx_tensor = torch.zeros((1, 0, 9), dtype=torch.float32, device=args.device)
                            else:
                                ctx_tensor = torch.stack(context_features).unsqueeze(0).to(args.device)  # (1, N, 9)
                            
                            group_i = symbolic_data[i].get('group_id', -1)
                            group_j = symbolic_data[j].get('group_id', -1)
                            label = torch.tensor([1.0 if (group_i == group_j and group_i != -1) else 0.0], 
                                                dtype=torch.float32).to(args.device)
                            
                            logits = model(pos_i, pos_j, ctx_tensor)
                            pred = (torch.sigmoid(logits) > 0.5).float()
                            val_correct += (pred == label).sum().item()
                            val_total += 1
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")
    return model


def load_and_test_group_model(task_name, train_principle, grp_test_data, args, group_model=None):
    """Load or use provided group model and test its performance.
    
    Args:
        task_name: Name of the task
        train_principle: Gestalt principle (e.g., 'proximity', 'similarity')
        grp_test_data: Test data with positive and negative samples
        args: Arguments containing device, remote, visualize flags
        group_model: Optional pre-trained model to use (if None, loads from file)
    
    Returns:
        tuple: (group_model, group_metrics) where group_metrics is a dict of performance metrics
    """
    # Load pre-trained scorer model for grouping if not provided
    if group_model is None:
        raise ValueError("No group model provided. Please train a scorer model and load it before testing.")
    else:
        print(f"Using provided trained model for testing")
    
    # Test grouping performance on test data
    print(f"Testing grouping performance on test data for task {task_name}...")
    test_groups_list = []
    test_gt_objects_list = []
    test_pred_objects_list = []
    
    # Process positive and negative test samples separately to track labels
    test_samples_with_labels = []
    
    # Add positive samples with label
    for sample in grp_test_data["positive"]:
        test_samples_with_labels.append((sample, "positive"))
    
    # Add negative samples with label
    for sample in grp_test_data["negative"]:
        test_samples_with_labels.append((sample, "negative"))
    
    for sample, label in test_samples_with_labels:
        symbolic_data = sample["symbolic_data"]
        
        if len(symbolic_data) < 2:
            continue
        
        # Prepare objects for group detection (using ground truth objects)
        objs = []
        for obj_data in symbolic_data:
            # Combine color components into a single array [r, g, b]
            color_array = [obj_data['color_r'] / 255.0, 
                          obj_data['color_g'] / 255.0, 
                          obj_data['color_b'] / 255.0]
            
            # Create shape one-hot encoding (4 dimensions for bk_shapes_clevr)
            shape_id = obj_data['shape']
            shape_one_hot = torch.zeros(4)  # 4 shapes: none, cube, sphere, cylinder
            shape_one_hot[shape_id] = 1.0
            
            obj = {
                's': {
                    'x': obj_data['x'],
                    'y': obj_data['y'],
                    'w': obj_data['size'],  # Use 'w' instead of 'size'
                    'color': color_array,  # Combined color array
                    'shape': shape_one_hot,  # One-hot encoding (4 dims)
                    'group_id': obj_data.get('group_id', None)  # Keep for ground truth
                },
                'h': torch.zeros((6, 16, 7), dtype=torch.float32)  # Dummy patches for scorer model (P, L, D)
            }
            objs.append(obj)
        
        # Run group detection
        pred_groups = eval_groups.eval_clevr_groups(
            objs=objs,
            group_model=group_model,
            principle=train_principle,
            device=args.device,
            dim=7,
            grp_th=0.5
        )
        
        test_groups_list.append(pred_groups)
        test_gt_objects_list.append(symbolic_data)
        test_pred_objects_list.append(objs)
    
    # Calculate grouping metrics
    from src.metric_od_gd import evaluate_group_detection
    group_metrics = evaluate_group_detection(
        groups_list=test_groups_list,
        gt_objects_list=test_gt_objects_list,
        obj_lists=test_pred_objects_list,
        iou_threshold=0.5
    )
    
    print(f"Grouping Performance on Test Data:")
    print(f"  mAP: {group_metrics['mAP']:.4f}")
    print(f"  Precision: {group_metrics['precision']:.4f}")
    print(f"  Recall: {group_metrics['recall']:.4f}")
    print(f"  F1: {group_metrics['f1']:.4f}")
    print(f"  Binary Accuracy: {group_metrics['binary_accuracy']:.4f}")
    print(f"  Group Count Accuracy: {group_metrics['group_count_accuracy']:.4f}")
    
    # Visualize results for a few samples
    if args.visualize and len(test_samples_with_labels) > 0:
        output_dir = config.get_proj_output_path(args.remote) / "group_testing_results"
        
        # Create separate directories for positive and negative samples
        vis_output_dir_positive = output_dir / task_name / "visualizations" / "positive"
        vis_output_dir_negative = output_dir / task_name / "visualizations" / "negative"
        os.makedirs(vis_output_dir_positive, exist_ok=True)
        os.makedirs(vis_output_dir_negative, exist_ok=True)
        
        # Visualize first 5 positive and first 5 negative samples
        positive_count = 0
        negative_count = 0
        max_per_class = 5
        
        for idx, (sample, label) in enumerate(test_samples_with_labels):
            # Skip if we've reached the limit for this class
            if label == "positive" and positive_count >= max_per_class:
                continue
            if label == "negative" and negative_count >= max_per_class:
                continue
            
            pred_groups = test_groups_list[idx]
            img_path = sample["image_path"]
            if isinstance(img_path, list):
                img_path = img_path[0]
            img_name = os.path.basename(img_path).replace(".png", "")
            
            # Convert predicted groups to format expected by visualize_group_results
            groups_data = []
            for g in pred_groups:
                group_dict = {
                    "group_id": g["id"],
                    "child_obj_ids": g["child_obj_ids"],
                    "num_members": len(g["members"]),
                    "principle": g["principle"]
                }
                groups_data.append(group_dict)
            
            # Choose output directory based on label
            vis_output_dir = vis_output_dir_positive if label == "positive" else vis_output_dir_negative
            vis_output_path = vis_output_dir / f"{img_name}_test_groups_vis.png"
            symbolic_data = sample["symbolic_data"]
            gt_groups = [obj.get('group_id', 0) for obj in symbolic_data]
            
            visualize_group_results(
                image_path=img_path,
                symbolic_data=symbolic_data,
                gt_groups=gt_groups,
                pred_groups=groups_data,
                output_path=str(vis_output_path)
            )
            
            # Increment counter for this class
            if label == "positive":
                positive_count += 1
            else:
                negative_count += 1
        
        print(f"Visualizations saved to:")
        print(f"  Positive: {vis_output_dir_positive}")
        print(f"  Negative: {vis_output_dir_negative}")
    
    return group_model, group_metrics


def compare_scorer_models(train_data, val_data, test_data, train_principle, args, epochs=100, lr=1e-3, mask_dims=None):
    """Compare NN-based and Transformer-based scorer models.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        train_principle: Gestalt principle
        args: Arguments
        epochs: Number of training epochs
        lr: Learning rate
        mask_dims: Dimensions to mask
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for model_type in ['nn', 'transformer']:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        # Train model
        model = train_group_scorer_model(
            train_data=train_data,
            val_data=val_data,
            train_principle=train_principle,
            args=args,
            epochs=epochs,
            lr=lr,
            mask_dims=mask_dims,
            model_type=model_type
        )
        
        # Test model
        task_name = train_data.get("task", "unknown")
        _, metrics = load_and_test_group_model(
            task_name=task_name,
            train_principle=train_principle,
            grp_test_data=test_data,
            args=args,
            group_model=model
        )
        
        results[model_type] = {
            'model': model,
            'metrics': metrics
        }
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Model Comparison Results")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'NN':<15} {'Transformer':<15} {'Winner':<10}")
    print(f"{'-'*65}")
    
    for metric_name in ['mAP', 'precision', 'recall', 'f1', 'binary_accuracy', 'group_count_accuracy']:
        nn_val = results['nn']['metrics'].get(metric_name, 0)
        tf_val = results['transformer']['metrics'].get(metric_name, 0)
        winner = 'NN' if nn_val > tf_val else 'Transformer' if tf_val > nn_val else 'Tie'
        print(f"{metric_name:<25} {nn_val:<15.4f} {tf_val:<15.4f} {winner:<10}")
    
    return results


def run_grm_clevr():
    args = args_utils.get_args()
    args.save_groups = True  # Enable saving group results for analysis
    args.visualize = True  # Enable visualization of group results
    args.use_gt_groups = False  # Set to True to use ground truth groups instead of training a model
    args.model_type="nn"  # Use 'transformer' for group scorer model
    args.mask_dims = ["color", "shape"]  # Mask color and shape dimensions for ablation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Load CLEVR data
    clevr_path = config.get_clevr_path(args.remote)
    clevr_grp_path = config.get_clevr_grp_path(args.remote)
    print(f"Loading CLEVR data from: {clevr_path}")
    combined_loader = load_clevr_combined_dataset(clevr_path)
    print(f"Found {len(combined_loader)} tasks")
    
    if len(combined_loader) == 0:
        print("ERROR: No tasks found in CLEVR dataset!")
        return
    
    # For principle, we'll use the principle from the first task or from args
    if hasattr(args, 'principle') and args.principle:
        train_principle = args.principle
    elif combined_loader and len(combined_loader) > 0:
        train_principle = combined_loader[0][0].get("principle", "proximity")
    else:
        train_principle = "proximity"  # default
    
    obj_model = eval_patch_classifier.load_model(args.device, args.remote)
    

    # Use a generic project name for CLEVR or include principle if available
    project_name = f"grm_clevr_{train_principle}" if train_principle else "grm_clevr"
    exp_name = args.exp_name if hasattr(args, 'exp_name') and args.exp_name else f"clevr_{timestamp}"
    # wandb.init(project=project_name, config=args.__dict__, name=exp_name)

    results_summary = defaultdict(lambda: defaultdict(list))
    # mode -> error_type -> list of counts
    error_summary = defaultdict(lambda: defaultdict(list))
    # mode -> topk_metric -> list
    topk_summary = defaultdict(lambda: defaultdict(list))
    per_task_results = defaultdict(list)  # mode -> list of dicts
    analysis_summary = defaultdict(lambda: defaultdict(list))

    all_f1 = {conf: [] for conf in ABLATED_CONFIGS}
    all_auc = {conf: [] for conf in ABLATED_CONFIGS}
    all_acc = {conf: [] for conf in ABLATED_CONFIGS}
    
    # Load group training data from clevr_grp_path
    print(f"Loading group training data from: {clevr_grp_path}")
    grp_combined_loader = load_clevr_combined_dataset(clevr_grp_path)
    print(f"Found {len(grp_combined_loader)} group training tasks")
    
    for task_idx, (train_data, val_data, test_data) in enumerate(combined_loader):
        if task_idx!=2:
            continue 
        
        task_name = train_data["task"]
        print(f"\nTask {task_idx + 1}/{len(combined_loader)}: {task_name}")
        
        # Get corresponding group training data for the same task
        grp_train_data, grp_val_data, grp_test_data = grp_combined_loader[task_idx]
        
        # Check if using ground truth groups or training a model
        if args.use_gt_groups:
            print(f"Using ground truth groups (--use_gt_groups flag is set)")
            group_model = None  # Not needed when using GT groups
        else:
            gd_epochs = getattr(args, 'gd_epochs', 100)
            gd_lr = getattr(args, 'gd_lr', 1e-3)
            mask_dims = getattr(args, 'mask_dims', None)
            compare_models = getattr(args, 'compare_models', False)
            
            # Check if we should compare models
            if compare_models:
                print(f"Comparing NN vs Transformer models for task {task_name}...")
                comparison_results = compare_scorer_models(
                    train_data=grp_train_data,
                    val_data=grp_val_data,
                    test_data=grp_test_data,
                    train_principle=train_principle,
                    args=args,
                    epochs=gd_epochs,
                    lr=gd_lr,
                    mask_dims=mask_dims
                )
                # Use the better model for downstream tasks
                nn_f1 = comparison_results['nn']['metrics'].get('f1', 0)
                tf_f1 = comparison_results['transformer']['metrics'].get('f1', 0)
                if tf_f1 > nn_f1:
                    group_model = comparison_results['transformer']['model']
                    print(f"Using Transformer model (F1: {tf_f1:.4f} > {nn_f1:.4f})")
                else:
                    group_model = comparison_results['nn']['model']
                    print(f"Using NN model (F1: {nn_f1:.4f} >= {tf_f1:.4f})")
            else:
                # Train single model
                print(f"Training group scorer model for task {task_name}...")
                model_type = getattr(args, 'model_type', 'nn')  # 'nn' or 'transformer'
                group_model = train_group_scorer_model(
                    train_data=grp_train_data,
                    val_data=grp_val_data,
                    train_principle=train_principle,
                    args=args,
                    epochs=gd_epochs,
                    lr=gd_lr,
                    mask_dims=mask_dims,
                    model_type=model_type
                )
                
                # Test the trained group model
                print(f"Testing group model on test data for task {task_name}...")
                _, group_metrics = load_and_test_group_model(
                    task_name=task_name,
                    train_principle=train_principle,
                    grp_test_data=grp_test_data,
                    args=args,
                    group_model=group_model  # Pass the trained model
                )
        
        log_dicts = {}
        for mode_name, ablation_flags in ABLATED_CONFIGS.items():

            t1 = time.time()
            test_metrics, final_rules = run_ablation_train_val(train_data, val_data, test_data, obj_model, group_model,
                                                               train_principle, args, mode_name, ablation_flags)
            t2 = time.time()
            print(f"  Running ablation: {mode_name} in {t2 - t1} seconds")
            for k in ["acc", "f1", "auc"]:
                results_summary[mode_name][k].append(test_metrics.get(k, 0))
                # print(f"task: {task_idx + 1}/{len(combined_loader)}: {k} {test_metrics.get(k, 0)}")

            test_acc = test_metrics.get("acc", 0)
            if test_acc > 0.9:
                print(f"High acc {test_acc} in {task_name} for {mode_name}")
            test_auc = test_metrics.get("auc", 0)
            test_f1 = test_metrics.get("f1", 0)

            all_f1[mode_name].append(test_f1)
            all_auc[mode_name].append(test_auc)
            all_acc[mode_name].append(test_acc)

            error_stats = test_metrics.get("error_stats", None)
            if error_stats:
                for err_type, count in error_stats.items():
                    error_summary[mode_name][err_type].append(count)

            # Top-k clause analysis
            for k in ["topk_clause_recall", "topk_clause_precision"]:
                if k in test_metrics:
                    topk_summary[mode_name][k].append(test_metrics[k])

            if "analysis" in test_metrics:
                for k, values in test_metrics["analysis"].items():
                    analysis_summary[mode_name][k].extend(values)

            log_dicts.update(
                {f"{mode_name}_{k}": test_metrics.get(k, 0) for k in test_metrics})
            log_dicts.update({f"{mode_name}_acc_avg": torch.tensor(all_acc[mode_name]).mean(),
                              f"{mode_name}_auc_avg": torch.tensor(all_auc[mode_name]).mean(),
                              f"{mode_name}_f1_avg": torch.tensor(all_f1[mode_name]).mean()
                              })
            # Store per-task results
            per_task_results[mode_name].append({
                "task_idx": task_idx,
                "task_name": task_name,
                **{k: test_metrics.get(k, 0) for k in test_metrics}
            })
        # wandb.log(log_dicts)

    # save and summarize
    final_summary = {
        mode: {f"avg_{k}": float(torch.tensor(v).mean())
               for k, v in metric_dict.items()}
        for mode, metric_dict in results_summary.items()
    }
    # Include top-k clause stats
    for mode, topk_metrics in topk_summary.items():
        for k, values in topk_metrics.items():
            final_summary[mode][f"avg_{k}"] = float(
                torch.tensor(values).mean())
    for mode, analysis_dict in analysis_summary.items():
        for k, values in analysis_dict.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                final_summary[mode][f"avg_{k}"] = float(
                    torch.tensor(valid_values, dtype=torch.float).mean())
            else:
                final_summary[mode][f"avg_{k}"] = None

    # Save both per-task and average results
    output_json = {
        "per_task_results": per_task_results,
        "summary": final_summary
    }
    output_path = config.get_proj_output_path(args.remote)
    with open(output_path / f"grm_clevr_summary_{train_principle}_{timestamp}.json", "w") as f:
        json.dump(output_json, f, indent=2)
    
    # Print results for each task individually
    print("\n=== Per-Task Results ===")
    for mode_name in ABLATED_CONFIGS.keys():
        print(f"\n{mode_name}:")
        for task_result in per_task_results[mode_name]:
            task_name = task_result["task_name"]
            acc = task_result.get("acc", 0)
            f1 = task_result.get("f1", 0)
            auc = task_result.get("auc", 0)
            print(f"  {task_name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    
    print("\n=== Average Summary ===")
    for mode, metrics in final_summary.items():
        print(f"{mode}: {metrics}")

    final_error_stats = {}
    for mode, err_dict in error_summary.items():
        total_errors = torch.tensor(err_dict.get(
            "total_errors", [1.0])).float()  # avoid div by zero
        mode_stats = {
            err_type: float(torch.tensor(counts).sum() / total_errors.sum())
            for err_type, counts in err_dict.items()
            if err_type != "total_errors"
        }
        final_error_stats[mode] = mode_stats
    # Save or print
    with open(output_path / f"grm_clevr_error_summary_{train_principle}_{timestamp}.json", "w") as f:
        json.dump(final_error_stats, f, indent=2)
    print("\n=== Error Summary ===")
    for mode, stats in final_error_stats.items():
        print(f"{mode}: {stats}")
    wandb.finish()


if __name__ == "__main__":
    run_grm_clevr()
