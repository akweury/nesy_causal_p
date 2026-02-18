import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from collections import defaultdict
from mbg.group import eval_groups
from src.utils import args_utils
from src import dataset
from mbg.object import eval_patch_classifier
from mbg.training import training
from mbg.evaluation import evaluation
import config
from mbg.scorer import scorer_config
from mbg.scorer import improved_calibrator
from mbg import patch_preprocess
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from itertools import combinations


from mbg.scorer.simplified_position_scorer import SimplifiedPositionScorer
from mbg.scorer.transformer_position_scorer import TransformerPositionScorer


class PairwiseGroupDataset(Dataset):
    """Dataset for training pairwise grouping models (SimplifiedPositionScorer, TransformerPositionScorer)."""
    
    def __init__(self, data_loader, obj_model, device, max_samples=100):
        """
        Args:
            data_loader: Combined data loader with train/val/test splits
            obj_model: Object detection model
            device: Device to use
            max_samples: Maximum number of tasks to process
        """
        self.samples = []
        self.device = device
        
        print("Preparing pairwise grouping dataset...")
        for task_idx, (train_data, val_data, test_data) in enumerate(data_loader):
            if task_idx >= max_samples:
                break
            
            # Use training data
            all_data = train_data["positive"] + train_data["negative"]
            
            for sample in all_data:
                img_path = sample["image_path"][0]
                symbolic_data = sample["symbolic_data"]
                
                # Load and process image
                img = patch_preprocess.load_images_fast([img_path], device=device)[0]
                
                # Detect objects
                objs = eval_patch_classifier.evaluate_image(obj_model, img, device)
                
                if len(objs) < 2:
                    continue
                
                # Extract ground truth groups
                gt_groups = self._extract_ground_truth_groups(symbolic_data)
                
                # Create pairwise labels
                for i, j in combinations(range(len(objs)), 2):
                    # Check if i and j are in the same group
                    same_group = any(i in group and j in group for group in gt_groups)
                    label = 1.0 if same_group else 0.0
                    
                    # Extract features: x, y, r, g, b, shape[4]
                    feat_i = self._extract_features(objs[i])
                    feat_j = self._extract_features(objs[j])
                    
                    # Context: all objects except i and j
                    context_feats = [self._extract_features(objs[k]) 
                                    for k in range(len(objs)) if k != i and k != j]
                    
                    self.samples.append((feat_i, feat_j, context_feats, label))
            
            if (task_idx + 1) % 10 == 0:
                print(f"  Processed {task_idx + 1} tasks, {len(self.samples)} pairs...")
        
        print(f"Dataset created with {len(self.samples)} pairwise samples")
    
    def _extract_ground_truth_groups(self, symbolic_data):
        """Extract ground truth groups from symbolic data."""
        gt_group_dict = defaultdict(list)
        for idx, obj in enumerate(symbolic_data):
            group_id = int(obj["group_id"].item()) if torch.is_tensor(obj["group_id"]) else int(obj["group_id"])
            gt_group_dict[group_id].append(idx)
        return list(gt_group_dict.values())
    
    def _extract_features(self, obj):
        """Extract features from object: x, y, r, g, b, shape[4]."""
        symbolic = obj['s']
        x = symbolic['x']
        y = symbolic['y']
        color = symbolic.get('color', [0.0, 0.0, 0.0])
        shape = symbolic.get('shape', torch.zeros(4))
        
        if isinstance(shape, torch.Tensor):
            shape_list = shape.tolist()
        else:
            shape_list = list(shape)
        
        # Concatenate: x, y, r, g, b, shape[4]
        return [x, y, color[0], color[1], color[2]] + shape_list
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat_i, feat_j, context_feats, label = self.samples[idx]
        
        # Convert to tensors
        pos_i = torch.tensor(feat_i, dtype=torch.float32)
        pos_j = torch.tensor(feat_j, dtype=torch.float32)
        
        if len(context_feats) == 0:
            context = torch.zeros((0, 9), dtype=torch.float32)
        else:
            context = torch.tensor(context_feats, dtype=torch.float32)
        
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return pos_i, pos_j, context, label_tensor


def collate_pairwise(batch):
    """Collate function for pairwise dataset."""
    pos_i_list, pos_j_list, context_list, label_list = zip(*batch)
    
    pos_i = torch.stack(pos_i_list)
    pos_j = torch.stack(pos_j_list)
    labels = torch.stack(label_list).squeeze(-1)
    
    # Contexts have variable lengths - we'll process them separately in the model
    # For now, pad to max length in batch
    max_ctx_len = max(ctx.shape[0] for ctx in context_list)
    
    padded_contexts = []
    for ctx in context_list:
        if ctx.shape[0] < max_ctx_len:
            padding = torch.zeros((max_ctx_len - ctx.shape[0], 9), dtype=torch.float32)
            padded_ctx = torch.cat([ctx, padding], dim=0)
        else:
            padded_ctx = ctx
        padded_contexts.append(padded_ctx)
    
    contexts = torch.stack(padded_contexts)
    
    return pos_i, pos_j, contexts, labels


def train_scorer_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train a scorer model (SimplifiedPositionScorer or TransformerPositionScorer)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for pos_i, pos_j, contexts, labels in train_loader:
            pos_i = pos_i.to(device)
            pos_j = pos_j.to(device)
            contexts = contexts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(pos_i, pos_j, contexts).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        train_acc = correct / total if total > 0 else 0
        train_loss = total_loss / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for pos_i, pos_j, contexts, labels in val_loader:
                pos_i = pos_i.to(device)
                pos_j = pos_j.to(device)
                contexts = contexts.to(device)
                labels = labels.to(device)
                
                logits = model(pos_i, pos_j, contexts).reshape(-1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss_avg = val_loss / val_total if val_total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}")
    
    print(f"  Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def extract_ground_truth_groups(symbolic_data):
    """Extract ground truth groups from symbolic data."""
    gt_group_dict = defaultdict(list)
    for idx, obj in enumerate(symbolic_data):
        group_id = int(obj["group_id"].item()) if torch.is_tensor(obj["group_id"]) else int(obj["group_id"])
        gt_group_dict[group_id].append(idx)
    gt_groups = list(gt_group_dict.values())
    return gt_groups


def compute_group_iou(pred_group, gt_group):
    """Compute IoU between predicted and ground truth group (as sets of object indices)."""
    pred_set = set(pred_group)
    gt_set = set(gt_group)
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union if union > 0 else 0.0


def evaluate_grouping_predictions(pred_groups_list, gt_groups_list, iou_threshold=0.5):
    """
    Evaluate grouping predictions using F1 and accuracy metrics.
    
    Args:
        pred_groups_list: List of predicted groups for each image (each group is a list of object indices)
        gt_groups_list: List of ground truth groups for each image
        iou_threshold: Threshold for matching groups
        
    Returns:
        Dictionary with metrics: precision, recall, f1, accuracy
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_correct_matches = 0
    all_gt_groups = 0
    
    for pred_groups, gt_groups in zip(pred_groups_list, gt_groups_list):
        if not gt_groups:
            continue
            
        all_gt_groups += len(gt_groups)
        matched_gt = set()
        matched_pred = set()
        
        # Match predicted groups to ground truth groups
        for pred_idx, pred_group in enumerate(pred_groups):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_group in enumerate(gt_groups):
                if gt_idx in matched_gt:
                    continue
                iou = compute_group_iou(pred_group, gt_group)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                all_tp += 1
                # Check if exact match
                if set(pred_group) == set(gt_groups[best_gt_idx]):
                    all_correct_matches += 1
        
        # False positives: unmatched predicted groups
        all_fp += len(pred_groups) - len(matched_pred)
        # False negatives: unmatched ground truth groups
        all_fn += len(gt_groups) - len(matched_gt)
    
    # Compute metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = all_correct_matches / all_gt_groups if all_gt_groups > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn
    }


def evaluate_model_on_dataset(model, model_name, obj_model, data_loader, principle, device, grp_threshold=0.5, max_samples=50):
    """Evaluate a single grouping model on the dataset."""
    pred_groups_list = []
    gt_groups_list = []
    
    print(f"\nEvaluating {model_name} model...")
    
    for task_idx, (train_data, val_data, test_data) in enumerate(data_loader):
        if task_idx >= max_samples:
            break
            
        # Use test data for evaluation
        all_data = test_data["positive"] + test_data["negative"]
        
        for sample in all_data:
            img_path = sample["image_path"][0]
            symbolic_data = sample["symbolic_data"]
            
            # Load and process image
            img = patch_preprocess.load_images_fast([img_path], device=device)[0]
            
            # Detect objects
            objs = eval_patch_classifier.evaluate_image(obj_model, img, device)    
            
            if len(objs) == 0:
                pred_groups_list.append([])
                gt_groups_list.append(extract_ground_truth_groups(symbolic_data))
                continue
            
            # Detect groups using the model
            try:
                groups = eval_groups.eval_clevr_groups(
                    objs, model, principle, device, dim=7, grp_th=grp_threshold
                )
                pred_group_ids = [g["child_obj_ids"] for g in groups]
            except Exception as e:
                print(f"Error in group detection: {e}")
                pred_group_ids = [[i] for i in range(len(objs))]  # Fallback: each object in its own group
            
            pred_groups_list.append(pred_group_ids)
            gt_groups_list.append(extract_ground_truth_groups(symbolic_data))
        
        if (task_idx + 1) % 10 == 0:
            print(f"  Processed {task_idx + 1} tasks...")
    
    # Compute metrics
    metrics = evaluate_grouping_predictions(pred_groups_list, gt_groups_list, iou_threshold=0.5)
    
    print(f"\n{model_name} Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def main():
    args = args_utils.get_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get principle (default to 'proximity' if not specified)
    train_principle = getattr(args, 'principle', 'proximity')
    
    # Get wandb setting (default to False)
    use_wandb = getattr(args, 'use_wandb', False)
    
    # Get training settings
    train_epochs = getattr(args, 'train_epochs', 30)
    max_train_samples = getattr(args, 'max_train_samples', 100)
    max_eval_samples = getattr(args, 'max_eval_samples', 10)
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="grouping_comparison",
            name=f"grp_comparison_{train_principle}_{timestamp}",
            config=vars(args)
        )
    else:
        print("wandb disabled (use --use_wandb to enable)")
    
    print(f"Comparing grouping detectors for principle: {train_principle}")
    print(f"Training epochs: {train_epochs}")
    print(f"Max training samples: {max_train_samples}")
    print(f"Max evaluation samples: {max_eval_samples}")
    
    # Load data
    principle_path = scorer_config.get_data_path(args.remote, train_principle)
    combined_loader = dataset.load_combined_dataset(principle_path)
    
    # Load object detection model
    obj_model = eval_patch_classifier.load_model(args.device, args.remote)
    
    # ============================================================
    # PHASE 1: TRAINING
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: TRAINING MODELS")
    print("="*60)
    
    trained_models = {}
    
    # 1. Train SimplifiedNN
    print("\n[1/2] Training SimplifiedNN...")
    nn_model = SimplifiedPositionScorer(
        position_dim=9,
        hidden_dim=64,
        context_embed_dim=32,
        mask_dims=[]  # No masking
    ).to(args.device)
    
    # Prepare datasets
    print("  Preparing training data...")
    train_dataset = PairwiseGroupDataset(
        combined_loader, obj_model, args.device, max_samples=max_train_samples
    )
    
    # Split into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_split, val_split = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_split, batch_size=32, shuffle=True, collate_fn=collate_pairwise)
    val_loader = DataLoader(val_split, batch_size=32, shuffle=False, collate_fn=collate_pairwise)
    
    nn_acc = train_scorer_model(nn_model, train_loader, val_loader, args.device, epochs=train_epochs)
    trained_models["SimplifiedNN"] = {"model": nn_model, "train_acc": nn_acc}
    
    # 2. Train TransformerScorer
    print("\n[2/2] Training TransformerScorer...")
    transformer_model = TransformerPositionScorer(
        position_dim=9,
        hidden_dim=64,
        context_embed_dim=32,
        mask_dims=[],  # No masking
        num_heads=4,
        num_layers=2
    ).to(args.device)
    
    # Reuse same data loaders
    tf_acc = train_scorer_model(transformer_model, train_loader, val_loader, args.device, epochs=train_epochs)
    trained_models["TransformerScorer"] = {"model": transformer_model, "train_acc": tf_acc}
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    for model_name, info in trained_models.items():
        print(f"  {model_name}: Train Acc = {info['train_acc']:.4f}")
    
    # ============================================================
    # PHASE 2: EVALUATION
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: EVALUATING MODELS")
    print("="*60)
    
    # Reload data for evaluation
    combined_loader_eval = dataset.load_combined_dataset(principle_path)
    
    # Store results
    all_results = {}
    
    # Evaluate each trained model
    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        
        # Evaluate
        metrics = evaluate_model_on_dataset(
            model=model,
            model_name=model_name,
            obj_model=obj_model,
            data_loader=combined_loader_eval,
            principle=train_principle,
            device=args.device,
            grp_threshold=0.5,
            max_samples=max_eval_samples
        )
        
        # Add training accuracy to results
        metrics["train_accuracy"] = model_info["train_acc"]
        
        # Store results
        all_results[model_name] = metrics
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                f"{model_name}/train_accuracy": model_info["train_acc"],
                f"{model_name}/test_precision": metrics["precision"],
                f"{model_name}/test_recall": metrics["recall"],
                f"{model_name}/test_f1": metrics["f1"],
                f"{model_name}/test_accuracy": metrics["accuracy"]
            })
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Train Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Test Acc':<12}")
    print("-"*80)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name:<25} {metrics.get('train_accuracy', 0.0):<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['accuracy']:<12.4f}")
    
    # Find best model
    best_f1_model = max(all_results.items(), key=lambda x: x[1]['f1'])
    best_acc_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n" + "="*80)
    print(f"Best F1 Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
    print(f"Best Test Accuracy: {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.4f})")
    print("="*80)
    
    # Save results
    results_path = config.output / f"elvis_grouping_comparison_{train_principle}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "principle": train_principle,
            "timestamp": timestamp,
            "training_config": {
                "epochs": train_epochs,
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples
            },
            "results": all_results,
            "best_f1": best_f1_model[0],
            "best_accuracy": best_acc_model[0]
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    if use_wandb:
        wandb.save(str(results_path))
        wandb.finish()
    
    return all_results


if __name__ == "__main__":
    main()