import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tqdm import tqdm
import json
from pathlib import Path
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available, training metrics will not be logged")
    WANDB_AVAILABLE = False
import argparse
import torch.nn.functional as F

import config
from src import bk
from mbg.group.gd_transformer import GroupingTransformer, ShapeEmbedding, contour_to_fd8
from mbg import patch_preprocess
from mbg.object import eval_patch_classifier


try:
    from rtpt import RTPT
    RTPT_AVAILABLE = True
except ImportError:
    RTPT_AVAILABLE = False
    RTPT = None


def get_data_list(root_dir, task_num, device="cpu", train_test_split=0.8, obj_model=None):
    """
    Args:
        root_dir: Path to data directory
        task_num: Number of tasks to process
        device: Device to place tensors on
        train_test_split: Fraction of data for training (rest for testing)
        obj_model: Trained object detection model (if None, will load default)
    
    Returns:
        train_data_list, test_data_list: Split datasets
    """
    data_list = []
    
    # Load object detection model if not provided
    if obj_model is None:
        print("Loading object detection model...")
        obj_model = eval_patch_classifier.load_model(device)
        print("Object detection model loaded.")

    # Get all task directories and randomly shuffle them
    task_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    random.shuffle(task_dirs)
    # select only up to task_num
    task_dirs = task_dirs[:task_num] if task_num < len(
        task_dirs) else task_dirs

    shape_encoder = ShapeEmbedding(
        num_shapes=len(bk.bk_shapes_2),
        contour_dim=8,
        hidden_dim=32,
        shape_dim=16
    ).to(device)
    
    for task_dir in tqdm(task_dirs, desc="Generate Dataset"):
        for label in ["positive", "negative"]:
            label_dir = task_dir / label
            if not label_dir.exists():
                continue
            json_files = sorted(label_dir.glob("*.json"))
            png_files = sorted(label_dir.glob("*.png"))
            for f_i, json_file in enumerate(json_files):
                # Load ground truth for group labels only
                with open(json_file) as f:
                    metadata = json.load(f)
                gt_objects = metadata.get("img_data", [])
                
                if len(gt_objects) < 2:
                    continue
                    
                # Use object detector to get detected objects
                img_path = png_files[f_i]
                img = patch_preprocess.load_images_fast([img_path], device=device)[0]
                detected_objs = eval_patch_classifier.evaluate_image(obj_model, img, device)
                
                if len(detected_objs) < 2:
                    continue
                
                # Match detected objects to ground truth for group labels
                # Simple matching based on position proximity
                matched_groups = []
                for det_obj in detected_objs:
                    det_pos = [det_obj['s']['x'], det_obj['s']['y']]  # x, y position
                    
                    # Find closest ground truth object
                    min_dist = float('inf')
                    best_match_group = 0  # default group
                    
                    for gt_obj in gt_objects:
                        gt_pos = [gt_obj['x'], gt_obj['y']]
                        dist = ((det_pos[0] - gt_pos[0])**2 + (det_pos[1] - gt_pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_match_group = gt_obj['group_id']
                    
                    matched_groups.append(best_match_group)

                # Extract features from detected objects
                positions = []
                colors = []
                sizes = []
                shapes = []
                contours = []
                
                for det_obj in detected_objs:
                    # Position
                    positions.append([det_obj['s']['x'], det_obj['s']['y']])  # x, y
                    
                    # Color (RGB)
                    colors.append(det_obj['s']['color'])  # r, g, b
                    
                    # Size
                    if isinstance(det_obj['s']['w'], (list, tuple)):
                        sizes.append([det_obj['s']['w'][0]])
                    else:
                        sizes.append([det_obj['s']['w']])
                    
                    # Shape - map detected shape to shape ID
                    detected_shape = bk.bk_shapes_2[det_obj['s']['shape'].argmax()]
                    shapes.append(detected_shape)
                
                
                # For contours, we need to extract them from detected objects
                # Assuming detected objects have contour information
                obj_contours = []
                for det_obj in detected_objs:
                    obj_contours.append(det_obj['h'].reshape(-1,2))
                                    
                obj_contours = torch.stack(obj_contours).to(device)     
                
                # Create shape embeddings
                shape_embeddings = patch_preprocess.patch2code(obj_contours, obj_labels=shapes, device=device)
                data_list.append({
                    "pos": positions,
                    "color": colors,
                    "size": sizes,
                    "shape": shape_embeddings,
                    "contour": shape_embeddings,
                    "group": matched_groups,
                })

    # Split data into train and test
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_test_split)
    train_data_list = data_list[:split_idx]
    test_data_list = data_list[split_idx:]
    
    print(f"Data split: {len(train_data_list)} train, {len(test_data_list)} test")
    return train_data_list, test_data_list

# -------------------------------------------------------------------------
# 1. Dataset Example
# -------------------------------------------------------------------------


class GroupDataset(Dataset):
    """
    Each sample:
        pos:   (N,2)
        color: (N,3)
        size:  (N,1)
        shape: (N,D_shape)
        groups: list of group indices, e.g. [0,0,1,1,1,2] length N
    """

    def __init__(self, data_list):
        super().__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        pos = torch.tensor(item["pos"], dtype=torch.float32)
        size = torch.tensor(item["size"], dtype=torch.float32)
        groups = torch.tensor(item["group"], dtype=torch.long)

        # Create ground-truth affinity matrix (N,N)
        N = len(groups)
        gt = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if groups[i] == groups[j]:
                    gt[i, j] = 1.0
        gt.fill_diagonal_(0.0)  # no self-links

        return pos, size, gt


def custom_collate_fn(batch):
    """
    Custom collate function to handle scenes with different numbers of objects.
    Since each scene can have different N (number of objects), we cannot batch them.
    Instead, we return a list of individual samples.
    """
    # For grouping tasks, we process one scene at a time
    # Return the first item in the batch (assuming batch_size=1)
    if len(batch) == 1:
        return batch[0]
    else:
        # If batch_size > 1, we need to process each scene separately
        # This is a limitation of grouping tasks with variable object counts
        raise ValueError(
            "Batch size > 1 not supported for scenes with variable object counts. "
            "Please use batch_size=1 for grouping tasks."
        )


# -------------------------------------------------------------------------
# 2. Grouping Loss (BCE + stability contrastive term)
# -------------------------------------------------------------------------
class GroupingLoss(nn.Module):
    """
    BCE on pairwise linking + optional stability regularization.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_affinity, gt_affinity):
        # pred_affinity: (B,N,N)
        # gt_affinity:   (B,N,N)
        loss = self.bce(pred_affinity, gt_affinity)
        return loss


# -------------------------------------------------------------------------
# 3. Training Script
# -------------------------------------------------------------------------
def train_grouping(model,
                   train_loader,
                   test_loader=None,
                   device="cuda",
                   lr=1e-4,
                   epochs=50,
                   log_interval=10,
                   save_path=None):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = GroupingLoss()

    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (pos, size, gt) in enumerate(train_loader):
            # Each scene has different N (number of objects)
            # pos: (N, 2), size: (N, 1), gt: (N, N)
            
            # Add batch dimension for single scene
            pos = pos.squeeze().unsqueeze(0).to(device)  # (1, N, 2)
            size = size.squeeze().unsqueeze(0).unsqueeze(2).to(device)  # (1, N, 1)
            gt = gt.squeeze().unsqueeze(0).to(device)  # (1, N, N)

            optimizer.zero_grad()

            # Forward pass
            pred = model(pos, size)

            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # if batch_idx % log_interval == 0:
            #     print(
            #         f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # Calculate train accuracy
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for pos, size, gt in train_loader:
                # Add batch dimension for single scene
                pos = pos.squeeze().unsqueeze(0).to(device)  # (1, N, 2)
                size = size.squeeze().unsqueeze(0).unsqueeze(2).to(device)  # (1, N, 1)
                gt = gt.squeeze().unsqueeze(0).to(device)  # (1, N, N)
                
                pred = model(pos, size)
                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                train_correct += (pred_binary == gt).sum().item()
                train_total += gt.numel()
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Evaluate on test data if available
        test_accuracy = 0
        test_loss = 0
        if test_loader:
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss_total = 0
            with torch.no_grad():
                for pos, size, gt in test_loader:
                    # Add batch dimension for single scene
                    pos = pos.squeeze().unsqueeze(0).to(device)  # (1, N, 2)
                    size = size.squeeze().unsqueeze(0).unsqueeze(2).to(device)  # (1, N, 1)
                    gt = gt.squeeze().unsqueeze(0).to(device)  # (1, N, N)
                    
                    pred = model(pos, size)
                    loss = criterion(pred, gt)
                    test_loss_total += loss.item()
                    
                    pred_binary = (torch.sigmoid(pred) > 0.5).float()
                    test_correct += (pred_binary == gt).sum().item()
                    test_total += gt.numel()
            
            test_accuracy = test_correct / test_total if test_total > 0 else 0
            test_loss = test_loss_total / len(test_loader)
        
        # Log train and test accuracy once per epoch
        if WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch,
                "train_accuracy": train_accuracy
            }
            if test_loader:
                log_dict["test_accuracy"] = test_accuracy
            # wandb.log(log_dict)

        # Save best model based on test accuracy if available, otherwise train loss
        current_metric = test_accuracy if test_loader else -avg_loss  # Use negative loss for minimization
        if save_path and current_metric > best_acc:
            best_loss = avg_loss
            best_acc = current_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'test_loss': test_loss if test_loader else None,
                'test_accuracy': test_accuracy if test_loader else None
            }, save_path)
            metric_name = "test accuracy" if test_loader else "train loss"
            print(
                f"Saved best model with {metric_name} {best_acc:.4f}")

        # Print epoch results
        if test_loader:
            print(
                f"==> Epoch {epoch} | Train Acc {train_accuracy:.4f} | Test Acc {test_accuracy:.4f}")
        else:
            print(
                f"==> Epoch {epoch} | Train Acc {train_accuracy:.4f}")

    return best_acc, best_loss


def load_group_transformer(model_path, device="cuda", shape_dim=16, app_dim=0, d_model=128, num_heads=4, depth=4, rel_dim=64):
    """
    Load a trained GroupingTransformer model from checkpoint
    
    Args:
        model_path: path to the saved model checkpoint
        device: device to load model on ('cuda' or 'cpu')
        shape_dim: shape embedding dimension (must match training config)
        app_dim: appearance dimension (must match training config)  
        d_model: transformer model dimension (must match training config)
        num_heads: number of attention heads (must match training config)
        depth: number of transformer layers (must match training config)
        rel_dim: relative geometry dimension (must match training config)
    
    Returns:
        model: loaded GroupingTransformer model in eval mode
        checkpoint_info: dict with training info (epoch, accuracy, loss, etc.)
    """
    # Initialize model with same architecture as training
    model = GroupingTransformer(
        shape_dim=shape_dim,
        app_dim=app_dim,
        d_model=d_model,
        num_heads=num_heads,
        depth=depth,
        rel_dim=rel_dim
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': checkpoint.get('train_loss', 'unknown'),
            'test_loss': checkpoint.get('test_loss', 'unknown'),
            'test_accuracy': checkpoint.get('test_accuracy', 'unknown')
        }
    else:
        # Assume checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
        checkpoint_info = {'epoch': 'unknown', 'train_loss': 'unknown', 'test_loss': 'unknown', 'test_accuracy': 'unknown'}
    
    model.eval()
    print(f"Loaded GroupingTransformer from {model_path}")
    print(f"Checkpoint info: {checkpoint_info}")
    
    return model, checkpoint_info


def evaluate_model(model, test_loader, device="cuda"):
    """
    Evaluate model on test data
    """
    model.eval()
    criterion = GroupingLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pos, color, size, shape, gt in test_loader:
            # Add batch dimension for single scene
            pos = pos.unsqueeze(0).to(device)
            size = size.unsqueeze(0).to(device)
            gt = gt.unsqueeze(0).to(device)
            
            pred = model(pos, size)
            loss = criterion(pred, gt)
            total_loss += loss.item()
            
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_binary == gt).sum().item()
            total += gt.numel()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def parse_device(device_str):
    if device_str.isdigit():
        return f"cuda:{device_str}"
    elif device_str.startswith("cuda") or device_str == "cpu":
        return device_str
    else:
        raise ValueError(f"Invalid device string: {device_str}")

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                        else "cpu", help="Device to train on")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--principle", type=str, default="proximity")
    parser.add_argument("--input_types", type=str, default="pos_color_size")
    parser.add_argument("--task_num", type=int, default=10,
                        help="Number of tasks to process")
    parser.add_argument("--remove_cache", action="store_true",
                        help="Remove existing cache files before processing")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    args.device = parse_device(args.device)
    if RTPT_AVAILABLE:
        rtpt = RTPT(name_initials='JIS',
                    experiment_name=f'GRMGDTR{args.principle}', max_iterations=1)
        rtpt.start()

    return args


# -------------------------------------------------------------------------
# 4. Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------
    # Dummy dataset (replace with your GRM object-extractor outputs)
    # -------------------------------------------------------------
    args = init()
    base_dir = config.get_raw_patterns_path(args.remote)
    task_num = args.task_num
    data_list_path = base_dir / args.principle / "train" / "data_list.pkl"
    device = args.device
    print(f"Using device: {device}")

    if data_list_path.exists():
        print(f"Loading cached data_list from {data_list_path}")
        with open(data_list_path, "rb") as f:
            cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and "train" in cached_data:
                # New format with train/test split
                train_data_list = cached_data["train"]
                test_data_list = cached_data["test"]
                print(f"Loaded train/test split: {len(train_data_list)} train, {len(test_data_list)} test")
            else:
                # Old format, need to split
                print("Converting old format to train/test split...")
                data_list = cached_data
                random.shuffle(data_list)
                split_idx = int(len(data_list) * 0.8)
                train_data_list = data_list[:split_idx]
                test_data_list = data_list[split_idx:]
                print(f"Split data: {len(train_data_list)} train, {len(test_data_list)} test")
                
        # Move data to device
        print("Moving data to device...")
        for i, single_data in enumerate(train_data_list):
            if isinstance(single_data, dict):
                train_data_list[i] = {"pos": torch.tensor(single_data["pos"]).to(device) if not isinstance(single_data["pos"], torch.Tensor) else single_data["pos"].to(device), 
                                     "color": torch.tensor(single_data["color"]).to(device) if not isinstance(single_data["color"], torch.Tensor) else single_data["color"].to(device), 
                                     "size": torch.tensor(single_data["size"]).to(device) if not isinstance(single_data["size"], torch.Tensor) else single_data["size"].to(device), 
                                     "contour": torch.tensor(single_data["contour"]).to(device) if not isinstance(single_data["contour"], torch.Tensor) else single_data["contour"].to(device), 
                                     "group": torch.tensor(single_data["group"]).to(device) if not isinstance(single_data["group"], torch.Tensor) else single_data["group"].to(device)}
        for i, single_data in enumerate(test_data_list):
            if isinstance(single_data, dict):
                test_data_list[i] = {"pos": torch.tensor(single_data["pos"]).to(device) if not isinstance(single_data["pos"], torch.Tensor) else single_data["pos"].to(device), 
                                    "color": torch.tensor(single_data["color"]).to(device) if not isinstance(single_data["color"], torch.Tensor) else single_data["color"].to(device), 
                                    "size": torch.tensor(single_data["size"]).to(device) if not isinstance(single_data["size"], torch.Tensor) else single_data["size"].to(device), 
                                    "contour": torch.tensor(single_data["contour"]).to(device) if not isinstance(single_data["contour"], torch.Tensor) else single_data["contour"].to(device), 
                                    "group": torch.tensor(single_data["group"]).to(device) if not isinstance(single_data["group"], torch.Tensor) else single_data["group"].to(device)}
    else:
        print("Generating new data...")
        # Load object detection model
        print("Loading object detection model for data generation...")
        obj_model = eval_patch_classifier.load_model(device)
        print("Object detection model loaded.")
        
        train_data_list, test_data_list = get_data_list(
            base_dir / args.principle / "train", task_num=task_num, device=device, train_test_split=0.8, obj_model=obj_model)
        # save data_list to a file for fast loading next time
        with open(data_list_path, "wb") as f:
            # Save in CPU format to avoid device issues when loading
            cpu_train_data = [{"pos":single_data["pos"], 
                             "color":single_data["color"], 
                             "size":single_data["size"], 
                             "contour":single_data["contour"].tolist(), 
                             "group":single_data["group"]} for single_data in train_data_list]
            cpu_test_data = [{"pos":single_data["pos"], 
                             "color":single_data["color"], 
                             "size":single_data["size"], 
                             "contour":single_data["contour"].tolist(), 
                             "group":single_data["group"]} for single_data in test_data_list]
            pickle.dump({"train": cpu_train_data, "test": cpu_test_data}, f)
            
    # Create train and test datasets
    train_dataset = GroupDataset(train_data_list)
    test_dataset = GroupDataset(test_data_list)
    
    # IMPORTANT: Use batch_size=1 because each scene has different number of objects
    # Variable object counts make true batching impossible without padding/masking
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # -------------------------------------------------------------
    # Initialize model
    # -------------------------------------------------------------
    print(f"Initializing model on {device}...")
    model = GroupingTransformer(
        shape_dim=16,
        app_dim=0,
        d_model=128,
        num_heads=4,
        depth=4,
        rel_dim=64
    ).to(device)

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------
    # Initialize wandb for this training run

    wandb.init(project=f"gd_transformer_{args.principle}", config={
        "epochs": args.epochs,
        "lr": 1e-4,
        "device": args.device,
        "principle": args.principle if hasattr(args, 'principle') else "unknown"
    })

    # Setup model saving
    model_dir = config.get_proj_output_path(
        getattr(args, 'remote', False)) / "models"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / \
        f"gd_transformer_{getattr(args, 'principle', 'unknown')}_standalone.pt"

    train_grouping(model,
                   train_loader,
                   test_loader,
                   device=device,
                   lr=1e-4,
                   epochs=args.epochs,
                   save_path=str(save_path))

    # Training is complete - test evaluation happens during training
    print("\n=== Training Complete ===")
    wandb.finish()
