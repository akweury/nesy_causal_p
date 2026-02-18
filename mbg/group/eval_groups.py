# Created by MacBook Pro at 23.04.25

import csv
import os
import networkx as nx
from itertools import combinations
import torch
import torch.nn.functional as F
import numpy as np

from mbg.scorer import scorer_config
from mbg.group import proximity_grouping
from mbg.group import symbolic_group_features
from mbg.group.neural_group_features import NeuralGroupEncoder
from mbg.group.gd_transformer import GroupingTransformer
from mbg.group.train_gd_transformer import load_group_transformer, train_grouping, GroupDataset
from src import bk
from torch.utils.data import DataLoader
from mbg.patch_preprocess import patch2code
import config


def embedding_principles(group_principle):
    principles = bk.gestalt_principles
    p_id = principles.index(group_principle)
    p_g = F.one_hot(torch.tensor(p_id), num_classes=len(principles)).float()
    return p_g


def embedding_group_neural_features(group_objs, device, input_dim=7):
    group_patches = torch.stack([o["h"] for o in group_objs])[
        :, :, :, :input_dim].to(device)  # (G, P, L, D)
    # build the encoder (dims must match your patch-encoder settings):
    group_encoder = NeuralGroupEncoder(
        input_dim=input_dim,
        obj_embed_dim=64,
        hidden_dim=128,
        group_embed_dim=128
    ).to(device)

    # get one group-embedding h_g of size 128:
    h_g = group_encoder(group_patches)  # → torch.Size([128])
    return h_g


def dict_group_features(group_objs):
    group_feature = symbolic_group_features.compute_symbolic_group_features(
        group_objs, 1024, 1024)
    s_g = group_feature.to_dict()
    return s_g


def construct_group_representations(objs, group_obj_ids, principle, input_dim, device):
    rep_gs = []
    for g_i, group_obj_id in enumerate(group_obj_ids):
        group_objs = [objs[i] for i in group_obj_id]
        # s_g = dict_group_features(group_objs)
        h_g = embedding_group_neural_features(group_objs, device, input_dim)
        p_g = principle
        grp = {
            "id": g_i,
            "child_obj_ids": group_obj_id,
            "members": group_objs,
            "h": h_g,
            "principle": p_g
        }
        rep_gs.append(grp)
    return rep_gs

def construct_clevr_group_representations(objs, group_obj_ids, principle, input_dim, device):
    rep_gs = []
    for g_i, group_obj_id in enumerate(group_obj_ids):
        group_objs = [objs[i] for i in group_obj_id]
        # s_g = dict_group_features(group_objs)
        h_g = None
        p_g = principle
        grp = {
            "id": g_i,
            "child_obj_ids": group_obj_id,
            "members": group_objs,
            "h": None,
            "principle": p_g
        }
        rep_gs.append(grp)
    return rep_gs

@torch.no_grad()
def group_clevr_objects_with_model(model, objs, device, threshold=0.5):
    """
    Group CLEVR objects using scorer models (SimplifiedPositionScorer or TransformerPositionScorer).
    Both models share the same interface: model(pos_i, pos_j, context_positions).
    
    Args:
        model: trained scorer model (SimplifiedPositionScorer or TransformerPositionScorer)
        objs: list of object dicts with 's' (symbolic) and 'h' (neural) features
              symbolic features contain 'x', 'y' position, 'color' RGB, and 'shape' one-hot
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped
    
    Returns:
        List of groups, each group is a list of object indices
    """
    model = model.to(device).eval()
    n = len(objs)
    
    if n <= 1:
        return [[i] for i in range(n)]
    
    # Build graph for connected components
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Extract positions, colors, and shapes from symbolic data
    features = []
    for obj in objs:
        symbolic = obj['s']
        x = symbolic['x']
        y = symbolic['y']
        # Extract RGB color values
        color = symbolic.get('color', [0.0, 0.0, 0.0])
        # Extract shape one-hot (4 dimensions for bk_shapes_clevr)
        shape = symbolic.get('shape', torch.zeros(4))
        if isinstance(shape, torch.Tensor):
            shape_list = shape.tolist()
        else:
            shape_list = list(shape)
        # Concatenate: x, y, r, g, b, shape[4]
        feat = [x, y, color[0], color[1], color[2]] + shape_list
        features.append(feat)
    
    # Check all pairs of objects
    for i, j in combinations(range(n), 2):
        # Get features (position + color + shape) for objects i and j
        pos_i = torch.tensor([features[i]], dtype=torch.float32).to(device)  # (1, 9)
        pos_j = torch.tensor([features[j]], dtype=torch.float32).to(device)  # (1, 9)
        
        # Get context features (all other objects)
        context_features = [features[k] for k in range(n) if k != i and k != j]
        
        if len(context_features) == 0:
            ctx_tensor = torch.zeros((1, 0, 9), dtype=torch.float32, device=device)
        else:
            ctx_tensor = torch.tensor([context_features], dtype=torch.float32).to(device)  # (1, N, 9)
        
        # Get grouping score
        logit = model(pos_i, pos_j, ctx_tensor)
        prob = torch.sigmoid(logit).item()
        
        if prob > threshold:
            G.add_edge(i, j)
    
    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups


@torch.no_grad()
def group_objects_with_model(model, objects, device, input_type="pos_color_size", threshold=0.5, dim=7):
    """
    Args:
        model: trained ContextContourScorer model
        objects: list of dicts, each with keys like 'position', 'color', 'size' depending on input_type
        input_type: one of 'pos', 'pos_color', 'pos_color_size'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped
    Returns:
        List of groups, each group is a list of object indices
    """
    model = model.to(device).eval()
    n = len(objects)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        ci, cj = objects[i].unsqueeze(0), objects[j].unsqueeze(0)
        context = [x for k, x in enumerate(objects) if k != i and k != j]
        if len(context) == 0:
            ctx_tensor = torch.zeros((1, 1, 6, 16, 7), device=device)
        else:
            ctx_tensor = torch.stack(context).unsqueeze(0).to(device)

        logit = model(ci[:, :, :, :dim], cj[:, :, :, :dim],
                      ctx_tensor[:, :, :, :, :dim])
        prob = torch.sigmoid(logit).item()
        if prob > threshold:
            G.add_edge(i, j)
    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups


def eval_groups(objs, group_model, principle, device, dim, grp_th=0.5):
    # symbolic_objs = [o["s"] for o in objs]
    neural_objs = [o["h"] for o in objs]
    if principle == "similarity":
        dim = 5

    group_ids = get_transformer_group_ids(
        transformer_model=group_model,
        objects=objs,
        device=device,
        threshold=grp_th)

    # group_ids = group_objects_with_model(
    #     group_model, neural_objs, device, dim=dim, threshold=grp_th)
    # encoding the groups
    groups = construct_group_representations(
        objs, group_ids, principle, dim, device)
    return groups



@torch.no_grad()
def get_transformer_group_ids(transformer_model, objects, device, threshold=0.5):
    """
    Use transformer group detector to get group IDs

    Args:
        transformer_model: trained GroupingTransformer model
        objects: list of object dicts with keys 'position', 'color', 'size', 'contour'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped

    Returns:
        List of groups, each group is a list of object indices
    """
    transformer_model = transformer_model.to(device).eval()

    n = len(objects)
    if n <= 1:
        return [[i] for i in range(n)]  # Each object in its own group

    # Extract features from objects
    positions = []
    colors = []
    sizes = []
    shapes = []
    positions = torch.tensor([[obj['s']['x'], obj['s']['y']]
                             for obj in objects]).to(device).unsqueeze(0)
    colors = torch.tensor([obj['s']['color']
                          for obj in objects]).to(device).unsqueeze(0)
    sizes = torch.tensor([[obj['s']['w']]
                         for obj in objects]).to(device).unsqueeze(0)

    obj_labels = [np.array(bk.bk_shapes_2)[
        obj['s']['shape'].bool().numpy()][0] for obj in objects]
    obj_patches = torch.stack([obj['h'].reshape(-1, 2) for obj in objects])
    shape_code = patch2code(
        obj_patches, obj_labels=obj_labels, device=device).unsqueeze(0)
    # (N, 16)

    # Get predictions from transformer
    with torch.no_grad():
        pred = transformer_model(
            positions, sizes)  # (1, N, N)
        pred = torch.sigmoid(pred).squeeze(0)  # (N, N)

    # Build graph from predictions
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if pred[i, j].item() > threshold:
                G.add_edge(i, j)

    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups


@torch.no_grad()
def get_transformer_clevr_group_ids(transformer_model, objects, device, threshold=0.5):
    """
    Use transformer group detector to get group IDs for CLEVR dataset.
    Uses the same shape encoding as training (simple one-hot based on shape ID).

    Args:
        transformer_model: trained GroupingTransformer model
        objects: list of object dicts with keys 'position', 'color', 'size', 'shape'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped

    Returns:
        List of groups, each group is a list of object indices
    """
    transformer_model = transformer_model.to(device).eval()

    n = len(objects)
    if n <= 1:
        return [[i] for i in range(n)]  # Each object in its own group

    # Extract features from objects - matching training format
    positions = torch.tensor([[obj['s']['x'], obj['s']['y']]
                             for obj in objects], dtype=torch.float32).to(device).unsqueeze(0)
    
    # Handle color - can be either a combined array or separate r,g,b values
    color_list = []
    for obj in objects:
        if 'color' in obj['s']:
            color_list.append(obj['s']['color'])
        else:
            # Construct from separate r,g,b values
            color_list.append([obj['s']['color_r'], obj['s']['color_g'], obj['s']['color_b']])
    colors = torch.tensor(color_list, dtype=torch.float32).to(device).unsqueeze(0)
    # Normalize colors to [0, 1] if they're in [0, 255] range
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # Handle size - can be 'w' or 'size'
    size_list = []
    for obj in objects:
        if 'w' in obj['s']:
            size_list.append([obj['s']['w']])
        else:
            size_list.append([obj['s']['size']])
    sizes = torch.tensor(size_list, dtype=torch.float32).to(device).unsqueeze(0)

    # Create shape embeddings the same way as training
    # Extract shape_id from one-hot encoding
    shape_embeddings = []
    for obj in objects:
        shape_one_hot = obj['s']['shape']  # Already a tensor
        # Find which index is 1 (the shape_id + 1 due to background offset)
        shape_indices = shape_one_hot.nonzero(as_tuple=True)[0]
        if len(shape_indices) > 0:
            shape_idx = shape_indices[0].item() - 1  # Remove background offset
        else:
            shape_idx = 0  # Default
        
        # Create simple one-hot embedding (16-dim) - same as training
        emb = torch.zeros(16, dtype=torch.float32)
        if 0 <= shape_idx < 16:
            emb[shape_idx] = 1.0
        shape_embeddings.append(emb)
    
    shape_code = torch.stack(shape_embeddings).to(device).unsqueeze(0)  # (1, N, 16)

    # Get predictions from transformer
    with torch.no_grad():
        pred = transformer_model(
            positions, sizes)  # (1, N, N)
        pred = torch.sigmoid(pred).squeeze(0)  # (N, N)

    # Build graph from predictions
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if pred[i, j].item() > threshold:
                G.add_edge(i, j)

    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups




def init_gd_transformer_model(train_data, val_data, device, obj_model, epochs=50, lr=1e-4):
    """Initialize and train a new group detector transformer model on task data.
    
    Args:
        train_data: Training dataset with positive and negative samples
        val_data: Validation dataset for evaluation during training
        device: Device to train on (cuda/cpu)
        obj_model: Object detection model (currently not used as we use ground truth)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained GroupingTransformer model
    """
    print("Creating dataset from training data...")
    
    # Prepare data list from train_data
    data_list = []
    
    # Process both positive and negative samples
    for samples in [train_data["positive"], train_data["negative"]]:
        for sample in samples:
            symbolic_data = sample["symbolic_data"]
            
            if len(symbolic_data) < 2:
                continue
            
            # Extract features from symbolic data
            positions = []
            colors = []
            sizes = []
            shapes = []
            groups = []
            
            for obj in symbolic_data:
                positions.append([obj['x'], obj['y']])
                colors.append([0,0,0])
                sizes.append([obj['size']])
                shapes.append(0)  # Already converted to shape_id
                groups.append(obj.get('group_id', 0) if obj.get('group_id') is not None else 0)
            
            # Convert shapes to embeddings (use simple one-hot or ID for now)
            # Since we don't have contours, we'll use a simple embedding
            shape_embeddings = []
            for shape_id in shapes:
                # Create a simple shape embedding (16-dim)
                emb = torch.zeros(16)
                if shape_id < 16:
                    emb[shape_id] = 1.0
                shape_embeddings.append(emb)
            
            data_list.append({
                "pos": positions,
                "size": sizes,
                "group": groups,
            })
    
    # Also process validation data
    val_data_list = []
    for samples in [val_data["positive"], val_data["negative"]]:
        for sample in samples:
            symbolic_data = sample["symbolic_data"]
            
            if len(symbolic_data) < 2:
                continue
            
            positions = []
            colors = []
            sizes = []
            shapes = []
            groups = []
            
            for obj in symbolic_data:
                positions.append([obj['x'], obj['y']])
                colors.append([0,0,0])
                sizes.append([obj['size']])
                shapes.append(0)  # Already converted to shape_id
                groups.append(obj.get('group_id', 0) if obj.get('group_id') is not None else 0)
            
            shape_embeddings = []
            for shape_id in shapes:
                emb = torch.zeros(16)
                if shape_id < 16:
                    emb[shape_id] = 1.0
                shape_embeddings.append(emb)
            
            val_data_list.append({
                "pos": positions,
                "size": sizes,
                "group": groups,
            })
    
    print(f"Created dataset with {len(data_list)} training samples and {len(val_data_list)} validation samples")
    
    # Save training data to CSV file
    csv_output_dir = config.get_proj_output_path(False) / "group_training_data"
    os.makedirs(csv_output_dir, exist_ok=True)
    task_name = train_data.get("task", "unknown")
    csv_path = csv_output_dir / f"{task_name}_training_data.csv"
    
    print(f"Saving training data to CSV: {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header - determine max number of objects from the data
        max_objs = max([len(d["pos"]) for d in data_list])
        header = ["sample_id", "num_objects"]
        for i in range(max_objs):
            header.extend([f"obj{i}_x", f"obj{i}_y", f"obj{i}_size", f"obj{i}_group_id"])
        csv_writer.writerow(header)
        
        # Write data rows
        for sample_id, sample in enumerate(data_list):
            num_objs = len(sample["pos"])
            row = [sample_id, num_objs]
            
            for i in range(num_objs):
                row.append(sample["pos"][i][0])  # x
                row.append(sample["pos"][i][1])  # y
                row.append(sample["size"][i][0])  # size
                row.append(sample["group"][i])   # group_id
            
            # Pad with None for missing objects
            for i in range(num_objs, max_objs):
                row.extend([None, None, None, None])
            
            csv_writer.writerow(row)
    
    print(f"Saved {len(data_list)} training samples to {csv_path}")
    
    # Save validation data to CSV file
    if len(val_data_list) > 0:
        val_csv_path = csv_output_dir / f"{task_name}_validation_data.csv"
        print(f"Saving validation data to CSV: {val_csv_path}")
        
        with open(val_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header
            max_objs = max([len(d["pos"]) for d in val_data_list])
            header = ["sample_id", "num_objects"]
            for i in range(max_objs):
                header.extend([f"obj{i}_x", f"obj{i}_y", f"obj{i}_size", f"obj{i}_group_id"])
            csv_writer.writerow(header)
            
            # Write data rows
            for sample_id, sample in enumerate(val_data_list):
                num_objs = len(sample["pos"])
                row = [sample_id, num_objs]
                
                for i in range(num_objs):
                    row.append(sample["pos"][i][0])  # x
                    row.append(sample["pos"][i][1])  # y
                    row.append(sample["size"][i][0])  # size
                    row.append(sample["group"][i])   # group_id
                
                # Pad with None for missing objects
                for i in range(num_objs, max_objs):
                    row.extend([None, None, None, None])
                
                csv_writer.writerow(row)
        
        print(f"Saved {len(val_data_list)} validation samples to {val_csv_path}")
    
    if len(data_list) == 0:
        print("WARNING: No training data available, returning untrained model")
        model = GroupingTransformer(
            shape_dim=16,
            app_dim=0,
            d_model=128,
            num_heads=4,
            depth=4,
            rel_dim=64
        ).to(device)
        return model
    
    # Create datasets and loaders
    train_dataset = GroupDataset(data_list)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    val_loader = None
    if len(val_data_list) > 0:
        val_dataset = GroupDataset(val_data_list)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = GroupingTransformer(
        shape_dim=16,
        app_dim=0,
        d_model=128,
        num_heads=4,
        depth=4,
        rel_dim=64
    ).to(device)
    
    print(f"Training group detector for {epochs} epochs...")
    
    # Train the model (no save path as we return the model directly)
    best_acc, best_loss = train_grouping(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        log_interval=max(1, len(train_loader) // 5),  # Log 5 times per epoch
        save_path=None  # Don't save to disk
    )
    
    print(f"Training completed. Best metric: {best_acc:.4f}")
    
    return model

def eval_clevr_groups(objs, group_model, principle, device, dim, grp_th=0.5):
    """Evaluate groups for CLEVR dataset using trained group detector.
    
    Args:
        objs: list of object dicts with 's' (symbolic) and 'h' (neural) features
        group_model: trained model - one of:
            - SimplifiedPositionScorer (NN-based scorer)
            - TransformerPositionScorer (Transformer-based scorer)
            - GroupingTransformer (old transformer model)
        principle: gestalt principle (e.g., 'proximity', 'similarity')
        device: cuda or cpu
        dim: dimension for neural features
        grp_th: threshold for grouping probability
    
    Returns:
        List of group dicts with id, child_obj_ids, members, h (embedding), principle
    """
    if principle == "similarity":
        dim = 5

    # Check model type and use appropriate grouping function
    model_type = type(group_model).__name__
    
    if model_type in ["SimplifiedPositionScorer", "TransformerPositionScorer"]:
        # Use scorer models (SimplifiedPositionScorer or TransformerPositionScorer)
        # Both use the same interface with pos_i, pos_j, context_positions
        group_ids = group_clevr_objects_with_model(
            model=group_model,
            objs=objs,
            device=device,
            threshold=grp_th
        )
    else:
        # Use old transformer model (GroupingTransformer with different interface)
        group_ids = get_transformer_clevr_group_ids(
            transformer_model=group_model,
            objects=objs,
            device=device,
            threshold=grp_th
        )

    # Construct group representations
    groups = construct_clevr_group_representations(
        objs, group_ids, principle, dim, device
    )
    
    return groups