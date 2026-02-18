import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import random
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

from mbg.grounding.candidate_generator import generate_candidate_groups
from mbg.grounding.predicates_real import ObjectInstance, describe_group
from src.coco_model.coco_group_reasoner import (
    reason_over_groups
)

def load_coco_objects_from_image(coco_json, depth_dir, conf_dir, image_idx=0):
    """
    Load objects from a real COCO image with depth information.
    
    Args:
        coco_json: Path to COCO annotations JSON file
        depth_dir: Directory containing depth maps
        conf_dir: Directory containing confidence maps
        image_idx: Index of image to load (default: 0)
    
    Returns:
        objects: Dict of ObjectInstance objects
        image_path: Path to the image file
        image_diag: Image diagonal for normalization
    """
    # Load COCO data
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Build category ID to name mapping
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    
    # Get image and its annotations
    images = coco["images"]
    if image_idx >= len(images):
        image_idx = 0
    
    img_info = images[image_idx]
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_width = img_info["width"]
    img_height = img_info["height"]
    
    print(f"Loading image: {file_name} (ID: {img_id})")
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Get all annotations for this image
    annotations = [ann for ann in coco["annotations"] 
                   if ann["image_id"] == img_id and ann.get("iscrowd", 0) == 0]
    
    # Load depth map
    stem = file_name.replace(".jpg", "")
    depth_path = os.path.join(depth_dir, f"{stem}_depth.npz")
    conf_path = os.path.join(conf_dir, f"{stem}_conf.png")
    
    if os.path.exists(depth_path):
        depth = np.load(depth_path)["depth"]
        conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    else:
        print(f"Warning: Depth map not found at {depth_path}, using default depth")
        depth = np.ones((img_height, img_width)) * 2.0
        conf = np.ones((img_height, img_width))
    
    # Pool depth for each object
    def pool_object_depth(depth, conf, bbox):
        x, y, w, h = map(int, bbox)
        patch_d = depth[y:y+h, x:x+w]
        patch_c = conf[y:y+h, x:x+w]
        
        if patch_d.size == 0:
            return 2.0  # default depth
        
        return np.median(patch_d.flatten())
    
    # Create ObjectInstance objects
    objects = {}
    for i, ann in enumerate(annotations):
        bbox = ann["bbox"]  # [x, y, w, h]
        obj_depth = pool_object_depth(depth, conf, bbox)
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, f"category_{cat_id}")
        
        objects[i] = ObjectInstance(
            oid=i,
            bbox=tuple(bbox),
            depth=float(obj_depth),
            category=cat_name
        )
    
    print(f"Loaded {len(objects)} objects from image")
    
    # Calculate image diagonal for normalization
    image_diag = np.sqrt(img_width**2 + img_height**2)
    
    # Build image path
    coco_path = Path(coco_json).parent.parent
    image_path = coco_path / "val2017" / file_name
    
    return objects, str(image_path), image_diag


def get_group_type_name(group_id):
    """
    Extract group type name from group ID.
    
    Args:
        group_id: Group ID string (e.g., 'prox_0', 'depth_near', 'cat_orange', 'size_0')
    
    Returns:
        Human-readable group type name
    """
    if group_id.startswith('prox_'):
        return "Proximity Group"
    elif group_id.startswith('depth_'):
        return "Depth Group"
    elif group_id.startswith('cat_'):
        return "Category Group"
    elif group_id.startswith('size_'):
        return "Size Similarity Group"
    elif group_id.startswith('func_'):
        return "Functional Group"
    else:
        return "Group"


def visualize_all_objects_clean(image_path, objects, output_path):
    """
    Visualize all objects in the image with bounding boxes only (no text, no legend).
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        output_path: Path to save the visualization
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # Draw bounding boxes for all objects with different colors
    num_objects = len(objects)
    for idx, (oid, obj) in enumerate(objects.items()):
        x, y, w, h = obj.bbox
        color = hsv_to_rgb([idx / max(num_objects, 1), 0.8, 0.9])
        
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"  Saved all objects: {output_path}")


def visualize_group_clean(image_path, objects, group, output_path):
    """
    Visualize a group with only bounding boxes (no text, no legend, no titles).
    Shows group bounding box and member bounding boxes.
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        group: Group object with members list
        output_path: Path to save the visualization
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # Use a single color for the entire group
    group_color = hsv_to_rgb([hash(group.gid) % 100 / 100.0, 0.8, 0.9])
    
    # Calculate bounding box for the entire group
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    # Draw bounding boxes for each member
    for oid in group.members:
        obj = objects[oid]
        x, y, w, h = obj.bbox
        
        # Update group bounding box
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        
        # Draw member rectangle
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor=group_color,
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Draw the group bounding box
    group_bbox_width = max_x - min_x
    group_bbox_height = max_y - min_y
    group_rect = patches.Rectangle(
        (min_x, min_y), group_bbox_width, group_bbox_height,
        linewidth=3,
        edgecolor=group_color,
        facecolor='none',
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(group_rect)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"  Saved clean: {output_path}")


def visualize_group(image_path, objects, group, output_path, group_id, reasoning_info=None):
    """
    Visualize a single group by drawing bounding boxes around member objects.
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        group: Group object with members list
        output_path: Path to save the visualization
        group_id: ID of the group for the filename
        reasoning_info: Optional dict with reasoning information (score, role, etc.)
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with image on left and legend on right
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.1)
    
    # Left subplot: Image with bounding boxes only
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img)
    
    # Right subplot: Legend with object information
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')
    
    # Use a single color for the entire group
    group_color = hsv_to_rgb([hash(group.gid) % 100 / 100.0, 0.8, 0.9])
    
    # Calculate bounding box for the entire group
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    # Draw bounding boxes for each member (no text on image)
    legend_info = []
    for oid in group.members:
        obj = objects[oid]
        x, y, w, h = obj.bbox
        
        # Update group bounding box
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        
        # Draw rectangle on image with same color for all members
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor=group_color,
            facecolor='none'
        )
        ax_img.add_patch(rect)
        
        # Collect info for legend
        legend_info.append({
            'oid': oid,
            'category': obj.category,
            'depth': obj.depth,
        })
    
    # Draw the group bounding box
    group_bbox_width = max_x - min_x
    group_bbox_height = max_y - min_y
    group_rect = patches.Rectangle(
        (min_x, min_y), group_bbox_width, group_bbox_height,
        linewidth=3,
        edgecolor=group_color,
        facecolor='none',
        linestyle='--',
        alpha=0.7
    )
    ax_img.add_patch(group_rect)
    
    # Add title with group type and information
    group_type = get_group_type_name(group.gid)
    member_ids = ', '.join(map(str, group.members))
    title = f"{group_type}\nGroup {group_id}: [{member_ids}] ({len(group.members)} objects)"
    
    # Add reasoning role if provided
    if reasoning_info and 'role' in reasoning_info:
        role = reasoning_info['role']
        if role == 'main':
            title = f"⭐ MAIN GROUP ⭐\n{title}"
        elif role == 'background':
            title = f"🌫️ BACKGROUND\n{title}"
        else:
            title = f"OTHER\n{title}"
    
    ax_img.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax_img.axis('off')
    
    # Create legend on the right side
    legend_text = "Object Information:\n" + "="*30 + "\n\n"
    for info in legend_info:
        legend_text += f"● ID: {info['oid']}\n"
        legend_text += f"  Category: {info['category']}\n"
        legend_text += f"  Depth: {info['depth']:.2f}m\n\n"
    
    # Add reasoning information if provided
    if reasoning_info:
        legend_text += "\n" + "="*30 + "\n"
        legend_text += "Reasoning:\n" + "="*30 + "\n"
        if 'score' in reasoning_info:
            legend_text += f"Score: {reasoning_info['score']}\n"
        if 'summary' in reasoning_info:
            summary = reasoning_info['summary']
            legend_text += f"Size: {summary.get('size', 'N/A')}\n"
            legend_text += f"Depth: {summary.get('depth', 'N/A')}\n"
            legend_text += f"Category: {summary.get('dominant_category', 'N/A')}\n"
            legend_text += f"Compact: {summary.get('compact', False)}\n"
            legend_text += f"Functional: {summary.get('functional', False)}\n"
    
    ax_legend.text(
        0.05, 0.95, legend_text,
        transform=ax_legend.transAxes,
        fontsize=11,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor='gray', alpha=0.9)
    )
    
    # Add group color indicator at the top
    group_color_patch = patches.Rectangle(
        (0.01, 0.97), 0.03, 0.02,
        transform=ax_legend.transAxes,
        facecolor=group_color,
        edgecolor='black',
        linewidth=1
    )
    ax_legend.add_patch(group_color_patch)
    ax_legend.text(
        0.05, 0.98, "Group Color",
        transform=ax_legend.transAxes,
        fontsize=9,
        verticalalignment='center',
        color='black'
    )
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def visualize_all_groups(image_path, objects, groups, output_dir):
    """
    Visualize all groups and save each as a separate PNG file.
    Saves both detailed versions (with legend) and clean versions (only bboxes).
    Also saves one image showing all objects.
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        groups: List of Group objects
        output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving group visualizations to: {output_dir}")
    print(f"Total groups: {len(groups)}")
    
    # First, save visualization of all objects in the image
    all_objects_path = os.path.join(output_dir, "all_objects.png")
    visualize_all_objects_clean(image_path, objects, all_objects_path)
    
    # Visualize each group (both detailed and clean versions)
    for i, group in enumerate(groups):
        # Detailed version with legend
        output_path = os.path.join(output_dir, f"group_{i:03d}.png")
        visualize_group(image_path, objects, group, output_path, group_id=i)
        
        # Clean version with only bounding boxes
        clean_path = os.path.join(output_dir, f"group_{i:03d}_clean.png")
        visualize_group_clean(image_path, objects, group, clean_path)
    
    print(f"\nVisualization complete! {len(groups)} groups saved (detailed + clean versions) + all objects image.")


def visualize_groups_with_reasoning(image_path, objects, groups, reasoning_result, output_dir):
    """
    Visualize all groups with reasoning annotations and save each as a separate PNG file.
    Saves both detailed versions (with reasoning) and clean versions (only bboxes).
    Also saves one image showing all objects.
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        groups: List of Group objects
        reasoning_result: GroupReasoningResult from reasoning
        output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving reasoning-annotated visualizations to: {output_dir}")
    print(f"Total groups: {len(groups)}")
    
    # First, save visualization of all objects in the image
    all_objects_path = os.path.join(output_dir, "all_objects.png")
    visualize_all_objects_clean(image_path, objects, all_objects_path)
    
    # Create a mapping from gid to group object
    gid_to_group = {g.gid: g for g in groups}
    
    # Get reasoning information
    scores = reasoning_result.explanation['scores']
    summaries = reasoning_result.explanation['summaries']
    
    # Visualize each group with reasoning info
    for i, group in enumerate(groups):
        reasoning_info = {
            'score': scores.get(group.gid, 0),
            'summary': summaries.get(group.gid, {}),
        }
        
        # Determine role
        if group.gid == reasoning_result.main_group.gid:
            reasoning_info['role'] = 'main'
        elif any(g.gid == group.gid for g in reasoning_result.background_groups):
            reasoning_info['role'] = 'background'
        else:
            reasoning_info['role'] = 'other'
        
        # Detailed version with reasoning annotations
        output_path = os.path.join(output_dir, f"group_{i:03d}_reasoning.png")
        visualize_group(image_path, objects, group, output_path, group_id=i, 
                       reasoning_info=reasoning_info)
        
        # Clean version with only bounding boxes
        clean_path = os.path.join(output_dir, f"group_{i:03d}_clean.png")
        visualize_group_clean(image_path, objects, group, clean_path)
    
    print(f"\nReasoning visualization complete! {len(groups)} groups saved (detailed + clean versions) + all objects image.")


def visualize_three_panel_summary(image_path, objects, candidate_groups, refined_groups, output_path):
    """
    Create a single visualization with three subfigures:
    - Left: All object bounding boxes (light color)
    - Middle: All candidate group bounding boxes (only group boxes)
    - Right: All refined group bounding boxes (only group boxes)
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        candidate_groups: List of candidate Group objects
        refined_groups: List of refined Group objects after reasoning
        output_path: Output file path
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors
    object_color = (1,0,0)  # Light red for objects
    candidate_color = (1.0, 0, 0)  # Light orange for candidate groups
    refined_color = (0, 1.0, 0)  # Light green for refined groups
    
    # LEFT PANEL: All objects with bounding boxes
    axes[0].imshow(img)
    axes[0].set_title(f'All Objects ({len(objects)})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    for oid, obj in objects.items():
        x, y, bw, bh = obj.bbox
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2,
            edgecolor=object_color,
            facecolor='none'
        )
        axes[0].add_patch(rect)
    
    # MIDDLE PANEL: All candidate group bounding boxes
    axes[1].imshow(img)
    axes[1].set_title(f'Candidate Groups ({len(candidate_groups)})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    for group in candidate_groups:
        # Get group bounding box
        member_objs = [objects[oid] for oid in group.members if oid in objects]
        if not member_objs:
            continue
        
        # Compute group bounding box
        all_bboxes = [obj.bbox for obj in member_objs]
        xs = [bbox[0] for bbox in all_bboxes]
        ys = [bbox[1] for bbox in all_bboxes]
        x_maxs = [bbox[0] + bbox[2] for bbox in all_bboxes]
        y_maxs = [bbox[1] + bbox[3] for bbox in all_bboxes]
        
        gx = min(xs)
        gy = min(ys)
        gw = max(x_maxs) - gx
        gh = max(y_maxs) - gy
        
        rect = patches.Rectangle(
            (gx, gy), gw, gh,
            linewidth=2,
            edgecolor=candidate_color,
            facecolor='none'
        )
        axes[1].add_patch(rect)
    
    # RIGHT PANEL: All refined group bounding boxes
    axes[2].imshow(img)
    axes[2].set_title(f'Refined Groups ({len(refined_groups)})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    for group in refined_groups:
        # Get group bounding box
        member_objs = [objects[oid] for oid in group.members if oid in objects]
        if not member_objs:
            continue
        
        # Compute group bounding box
        all_bboxes = [obj.bbox for obj in member_objs]
        xs = [bbox[0] for bbox in all_bboxes]
        ys = [bbox[1] for bbox in all_bboxes]
        x_maxs = [bbox[0] + bbox[2] for bbox in all_bboxes]
        y_maxs = [bbox[1] + bbox[3] for bbox in all_bboxes]
        
        gx = min(xs)
        gy = min(ys)
        gw = max(x_maxs) - gx
        gh = max(y_maxs) - gy
        
        rect = patches.Rectangle(
            (gx, gy), gw, gh,
            linewidth=2,
            edgecolor=refined_color,
            facecolor='none',
        )
        axes[2].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Three-panel summary saved: {output_path}")


# ============================================================
# Batch experiment configuration
# ============================================================

NUM_IMAGES = 20          # number of random images per run
RANDOM_SEED = None      # set to int for reproducibility, or None
MODES = ["intersection"]

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# Load real COCO data
coco_path = config.get_coco_path()
coco_json = coco_path / "selected" / "annotations" / "instances_val2017.json"
depth_dir = coco_path / "selected" / "depth_maps"
conf_dir = coco_path / "selected" / "depth_maps"

# Load COCO metadata once to get image indices
with open(coco_json) as f:
    coco_data = json.load(f)

num_total_images = len(coco_data["images"])
all_indices = list(range(num_total_images))
selected_indices = random.sample(all_indices, k=min(NUM_IMAGES, num_total_images))

print("\n" + "=" * 60)
print(f"Running GRM real-image batch experiment")
print(f"Selected image indices: {selected_indices}")
print("=" * 60)

# Create a unique output folder for this batch
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_output_dir = config.get_proj_output_path() / f"grm_real_batch_{timestamp}"
batch_output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving batch results to: {batch_output_dir}")

# ============================================================
# Main batch loop
# ============================================================


for run_idx, image_idx in enumerate(selected_indices):

    print("\n" + "-" * 60)
    print(f"[{run_idx+1}/{len(selected_indices)}] Processing image_idx={image_idx}")
    print("-" * 60)

    # --------------------------------------------------------
    # Load objects for one image
    # --------------------------------------------------------
    objects, image_path, diag = load_coco_objects_from_image(
        str(coco_json),
        str(depth_dir),
        str(conf_dir),
        image_idx=image_idx
    )

    if len(objects) == 0:
        print("No objects found, skipping image.")
        continue

    # --------------------------------------------------------
    # Generate candidate groups
    # --------------------------------------------------------
    groups = generate_candidate_groups(objects, image_diag_norm=diag)

    if len(groups) == 0:
        print("No candidate groups generated, skipping image.")
        continue

    # --------------------------------------------------------
    # Per-image output directories
    # --------------------------------------------------------
    image_stem = Path(image_path).stem
    image_out_dir = batch_output_dir / image_stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Run symbolic reasoning (intersection-based, single mode)
    # --------------------------------------------------------
    new_groups = reason_over_groups(groups, objects)

    # --------------------------------------------------------
    # Create three-panel summary visualization
    # --------------------------------------------------------
    summary_path = image_out_dir / f"{image_stem}_summary.png"
    visualize_three_panel_summary(
        image_path=image_path,
        objects=objects,
        candidate_groups=groups,
        refined_groups=new_groups,
        output_path=str(summary_path)
    )

print("\n" + "=" * 60)
print("Batch experiment completed.")
print(f"All results saved to: {batch_output_dir}")
print("=" * 60)
