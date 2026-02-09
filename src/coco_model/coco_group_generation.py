import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

from mbg.grounding.candidate_generator import generate_candidate_groups
from mbg.grounding.predicates_real import ObjectInstance, describe_group


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
        group_id: Group ID string (e.g., 'prox_0', 'depth_near', 'cat_orange')
    
    Returns:
        Human-readable group type name
    """
    if group_id.startswith('prox_'):
        return "Proximity Group"
    elif group_id.startswith('depth_'):
        return "Depth Group"
    elif group_id.startswith('cat_'):
        return "Category Group"
    elif group_id.startswith('func_'):
        return "Functional Group"
    else:
        return "Group"


def visualize_group(image_path, objects, group, output_path, group_id):
    """
    Visualize a single group by drawing bounding boxes around member objects.
    
    Args:
        image_path: Path to the image file
        objects: Dict of ObjectInstance objects
        group: Group object with members list
        output_path: Path to save the visualization
        group_id: ID of the group for the filename
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
    
    # Generate distinct colors for each object in the group
    num_members = len(group.members)
    colors = [hsv_to_rgb([i / max(num_members, 1), 0.8, 0.9]) for i in range(num_members)]
    
    # Draw bounding boxes (no text on image)
    legend_info = []
    for idx, oid in enumerate(group.members):
        obj = objects[oid]
        x, y, w, h = obj.bbox
        color = colors[idx]
        
        # Draw rectangle on image
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        )
        ax_img.add_patch(rect)
        
        # Collect info for legend
        legend_info.append({
            'oid': oid,
            'category': obj.category,
            'depth': obj.depth,
            'color': color
        })
    
    # Add title with group type and information
    group_type = get_group_type_name(group.gid)
    member_ids = ', '.join(map(str, group.members))
    title = f"{group_type}\nGroup {group_id}: [{member_ids}] ({len(group.members)} objects)"
    ax_img.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax_img.axis('off')
    
    # Create legend on the right side
    legend_text = "Object Information:\n" + "="*30 + "\n\n"
    for info in legend_info:
        legend_text += f"● ID: {info['oid']}\n"
        legend_text += f"  Category: {info['category']}\n"
        legend_text += f"  Depth: {info['depth']:.2f}m\n\n"
    
    ax_legend.text(
        0.05, 0.95, legend_text,
        transform=ax_legend.transAxes,
        fontsize=11,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor='gray', alpha=0.9)
    )
    
    # Add color patches to legend
    for idx, info in enumerate(legend_info):
        y_pos = 0.93 - (idx * 0.12)  # Adjust spacing for each object
        if y_pos > 0:
            color_patch = patches.Rectangle(
                (0.01, y_pos), 0.02, 0.025,
                transform=ax_legend.transAxes,
                facecolor=info['color'],
                edgecolor='black',
                linewidth=1
            )
            ax_legend.add_patch(color_patch)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")


def visualize_all_groups(image_path, objects, groups, output_dir):
    """
    Visualize all groups and save each as a separate PNG file.
    
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
    
    # Visualize each group
    for i, group in enumerate(groups):
        output_path = os.path.join(output_dir, f"group_{i:03d}.png")
        visualize_group(image_path, objects, group, output_path, group_id=i)
    
    print(f"\nVisualization complete! {len(groups)} images saved.")


# Load real COCO data
coco_path = config.get_coco_path()
coco_json = coco_path / "selected" / "annotations" / "instances_val2017.json"
depth_dir = coco_path / "selected" / "depth_maps"
conf_dir = coco_path / "selected" / "depth_maps"

print("=" * 60)
print("Loading COCO objects from real image...")
print("=" * 60)
objects, image_path, diag = load_coco_objects_from_image(
    str(coco_json),
    str(depth_dir),
    str(conf_dir),
    image_idx=0  # Change this to load different images
)

# Print object information
print("\nObject Details:")
for oid, obj in objects.items():
    print(f"  Object {oid}: {obj.category}, bbox={obj.bbox}, depth={obj.depth:.2f}")

# Generate candidate groups
print("\n" + "=" * 60)
print("Generating candidate groups...")
print("=" * 60)
groups = generate_candidate_groups(objects, image_diag_norm=diag)

print(f"\nGenerated {len(groups)} candidate groups:\n")
for g in groups:
    print(describe_group(g, objects))
# Visualize and save groups
print("\n" + "=" * 60)
print("Visualizing groups...")
print("=" * 60)

output_base_dir = config.get_proj_output_path()
visualization_dir = output_base_dir / "group_visualizations"
visualize_all_groups(image_path, objects, groups, str(visualization_dir))
