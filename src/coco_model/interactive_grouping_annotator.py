#!/usr/bin/env python3
"""
Interactive Manual Grouping Annotation Tool for COCO Images

This script allows manual annotation of object groupings by:
1. Displaying COCO images with numbered bounding boxes
2. Accepting user input for groupings via terminal (format: "0,3;1,2,4")
3. Saving annotations after each image
4. Resuming from last annotated image when restarted

Usage: python interactive_grouping_annotator.py [--start_from IMAGE_IDX]
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Enable interactive matplotlib mode
plt.ion()


class InteractiveGroupingAnnotator:
    def __init__(self, coco_json_path, images_dir, annotation_file="grouping_annotations.json", skip_crowd=True):
        """
        Initialize the interactive annotation tool.
        
        Args:
            coco_json_path: Path to COCO annotations JSON
            images_dir: Directory containing images
            annotation_file: File to save grouping annotations
            skip_crowd: Skip crowd annotations
        """
        self.coco_json_path = coco_json_path
        self.images_dir = Path(images_dir)
        self.annotation_file = annotation_file
        self.skip_crowd = skip_crowd
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image ID
        self.imgid_to_anns = {}
        for ann in self.annotations:
            if skip_crowd and ann.get('iscrowd', 0) == 1:
                continue
            img_id = ann['image_id']
            if img_id not in self.imgid_to_anns:
                self.imgid_to_anns[img_id] = []
            self.imgid_to_anns[img_id].append(ann)
        
        # Filter images to only those with annotations
        self.images = [img for img in self.images if img['id'] in self.imgid_to_anns]
        
        # Load existing annotations
        self.grouping_annotations = self.load_existing_annotations()
        
        print(f"Loaded {len(self.images)} images with annotations")
        print(f"Existing annotations: {len(self.grouping_annotations)} images")
    
    def load_existing_annotations(self):
        """Load existing grouping annotations if file exists."""
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r') as f:
                    data = json.load(f)
                print(f"Loaded existing annotations from {self.annotation_file}")
                return data
            except Exception as e:
                print(f"Error loading annotations: {e}")
                return {}
        return {}
    
    def save_annotations(self):
        """Save current annotations to file."""
        try:
            with open(self.annotation_file, 'w') as f:
                json.dump(self.grouping_annotations, f, indent=2)
            print(f"Annotations saved to {self.annotation_file}")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def find_next_unannotated_image(self, start_from=0):
        """Find the next image that hasn't been annotated yet."""
        for i in range(start_from, len(self.images)):
            img_id = str(self.images[i]['id'])
            if img_id not in self.grouping_annotations:
                return i
        return None
    
    def display_image_with_boxes(self, img_info, annotations):
        """Display image with numbered bounding boxes."""
        img_path = self.images_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return False
        
        # Read and display image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with padding for object info
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img_rgb)
        
        # Draw numbered bounding boxes
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(annotations), 1)))
        
        object_info = []
        object_labels = []
        
        for i, ann in enumerate(annotations):
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            
            # Draw bounding box only (no text inside)
            rect = plt.Rectangle((x, y), w, h, fill=False, 
                               edgecolor=colors[i], linewidth=3)
            ax.add_patch(rect)
            
            # Add plain object ID number at top-left corner (no background, no outline)
            ax.text(x+15, y+15, str(i), fontsize=12,
                   color="red", ha='right', va='bottom')
            
            object_info.append(f"ID {i}: Category {category_id}, BBox ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")
            object_labels.append(f"ID {i}: Cat {category_id}")
        
        # Add object list at the bottom with padding
        img_height = img_rgb.shape[0]
        padding_start = img_height + 20
        
        # Create text showing all objects at bottom
        objects_per_row = 4
        for i, label in enumerate(object_labels):
            row = i // objects_per_row
            col = i % objects_per_row
            x_pos = col * (img_rgb.shape[1] // objects_per_row) + 50
            y_pos = padding_start + row * 30
            
            ax.text(x_pos, y_pos, label, fontsize=12, weight='bold',
                   color=colors[i], ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors[i], alpha=0.9))
        
        # Extend axis limits to show the object list
        max_rows = (len(object_labels) - 1) // objects_per_row + 1
        ax.set_ylim(img_height + max_rows * 40 + 40, -50)  # Flip Y and add space
        ax.set_xlim(-50, img_rgb.shape[1] + 50)
        
        ax.set_title(f"Image: {img_info['file_name']} ({len(annotations)} objects)", fontsize=14, pad=20)
        ax.axis('off')
        
        # Use non-blocking display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure display updates
        
        return object_info
    
    def parse_grouping_input(self, user_input, num_objects):
        """Parse user grouping input and validate it."""
        try:
            if user_input.strip().lower() in ['skip', 's']:
                return 'skip'
            
            if user_input.strip().lower() in ['quit', 'q']:
                return 'quit'
            
            if user_input.strip() == '':
                # Empty input - skip this image without annotation
                return 'skip'
            
            groups = []
            group_strs = user_input.split(';')
            
            used_objects = set()
            for group_str in group_strs:
                group_str = group_str.strip()
                if not group_str:
                    continue
                
                # Parse object IDs in this group
                try:
                    object_ids = [int(x.strip()) for x in group_str.split(',')]
                    
                    # Validate object IDs
                    for obj_id in object_ids:
                        if obj_id < 0 or obj_id >= num_objects:
                            print(f"Invalid object ID: {obj_id}. Valid range: 0-{num_objects-1}")
                            return None
                        if obj_id in used_objects:
                            print(f"Object ID {obj_id} used multiple times")
                            return None
                        used_objects.add(obj_id)
                    
                    groups.append(object_ids)
                
                except ValueError:
                    print(f"Invalid group format: '{group_str}'. Use comma-separated integers.")
                    return None
            
            # Add remaining objects as individual groups
            for i in range(num_objects):
                if i not in used_objects:
                    groups.append([i])
            
            return groups
            
        except Exception as e:
            print(f"Error parsing input: {e}")
            return None
    
    def annotate_image(self, img_idx):
        """Annotate a single image."""
        img_info = self.images[img_idx]
        img_id = img_info['id']
        annotations = self.imgid_to_anns[img_id]
        
        print(f"\n{'='*60}")
        print(f"Image {img_idx+1}/{len(self.images)}: {img_info['file_name']}")
        print(f"Image ID: {img_id}, Objects: {len(annotations)}")
        print(f"{'='*60}")
        
        # Display image with bounding boxes
        object_info = self.display_image_with_boxes(img_info, annotations)
        
        if not object_info:
            return False
        
        # Show object information
        print("\nObjects in this image:")
        for info in object_info:
            print(f"  {info}")
        
        print("\nGrouping Instructions:")
        print("- Enter groups using format: '0,3;1,2,4' (objects 0,3 in one group, 1,2,4 in another)")
        print("- Use semicolons (;) to separate different groups")
        print("- Objects not mentioned will be treated as individual groups")
        print("- Enter 'skip' or 's' to skip this image")
        print("- Enter 'quit' or 'q' to save and quit")
        print("- Press ENTER (empty input) to skip without annotation")
        
        while True:
            user_input = input(f"\nEnter grouping for image {img_info['file_name']}: ").strip()
            
            parsed_groups = self.parse_grouping_input(user_input, len(annotations))
            
            if parsed_groups == 'skip':
                print("Skipping this image")
                return True
            
            if parsed_groups == 'quit':
                print("Quitting annotation session")
                return False
            
            if parsed_groups is not None:
                # Save annotation
                annotation_entry = {
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'timestamp': datetime.now().isoformat(),
                    'num_objects': len(annotations),
                    'groups': parsed_groups,
                    'annotations': [
                        {
                            'object_id': i,
                            'bbox': ann['bbox'],
                            'category_id': ann['category_id'],
                            'annotation_id': ann['id']
                        }
                        for i, ann in enumerate(annotations)
                    ]
                }
                
                self.grouping_annotations[str(img_id)] = annotation_entry
                self.save_annotations()
                
                print(f"Saved grouping: {parsed_groups}")
                return True
            else:
                print("Invalid input format. Please try again.")
    
    def run_annotation_session(self, start_from=0):
        """Run the interactive annotation session."""
        print("Starting Interactive Grouping Annotation Session")
        print(f"Annotation file: {self.annotation_file}")
        
        # Find starting point
        start_idx = self.find_next_unannotated_image(start_from)
        if start_idx is None:
            print("All images have been annotated!")
            return
        
        print(f"Starting from image {start_idx + 1}/{len(self.images)}")
        
        try:
            for img_idx in range(start_idx, len(self.images)):
                img_id = str(self.images[img_idx]['id'])
                
                # Skip if already annotated
                if img_id in self.grouping_annotations:
                    print(f"Skipping already annotated image {img_idx+1}: {self.images[img_idx]['file_name']}")
                    continue
                
                # Annotate this image
                continue_annotation = self.annotate_image(img_idx)
                
                if not continue_annotation:
                    break
                
                plt.close('all')  # Close matplotlib windows
                plt.pause(0.1)  # Allow time for window to close
        
        except KeyboardInterrupt:
            print("\nAnnotation interrupted by user")
            self.save_annotations()
        
        print(f"Annotation session completed. Total annotations: {len(self.grouping_annotations)}")
    
    def show_annotation_stats(self):
        """Show statistics about current annotations."""
        if not self.grouping_annotations:
            print("No annotations found")
            return
        
        total_images = len(self.grouping_annotations)
        total_objects = sum(ann['num_objects'] for ann in self.grouping_annotations.values())
        total_groups = sum(len(ann['groups']) for ann in self.grouping_annotations.values())
        
        print(f"\nAnnotation Statistics:")
        print(f"  Annotated images: {total_images}")
        print(f"  Total objects: {total_objects}")
        print(f"  Total groups: {total_groups}")
        print(f"  Average objects per image: {total_objects/total_images:.1f}")
        print(f"  Average groups per image: {total_groups/total_images:.1f}")
        
        # Group size distribution
        group_sizes = []
        for ann in self.grouping_annotations.values():
            for group in ann['groups']:
                group_sizes.append(len(group))
        
        from collections import Counter
        size_counts = Counter(group_sizes)
        print(f"\nGroup size distribution:")
        for size in sorted(size_counts.keys()):
            print(f"  Size {size}: {size_counts[size]} groups")


def main():
    parser = argparse.ArgumentParser(description="Interactive Grouping Annotation Tool")
    parser.add_argument("--start_from", type=int, default=0, 
                       help="Start annotation from image index (default: 0)")
    parser.add_argument("--annotation_file", type=str, default="grouping_annotations.json",
                       help="Annotation file name (default: grouping_annotations.json)")
    parser.add_argument("--stats", action='store_true',
                       help="Show annotation statistics and exit")
    
    args = parser.parse_args()
    
    # Setup paths using config
    coco_path = config.get_coco_path()
    coco_json_path = coco_path / "selected" / "annotations" / "instances_val2017.json"
    images_dir = coco_path / "selected" / "val2017"
    
    # Check if paths exist
    if not coco_json_path.exists():
        print(f"COCO JSON not found: {coco_json_path}")
        print("Make sure you have run the COCO data processing script first")
        return
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    # Create annotator
    annotator = InteractiveGroupingAnnotator(
        coco_json_path=coco_json_path,
        images_dir=images_dir,
        annotation_file=args.annotation_file
    )
    
    if args.stats:
        annotator.show_annotation_stats()
        return
    
    # Run annotation session
    annotator.run_annotation_session(start_from=args.start_from)


if __name__ == "__main__":
    main()