#!/usr/bin/env python3
"""
Diagnostic script for Depth Anything outputs on COCO images.

Analyzes three types of files for each RGB image:
- *_depth.npz: Raw depth data
- *_depth.png: Depth visualization/scaled depth  
- *_conf.png: Confidence maps

The script automatically uses paths from config.py:
- config.get_coco_path(remote) / "original" / "depth_maps"
- config.get_coco_path(remote) / "selected" / "depth_maps"

Usage: 
    python gen_data/depth_diagnostic.py [path_to_depth_folder]
    python gen_data/depth_diagnostic.py --remote  # Use remote config paths
    python gen_data/depth_diagnostic.py --sample 000000123456 --visualize
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import args_utils


def analyze_npz_file(npz_path: str) -> Dict[str, Any]:
    """
    Analyze .npz file contents and return comprehensive statistics.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"\n=== Analyzing {os.path.basename(npz_path)} ===")
    
    try:
        data = np.load(npz_path)
        results = {
            'file_path': npz_path,
            'keys': list(data.keys()),
            'key_analysis': {}
        }
        
        print(f"Keys found: {results['keys']}")
        
        for key in data.keys():
            array = data[key]
            print(f"\nKey: '{key}'")
            print(f"  Shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            
            # Handle different dtypes appropriately
            if np.issubdtype(array.dtype, np.number):
                # Flatten for easier statistics
                flat = array.flatten()
                
                # Basic statistics
                stats = {
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'min': float(np.min(flat)),
                    'max': float(np.max(flat)),
                    'mean': float(np.mean(flat)),
                    'std': float(np.std(flat)),
                    'percentiles': {}
                }
                
                # Percentiles
                percentiles = [1, 5, 50, 95, 99]
                for p in percentiles:
                    stats['percentiles'][p] = float(np.percentile(flat, p))
                
                # Check for problematic values
                nan_count = np.sum(np.isnan(flat))
                inf_count = np.sum(np.isinf(flat))
                zero_count = np.sum(flat == 0)
                
                stats['nan_count'] = int(nan_count)
                stats['inf_count'] = int(inf_count)
                stats['zero_count'] = int(zero_count)
                
                print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"  Percentiles: {stats['percentiles']}")
                print(f"  NaNs: {nan_count}, Infs: {inf_count}, Zeros: {zero_count}")
                
                # Data range analysis
                if stats['min'] >= 0 and stats['max'] <= 1.1:
                    print("  → Likely normalized to [0,1] range")
                elif stats['max'] > 1000:
                    print("  → Large values detected (possibly pixel coordinates or raw depth)")
                elif abs(stats['mean']) < 0.01:
                    print("  → Values near zero (check if meaningful)")
                
                # Store array for further analysis
                stats['array'] = array
                results['key_analysis'][key] = stats
                
            else:
                print(f"  Non-numeric data type: {array.dtype}")
                results['key_analysis'][key] = {
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'non_numeric': True
                }
        
        data.close()  # Close the npz file
        return results
        
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return {'error': str(e)}


def analyze_depth_type(depth_array: np.ndarray) -> Dict[str, Any]:
    """
    Heuristically determine if depth is normal or inverse depth.
    
    Args:
        depth_array: 2D numpy array containing depth values
        
    Returns:
        Dictionary with depth type analysis
    """
    print(f"\n=== Depth Type Analysis ===")
    
    h, w = depth_array.shape
    center_h, center_w = h // 2, w // 2
    
    # Define center and corner regions
    center_size = min(h, w) // 10  # 10% of smaller dimension
    corner_size = min(h, w) // 20  # 5% of smaller dimension
    
    # Extract regions
    center_region = depth_array[
        center_h - center_size:center_h + center_size,
        center_w - center_size:center_w + center_size
    ]
    
    # Four corners
    corners = [
        depth_array[:corner_size, :corner_size],  # top-left
        depth_array[:corner_size, -corner_size:],  # top-right
        depth_array[-corner_size:, :corner_size],  # bottom-left
        depth_array[-corner_size:, -corner_size:]  # bottom-right
    ]
    
    # Calculate statistics
    center_mean = np.mean(center_region)
    corner_means = [np.mean(corner) for corner in corners]
    corner_mean_avg = np.mean(corner_means)
    
    print(f"Center region mean depth: {center_mean:.6f}")
    print(f"Corner regions mean depth: {corner_mean_avg:.6f}")
    print(f"Individual corner means: {[f'{m:.6f}' for m in corner_means]}")
    
    # Heuristic: in normal depth, objects in center are often closer (smaller depth)
    # In inverse depth, closer objects have larger values
    center_vs_corners = center_mean - corner_mean_avg
    print(f"Center - Corners difference: {center_vs_corners:.6f}")
    
    analysis = {
        'center_mean': float(center_mean),
        'corner_mean_avg': float(corner_mean_avg),
        'corner_means': [float(m) for m in corner_means],
        'center_vs_corners_diff': float(center_vs_corners)
    }
    
    if abs(center_vs_corners) < np.std(depth_array) * 0.1:
        print("  → Depth appears relatively uniform (possibly sky/far scene)")
        analysis['depth_type'] = 'uniform'
    elif center_vs_corners < 0 and abs(center_vs_corners) > np.std(depth_array) * 0.2:
        print("  → Center has lower values than corners (likely NORMAL depth)")
        analysis['depth_type'] = 'normal'
    elif center_vs_corners > 0 and abs(center_vs_corners) > np.std(depth_array) * 0.2:
        print("  ⚠️  CENTER HAS HIGHER VALUES - might be INVERSE depth!")
        analysis['depth_type'] = 'possibly_inverse'
    else:
        print("  → Unclear depth type pattern")
        analysis['depth_type'] = 'unclear'
    
    return analysis


def analyze_png_file(png_path: str, file_type: str = "depth") -> Dict[str, Any]:
    """
    Analyze PNG file (either depth or confidence).
    
    Args:
        png_path: Path to PNG file
        file_type: Either "depth" or "conf" for different analysis
        
    Returns:
        Dictionary containing PNG analysis results
    """
    print(f"\n=== Analyzing {os.path.basename(png_path)} ({file_type}) ===")
    
    try:
        # Load with IMREAD_UNCHANGED to preserve original data type
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return {'error': f'Could not load {png_path}'}
        
        # Basic properties
        results = {
            'file_path': png_path,
            'shape': img.shape,
            'dtype': str(img.dtype),
            'channels': len(img.shape) if len(img.shape) == 2 else img.shape[2]
        }
        
        print(f"Shape: {img.shape}")
        print(f"Dtype: {img.dtype}")
        print(f"Channels: {results['channels']}")
        
        # Convert to single channel if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print("  Converted BGR to grayscale")
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                print("  Converted BGRA to grayscale")
        
        flat = img.flatten()
        results.update({
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat))
        })
        
        print(f"Min: {results['min']}, Max: {results['max']}")
        print(f"Mean: {results['mean']:.2f}, Std: {results['std']:.2f}")
        
        if file_type == "depth":
            # Depth-specific analysis
            if results['dtype'] == 'uint8':
                print("  → 8-bit depth: likely visualization-only (0-255)")
            elif results['dtype'] == 'uint16':
                print("  → 16-bit depth: might contain scaled depth data")
                if results['max'] > 1000:
                    print("  → High values suggest scaled depth (not just visualization)")
            
        elif file_type == "conf":
            # Confidence-specific analysis
            unique_vals = len(np.unique(flat))
            results['unique_values'] = unique_vals
            
            print(f"Unique values: {unique_vals}")
            
            if unique_vals <= 2:
                print("  → Binary confidence (likely 0/1 or 0/255)")
                results['conf_type'] = 'binary'
            elif unique_vals < 20:
                print("  → Discrete confidence levels")
                results['conf_type'] = 'discrete'
            else:
                print("  → Continuous confidence values")
                results['conf_type'] = 'continuous'
            
            # Confidence quality assessment
            if results['mean'] > results['max'] * 0.7:
                print("  → Generally high confidence")
            elif results['mean'] < results['max'] * 0.3:
                print("  → Generally low confidence")
            else:
                print("  → Mixed confidence levels")
        
        results['array'] = img
        return results
        
    except Exception as e:
        print(f"Error loading {png_path}: {e}")
        return {'error': str(e)}


def check_consistency(depth_npz: Dict, depth_png: Dict, conf_png: Dict) -> None:
    """
    Check consistency between the three file types.
    """
    print(f"\n=== Cross-file Consistency Check ===")
    
    # Extract depth array from npz (assuming main key)
    npz_arrays = []
    if 'key_analysis' in depth_npz:
        for key, analysis in depth_npz['key_analysis'].items():
            if 'array' in analysis and len(analysis['array'].shape) == 2:
                npz_arrays.append((key, analysis['array']))
    
    issues = []
    
    # Shape consistency
    shapes = []
    if npz_arrays:
        for key, array in npz_arrays:
            shapes.append(f"NPZ[{key}]: {array.shape}")
    if 'array' in depth_png:
        shapes.append(f"depth.png: {depth_png['array'].shape}")
    if 'array' in conf_png:
        shapes.append(f"conf.png: {conf_png['array'].shape}")
    
    print(f"Shapes: {shapes}")
    
    # Check if all shapes match
    if len(set([s.split(': ')[1] for s in shapes])) > 1:
        issues.append("⚠️  Shape mismatch detected between files!")
        print("⚠️  Shape mismatch detected between files!")
    else:
        print("✓ All files have consistent shapes")
    
    # Value range consistency (if available)
    if npz_arrays and 'array' in depth_png:
        npz_key, npz_array = npz_arrays[0]  # Use first suitable array
        png_array = depth_png['array']
        
        if npz_array.shape == png_array.shape:
            corr = np.corrcoef(npz_array.flatten(), png_array.flatten())[0,1]
            print(f"Correlation between NPZ[{npz_key}] and depth.png: {corr:.4f}")
            
            if corr < 0.5:
                issues.append("⚠️  Low correlation between NPZ and PNG depth data")
                print("⚠️  Low correlation between NPZ and PNG depth data")
    
    if not issues:
        print("✓ No major consistency issues detected")
    
    return issues


def visualize_example(depth_array: np.ndarray, conf_array: np.ndarray, 
                     base_name: str, save_path: str = None) -> None:
    """
    Create visualization of depth and confidence maps.
    """
    print(f"\n=== Creating Visualization ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Depth heatmap
    im1 = axes[0].imshow(depth_array, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Depth Map: {base_name}')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[0], label='Depth Value')
    
    # Confidence heatmap
    im2 = axes[1].imshow(conf_array, cmap='hot', aspect='auto')
    axes[1].set_title(f'Confidence Map: {base_name}')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    plt.colorbar(im2, ax=axes[1], label='Confidence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    # Don't show plot in non-interactive environments
    # plt.show()
    plt.close()


def process_single_sample(base_path: str, base_name: str, 
                         visualize: bool = False) -> Dict[str, Any]:
    """
    Process a single sample (RGB + 3 depth files).
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING SAMPLE: {base_name}")
    print(f"{'='*60}")
    
    # File paths
    npz_path = os.path.join(base_path, f"{base_name}_depth.npz")
    depth_png_path = os.path.join(base_path, f"{base_name}_depth.png")
    conf_png_path = os.path.join(base_path, f"{base_name}_conf.png")
    
    # Check file existence
    files_exist = {
        'npz': os.path.exists(npz_path),
        'depth_png': os.path.exists(depth_png_path),
        'conf_png': os.path.exists(conf_png_path)
    }
    
    print(f"File existence: {files_exist}")
    
    results = {
        'base_name': base_name,
        'files_exist': files_exist
    }
    
    # Analyze each file type
    if files_exist['npz']:
        results['npz_analysis'] = analyze_npz_file(npz_path)
    
    if files_exist['depth_png']:
        results['depth_png_analysis'] = analyze_png_file(depth_png_path, "depth")
    
    if files_exist['conf_png']:
        results['conf_png_analysis'] = analyze_png_file(conf_png_path, "conf")
    
    # Depth type analysis (use npz data if available)
    if ('npz_analysis' in results and 'key_analysis' in results['npz_analysis']):
        for key, analysis in results['npz_analysis']['key_analysis'].items():
            if 'array' in analysis and len(analysis['array'].shape) == 2:
                results['depth_type_analysis'] = analyze_depth_type(analysis['array'])
                break
    
    # Consistency check
    if all(files_exist.values()):
        consistency_issues = check_consistency(
            results.get('npz_analysis', {}),
            results.get('depth_png_analysis', {}),
            results.get('conf_png_analysis', {})
        )
        results['consistency_issues'] = consistency_issues
    
    # Visualization
    if visualize and all(files_exist.values()):
        try:
            # Get arrays for visualization
            depth_array = None
            conf_array = None
            
            if 'npz_analysis' in results:
                for key, analysis in results['npz_analysis']['key_analysis'].items():
                    if 'array' in analysis and len(analysis['array'].shape) == 2:
                        depth_array = analysis['array']
                        break
            
            if depth_array is None and 'depth_png_analysis' in results:
                depth_array = results['depth_png_analysis'].get('array')
            
            if 'conf_png_analysis' in results:
                conf_array = results['conf_png_analysis'].get('array')
            
            if depth_array is not None and conf_array is not None:
                viz_path = os.path.join(base_path, f"{base_name}_diagnostic_viz.png")
                visualize_example(depth_array, conf_array, base_name, viz_path)
                results['visualization_saved'] = viz_path
                
        except Exception as e:
            print(f"Visualization failed: {e}")
            results['visualization_error'] = str(e)
    
    return results


def main():
    # Get args for remote flag
    try:
        args_obj = args_utils.get_args()
        remote = args_obj.remote
    except:
        remote = False
    
    parser = argparse.ArgumentParser(description='Analyze Depth Anything outputs')
    parser.add_argument('folder_path', nargs='?', 
                       help='Path to folder containing depth files (default: use config paths)')
    parser.add_argument('--sample', '-s', 
                       help='Analyze specific sample by base name (e.g., "000000123456")')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--max-samples', '-m', type=int, default=5,
                       help='Maximum number of samples to analyze (default: 5)')
    parser.add_argument('--remote', action='store_true',
                       help='Use remote paths from config')
    
    args = parser.parse_args()
    
    # Override remote flag if provided
    if args.remote:
        remote = True
    
    # Use config paths if no specific path provided
    if not args.folder_path:
        # Try original and selected depth_maps directories from config
        possible_paths = [
            config.get_coco_path(remote) / "original" / "depth_maps",
            config.get_coco_path(remote) / "selected" / "depth_maps",
            config.get_coco_path(remote) / "depth_maps"
        ]
        
        folder_path = None
        for path in possible_paths:
            if os.path.exists(path):
                test_files = [f for f in os.listdir(path) if f.endswith('_depth.npz')]
                if test_files:
                    folder_path = str(path)
                    print(f"Auto-detected depth files in: {os.path.abspath(folder_path)}")
                    break
        
        if folder_path is None:
            print("No depth files found in config-defined paths:")
            for path in possible_paths:
                status = "exists" if os.path.exists(path) else "missing"
                print(f"  {path} ({status})")
            print(f"\nTried: {[str(p) for p in possible_paths]}")
            print("Please specify a path explicitly or ensure depth files exist.")
            return
    else:
        folder_path = args.folder_path
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        print(f"Please specify a valid path containing depth files.")
        return
    
    print(f"Analyzing depth files in: {os.path.abspath(folder_path)}")
    print(f"Using remote={remote} for config paths")
    
    # Find all depth files
    all_files = os.listdir(folder_path)
    depth_files = []
    for file in all_files:
        if file.endswith('_depth.npz'):
            base_name = file.replace('_depth.npz', '')
            depth_files.append(base_name)
    
    depth_files.sort()
    print(f"Found {len(depth_files)} samples with depth data")
    
    # Better debugging when no files found
    if len(depth_files) == 0:
        print("\nNo depth files found!")
        print(f"Directory contents ({len(all_files)} files):")
        
        # Show first 20 files for debugging
        for i, f in enumerate(all_files[:20]):
            print(f"  {f}")
        if len(all_files) > 20:
            print(f"  ... and {len(all_files) - 20} more files")
            
        print("\nLooking for files ending with '_depth.npz'")
        print("Config-based depth directories:")
        print(f"  - {config.get_coco_path(remote) / 'original' / 'depth_maps'}")
        print(f"  - {config.get_coco_path(remote) / 'selected' / 'depth_maps'}")
        print("\nUsage: python gen_data/depth_diagnostic.py [/path/to/depth/folder]")
        return  # Exit gracefully instead of sys.exit(1)
    
    # Process specific sample or multiple samples
    if args.sample:
        if args.sample in depth_files:
            results = [process_single_sample(folder_path, args.sample, args.visualize)]
        else:
            print(f"Sample {args.sample} not found!")
            print(f"Available samples: {depth_files[:10]}{'...' if len(depth_files) > 10 else ''}")
            return  # Exit gracefully instead of sys.exit(1)
    else:
        # Process multiple samples
        max_samples = min(args.max_samples, len(depth_files))
        print(f"Processing first {max_samples} samples...")
        print(f"Processing first {max_samples} samples...")
        
        results = []
        for i, base_name in enumerate(depth_files[:max_samples]):
            result = process_single_sample(folder_path, base_name, 
                                         args.visualize and i == 0)  # Only visualize first
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_samples = len(results)
    complete_samples = sum(1 for r in results if all(r.get('files_exist', {}).values()))
    
    print(f"Total samples processed: {total_samples}")
    print(f"Complete samples (all 3 files): {complete_samples}")
    
    if complete_samples > 0:
        # Aggregate statistics
        depth_types = {}
        conf_types = {}
        
        for result in results:
            if 'depth_type_analysis' in result:
                dt = result['depth_type_analysis'].get('depth_type', 'unknown')
                depth_types[dt] = depth_types.get(dt, 0) + 1
            
            if 'conf_png_analysis' in result:
                ct = result['conf_png_analysis'].get('conf_type', 'unknown')
                conf_types[ct] = conf_types.get(ct, 0) + 1
        
        print(f"\nDepth types detected: {depth_types}")
        print(f"Confidence types detected: {conf_types}")
        
        # Check for common issues
        consistency_issues = sum(1 for r in results if r.get('consistency_issues'))
        if consistency_issues > 0:
            print(f"\n⚠️  {consistency_issues}/{total_samples} samples have consistency issues")
    
    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()