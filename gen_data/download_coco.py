"""
Download COCO val2017 dataset (images and annotations).
Saves to the path defined in config.py: storage/coco/
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import config


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"Download complete: {output_path}")


def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress bar"""
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in the archive
        file_list = zip_ref.namelist()
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)
    
    print(f"Extraction complete to {extract_to}")


def download_coco_val2017():
    """
    Download COCO val2017 dataset:
    - Validation images (1GB)
    - Validation annotations (241MB)
    """
    
    # Define paths from config
    coco_root = config.get_coco_path(remote=False)  # Local path for COCO dataset
    coco_root.mkdir(parents=True, exist_ok=True)
    
    # COCO download URLs
    urls = {
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # Download validation images
    val_images_zip = coco_root / 'val2017.zip'
    if not (coco_root / 'val2017').exists():
        if not val_images_zip.exists():
            print("\n" + "="*60)
            print("Downloading COCO val2017 images (~1GB)")
            print("="*60)
            download_url(urls['val_images'], val_images_zip)
        
        print("\n" + "="*60)
        print("Extracting validation images")
        print("="*60)
        extract_zip(val_images_zip, coco_root)
        
        # Clean up zip file
        print(f"Removing zip file: {val_images_zip}")
        val_images_zip.unlink()
    else:
        print(f"✓ Validation images already exist at {coco_root / 'val2017'}")
    
    # Download annotations
    annotations_zip = coco_root / 'annotations_trainval2017.zip'
    if not (coco_root / 'annotations').exists():
        if not annotations_zip.exists():
            print("\n" + "="*60)
            print("Downloading COCO annotations (~241MB)")
            print("="*60)
            download_url(urls['annotations'], annotations_zip)
        
        print("\n" + "="*60)
        print("Extracting annotations")
        print("="*60)
        extract_zip(annotations_zip, coco_root)
        
        # Clean up zip file
        print(f"Removing zip file: {annotations_zip}")
        annotations_zip.unlink()
    else:
        print(f"✓ Annotations already exist at {coco_root / 'annotations'}")
    
    # Verify the expected files exist
    val_images_dir = coco_root / 'val2017'
    annotations_file = coco_root / 'annotations' / 'instances_val2017.json'
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"COCO dataset saved to: {coco_root}")
    print(f"  - Images: {val_images_dir}")
    print(f"  - Annotations: {annotations_file}")
    
    if val_images_dir.exists():
        num_images = len(list(val_images_dir.glob('*.jpg')))
        print(f"\n✓ Found {num_images} validation images")
    
    if annotations_file.exists():
        print(f"✓ Annotations file exists")
    
    print("\nYou can now use the dataset with config.get_coco_path()")


if __name__ == "__main__":
    print("COCO val2017 Dataset Downloader")
    print("="*60)
    print("This script will download:")
    print("  1. COCO val2017 images (~1GB)")
    print("  2. COCO annotations (~241MB)")
    print(f"\nDataset will be saved to: {config.storage / 'coco'}")
    print("="*60)
    
    response = input("\nProceed with download? (y/n): ")
    if response.lower() in ['y', 'yes']:
        download_coco_val2017()
    else:
        print("Download cancelled.")
