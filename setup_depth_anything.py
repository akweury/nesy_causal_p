"""
Setup script for Depth Anything V3
Clones the repository and installs dependencies
"""

import os
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
DEPTH_ANYTHING_DIR = EXTERNAL_DIR / "Depth-Anything-3"

def run_command(cmd, cwd=None, description=""):
    """Run a command and print output"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)
    
    # Use the same Python interpreter that's running this script
    if cmd.startswith("pip "):
        cmd = f"{sys.executable} -m {cmd}"
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True

def setup_depth_anything_v3():
    """Clone and install Depth Anything V3"""
    
    # Create external directory
    EXTERNAL_DIR.mkdir(exist_ok=True)
    print(f"✓ External directory: {EXTERNAL_DIR}")
    
    # Clone repository if not exists
    if not DEPTH_ANYTHING_DIR.exists():
        print(f"\n📦 Cloning Depth Anything V3...")
        if not run_command(
            "git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            cwd=EXTERNAL_DIR,
            description="Cloning Depth Anything V3 repository"
        ):
            return False
        print("✓ Repository cloned successfully")
    else:
        print(f"✓ Repository already exists at {DEPTH_ANYTHING_DIR}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Install xformers, torch>=2, torchvision (already have torch 2.10.0)
    if not run_command(
        "pip install xformers",
        description="Installing xformers"
    ):
        print("⚠️ xformers installation failed (may need CUDA), continuing...")
    
    # Install the package in editable mode (basic)
    if not run_command(
        "pip install -e .",
        cwd=DEPTH_ANYTHING_DIR,
        description="Installing Depth Anything V3 (basic)"
    ):
        return False
    
    print("\n" + "="*60)
    print("✅ Depth Anything V3 setup complete!")
    print("="*60)
    print(f"Repository location: {DEPTH_ANYTHING_DIR}")
    print("\nOptional installations:")
    print("  - For Gaussian head: pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70")
    print("  - For Gradio app: pip install -e \".[app]\" (requires Python>=3.10)")
    print("  - For all features: pip install -e \".[all]\"")
    
    return True

if __name__ == "__main__":
    print("Depth Anything V3 Setup")
    print("="*60)
    
    success = setup_depth_anything_v3()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nYou can now use Depth Anything V3 in your project.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
