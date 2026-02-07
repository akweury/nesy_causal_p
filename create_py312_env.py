"""
Create a Python 3.12 virtual environment for Depth Anything V3 compatibility
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_312_DIR = PROJECT_ROOT / ".venv312"

def run_command(cmd, description=""):
    """Run a command and print output"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True

print("="*60)
print("Creating Python 3.12 Virtual Environment")
print("="*60)
print("\nDepth Anything V3 requires Python <=3.13 (specifically 3.9-3.12)")
print("Your current Python is 3.13.12, which is not compatible.")
print("\nOptions:")
print("1. Install Python 3.12 from python.org")
print("2. Use a different Python version already on your system")
print("3. Try modifying the package requirements (not recommended)")
print("\n" + "="*60)
print("\nTo check available Python versions, run:")
print("  py -0  (Windows)")
print("\nIf you have Python 3.12 installed, create a new venv:")
print("  py -3.12 -m venv .venv312")
print("  .venv312\\Scripts\\activate")
print("  pip install -r requirements.txt")
print("  python setup_depth_anything.py")
