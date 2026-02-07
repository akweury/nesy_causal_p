"""
Switch to Python 3.10 for Depth Anything V3 compatibility
This script will create a new virtual environment and reinstall packages
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OLD_VENV = PROJECT_ROOT / ".venv"
NEW_VENV = PROJECT_ROOT / ".venv310"

print("="*60)
print("Switching to Python 3.10")
print("="*60)
print(f"\nCurrent venv: {OLD_VENV}")
print(f"New venv: {NEW_VENV}")
print("\nThis will:")
print("1. Create a new Python 3.10 virtual environment")
print("2. Reinstall all packages from requirements.txt")
print("3. Keep your old .venv folder (you can delete it later)")
print("\n" + "="*60)

response = input("\nProceed? (y/n): ")
if response.lower() not in ['y', 'yes']:
    print("Cancelled.")
    sys.exit(0)

def run_cmd(cmd, description=""):
    if description:
        print(f"\n{'='*60}")
        print(description)
        print('='*60)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed")
        return False
    print("✓ Success")
    return True

# Create new venv with Python 3.10
if not run_cmd(
    f"py -3.10 -m venv {NEW_VENV}",
    "Creating Python 3.10 virtual environment"
):
    sys.exit(1)

# Upgrade pip
if not run_cmd(
    f"{NEW_VENV}\\Scripts\\python.exe -m pip install --upgrade pip setuptools wheel",
    "Upgrading pip, setuptools, and wheel"
):
    sys.exit(1)

# Install requirements
if not run_cmd(
    f"{NEW_VENV}\\Scripts\\python.exe -m pip install -r requirements.txt",
    "Installing requirements.txt"
):
    sys.exit(1)

print("\n" + "="*60)
print("✅ Python 3.10 environment created successfully!")
print("="*60)
print(f"\nNew virtual environment: {NEW_VENV}")
print("\nTo activate it:")
print(f"  {NEW_VENV}\\Scripts\\Activate.ps1")
print("\nOr in VS Code:")
print("  1. Press Ctrl+Shift+P")
print("  2. Type 'Python: Select Interpreter'")
print(f"  3. Select the interpreter at: {NEW_VENV}\\Scripts\\python.exe")
print("\nAfter activating, run:")
print("  python setup_depth_anything.py")
