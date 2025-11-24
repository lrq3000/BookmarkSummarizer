#!/usr/bin/env python3
"""
Build script for packaging the BookmarkSummarizer app into a standalone executable using PyInstaller.

This script:
- Installs PyInstaller if not already present
- Packages the main script 'fuzzy_bookmark_search.py' into a single executable
- Includes hidden imports for custom parsers: youtube, zhihu, a_suspended_tabs, 0_suspended_tabs
- Uses --onefile option for a single executable file
"""

import subprocess
import sys
import os

def install_pyinstaller():
    """Install PyInstaller if not present."""
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller installed successfully.")

def build_executable():
    """Build the standalone executable using PyInstaller."""
    main_script = "fuzzy_bookmark_search.py"

    # Hidden imports for custom parsers
    hidden_imports = [
        "custom_parsers.youtube",
        "custom_parsers.zhihu",
        "custom_parsers.a_suspended_tabs",
        "custom_parsers.0_suspended_tabs"
    ]

    # Build the PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",  # Create a single executable file
        "--name", "BookmarkSummarizer",  # Name of the executable
        main_script
    ]

    # Add hidden imports
    for hidden_import in hidden_imports:
        cmd.extend(["--hidden-import", hidden_import])

    print("Building executable with PyInstaller...")
    print("Command:", " ".join(cmd))

    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
        print("Build completed successfully!")
        print("Executable created in the 'dist' directory.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting build process for BookmarkSummarizer...")
    install_pyinstaller()
    build_executable()
    print("Build script execution completed.")