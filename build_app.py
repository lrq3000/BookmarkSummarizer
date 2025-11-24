#!/usr/bin/env python3
"""
Build script for packaging the BookmarkSummarizer app into standalone executables using PyInstaller.

This script:
- Installs PyInstaller if not already present
- Packages the scripts 'index.py', 'crawl.py', and 'fuzzy_bookmark_search.py'
- Includes necessary data and hidden imports
- Uses --onefile option for single executable files
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
    """Build the standalone executables using PyInstaller."""

    # Configuration for each script
    scripts = [
        {
            "name": "index",
            "script": "index.py",
            "add_data": [],
            "hidden_imports": ["browser_history"]
        },
        {
            "name": "crawl",
            "script": "crawl.py",
            "add_data": [("custom_parsers", "custom_parsers")],
            "hidden_imports": [
                "requests", "bs4", "chardet", "tqdm", "selenium", "webdriver_manager",
                "lxml", "whoosh", "fastapi", "uvicorn", "browser_history", "lmdb",
                "custom_parsers.youtube", "custom_parsers.zhihu",
                "custom_parsers.a_suspended_tabs", "custom_parsers.0_suspended_tabs"
            ]
        },
        {
            "name": "fuzzy-search",
            "script": "fuzzy_bookmark_search.py",
            "add_data": [],
            "hidden_imports": [
                "whoosh", "whoosh.index", "whoosh.fields", "whoosh.qparser",
                "whoosh.scoring", "fastapi", "uvicorn", "lmdb", "pickle"
            ]
        }
    ]

    for script_config in scripts:
        name = script_config["name"]
        script = script_config["script"]
        add_data = script_config["add_data"]
        hidden_imports = script_config["hidden_imports"]

        print(f"Building {name} from {script}...")

        # Build the PyInstaller command
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",  # Create a single executable file
            "--name", name,  # Name of the executable
            script
        ]

        # Add data files
        separator = ";" if sys.platform.startswith("win") else ":"
        for src, dest in add_data:
            # Check if source exists
            if os.path.exists(src):
                cmd.extend(["--add-data", f"{src}{separator}{dest}"])
            else:
                print(f"Warning: Source path '{src}' for add-data does not exist. Skipping.")

        # Add hidden imports
        for hidden_import in hidden_imports:
            cmd.extend(["--hidden-import", hidden_import])

        print("Command:", " ".join(cmd))

        # Run PyInstaller
        try:
            subprocess.check_call(cmd)
            print(f"Build for {name} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Build for {name} failed with error: {e}")
            sys.exit(1)

    print("All builds completed successfully. Executables are in the 'dist' directory.")

if __name__ == "__main__":
    print("Starting build process for BookmarkSummarizer...")
    install_pyinstaller()
    build_executable()
    print("Build script execution completed.")
