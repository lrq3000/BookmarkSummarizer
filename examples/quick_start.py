#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BookmarkSummarizer Quick Start Example
"""

import os
import sys
import subprocess

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the program
from index import get_bookmarks

def main():
    """Quick start example"""
    # Step 1: Extract bookmarks
    print("Step 1: Extract Chrome bookmarks")
    bookmark_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/Bookmarks")

    if not os.path.exists(bookmark_path):
        print(f"Error: Chrome bookmarks file not found: {bookmark_path}")
        print("Please confirm Chrome is installed, or modify the bookmark_path")
        sys.exit(1)

    # Use index.py to extract bookmarks
    bookmarks = get_bookmarks(bookmark_path)
    print(f"Successfully extracted {len(bookmarks)} bookmarks")

    # Step 2: Crawl content and generate summaries
    print("\nStep 2: Crawl content and generate summaries")
    print("Run the following command to start processing:")
    print("python crawl.py --limit 5")  # Only process 5 bookmarks as an example

    # Prompt user for confirmation
    confirmation = input("\nDo you want to start processing 5 bookmarks immediately? (y/n): ")
    if confirmation.lower() == 'y':
        try:
            subprocess.run(["python", "../crawl.py", "--limit", "5"], check=True)
            print("\nProcessing completed! Please check the generated JSON files")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during processing: {e}")
    else:
        print("You can manually run the above command later")

    print("\nQuick start completed!")

if __name__ == "__main__":
    main()