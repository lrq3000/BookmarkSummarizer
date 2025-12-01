#!/usr/bin/env python3
"""
Test script to run crawl.py with test bookmarks data.
This script loads test bookmarks and runs the crawling logic directly.
"""

import json
import sys
import os

# Add project root to path to import crawl.py functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fuzzy_bookmark_search import lmdb_open

def main():
    print("=== Fuzzy Search LMDB Open Test ===")

    # Initialize LMDB
    print("Opening LMDB database...")
    lmdb_open()

    print("=== LMDB Open Test Complete ===")

if __name__ == "__main__":
    main()