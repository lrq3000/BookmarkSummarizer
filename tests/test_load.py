#!/usr/bin/env python3
"""
Test script to reproduce the LMDB loading error.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fuzzy_bookmark_search import lmdb_open, load_bookmarks_from_lmdb, cleanup_lmdb

def test_load():
    print("Opening LMDB...")
    lmdb_open()
    print("Loading bookmarks...")
    try:
        bookmarks = load_bookmarks_from_lmdb()
        print(f"Loaded {len(bookmarks)} bookmarks")
    except Exception as e:
        print(f"Error loading: {e}")
    finally:
        cleanup_lmdb()

if __name__ == "__main__":
    test_load()