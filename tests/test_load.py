#!/usr/bin/env python3
"""
Test script to reproduce the LMDB loading error.
"""

import sys
import os
import lmdb
import pickle
import pytest
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fuzzy_bookmark_search import FuzzyBookmarkSearch

@pytest.fixture
def dummy_lmdb(tmp_path):
    lmdb_dir = tmp_path / "test_bookmarks.lmdb"
    str_path = str(lmdb_dir)

    # Create dummy LMDB
    env = lmdb.open(str_path, map_size=10485760, max_dbs=5) # 10MB
    try:
        with env.begin(write=True) as txn:
            bookmarks_db = env.open_db(b'bookmarks', txn=txn)
            # Must create these too because FuzzyBookmarkSearch expects them
            env.open_db(b'domain_index', txn=txn)
            env.open_db(b'date_index', txn=txn)

            bookmark = {
                'url': 'https://example.com',
                'title': 'Example',
                'content': 'This is an example content.',
                'summary': 'Summary of example.',
                'guid': '123',
                'id': '1'
            }

            # Key must be bytes
            key = b'1'
            value = pickle.dumps(bookmark)
            txn.put(key, value, db=bookmarks_db)
    finally:
        env.close()
    return str_path

def test_load(dummy_lmdb):
    print("Opening LMDB...")
    searcher = FuzzyBookmarkSearch(lmdb_path=dummy_lmdb)
    searcher.lmdb_open()
    print("Loading bookmarks...")
    try:
        bookmarks = searcher.load_bookmarks_from_lmdb()
        print(f"Loaded {len(bookmarks)} bookmarks")
    except Exception as e:
        pytest.fail(f"Error loading: {e}")

    # Assertions outside try block for clearer failure messages
    try:
        assert len(bookmarks) == 1
        assert bookmarks[0]['title'] == 'Example'
    finally:
        searcher.cleanup_lmdb()
