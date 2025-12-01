#!/usr/bin/env python3
"""
Test script to test fuzzy_bookmark_search.py functionality with LMDB backend.
This script tests loading bookmarks from LMDB and performing searches.
"""

import sys
import os
import lmdb
import pickle
import pytest

# Add parent directory to path to import fuzzy_bookmark_search functions
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fuzzy_bookmark_search import FuzzyBookmarkSearch, index_bookmarks, search_bookmarks

@pytest.fixture
def test_env(tmp_path):
    lmdb_path = str(tmp_path / "bookmark_index.lmdb")
    index_dir = str(tmp_path / "whoosh_index")

    # Create dummy LMDB
    env = lmdb.open(lmdb_path, map_size=10485760, max_dbs=10)
    try:
        with env.begin(write=True) as txn:
            bookmarks_db = env.open_db(b'bookmarks', txn=txn)
            domain_index_db = env.open_db(b'domain_index', txn=txn)
            date_index_db = env.open_db(b'date_index', txn=txn)

            bookmarks = [
                {
                    'url': 'https://python.org',
                    'title': 'Python Programming',
                    'content': 'Python is a programming language.',
                    'summary': 'Official Python website.',
                    'guid': '1',
                    'id': '1'
                },
                {
                    'url': 'https://github.com',
                    'title': 'GitHub',
                    'content': 'GitHub is a code hosting platform.',
                    'summary': 'Where code lives.',
                    'guid': '2',
                    'id': '2'
                }
            ]

            for i, b in enumerate(bookmarks):
                key = str(i).encode('utf-8')
                txn.put(key, pickle.dumps(b), db=bookmarks_db)
    finally:
        env.close()

    return lmdb_path, index_dir

def test_lmdb_loading(test_env):
    """Test loading bookmarks from LMDB database."""
    print("=== Testing LMDB Bookmark Loading ===")
    lmdb_path, _ = test_env

    # Initialize LMDB
    print("Opening LMDB database...")
    searcher = FuzzyBookmarkSearch(lmdb_path=lmdb_path)
    searcher.lmdb_open()

    try:
        # Load bookmarks from LMDB
        print("Loading bookmarks from LMDB...")
        bookmarks_gen = searcher.load_bookmarks_data()

        # Convert generator to list for counting
        bookmarks_list = list(bookmarks_gen)
        print(f"Loaded {len(bookmarks_list)} bookmarks from LMDB")

        assert len(bookmarks_list) == 2
        titles = [b['title'] for b in bookmarks_list]
        assert 'Python Programming' in titles
        assert 'GitHub' in titles

    finally:
        searcher.cleanup_lmdb()

def test_search_functionality(test_env):
    """Test search functionality."""
    print("\n=== Testing Search Functionality ===")
    lmdb_path, index_dir = test_env

    searcher = FuzzyBookmarkSearch(lmdb_path=lmdb_path)
    searcher.lmdb_open()

    try:
        # Index bookmarks
        bookmarks_gen = searcher.load_bookmarks_data()
        index_bookmarks(bookmarks_gen, index_dir=index_dir)

        # Test searches
        # Search for "python"
        results = search_bookmarks("python", index_dir=index_dir)
        assert len(results['results']) > 0
        assert results['results'][0]['title'] == 'Python Programming'

        # Search for "code"
        results = search_bookmarks("code", index_dir=index_dir)
        assert len(results['results']) > 0
        assert results['results'][0]['title'] == 'GitHub'

    finally:
        searcher.cleanup_lmdb()

def test_persistence(test_env):
    """Test data persistence by checking if data survives LMDB operations."""
    print("\n=== Testing Data Persistence ===")
    lmdb_path, _ = test_env

    # Load bookmarks again to verify persistence
    searcher = FuzzyBookmarkSearch(lmdb_path=lmdb_path)
    searcher.lmdb_open()
    try:
        bookmarks_gen = searcher.load_bookmarks_data()
        bookmarks_list = list(bookmarks_gen)
        print(f"Persistence check: {len(bookmarks_list)} bookmarks still available")
        assert len(bookmarks_list) == 2
    finally:
        searcher.cleanup_lmdb()
    print("LMDB cleanup completed")
