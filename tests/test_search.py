#!/usr/bin/env python3
"""
Test script to test fuzzy_bookmark_search.py functionality with LMDB backend.
This script tests loading bookmarks from LMDB and performing searches.
"""

import sys
import os
import json

# Add current directory to path to import fuzzy_bookmark_search functions
sys.path.insert(0, os.path.dirname(__file__))

from fuzzy_bookmark_search import load_bookmarks_data, init_lmdb, cleanup_lmdb, search_bookmarks

def test_lmdb_loading():
    """Test loading bookmarks from LMDB database."""
    print("=== Testing LMDB Bookmark Loading ===")

    # Initialize LMDB
    print("Initializing LMDB database...")
    init_lmdb()

    # Load bookmarks from LMDB
    print("Loading bookmarks from LMDB...")
    bookmarks_gen = load_bookmarks_data()

    # Convert generator to list for counting
    bookmarks_list = list(bookmarks_gen)
    print(f"Loaded {len(bookmarks_list)} bookmarks from LMDB")

    # Show sample bookmark
    if bookmarks_list:
        sample = bookmarks_list[0]
        print(f"Sample bookmark: {sample.get('title', 'No Title')} - {sample.get('url', 'No URL')}")
        print(f"Content length: {len(sample.get('content', ''))} characters")
    else:
        print("No bookmarks found - checking LMDB directly...")
        # Direct LMDB check
        import lmdb
        import json
        try:
            env = lmdb.open('./bookmark_index.lmdb', max_dbs=5)
            txn = env.begin()
            bookmarks_db = env.open_db(b'bookmarks')
            cursor = txn.cursor(bookmarks_db)
            count = 0
            for key, value in cursor:
                count += 1
                if count == 1:
                    bookmark = json.loads(value.decode('utf-8'))
                    print(f"Direct LMDB check - Sample bookmark: {bookmark.get('title', 'No Title')}")
            print(f"Direct LMDB check - Found {count} bookmarks")
            env.close()
        except Exception as e:
            print(f"Direct LMDB check failed: {str(e).encode('ascii', 'replace').decode('ascii')}")

    return len(bookmarks_list)

def test_search_functionality():
    """Test search functionality."""
    print("\n=== Testing Search Functionality ===")

    # Test searches
    test_queries = [
        "python",
        "github",
        "documentation",
        "web"
    ]

    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = search_bookmarks(query, limit=5)
            print(f"Found {len(results['results'])} results in {results['search_time']:.3f} seconds")
            if results['results']:
                top_result = results['results'][0]
                print(f"Top result: {top_result['title']} (score: {top_result['score']:.2f})")
        except Exception as e:
            print(f"Search failed for '{query}': {e}")

def test_persistence():
    """Test data persistence by checking if data survives LMDB operations."""
    print("\n=== Testing Data Persistence ===")

    # Load bookmarks again to verify persistence
    bookmarks_gen = load_bookmarks_data()
    bookmarks_list = list(bookmarks_gen)
    print(f"Persistence check: {len(bookmarks_list)} bookmarks still available")

    # Cleanup
    cleanup_lmdb()
    print("LMDB cleanup completed")

    return len(bookmarks_list)

def main():
    print("=== LMDB Migration Test: Search Phase ===")

    try:
        # Test loading
        bookmark_count = test_lmdb_loading()
        if bookmark_count == 0:
            print("ERROR: No bookmarks found in LMDB. Make sure crawl.py was run first.")
            return

        # Test searching
        test_search_functionality()

        # Test persistence
        final_count = test_persistence()

        print("\n=== Search Phase Test Results ===")
        print(f"- Bookmarks loaded: {bookmark_count}")
        print(f"- Bookmarks persisted: {final_count}")
        print(f"- Data persistence: {'PASS' if final_count == bookmark_count else 'FAIL'}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("=== Search Phase Test Complete ===")

if __name__ == "__main__":
    main()