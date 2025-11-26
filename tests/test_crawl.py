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

from crawl import parallel_fetch_bookmarks, cleanup_lmdb, ModelConfig, generate_summaries_for_bookmarks, test_api_connection, init_lmdb

def main():
    print("=== LMDB Migration Test: Crawling Phase ===")

    # Initialize LMDB
    print("Initializing LMDB database...")
    init_lmdb()

    # Load test bookmarks
    print("Loading test bookmarks...")
    test_bookmarks_path = os.path.join(os.path.dirname(__file__), 'test_bookmarks.json')
    with open(test_bookmarks_path, 'r', encoding='utf-8') as f:
        test_bookmarks = json.load(f)

    print(f"Loaded {len(test_bookmarks)} test bookmarks")

    # Run crawling with limit
    print("Starting crawl with LMDB backend...")
    bookmarks_with_content, failed_records, new_bookmarks_added = parallel_fetch_bookmarks(
        test_bookmarks,
        max_workers=2,  # Use fewer workers for testing
        limit=5,  # Limit to 5 bookmarks
        flush_interval=30,  # Shorter flush interval for testing
        skip_unreachable=False
    )

    print("\nCrawl Results:")
    print(f"- Successfully crawled: {len(bookmarks_with_content)} bookmarks")
    print(f"- Failed to crawl: {len(failed_records)} bookmarks")
    print(f"- New bookmarks added: {new_bookmarks_added}")

    # Test summary generation if we have content
    if bookmarks_with_content:
        print("\n=== Testing Summary Generation ===")
        model_config = ModelConfig()

        # Test API connection first
        print("Testing API connection...")
        if not test_api_connection(model_config):
            print("API connection failed - skipping summary generation")
        else:
            print("API connection successful - generating summaries...")
            try:
                bookmarks_with_summaries = generate_summaries_for_bookmarks(
                    bookmarks_with_content,
                    model_config,
                    force_recompute=False
                )
                print(f"Summary generation completed for {len(bookmarks_with_summaries)} bookmarks")
            except Exception as e:
                print(f"Summary generation failed: {e}")

    # Cleanup
    print("\nCleaning up LMDB resources...")
    cleanup_lmdb()

    print("=== Crawling Phase Test Complete ===")

if __name__ == "__main__":
    main()