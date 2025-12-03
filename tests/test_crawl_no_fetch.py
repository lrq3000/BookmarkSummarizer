#!/usr/bin/env python3
"""
Test script to verify --no-fetch functionality.
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import shutil
import time

# Add project root to path to import crawl.py functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl

@pytest.fixture
def cleanup_globals():
    """Save and restore global state."""
    # Save original values
    orig_env = crawl.lmdb_env
    orig_path = crawl.lmdb_storage_path
    orig_shutdown = crawl.shutdown_flag

    yield

    # Restore values
    crawl.shutdown_flag = orig_shutdown
    if crawl.lmdb_env:
        try:
            crawl.lmdb_env.close()
        except:
            pass
    crawl.lmdb_env = orig_env
    crawl.lmdb_storage_path = orig_path

def test_no_fetch_argument(tmp_path, cleanup_globals):
    """Test that parallel_fetch_bookmarks respects no_fetch=True."""
    lmdb_path = str(tmp_path / "test_no_fetch.lmdb")
    crawl.lmdb_storage_path = lmdb_path

    bookmarks = [
        {"url": "https://example.com", "name": "Example", "type": "url"},
        {"url": "https://google.com", "name": "Google", "type": "url"}
    ]

    # We use wraps to spy on the real function
    with patch('crawl.fetch_webpage_content', side_effect=crawl.fetch_webpage_content) as mock_fetch:

        # We need to mock requests.get and fetch_with_selenium to ensure they are NOT called
        with patch('crawl.requests.Session.get') as mock_get, \
             patch('crawl.fetch_with_selenium') as mock_selenium, \
             patch('crawl.apply_custom_parsers') as mock_parsers, \
             patch('crawl.check_disk_space', return_value=True):

            crawl.init_lmdb(map_size=10485760)

            try:
                # Run crawling with no_fetch=True
                bookmarks_with_content, failed_records, new_bookmarks_added = crawl.parallel_fetch_bookmarks(
                    bookmarks,
                    max_workers=2,
                    limit=5,
                    flush_interval=1,
                    skip_unreachable=False,
                    no_fetch=True
                )

                # Wait for threads to complete if needed, but parallel_fetch_bookmarks waits for completion.

                # Assert fetch_webpage_content was called twice
                assert mock_fetch.call_count == 2

                # Check call args to ensure no_fetch was passed
                # The arguments might be positional or keyword
                # fetch_webpage_content(bookmark, idx+1, total_count, min_delay, max_delay, no_fetch=no_fetch)

                # Inspecting call args of the spy
                calls = mock_fetch.call_args_list
                for args, kwargs in calls:
                    assert kwargs.get('no_fetch') is True

                # Assert requests.get and selenium were NOT called
                mock_get.assert_not_called()
                mock_selenium.assert_not_called()

                # Assert custom parsers were NOT called
                mock_parsers.assert_not_called()

                # Check results
                assert len(bookmarks_with_content) == 2
                assert new_bookmarks_added == 2
                assert failed_records == []

                for b in bookmarks_with_content:
                    assert b['content'] == ""
                    assert b['content_length'] == 0
                    assert b['crawl_method'] == "no-fetch"

            finally:
                crawl.cleanup_lmdb()
                # Ensure DB file is closed before cleanup
                if os.path.exists(lmdb_path):
                    # In Windows, we might have issues deleting if still open,
                    # but tmp_path fixture handles cleanup usually.
                    pass

def test_fetch_webpage_content_no_fetch(cleanup_globals):
    """Unit test for fetch_webpage_content with no_fetch=True."""
    bookmark = {"url": "https://test.com", "name": "Test"}

    with patch('crawl.requests.Session.get') as mock_get:
        result, error = crawl.fetch_webpage_content(
            bookmark,
            current_idx=1,
            total_count=1,
            no_fetch=True
        )

        mock_get.assert_not_called()
        assert result['content'] == ""
        assert result['crawl_method'] == "no-fetch"
        assert error is None
