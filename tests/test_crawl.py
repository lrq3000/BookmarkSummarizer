#!/usr/bin/env python3
"""
Test script to run crawl.py logic using pytest and mocks.
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path to import crawl.py functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl

@pytest.fixture
def test_bookmarks_file(tmp_path):
    bookmarks_file = tmp_path / "test_bookmarks.json"
    bookmarks = [
        {
            "url": "https://example.com",
            "name": "Example",
            "type": "url"
        },
        {
            "url": "https://google.com",
            "name": "Google",
            "type": "url"
        }
    ]
    with open(bookmarks_file, 'w', encoding='utf-8') as f:
        json.dump(bookmarks, f)
    return bookmarks_file, bookmarks

def test_api_connection_wrapper():
    """Test the API connection check wrapper."""
    model_config = crawl.ModelConfig()

    # Mock API calls to prevent network usage
    with patch('crawl.call_ollama_api', return_value="Response"), \
         patch('crawl.call_qwen_api', return_value="Response"), \
         patch('crawl.call_deepseek_api', return_value="Response"):

        result = crawl.test_api_connection(model_config)
        assert result is True

def test_crawl_workflow(tmp_path, test_bookmarks_file):
    """Test the main crawling workflow with mocked network calls."""
    _bookmarks_path, bookmarks_data = test_bookmarks_file
    lmdb_path = str(tmp_path / "test_crawl.lmdb")

    # Mock fetch_webpage_content to return dummy data
    mock_result = ({
        "url": "https://example.com",
        "title": "Example",
        "content": "Mock content",
        "content_length": 12,
        "crawl_time": "2024-01-01",
        "crawl_method": "mock"
    }, None)

    with patch('crawl.lmdb_storage_path', lmdb_path), \
         patch('crawl.fetch_webpage_content', return_value=mock_result):

        # Initialize LMDB
        crawl.init_lmdb(map_size=10485760)

        try:
            # Run crawling
            bookmarks_with_content, failed_records, new_bookmarks_added, _skipped = crawl.parallel_fetch_bookmarks(
                bookmarks_data,
                max_workers=2,
                limit=5,
                flush_interval=1,
                skip_unreachable=False
            )

            # Both bookmarks return same mock result, but they are processed
            assert len(bookmarks_with_content) == 2
            assert new_bookmarks_added == 2
            assert failed_records == []
            assert bookmarks_with_content[0]['content'] == "Mock content"

        finally:
            crawl.cleanup_lmdb()

def test_crawl_deduplication(tmp_path):
    """Test URL deduplication."""
    lmdb_path = str(tmp_path / "test_dedup.lmdb")

    bookmarks = [
        {"url": "https://unique.com", "name": "Unique", "type": "url"},
        {"url": "https://unique.com", "name": "Unique 2", "type": "url"} # Duplicate
    ]

    mock_result = ({"url": "https://unique.com", "content": "C", "title": "T"}, None)

    with patch('crawl.lmdb_storage_path', lmdb_path), \
         patch('crawl.fetch_webpage_content', return_value=mock_result):

        crawl.init_lmdb(map_size=10485760)
        try:
            results, _failed, _added, _skipped = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1)

            # Should process first, skip second
            assert len(results) == 1

        finally:
            crawl.cleanup_lmdb()
