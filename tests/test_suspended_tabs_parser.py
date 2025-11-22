#!/usr/bin/env python3
"""
Comprehensive test suite for the suspended tabs parser.
Tests various scenarios including encoding levels, malformed URLs, and error handling.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from custom_parsers.a_suspended_tabs import main


class TestSuspendedTabsParser(unittest.TestCase):
    """Test cases for the suspended tabs parser."""

    def test_single_encoded_url(self):
        """Test parsing of single-encoded suspended tab URLs."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=https%3A//www.example.com/path',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        self.assertEqual(result['url'], 'https://www.example.com/path')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_double_encoded_url(self):
        """Test parsing of double-encoded suspended tab URLs."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=https%253A//www.example.com/path',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        self.assertEqual(result['url'], 'https://www.example.com/path')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_non_encoded_url_with_protocol(self):
        """Test parsing of non-encoded URLs that already contain ://."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=https://www.example.com/path',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        self.assertEqual(result['url'], 'https://www.example.com/path')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_no_url_parameter(self):
        """Test handling of chrome-extension URLs without url= parameter."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?other=param',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        # Should return unchanged
        self.assertEqual(result['url'], 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?other=param')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_malformed_chrome_extension_url(self):
        """Test handling of malformed chrome-extension URLs."""
        bookmark = {
            'url': 'chrome-extension://invalid/suspended.html?url=https%3A//www.example.com',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        # Should attempt to parse but may fail if extension ID is invalid
        # The parser doesn't validate extension ID format, so it should try to decode
        self.assertEqual(result['url'], 'https://www.example.com')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_invalid_decoded_url_no_protocol(self):
        """Test handling of URLs that decode to invalid format without protocol."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=invalid%2Durl%2Dno%2Dprotocol',
            'title': 'Suspended Tab'
        }
        result = main(bookmark)
        # Should return unchanged due to lack of protocol after decoding
        self.assertEqual(result['url'], 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=invalid%2Durl%2Dno%2Dprotocol')
        self.assertEqual(result['title'], 'Suspended Tab')

    def test_normal_url_passthrough(self):
        """Test that normal URLs pass through unchanged."""
        bookmark = {
            'url': 'https://www.example.com/path',
            'title': 'Normal Bookmark'
        }
        result = main(bookmark)
        self.assertEqual(result['url'], 'https://www.example.com/path')
        self.assertEqual(result['title'], 'Normal Bookmark')

    def test_empty_url(self):
        """Test handling of bookmarks with empty URL."""
        bookmark = {
            'url': '',
            'title': 'Empty URL'
        }
        result = main(bookmark)
        self.assertEqual(result['url'], '')
        self.assertEqual(result['title'], 'Empty URL')

    def test_none_url(self):
        """Test handling of bookmarks with None URL."""
        bookmark = {
            'title': 'None URL'
        }
        result = main(bookmark)
        self.assertNotIn('url', result)
        self.assertEqual(result['title'], 'None URL')

    def test_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        bookmark = {
            'url': 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=%ZZ',  # Invalid percent encoding
            'title': 'Invalid Encoding'
        }
        result = main(bookmark)
        # Should return unchanged due to exception
        self.assertEqual(result['url'], 'chrome-extension://abcdefghijklmnopabcdefghijklmnop/suspended.html?url=%ZZ')
        self.assertEqual(result['title'], 'Invalid Encoding')


if __name__ == '__main__':
    unittest.main()