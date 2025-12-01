
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import custom_parsers.a_suspended_tabs as parser

class TestSuspendedTabsParser(unittest.TestCase):

    def test_normal_url(self):
        bookmark = {'url': 'https://example.com', 'name': 'Example'}
        result = parser.main(bookmark)
        self.assertEqual(result, bookmark)
        self.assertEqual(result['url'], 'https://example.com')

    def test_suspended_url(self):
        # chrome-extension://klbibkeccnjlkjkiokjodocebajanakg/suspended.html#ttl=Example&uri=https://example.com
        # Actual format might vary, but usually uri or url param
        # The parser code looks for 'url' param.

        # Construct a suspended URL
        target_url = 'https://example.com/page'
        encoded_url = 'https%3A%2F%2Fexample.com%2Fpage'
        suspended_url = f'chrome-extension://extid/suspended.html?url={encoded_url}'

        bookmark = {'url': suspended_url, 'name': 'Suspended'}
        result = parser.main(bookmark)
        self.assertEqual(result['url'], target_url)

    def test_nested_encoding(self):
        # Test recursive decoding
        target_url = 'https://example.com'
        encoded_1 = 'https%3A%2F%2Fexample.com'
        encoded_2 = 'https%253A%252F%252Fexample.com' # Double encoded

        suspended_url = f'chrome-extension://extid/suspended.html?url={encoded_2}'

        bookmark = {'url': suspended_url}
        result = parser.main(bookmark)
        self.assertEqual(result['url'], target_url)

    def test_missing_url_param(self):
        suspended_url = 'chrome-extension://extid/suspended.html?other=123'
        bookmark = {'url': suspended_url}
        result = parser.main(bookmark)
        self.assertEqual(result['url'], suspended_url)

    def test_malformed_url(self):
        # Should catch exception and return original
        # To trigger exception in urlparse or parse_qs might be hard with strings, but main wraps in try/except.
        # We can pass something that causes error?
        # Maybe a bookmark without url key? code uses bookmark.get('url', '') so it handles it.

        # Passing an object that raises exception on get?
        class BadDict(dict):
            def get(self, k, d=None):
                raise Exception("Boom")

        bookmark = BadDict()
        result = parser.main(bookmark)
        self.assertEqual(result, bookmark)

if __name__ == '__main__':
    unittest.main()
