
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import json
import datetime
import index

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestIndex(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.test_dir, "bookmarks.json")
        self.patcher = patch('index.output_path', self.output_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_get_bookmarks_structure(self):
        # Create a mock browser class
        class MockBrowser:
            def fetch_bookmarks(self, sort=False):
                return MagicMock(bookmarks=[
                    (datetime.datetime(2023, 1, 1), "https://example.com", "Example", "Folder"),
                    (None, "https://nodate.com", None, None)
                ])

        # Patch the module's contents directly instead of patching builtins.dir
        # We can mock `inspect.isclass` and `issubclass` to control which attributes are treated as valid browsers.
        # We also need `getattr` to return our mock class.

        with patch('index.browsers_module') as mock_module:
            # Setup the mock module to behave like an object with attributes
            # IMPORTANT: For MagicMock, dir() will include attributes we set on it.
            # So we don't need to patch builtins.dir or __dir__.

            mock_module.MockBrowser = MockBrowser

            # We need to make sure MockBrowser is treated as a class and subclass of Browser

            # Since index.py iterates over dir(browsers_module), and we confirmed `MockBrowser` will be in it,
            # we just need inspect.isclass and issubclass to return True for it.

            # However, dir(mock) also contains other standard mock methods/attributes (assert_called, etc).
            # We need to ensure inspect.isclass returns False for those, or handle them.
            # It's easier to mock inspect.isclass to return True ONLY for our MockBrowser.

            def side_effect_isclass(obj):
                return obj is MockBrowser

            with patch('index.inspect.isclass', side_effect=side_effect_isclass):
                with patch('index.issubclass', return_value=True):

                    bookmarks = index.get_bookmarks()
                    self.assertEqual(len(bookmarks), 2)
                    self.assertEqual(bookmarks[0]['url'], 'https://example.com')
                    self.assertEqual(bookmarks[1]['name'], '')

    def test_main(self):
         # Test main execution
         mock_bookmarks = [
             {
            "date_added": 1672531200.0,
            "date_last_used": "N/A",
            "guid": "N/A",
            "id": "N/A",
            "name": "Example",
            "type": "url",
            "url": "https://example.com",
            "folder": "Folder",
            },
            {
             "date_added": "N/A",
             "name": "Empty",
             "type": "url",
             "url": "", # Should be filtered
             "folder": "Folder"
            },
            {
             "date_added": "N/A",
             "name": "Extension",
             "type": "url",
             "url": "chrome-extension://...",
             "folder": "Extensions" # Should be filtered
            }
         ]

         with patch('index.get_bookmarks', return_value=mock_bookmarks):
             index.main()

             self.assertTrue(os.path.exists(self.output_path))
             with open(self.output_path, 'r') as f:
                 saved = json.load(f)
                 self.assertEqual(len(saved), 1)
                 self.assertEqual(saved[0]['url'], "https://example.com")

if __name__ == '__main__':
    unittest.main()
