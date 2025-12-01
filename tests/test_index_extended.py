
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
        # Let's test get_bookmarks by mocking the browser instantiation loop
        # Since get_bookmarks dynamically finds classes, we can inject a mock class into browsers_module

        # Create a mock browser class
        class MockBrowser:
            def fetch_bookmarks(self, sort=False):
                return MagicMock(bookmarks=[
                    (datetime.datetime(2023, 1, 1), "https://example.com", "Example", "Folder"),
                    (None, "https://nodate.com", None, None)
                ])

        # We need to temporarily add this to index.browsers_module or patch inspect/dir
        # Easier: patch the list comprehension or the loop.
        # The list comprehension is:
        # browser_classes = [ ... ]

        # Since it is inside the function, we can't easily patch the local variable.
        # But we can patch inspect.isclass and dir(browsers_module).

        with patch('index.browsers_module') as mock_module:
            with patch('index.inspect.isclass', return_value=True):
                with patch('index.issubclass', return_value=True):
                    # Mock module attributes
                    mock_module.MockBrowser = MockBrowser
                    mock_module.Browser = object # Different class
                    mock_module.ChromiumBasedBrowser = object # Different class

                    # Mock dir to return our mock browser name
                    # We need to patch the dir() call in index.py
                    # index.py imports browsers_module.

                    # We can patch 'dir' to return only our MockBrowser
                    with patch('builtins.dir', return_value=['MockTestBrowser']):
                         # Mock attributes on the module dynamically
                         mock_module.MockTestBrowser = MockBrowser

                         # Also need to make sure issubclass returns true for this against Browser
                         # Real MockBrowser doesn't inherit from real Browser.
                         # So we might need to rely on duck typing or actually inherit.

                         bookmarks = index.get_bookmarks()
                         # We expect 2 bookmarks from MockBrowser, and maybe more if fetch_bookmarks called multiple times?
                         # The loop runs for each class in dir(). dir() returns ['MockTestBrowser'].
                         # So it runs once.
                         self.assertEqual(len(bookmarks), 2)
                         # Sorting happens: (2023-01-01, example) vs (None, nodate).
                         # Key is (x[3] or "", x[2] or ""). Folder and Title.
                         # Folder: "Folder" vs None -> "Folder" vs ""
                         # "Folder" > ""
                         # So Example comes first.
                         self.assertEqual(bookmarks[0]['url'], 'https://example.com')
                         self.assertEqual(bookmarks[1]['name'], '') # None converted to ""

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
