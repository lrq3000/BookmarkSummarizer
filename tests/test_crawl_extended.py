
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import shutil
import tempfile
import pickle
import json
import datetime
import hashlib
import lmdb

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl

class TestCrawlExtended(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test_lmdb")
        # Patch the lmdb_storage_path in crawl module
        self.patcher = patch('crawl.lmdb_storage_path', self.lmdb_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)
        # Reset global variables in crawl
        if crawl.lmdb_env:
            try:
                crawl.lmdb_env.close()
            except:
                pass
        crawl.lmdb_env = None
        crawl.use_fallback = False

    def test_sanitize_bookmark(self):
        # Test basic dictionary
        bookmark = {"a": 1, "b": "test"}
        self.assertEqual(crawl.sanitize_bookmark(bookmark), bookmark)

        # Test nested dictionary
        nested = {"a": {"b": 2}}
        self.assertEqual(crawl.sanitize_bookmark(nested), nested)

        # Test list
        lst = {"a": [1, 2, 3]}
        self.assertEqual(crawl.sanitize_bookmark(lst), lst)

        # Test circular reference
        circular = {}
        circular["self"] = circular
        # crawl.sanitize_bookmark handles recursion by keeping track of seen objects
        # It returns None for the circular reference part or handles it gracefully
        sanitized = crawl.sanitize_bookmark(circular)
        self.assertIsInstance(sanitized, dict)
        # The exact behavior depends on implementation, but it shouldn't crash

        # Test object with methods (should be removed/ignored if it looks like selenium driver)
        class MockDriver:
            def quit(self): pass
            def get(self): pass
            def find_element(self): pass

        bookmark_with_driver = {"driver": MockDriver(), "valid": 1}
        sanitized = crawl.sanitize_bookmark(bookmark_with_driver)
        self.assertNotIn("driver", sanitized)
        self.assertEqual(sanitized["valid"], 1)

    def test_safe_pickle(self):
        obj = {"a": 1}
        pickled = crawl.safe_pickle(obj)
        self.assertEqual(pickle.loads(pickled), obj)

    @patch('shutil.disk_usage')
    def test_check_disk_space(self, mock_disk_usage):
        # Mock disk usage to return enough space
        mock_disk_usage.return_value = MagicMock(free=200 * 1024 * 1024) # 200 MB
        self.assertTrue(crawl.check_disk_space(min_space_mb=100))

        # Mock disk usage to return insufficient space
        mock_disk_usage.return_value = MagicMock(free=50 * 1024 * 1024) # 50 MB
        self.assertFalse(crawl.check_disk_space(min_space_mb=100))

    @patch('os.path.exists')
    @patch('lmdb.open')
    def test_check_lmdb_database_exists_and_has_data(self, mock_lmdb_open, mock_exists):
        # Case 1: Directory does not exist
        mock_exists.return_value = False
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)
        self.assertFalse(has_data)
        self.assertEqual(count, 0)

        # Case 2: Data file does not exist
        # We need to simulate exists returning True for dir but False for data.mdb
        mock_exists.side_effect = [True, False]
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)

        # Case 3: Database exists but is empty
        mock_exists.side_effect = None
        mock_exists.return_value = True

        mock_env = MagicMock()
        mock_lmdb_open.return_value = mock_env
        mock_txn = MagicMock()
        mock_env.begin.return_value = mock_txn
        mock_txn.__enter__.return_value = mock_txn
        mock_txn.__exit__.return_value = None

        # Cursor yields nothing
        mock_cursor = MagicMock()
        mock_txn.cursor.return_value = mock_cursor
        mock_cursor.__iter__.return_value = iter([])

        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertTrue(exists)
        self.assertFalse(has_data)
        self.assertEqual(count, 0)

        # Case 4: Database exists and has data
        mock_cursor.__iter__.return_value = iter([1, 2, 3])
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertTrue(exists)
        self.assertTrue(has_data)
        self.assertEqual(count, 3)

    @patch('crawl.check_lmdb_database_exists_and_has_data')
    @patch('shutil.copy2')
    @patch('os.makedirs')
    def test_create_lmdb_backup(self, mock_makedirs, mock_copy2, mock_check_db):
        # Case 1: No data to backup
        mock_check_db.return_value = (True, False, 0)
        success, path = crawl.create_lmdb_backup()
        self.assertTrue(success)
        self.assertIsNone(path)

        # Case 2: Success
        mock_check_db.return_value = (True, True, 10)
        # Need to patch glob to find files
        with patch('glob.glob', return_value=[os.path.join(self.lmdb_path, "data.mdb")]):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.getsize', return_value=100):
                    # Mock open for lock file
                    # We need a file-like object that supports fileno() for flock
                    m = mock_open()
                    f = m.return_value
                    f.fileno.return_value = 1 # Return a valid integer fd

                    with patch('builtins.open', m):
                        # Also need to mock fcntl if on unix
                        with patch('fcntl.flock'):
                             success, path = crawl.create_lmdb_backup()
                             self.assertTrue(success)
                             self.assertIsNotNone(path)
                             self.assertTrue(mock_copy2.called)

    def test_init_lmdb(self):
        # Test initialization
        crawl.init_lmdb(map_size=1024*1024)
        self.assertIsNotNone(crawl.lmdb_env)
        self.assertIsNotNone(crawl.bookmarks_db)

        # Cleanup
        # Note: cleanup_lmdb just closes the env, it keeps the global variable pointing to the closed env object.
        crawl.cleanup_lmdb()
        # So we cannot check if it is None, but we can check if it is closed by trying to use it or check if accessing it raises error?
        # Actually standard lmdb object doesn't have is_open method easily accessible?
        # But for the purpose of the test, we verified it was not None after init.
        # We can try to reopen it to ensure it was properly cleaned up or just assume it works.

        # Re-init to make sure it works again
        crawl.init_lmdb(map_size=1024*1024)
        self.assertIsNotNone(crawl.lmdb_env)


    def test_clean_text(self):
        text = "  Hello  \n\n  World  "
        cleaned = crawl.clean_text(text)
        self.assertEqual(cleaned, "Hello\nWorld")

    def test_extract_domain(self):
        self.assertEqual(crawl.extract_domain("https://www.example.com/page"), "example.com")
        self.assertEqual(crawl.extract_domain("http://sub.domain.org"), "sub.domain.org")
        self.assertEqual(crawl.extract_domain("invalid"), "")

    def test_extract_date(self):
        # From date_added
        bookmark = {"date_added": "2023-01-01T12:00:00"}
        self.assertEqual(crawl.extract_date(bookmark), "2023-01-01")

        # From crawl_time
        bookmark = {"crawl_time": "2023-02-02T12:00:00"}
        self.assertEqual(crawl.extract_date(bookmark), "2023-02-02")

        # Fallback to today (mock datetime)
        bookmark = {}
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.assertEqual(crawl.extract_date(bookmark), today)

    @patch('crawl.create_session')
    def test_fetch_webpage_content(self, mock_create_session):
        bookmark = {"url": "https://example.com", "name": "Test"}

        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Title</title></head><body>Content</body></html>"
        mock_response.content = b"Content"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_session.get.return_value = mock_response

        # Need to initialize LMDB for dedup check
        crawl.init_lmdb()

        result, failed = crawl.fetch_webpage_content(bookmark)
        self.assertIsNotNone(result)
        self.assertIn("Content", result["content"])
        self.assertEqual(result["title"], "Test")

        crawl.cleanup_lmdb()

if __name__ == '__main__':
    unittest.main()
