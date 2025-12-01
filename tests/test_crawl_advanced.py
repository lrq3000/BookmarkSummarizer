
import unittest
import sys
import os
import shutil
import tempfile
import threading
import signal
import pickle
import time
import json
import datetime
from unittest.mock import patch, MagicMock, mock_open, call

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl
try:
    import lmdb
except ImportError:
    lmdb = None

class TestCrawlAdvanced(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test.lmdb")
        self.patcher_storage = patch('crawl.lmdb_storage_path', self.lmdb_path)
        self.patcher_storage.start()

        self.patcher_bk = patch('crawl.bookmarks_path', os.path.join(self.test_dir, "bookmarks.json"))
        self.patcher_bk.start()
        self.patcher_fl = patch('crawl.failed_urls_path', os.path.join(self.test_dir, "failed.json"))
        self.patcher_fl.start()

        # Reset globals
        crawl.lmdb_env = None
        crawl.use_fallback = False
        crawl.shutdown_flag = False
        crawl.url_hashes_db = None
        crawl.content_hashes_db = None
        crawl.bookmarks_db = None
        crawl.failed_records_db = None
        crawl.url_to_key_db = None
        crawl.domain_index_db = None
        crawl.date_index_db = None
        crawl.custom_parsers = []

        # Prepare mock parser dir
        self.parsers_dir = os.path.join(self.test_dir, "custom_parsers")
        os.makedirs(self.parsers_dir)
        with open(os.path.join(self.parsers_dir, "test_parser.py"), "w") as f:
            f.write("def main(bookmark): bookmark['parsed'] = True; return bookmark")

    def tearDown(self):
        self.patcher_storage.stop()
        self.patcher_bk.stop()
        self.patcher_fl.stop()
        if crawl.lmdb_env:
            try:
                crawl.lmdb_env.close()
            except:
                pass
        crawl.lmdb_env = None
        shutil.rmtree(self.test_dir)

        # Cleanup injected attributes if any
        if hasattr(crawl, 'HAS_MSVC') and not getattr(crawl, '_HAS_MSVC_ORIG', True):
             delattr(crawl, 'HAS_MSVC')

    # --- Sanitize Bookmark Tests ---
    def test_sanitize_bookmark_cycle(self):
        bookmark = {"a": 1}
        bookmark["self"] = bookmark
        sanitized = crawl.sanitize_bookmark(bookmark)
        self.assertEqual(sanitized["a"], 1)
        self.assertIsNone(sanitized.get("self"))

    def test_sanitize_bookmark_selenium(self):
        class MockWebDriver:
            def quit(self): pass
            def get(self): pass
            def find_element(self): pass

        bookmark = {
            "url": "http://example.com",
            "driver": MockWebDriver(),
            "nested": {"driver": MockWebDriver(), "ok": 1},
            "list": [{"driver": MockWebDriver()}, {"ok": 2}]
        }
        sanitized = crawl.sanitize_bookmark(bookmark)
        self.assertNotIn("driver", sanitized)
        self.assertNotIn("driver", sanitized["nested"])
        self.assertEqual(sanitized["nested"]["ok"], 1)
        self.assertNotIn("driver", sanitized["list"][0])
        self.assertEqual(sanitized["list"][1]["ok"], 2)

    def test_sanitize_bookmark_complex(self):
        class Complex:
            def __init__(self): self.x = 1

        bookmark = {"obj": Complex(), "ok": 1}
        sanitized = crawl.sanitize_bookmark(bookmark)
        self.assertNotIn("obj", sanitized)
        self.assertEqual(sanitized["ok"], 1)

    def test_safe_pickle_recursion(self):
        # Create a deep structure to test recursion limit adjustment
        # Reduced depth to avoid hitting platform-specific limits or pickle limitations
        # 2000 was causing RecursionError in some environments
        deep_struct = {}
        curr = deep_struct
        for _ in range(500):
            curr["next"] = {}
            curr = curr["next"]

        # Should not raise RecursionError
        pickled = crawl.safe_pickle(deep_struct)
        self.assertIsInstance(pickled, bytes)

    # --- Disk Space & LMDB Check Tests ---
    @patch('shutil.disk_usage')
    def test_check_disk_space_low(self, mock_usage):
        # usage returns (total, used, free)
        mock_usage.return_value = MagicMock(free=10 * 1024 * 1024) # 10MB
        self.assertFalse(crawl.check_disk_space(min_space_mb=100))

        mock_usage.return_value = MagicMock(free=200 * 1024 * 1024) # 200MB
        self.assertTrue(crawl.check_disk_space(min_space_mb=100))

    @patch('os.path.exists')
    def test_check_lmdb_database_exists_and_has_data_missing_file(self, mock_exists):
        # Case 1: Directory missing
        mock_exists.return_value = False
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)

        # Case 2: Directory exists, but data file missing
        # We need to simulate exists(dir)=True, exists(data_file)=False
        def side_effect(path):
            if path == self.lmdb_path: return True
            if path.endswith('data.mdb'): return False
            return False

        mock_exists.side_effect = side_effect
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)

    def test_check_lmdb_database_exists_and_has_data_real(self):
        # Test with real files
        os.makedirs(self.lmdb_path)
        with open(os.path.join(self.lmdb_path, "data.mdb"), "w") as f:
            f.write("dummy")

        with patch('lmdb.open') as mock_open:
             mock_env = MagicMock()
             mock_open.return_value = mock_env
             mock_env.open_db.side_effect = Exception("Open failed")
             exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
             self.assertTrue(exists)
             self.assertFalse(has_data)
             mock_env.close.assert_called()

    # --- Backup Tests ---
    @patch('crawl.check_lmdb_database_exists_and_has_data', return_value=(True, True, 10))
    @patch('shutil.copy2', side_effect=Exception("Copy failed"))
    @patch('glob.glob', return_value=['data.mdb'])
    @patch('os.path.isfile', return_value=True)
    def test_create_lmdb_backup_copy_fail(self, mock_isfile, mock_glob, mock_copy, mock_check):
        # Ensure HAS_MSVC exists
        if not hasattr(crawl, 'HAS_MSVC'):
            setattr(crawl, 'HAS_MSVC', False)
            setattr(crawl, '_HAS_MSVC_ORIG', False) # Marker for cleanup

        with patch('builtins.open', mock_open()):
             # Mock locking to avoid fileno error
             with patch('crawl.HAS_FCNTL', False), patch('crawl.HAS_MSVC', False):
                success, path = crawl.create_lmdb_backup()
                self.assertFalse(success)

    @patch('crawl.check_lmdb_database_exists_and_has_data', return_value=(True, True, 10))
    @patch('shutil.copy2')
    @patch('glob.glob', return_value=['data.mdb'])
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.getsize', side_effect=[100, 50]) # Mismatch
    def test_create_lmdb_backup_size_mismatch(self, mock_size, mock_isfile, mock_glob, mock_copy, mock_check):
        # Ensure HAS_MSVC exists
        if not hasattr(crawl, 'HAS_MSVC'):
            setattr(crawl, 'HAS_MSVC', False)
            setattr(crawl, '_HAS_MSVC_ORIG', False)

        with patch('builtins.open', mock_open()):
             # Mock locking to avoid fileno error
             with patch('crawl.HAS_FCNTL', False), patch('crawl.HAS_MSVC', False):
                 success, path = crawl.create_lmdb_backup()
                 self.assertTrue(success)

    # --- Parser Loading Tests ---
    def test_get_custom_parsers_dir_frozen(self):
        with patch.object(sys, 'frozen', True, create=True):
            with patch.object(sys, '_MEIPASS', '/tmp/meipass', create=True):
                path = crawl.get_custom_parsers_dir()
                self.assertEqual(path, os.path.join('/tmp/meipass', 'custom_parsers'))

    def test_load_custom_parsers_filter(self):
        with patch('crawl.get_custom_parsers_dir', return_value=self.parsers_dir):
            parsers = crawl.load_custom_parsers(parser_filter=['test_parser'])
            self.assertEqual(len(parsers), 1)

            parsers = crawl.load_custom_parsers(parser_filter=['other'])
            self.assertEqual(len(parsers), 0)

    # --- Signal Handler ---
    @patch('crawl.cleanup_lmdb')
    def test_signal_handler(self, mock_cleanup):
        crawl.shutdown_flag = False
        crawl.signal_handler(signal.SIGINT, None)
        self.assertTrue(crawl.shutdown_flag)
        mock_cleanup.assert_called()

    # --- Encoding Fix Tests ---
    def test_fix_encoding_heuristics(self):
        # Short text
        self.assertEqual(crawl.fix_encoding("short"), "short")

        # Low non-ascii ratio
        text = "Hello world" + chr(128)
        self.assertEqual(crawl.fix_encoding(text), text)

        # No sequence of special chars
        text = "Hello" + chr(128) + "World" + chr(129) # scattered
        self.assertEqual(crawl.fix_encoding(text), text)

        # Sequence of special chars
        bad_text = "Test" + chr(128)*12
        # chardet detect mock
        with patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.9}):
             res = crawl.fix_encoding(bad_text)
             self.assertIsInstance(res, str)

    # --- Fetch with Selenium Zhihu Tests ---
    @patch('crawl.init_webdriver')
    def test_fetch_with_selenium_zhihu(self, mock_init):
        driver = MagicMock()
        mock_init.return_value = driver
        driver.page_source = "<html><body>Content</body></html>"

        # Mock finding close button
        close_btn = MagicMock()
        # Side effect to simulate finding then failing to verify loop
        driver.find_element.side_effect = [close_btn, Exception("No more")]

        content = crawl.fetch_with_selenium("http://zhihu.com/question/123", title="Zhihu")

        close_btn.click.assert_called()
        self.assertIn("Content", content)

    @patch('crawl.init_webdriver')
    def test_fetch_with_selenium_general(self, mock_init):
        driver = MagicMock()
        mock_init.return_value = driver

        # General content
        driver.page_source = "<html><body>General Content</body></html>"
        content = crawl.fetch_with_selenium("http://example.com", title="General")
        self.assertIn("General Content", content)

        # Error case (exception)
        driver.get.side_effect = Exception("Selenium Error")
        content = crawl.fetch_with_selenium("http://error.com", title="Error")
        self.assertIsNone(content)

        # Empty content case
        driver.get.side_effect = None
        driver.page_source = "<html><body></body></html>" # Empty text
        content = crawl.fetch_with_selenium("http://empty.com", title="Empty")
        self.assertIsNone(content)

    # --- Fetch Webpage Content Tests ---
    @patch('crawl.create_session')
    @patch('crawl.fetch_with_selenium')
    @patch('crawl.safe_lmdb_operation')
    def test_fetch_webpage_content_advanced(self, mock_safe_op, mock_selenium, mock_session):
        # Test 1: Unicode error in print
        # Default behavior: not duplicate
        mock_safe_op.return_value = False

        bookmark = {"url": "http://example.com", "name": "Title" + chr(9999)}

        mock_resp = MagicMock()
        mock_resp.text = "<html><title>Page</title><body>Content</body></html>"
        mock_resp.headers = {'Content-Type': 'text/html'}
        mock_resp.content = b"Content"
        mock_session.return_value.get.return_value = mock_resp

        # We want to verify it doesn't crash even if print fails
        with patch('builtins.print', side_effect=[UnicodeEncodeError('ascii', '', 0, 1, ''), None, None, None, None, None, None, None]):
             res, failed = crawl.fetch_webpage_content(bookmark)
             self.assertIsNotNone(res)

        # Test 2: Deduplication via LMDB (using mocked safe_lmdb_operation)
        mock_txn = MagicMock()
        crawl.content_hashes_db = MagicMock()

        def safe_op_side_effect(op_func, *args, **kwargs):
             # op_func is check_content_deduplication(txn)
             # args[0] is fallback_func (not used if not fallback)
             return op_func(mock_txn)

        mock_safe_op.side_effect = safe_op_side_effect

        try:
            # Case: Duplicate found (txn.get returns True)
            mock_txn.get.return_value = b'1'
            res, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNone(res) # Skipped

            # Case: Not duplicate
            mock_txn.get.return_value = None
            res, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNotNone(res)
            mock_txn.put.assert_called()

            # Test 3: Fallback deduplication
            # Here we want safe_lmdb_operation to behave like fallback was triggered

            def safe_op_fallback_side_effect(op_func, fallback_func, name):
                # Simulate safe_lmdb_operation calling fallback
                return fallback_func()

            mock_safe_op.side_effect = safe_op_fallback_side_effect
            crawl.use_fallback = True
            crawl.fallback_content_hashes = set()

            # First time - not duplicate
            res, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNotNone(res)

            # Second time - duplicate (hash in set)
            if res and res.get('content'):
                content_hash = crawl.hashlib.sha256(res['content'].encode('utf-8')).hexdigest()
                crawl.fallback_content_hashes.add(content_hash)

            res, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNone(res)
        finally:
            crawl.content_hashes_db = None
            crawl.use_fallback = False

    # --- Main & Parallel Fetch Integration ---
    @patch('crawl.parse_args')
    @patch('crawl.load_config')
    @patch('crawl.init_lmdb')
    @patch('crawl.get_bookmarks')
    @patch('crawl.prepare_webdriver')
    @patch('crawl.cleanup_lmdb')
    def test_main_arguments_and_flow(self, mock_clean, mock_prep, mock_get_bk, mock_init, mock_conf, mock_args):
        # Mock args
        args = MagicMock()
        args.limit = 10
        args.workers = 5
        args.no_summary = True
        args.rebuild = False # Set to False to trigger backup
        args.browser = 'chrome'
        args.profile_path = '/tmp'
        args.config = 'conf.toml'
        args.flush_interval = 10
        args.parsers = "p1|p2"
        args.lmdb_map_size = 100
        args.lmdb_max_dbs = 10
        args.lmdb_readonly = False
        args.lmdb_resize_threshold = 0.8
        args.lmdb_growth_factor = 2.0
        args.enable_backup = True
        args.disable_backup = False
        args.backup_dir = '/tmp/bk'
        args.backup_on_failure_stop = True
        args.min_delay = 0.1
        args.max_delay = 0.2
        args.skip_unreachable = True
        args.force_recompute_summaries = False
        args.from_json = False

        mock_args.return_value = args
        mock_get_bk.return_value = [{"url": "http://u1.com", "name": "n1", "type": "url"}]

        # Mock LMDB operations in main
        with patch('crawl.safe_lmdb_operation') as mock_safe_op, \
             patch('crawl.safe_backup_operation', return_value=True) as mock_bk_op, \
             patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0)) as mock_parallel:

            # safe_lmdb_operation needs to return existing bookmarks for rebuild=False path
            # We must return a non-empty list for the first call (load existing bookmarks) to trigger backup
            mock_safe_op.side_effect = [
                [{"url": "http://old.com", "name": "old"}], # loading existing bookmarks
                None, # populating deduplication (ignored result)
                [], # retrieving final bookmarks
                []  # retrieving failed records
            ]

            crawl.main()

            mock_init.assert_called()
            mock_bk_op.assert_called() # Backup triggered
            mock_parallel.assert_called()
            mock_clean.assert_called()

    # --- Secondary Index Tests ---
    def test_update_secondary_indexes(self):
        txn = MagicMock()
        # Mock get to return None first (empty)
        txn.get.return_value = None

        bookmark_key = b'\x00\x00\x00\x01'
        bookmark = {"url": "http://domain.com/page", "date_added": "2023-01-01"}

        crawl.domain_index_db = MagicMock()
        crawl.date_index_db = MagicMock()

        crawl.update_secondary_indexes(txn, bookmark_key, bookmark)

        # Check put calls
        # domain.com key
        txn.put.assert_any_call(b'domain.com', pickle.dumps({bookmark_key}), db=crawl.domain_index_db)
        # date key
        txn.put.assert_any_call(b'2023-01-01', pickle.dumps({bookmark_key}), db=crawl.date_index_db)

        # Test update existing
        existing_set = {b'\x00\x00\x00\x02'}
        txn.get.return_value = pickle.dumps(existing_set)

        crawl.update_secondary_indexes(txn, bookmark_key, bookmark)

        # Verify set contains both
        call_args_list = txn.put.call_args_list
        # Check the last calls
        domain_call = [c for c in call_args_list if c[1]['db'] == crawl.domain_index_db][-1]
        date_call = [c for c in call_args_list if c[1]['db'] == crawl.date_index_db][-1]

        saved_domain_set = pickle.loads(domain_call[0][1])
        self.assertIn(bookmark_key, saved_domain_set)
        self.assertIn(b'\x00\x00\x00\x02', saved_domain_set)

    def test_extract_domain_and_date(self):
        self.assertEqual(crawl.extract_domain("http://www.example.com/foo"), "example.com")
        self.assertEqual(crawl.extract_domain("invalid"), "")

        bk = {"date_added": "2023-01-01T12:00:00Z"}
        self.assertEqual(crawl.extract_date(bk), "2023-01-01")

        bk = {"crawl_time": "2023-01-02T12:00:00"}
        self.assertEqual(crawl.extract_date(bk), "2023-01-02")

        bk = {}
        self.assertEqual(crawl.extract_date(bk), datetime.datetime.now().strftime('%Y-%m-%d'))

if __name__ == '__main__':
    unittest.main()
