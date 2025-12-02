import unittest
import sys
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock, call
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl
try:
    import lmdb
except ImportError:
    lmdb = None

class TestCrawlExpert(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test.lmdb")
        self.patcher_storage = patch('crawl.lmdb_storage_path', self.lmdb_path)
        self.patcher_storage.start()

        self.patcher_bk = patch('crawl.bookmarks_path', os.path.join(self.test_dir, "bookmarks.json"))
        self.patcher_bk.start()
        self.patcher_fl = patch('crawl.failed_urls_path', os.path.join(self.test_dir, "failed.json"))
        self.patcher_fl.start()

        # Mock global state
        crawl.lmdb_env = None
        crawl.use_fallback = False
        crawl.shutdown_flag = False

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

    @patch('crawl.parse_args')
    @patch('crawl.load_config')
    @patch('crawl.init_lmdb')
    @patch('crawl.get_bookmarks')
    @patch('crawl.prepare_webdriver')
    @patch('crawl.cleanup_lmdb')
    @patch('crawl.parallel_fetch_bookmarks')
    @patch('crawl.test_api_connection')
    @patch('crawl.generate_summaries_for_bookmarks')
    @patch('crawl.safe_lmdb_operation')
    def test_main_full_flow(self, mock_safe_op, mock_gen_sum, mock_test_api, mock_parallel, mock_clean, mock_prep, mock_get_bk, mock_init, mock_conf, mock_args):
        # Setup Args
        args = MagicMock()
        args.limit = 0
        args.workers = 5
        args.no_summary = False # Enable summary
        args.rebuild = False
        args.browser = None
        args.profile_path = None
        args.config = 'conf.toml'
        args.flush_interval = 60
        args.parsers = None
        args.lmdb_map_size = None
        args.lmdb_max_dbs = None
        args.lmdb_readonly = False
        args.lmdb_resize_threshold = 0.8
        args.lmdb_growth_factor = 2.0
        args.enable_backup = True
        args.disable_backup = False
        args.backup_dir = self.test_dir
        args.backup_on_failure_stop = False
        args.min_delay = 0
        args.max_delay = 0
        args.skip_unreachable = False
        args.force_recompute_summaries = False
        args.from_json = False
        args.watch = None
        mock_args.return_value = args

        # Setup parallel fetch return
        item1 = {"url": "u1", "content": "c1", "content_length": 100, "crawl_method": "selenium"}
        item2 = {"url": "u2", "content": "c2", "content_length": 200, "crawl_method": "requests"}
        failed1 = {"url": "u3", "reason": "timeout", "title": "Fail"}
        mock_parallel.return_value = ([item1, item2], [failed1], 5, 0)

        # Setup API connection
        mock_test_api.return_value = True

        # Setup generate summaries
        item1_sum = item1.copy()
        item1_sum["summary"] = "Sum1"
        item2_sum = item2.copy()
        item2_sum["summary"] = "Sum2"
        mock_gen_sum.return_value = [item1_sum, item2_sum]

        # Setup safe_lmdb_operation behavior
        existing_bk = [{"url": "u4", "name": "old"}]
        final_bks = [item1_sum, item2_sum]
        final_failed = [failed1]

        mock_safe_op.side_effect = [
            existing_bk, # load existing
            None,        # populate dedup
            final_bks,   # final retrieval
            final_failed # failed retrieval
        ]

        # Setup Backup dir for counting
        os.makedirs(os.path.join(self.test_dir, "lmdb_backup_1"))
        os.makedirs(os.path.join(self.test_dir, "lmdb_backup_2"))

        # Run main
        with patch('crawl.safe_backup_operation', return_value=True):
            crawl.main()

        mock_test_api.assert_called()
        mock_gen_sum.assert_called()

    @patch('crawl.resize_lmdb_database')
    @patch('lmdb.open')
    def test_init_lmdb_resize_retry(self, mock_open, mock_resize):
        crawl.lmdb_env = None
        crawl.use_fallback = False
        crawl.current_lmdb_map_size = 100

        mock_env = MagicMock()
        mock_open.side_effect = [lmdb.MapFullError("Full"), mock_env]
        mock_resize.return_value = (True, 200)

        with patch('crawl.check_disk_space', return_value=True):
            crawl.init_lmdb()

        self.assertTrue(crawl.use_fallback)

    @patch('crawl.resize_lmdb_database')
    def test_safe_lmdb_operation_retry(self, mock_resize):
        crawl.lmdb_env = MagicMock()
        crawl.use_fallback = False
        crawl.current_lmdb_map_size = 100
        crawl.lmdb_growth_factor = 2.0

        op_func = MagicMock()
        op_func.side_effect = [lmdb.MapFullError("Full"), "Success"]

        mock_resize.return_value = (True, 200)

        result = crawl.safe_lmdb_operation(op_func)

        self.assertEqual(result, "Success")
        self.assertEqual(op_func.call_count, 2)
        mock_resize.assert_called()
        self.assertEqual(crawl.current_lmdb_map_size, 200)

    @patch('lmdb.open')
    def test_resize_lmdb_database_attempts(self, mock_open):
        crawl.lmdb_env = MagicMock()
        mock_new_env = MagicMock()
        mock_open.side_effect = [Exception("Fail 1"), mock_new_env]

        success, new_size = crawl.resize_lmdb_database(100, growth_factor=2.0)

        self.assertTrue(success)
        self.assertEqual(new_size, 200)
        self.assertEqual(mock_open.call_count, 2)

    @patch('crawl.init_webdriver')
    def test_fetch_with_selenium_zhihu_fail_loop(self, mock_init):
        driver = MagicMock()
        mock_init.return_value = driver
        driver.page_source = "<html><body>Content</body></html>"
        driver.find_element.side_effect = Exception("Not found")

        content = crawl.fetch_with_selenium("http://zhihu.com/question/123", title="Zhihu")
        self.assertIn("Content", content)

    def test_api_connection_branches(self):
        config = crawl.ModelConfig()

        config.model_type = crawl.ModelConfig.OLLAMA
        with patch('crawl.call_ollama_api', return_value="OK") as mock_ollama:
            self.assertTrue(crawl.test_api_connection(config))

        config.model_type = crawl.ModelConfig.QWEN
        with patch('crawl.call_qwen_api', return_value="OK") as mock_qwen:
            self.assertTrue(crawl.test_api_connection(config))

        config.model_type = crawl.ModelConfig.DEEPSEEK
        with patch('crawl.call_deepseek_api', return_value="OK") as mock_ds:
            self.assertTrue(crawl.test_api_connection(config))

        config.model_type = "unknown"
        with patch('crawl.call_deepseek_api', return_value="OK") as mock_ds:
            self.assertTrue(crawl.test_api_connection(config))

        config.model_type = crawl.ModelConfig.OLLAMA
        with patch('crawl.call_ollama_api', return_value="") as mock_ollama:
            self.assertFalse(crawl.test_api_connection(config))

        with patch('crawl.call_ollama_api', side_effect=Exception("Conn fail")):
            self.assertFalse(crawl.test_api_connection(config))

    @patch('crawl.create_session')
    def test_fetch_webpage_content_requests_fail(self, mock_session):
        mock_session.return_value.get.side_effect = Exception("Net fail")
        with patch('crawl.fetch_with_selenium', return_value=None):
            bookmark = {"url": "http://test.com", "name": "Test"}
            res, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNone(res)
            self.assertEqual(failed['reason'], "Request failed: Net fail")

    def test_parallel_fetch_sync_execution(self):
        # Mock ThreadPoolExecutor to execute synchronously
        class SyncExecutor:
            def __init__(self, max_workers): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, fn, *args, **kwargs):
                class Future:
                    def result(self): return fn(*args, **kwargs)
                    def cancel(self): pass
                return Future()

        with patch('concurrent.futures.ThreadPoolExecutor', side_effect=SyncExecutor):
            # Setup LMDB
            crawl.init_lmdb(map_size=1024*1024)

            bookmarks = [
                {"url": "u1", "name": "n1", "type": "url"},
                {"url": "u2", "name": "n2", "type": "url"},
                {"url": "u3", "name": "n3", "type": "url"} # fail case
            ]

            # Mock fetch_webpage_content
            def fetch_side_effect(bm, *args, **kwargs):
                if bm['url'] == 'u3':
                    return None, {"url": "u3", "reason": "fail"}
                return {"url": bm['url'], "content": "content"}, None

            with patch('crawl.fetch_webpage_content', side_effect=fetch_side_effect):
                # Run with flush_interval=0 to force flushing every item
                results, failed, count, _ = crawl.parallel_fetch_bookmarks(
                    bookmarks, max_workers=1, flush_interval=0
                )

                self.assertEqual(len(results), 3) # Includes failed one
                self.assertEqual(len(failed), 1)
                self.assertEqual(count, 3)

                # Verify flush_to_disk was called implicitly (check LMDB content)
                with crawl.lmdb_env.begin() as txn:
                    # Check bookmarks db
                    cursor = txn.cursor(crawl.bookmarks_db)
                    self.assertEqual(sum(1 for _ in cursor), 3) # All 3 flushed (1 failed record + 2 success)

                    # Check failed records db
                    cursor = txn.cursor(crawl.failed_records_db)
                    self.assertEqual(sum(1 for _ in cursor), 1)

    def test_fix_encoding_detailed(self):
        # Case 1: Short text
        self.assertEqual(crawl.fix_encoding("abc"), "abc")

        # Case 2: Low non-ascii
        self.assertEqual(crawl.fix_encoding("a"*100 + chr(128)), "a"*100 + chr(128))

        # Case 3: High non-ascii but scattered
        text = (chr(128) + "a") * 20
        self.assertEqual(crawl.fix_encoding(text), text)

        # Case 4: Consecutive special chars -> Trigger detection
        bad_text = chr(128) * 20
        with patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.9}):
             res = crawl.fix_encoding(bad_text)
             self.assertTrue(res)

        # Case 5: Exception during detection
        with patch('chardet.detect', side_effect=Exception("Fail")):
             res = crawl.fix_encoding(bad_text)
             self.assertEqual(res, bad_text)

    def test_apply_custom_parsers(self):
        # Test applying parsers
        bookmark = {"url": "u", "name": "n"}

        # Parser that modifies bookmark
        def parser1(bm):
            bm['p1'] = True
            return bm

        # Parser that returns nothing (should skip)
        def parser2(bm):
            return None

        # Parser that raises exception (should catch and continue)
        def parser3(bm):
            raise Exception("Fail")

        parsers = [parser1, parser2, parser3]

        result = crawl.apply_custom_parsers(bookmark, parsers)

        self.assertTrue(result.get('p1'))
        self.assertEqual(result['name'], 'n')

    @patch('crawl.ChromeDriverManager')
    def test_prepare_webdriver(self, mock_manager):
        mock_manager.return_value.install.return_value = "/path/to/driver"

        # Test normal
        crawl.prepare_webdriver()
        self.assertEqual(crawl.webdriver_path, "/path/to/driver")

        # Test frozen (mocking sys.frozen)
        with patch.object(sys, 'frozen', True, create=True):
            crawl.prepare_webdriver()
            self.assertEqual(crawl.webdriver_path, "/path/to/driver")

        # Test exception
        mock_manager.return_value.install.side_effect = Exception("Fail")
        crawl.prepare_webdriver()
        # Should log warning, not crash

    @patch('selenium.webdriver.Chrome')
    @patch('crawl.Service')
    def test_init_webdriver_execution(self, mock_service, mock_chrome):
        # Setup webdriver_path
        crawl.webdriver_path = "/path/to/driver"

        # Success case
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        driver = crawl.init_webdriver()
        self.assertEqual(driver, mock_driver)

        # Failure case
        mock_chrome.side_effect = Exception("Init fail")
        driver = crawl.init_webdriver()
        self.assertIsNone(driver)

        # No path case
        crawl.webdriver_path = None
        driver = crawl.init_webdriver()
        self.assertIsNone(driver)

if __name__ == '__main__':
    unittest.main()
