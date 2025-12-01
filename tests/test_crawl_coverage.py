import sys
import os
import unittest
import shutil
import tempfile
import pickle
import time
import datetime
from unittest.mock import patch, MagicMock, mock_open
import importlib
import requests

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl
try:
    import lmdb
except ImportError:
    lmdb = None

class TestCrawlCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test_lmdb")
        self.patcher_path = patch('crawl.lmdb_storage_path', self.lmdb_path)
        self.patcher_path.start()

        self.patcher_bk = patch('crawl.bookmarks_path', os.path.join(self.test_dir, "bookmarks.json"))
        self.patcher_bk.start()
        self.patcher_fl = patch('crawl.failed_urls_path', os.path.join(self.test_dir, "failed.json"))
        self.patcher_fl.start()

        # Reset globals
        crawl.lmdb_env = None
        crawl.use_fallback = False
        crawl.url_hashes_db = None
        crawl.content_hashes_db = None
        crawl.bookmarks_db = None
        crawl.failed_records_db = None
        crawl.url_to_key_db = None
        crawl.domain_index_db = None
        crawl.date_index_db = None
        crawl.shutdown_flag = False

    def tearDown(self):
        if crawl.lmdb_env:
            try:
                crawl.lmdb_env.close()
            except:
                pass
        crawl.lmdb_env = None
        self.patcher_fl.stop()
        self.patcher_bk.stop()
        self.patcher_path.stop()
        shutil.rmtree(self.test_dir)

    # --- Config Tests ---
    def test_load_config(self):
        with patch('builtins.open', mock_open(read_data=b'key="val"')):
             with patch('crawl.tomllib.load', return_value={"key": "val"}):
                 conf = crawl.load_config("exist.toml")
                 self.assertEqual(conf, {"key": "val"})

        with patch('builtins.open', side_effect=FileNotFoundError):
             conf = crawl.load_config("no.toml")
             self.assertEqual(conf, {})

        with patch('builtins.open', side_effect=Exception("Bad")):
             conf = crawl.load_config("bad.toml")
             self.assertEqual(conf, {})

    def test_model_config(self):
        conf = crawl.ModelConfig()
        self.assertEqual(conf.model_type, "openai")
        conf = crawl.ModelConfig(None)
        self.assertEqual(conf.model_type, "openai")
        conf = crawl.ModelConfig({"model": {"model_type": "qwen"}})
        self.assertEqual(conf.model_type, "qwen")

    # --- Disk Space Tests ---
    @patch('shutil.disk_usage')
    @patch('os.makedirs')
    def test_check_disk_space_create_dir_fail(self, mock_makedirs, mock_disk_usage):
        with patch('os.path.exists', return_value=False):
            mock_makedirs.side_effect = OSError("Permission denied")
            result = crawl.check_disk_space()
            self.assertFalse(result)
            mock_makedirs.assert_called()

    @patch('shutil.disk_usage')
    def test_check_disk_space_usage_fail(self, mock_disk_usage):
        mock_disk_usage.side_effect = OSError("Error")
        result = crawl.check_disk_space()
        self.assertFalse(result)

    # --- LMDB Existence Tests ---
    @patch('os.path.exists')
    def test_check_lmdb_existence_errors(self, mock_exists):
        mock_exists.side_effect = OSError("Error")
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)

        mock_exists.side_effect = [True, OSError("Error")]
        exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
        self.assertFalse(exists)

    # --- Backup Tests ---
    @patch('crawl.check_lmdb_database_exists_and_has_data')
    def test_create_lmdb_backup_no_data(self, mock_check):
        mock_check.return_value = (True, False, 0)
        success, path = crawl.create_lmdb_backup()
        self.assertTrue(success)
        self.assertIsNone(path)

    @patch('crawl.check_lmdb_database_exists_and_has_data')
    @patch('builtins.open', new_callable=mock_open)
    @patch('glob.glob')
    @patch('os.path.isfile')
    @patch('shutil.copy2')
    @patch('os.path.getsize')
    def test_create_lmdb_backup_locking_linux(self, mock_getsize, mock_copy2, mock_isfile, mock_glob, mock_file, mock_check):
        mock_check.return_value = (True, True, 100)
        mock_glob.return_value = ['/path/to/data.mdb']
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024

        with patch('crawl.HAS_FCNTL', True), patch('crawl.HAS_MSVC', False, create=True), patch('crawl.fcntl', create=True) as mock_flock:
            mock_flock.LOCK_EX = 2
            mock_flock.LOCK_NB = 4
            mock_flock.LOCK_UN = 8
            success, path = crawl.create_lmdb_backup()
            self.assertTrue(success)
            self.assertIsNotNone(path)
            self.assertTrue(mock_flock.flock.called)

    @patch('crawl.check_lmdb_database_exists_and_has_data')
    @patch('builtins.open', new_callable=mock_open)
    @patch('glob.glob')
    @patch('os.path.isfile')
    @patch('shutil.copy2')
    @patch('os.path.getsize')
    def test_create_lmdb_backup_locking_windows(self, mock_getsize, mock_copy2, mock_isfile, mock_glob, mock_file, mock_check):
        mock_check.return_value = (True, True, 100)
        mock_glob.return_value = ['/path/to/data.mdb']
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024

        with patch('crawl.HAS_FCNTL', False), patch('crawl.HAS_MSVC', True, create=True), patch('crawl.msvcrt', create=True) as mock_msvc:
            mock_msvc.LK_NBLCK = 1
            mock_msvc.LK_UNLCK = 0
            success, path = crawl.create_lmdb_backup()
            self.assertTrue(success)
            self.assertTrue(mock_msvc.locking.called)

    @patch('crawl.check_lmdb_database_exists_and_has_data')
    def test_create_lmdb_backup_exception(self, mock_check):
        mock_check.side_effect = Exception("Backup fail")
        success, path = crawl.create_lmdb_backup()
        self.assertFalse(success)

    @patch('crawl.create_lmdb_backup')
    def test_safe_backup_operation(self, mock_create):
        mock_create.return_value = (True, "/backup/path")
        self.assertTrue(crawl.safe_backup_operation())

        mock_create.return_value = (False, None)
        self.assertTrue(crawl.safe_backup_operation(continue_on_failure=True))
        self.assertFalse(crawl.safe_backup_operation(continue_on_failure=False))

        mock_create.side_effect = Exception("Error")
        self.assertTrue(crawl.safe_backup_operation(continue_on_failure=True))
        self.assertFalse(crawl.safe_backup_operation(continue_on_failure=False))

    # --- LMDB Init and Resize Tests ---
    @patch('crawl.check_disk_space', return_value=True)
    @patch('lmdb.open')
    def test_init_lmdb_exceptions(self, mock_lmdb_open, mock_space):
        errors = [lmdb.MapFullError("Full"), lmdb.MapResizedError("Resized"), lmdb.DiskError("Disk"),
                  lmdb.InvalidError("Invalid"), lmdb.VersionMismatchError("Version"), lmdb.BadRslotError("BadRslot"),
                  Exception("Generic")]
        for error in errors:
            mock_lmdb_open.side_effect = error
            crawl.use_fallback = False
            crawl.init_lmdb()
            self.assertTrue(crawl.use_fallback, f"Should use fallback on {type(error)}")

    @patch('lmdb.open')
    def test_resize_lmdb_database(self, mock_lmdb_open):
        mock_env = MagicMock()
        mock_lmdb_open.return_value = mock_env
        success, size = crawl.resize_lmdb_database(100)
        self.assertTrue(success)
        self.assertEqual(size, 200)

        mock_lmdb_open.side_effect = Exception("Fail")
        crawl.lmdb_env = MagicMock()
        success, size = crawl.resize_lmdb_database(100, max_attempts=2)
        self.assertFalse(success)
        self.assertEqual(size, 100)
        self.assertTrue(crawl.use_fallback)

    @patch('crawl.resize_lmdb_database')
    def test_safe_lmdb_operation_resize(self, mock_resize):
        crawl.lmdb_env = MagicMock()
        crawl.current_lmdb_map_size = 100
        crawl.lmdb_growth_factor = 2.0
        crawl.use_fallback = False
        op = MagicMock(side_effect=[lmdb.MapFullError("Full"), "Success"])
        mock_resize.return_value = (True, 200)
        result = crawl.safe_lmdb_operation(op, readonly=False)
        self.assertEqual(result, "Success")
        mock_resize.assert_called()
        self.assertEqual(op.call_count, 2)

    # --- Custom Parsers Tests ---
    @patch('crawl.get_custom_parsers_dir')
    @patch('os.listdir')
    def test_load_custom_parsers_edge_cases(self, mock_listdir, mock_dir):
        mock_dir.return_value = "/mock/dir"
        with patch('os.path.exists', return_value=False):
            self.assertEqual(crawl.load_custom_parsers(), [])

        with patch('os.path.exists', return_value=True):
            mock_listdir.return_value = ['bad.py']
            with patch('importlib.util.spec_from_file_location', side_effect=Exception("Import fail")):
                parsers = crawl.load_custom_parsers()
                self.assertEqual(len(parsers), 0)

    # --- LLM API Tests ---
    @patch('requests.post')
    def test_call_ollama_api(self, mock_post):
        config = crawl.ModelConfig()
        config.api_base = "http://localhost:11434"
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"message": {"content": "Response"}}
        self.assertEqual(crawl.call_ollama_api("prompt", config), "Response")
        mock_post.side_effect = requests.exceptions.RequestException("Fail")
        with self.assertRaises(Exception):
            crawl.call_ollama_api("prompt", config)

    @patch('requests.post')
    def test_call_qwen_api(self, mock_post):
        config = crawl.ModelConfig()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"choices": {"message": {"content": "Qwen"}}}
        self.assertEqual(crawl.call_qwen_api("prompt", config), "Qwen")
        mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "Qwen List"}}]}
        res = crawl.call_qwen_api("prompt", config)
        self.assertIn("Qwen List", res)

    @patch('requests.post')
    def test_call_deepseek_api(self, mock_post):
        config = crawl.ModelConfig()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"choices": {"message": {"content": "DeepSeek"}}}
        self.assertEqual(crawl.call_deepseek_api("prompt", config), "DeepSeek")

    def test_generate_summary_model_selection(self):
        with patch('crawl.call_ollama_api') as mock_ollama:
            config = crawl.ModelConfig()
            config.model_type = crawl.ModelConfig.OLLAMA
            crawl.generate_summary("T", "C", "U", config)
            mock_ollama.assert_called()
        with patch('crawl.call_qwen_api') as mock_qwen:
            config = crawl.ModelConfig()
            config.model_type = crawl.ModelConfig.QWEN
            crawl.generate_summary("T", "C", "U", config)
            mock_qwen.assert_called()
        with patch('crawl.call_deepseek_api') as mock_ds:
            config = crawl.ModelConfig()
            config.model_type = crawl.ModelConfig.DEEPSEEK
            crawl.generate_summary("T", "C", "U", config)
            mock_ds.assert_called()
        config = crawl.ModelConfig()
        config.model_type = "unknown"
        res = crawl.generate_summary("T", "C", "U", config)
        self.assertIn("failed", res)

    @patch('crawl.generate_summary')
    @patch('time.sleep')
    def test_generate_summaries_for_bookmarks_logic(self, mock_sleep, mock_gen):
        bookmarks = [
            {"url": "u1", "title": "t1", "content": "c1"},
            {"url": "u2", "title": "t2", "content": "c2", "summary": "s2"},
        ]

        crawl.lmdb_env = MagicMock()
        mock_txn = MagicMock()
        crawl.lmdb_env.begin.return_value = mock_txn
        mock_txn.__enter__.return_value = mock_txn

        b_db = MagicMock(name='b_db')
        u_db = MagicMock(name='u_db')
        crawl.bookmarks_db = b_db
        crawl.url_to_key_db = u_db

        cursor_b = MagicMock(name='cursor_b')
        cursor_u = MagicMock(name='cursor_u')

        def cursor_se(db=None, **kwargs):
            if db == b_db: return cursor_b
            if db == u_db: return cursor_u
            return MagicMock()

        mock_txn.cursor.side_effect = cursor_se

        u2_key = b'\x00\x00\x00\x02'
        u2_val = pickle.dumps({"url": "u2", "summary": "existing_summary"})

        cursor_u.__iter__.side_effect = lambda: iter([ (b'u2', u2_key) ])
        cursor_b.__iter__.side_effect = lambda: iter([])
        mock_txn.get.side_effect = lambda k, db=None: u2_val if k == u2_key else None

        mock_gen.return_value = "Generated Summary"

        result = crawl.generate_summaries_for_bookmarks(bookmarks, force_recompute=False)
        self.assertEqual(mock_gen.call_count, 1)

        mock_gen.reset_mock()
        result = crawl.generate_summaries_for_bookmarks(bookmarks, force_recompute=True)
        self.assertEqual(mock_gen.call_count, 2)

    # --- Fetch Content Tests ---
    @patch('crawl.create_session')
    def test_fetch_webpage_content_failures(self, mock_create_session):
        bookmark = {"url": "http://example.com", "name": "Test"}
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        mock_session.get.side_effect = Exception("Conn Error")
        crawl.use_fallback = True
        result, failed = crawl.fetch_webpage_content(bookmark)
        self.assertIsNone(result)
        self.assertIsNotNone(failed)
        self.assertIn("Conn Error", failed["reason"])

    @patch('crawl.create_session')
    @patch('crawl.fetch_with_selenium')
    def test_fetch_webpage_content_selenium_fallback(self, mock_selenium, mock_create_session):
        bookmark = {"url": "http://example.com", "name": "Test"}
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_resp.content = b""
        mock_resp.headers = {'Content-Type': 'text/html'}
        mock_session.get.return_value = mock_resp
        mock_selenium.return_value = "Selenium Content"
        crawl.use_fallback = True
        result, failed = crawl.fetch_webpage_content(bookmark)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "Selenium Content")
        mock_selenium.assert_called()

    # --- Parallel Fetch Tests ---
    @patch('crawl.fetch_webpage_content')
    def test_parallel_fetch_bookmarks_limit(self, mock_fetch):
        bookmarks = [{"url": f"http://ex{i}.com", "name": f"Ex{i}", "type": "url"} for i in range(10)]
        mock_fetch.return_value = ({"url": "u", "content": "c"}, None)
        crawl.init_lmdb(map_size=1024*1024)
        try:
            results, failed, added = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=2, limit=2)
            self.assertEqual(added, 2)
        finally:
            crawl.cleanup_lmdb()

    @patch('crawl.fetch_webpage_content')
    def test_parallel_fetch_flush(self, mock_fetch):
        bookmarks = [{"url": f"http://f{i}.com", "name": f"F{i}", "type": "url"} for i in range(5)]
        mock_fetch.return_value = ({"url": "u", "content": "c"}, None)
        crawl.init_lmdb(map_size=1024*1024)
        start_time = 1000
        times = [start_time + i*100 for i in range(20)]
        with patch('time.time', side_effect=times):
            crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1, flush_interval=50)
        crawl.cleanup_lmdb()

    # --- Parse Args Tests ---
    def test_parse_args(self):
        with patch.object(sys, 'argv', ['crawl.py']):
            args = crawl.parse_args()
            self.assertEqual(args.limit, None)
        with patch.object(sys, 'argv', ['crawl.py', '--limit', '5', '--no-summary', '--parsers', 'youtube']):
            args = crawl.parse_args()
            self.assertEqual(args.limit, 5)
            self.assertTrue(args.no_summary)
            self.assertEqual(args.parsers, 'youtube')

    # --- Selenium Execution Tests ---
    @patch('crawl.init_webdriver')
    def test_fetch_with_selenium_execution(self, mock_init):
        mock_driver = MagicMock()
        mock_init.return_value = mock_driver
        mock_driver.page_source = "<html><body>Selenium Body Content</body></html>"

        content = crawl.fetch_with_selenium("http://example.com", 1, 1, "Title")
        self.assertIn("Selenium Body Content", content)
        mock_driver.get.assert_called_with("http://example.com")
        mock_driver.quit.assert_called()

        mock_driver.page_source = "<html><body><div class='Post-RichText'>Zhihu Content</div></body></html>"
        content = crawl.fetch_with_selenium("http://zhihu.com/question/123", 1, 1, "Zhihu")
        self.assertIn("Zhihu Content", content)

        mock_driver.find_element.return_value = MagicMock()
        content = crawl.fetch_with_selenium("http://zhihu.com/question/123", 1, 1, "Zhihu")
        mock_driver.find_element.assert_called()

        mock_init.return_value = None
        content = crawl.fetch_with_selenium("http://fail.com")
        self.assertIsNone(content)

        mock_init.return_value = mock_driver
        mock_driver.get.side_effect = Exception("Selenium Error")
        content = crawl.fetch_with_selenium("http://error.com")
        self.assertIsNone(content)

    # --- Get Bookmarks Test ---
    def test_get_bookmarks(self):
        with patch('crawl.Chrome') as MockChrome, \
             patch('crawl.Firefox') as MockFirefox:

             mock_chrome_inst = MockChrome.return_value
             mock_chrome_inst.fetch_bookmarks.return_value.bookmarks = [
                 (datetime.datetime(2023,1,1), "u1", "t1", "f1")
             ]

             b = crawl.get_bookmarks(browser='chrome')
             self.assertEqual(len(b), 1)
             self.assertEqual(b[0]['url'], 'u1')

             MockFirefox.side_effect = Exception("Not installed")
             b = crawl.get_bookmarks()
             self.assertGreaterEqual(len(b), 1)

    # --- More Main Tests ---
    @patch('crawl.parse_args')
    @patch('crawl.load_config')
    @patch('crawl.init_lmdb')
    @patch('crawl.get_bookmarks')
    @patch('crawl.parallel_fetch_bookmarks')
    @patch('crawl.prepare_webdriver')
    def test_main_branches(self, mock_prep, mock_parallel, mock_get_bookmarks, mock_init, mock_load_config, mock_args):
        mock_args.return_value = MagicMock(
            limit=0, workers=1, no_summary=True, browser=None, profile_path=None, config="c",
            rebuild=True, flush_interval=60, lmdb_map_size=None, lmdb_max_dbs=None, lmdb_readonly=False,
            lmdb_resize_threshold=0.8, lmdb_growth_factor=2.0, enable_backup=False, disable_backup=True,
            backup_dir=".", backup_on_failure_stop=False, min_delay=0, max_delay=0, parsers=None,
            skip_unreachable=False, from_json=False
        )
        mock_get_bookmarks.return_value = []
        mock_parallel.return_value = ([], [], 0)

        # Manually set lmdb_env for this test since init_lmdb is mocked
        mock_env = MagicMock()
        crawl.lmdb_env = mock_env
        crawl.bookmarks_db = MagicMock()
        crawl.url_hashes_db = MagicMock()
        crawl.content_hashes_db = MagicMock()
        crawl.failed_records_db = MagicMock()
        crawl.url_to_key_db = MagicMock()

        mock_txn = MagicMock()
        mock_env.begin.return_value = mock_txn
        mock_txn.__enter__.return_value = mock_txn

        crawl.main()

        mock_args.return_value.rebuild = False
        mock_args.return_value.from_json = True

        # Setup bookmarks return for from_json path
        mock_cursor = MagicMock()
        mock_txn.cursor.return_value = mock_cursor
        bookmark_data = {"url": "u", "content": "c"}
        mock_cursor.__iter__.side_effect = lambda: iter([(b'k', pickle.dumps(bookmark_data))])

        with patch('crawl.test_api_connection', return_value=True):
             with patch('crawl.generate_summaries_for_bookmarks') as mock_gen:
                 crawl.main()
                 mock_gen.assert_called()

    # --- Init LMDB with fallback tests ---
    @patch('crawl.check_disk_space', return_value=False)
    def test_init_lmdb_no_space(self, mock_space):
         crawl.use_fallback = False
         crawl.init_lmdb(readonly=False)
         self.assertTrue(crawl.use_fallback)

    def test_fix_encoding(self):
        text = "valid text"
        self.assertEqual(crawl.fix_encoding(text), text)
        self.assertEqual(crawl.fix_encoding("short"), "short")

if __name__ == '__main__':
    unittest.main()
