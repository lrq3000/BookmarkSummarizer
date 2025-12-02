import unittest
import sys
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import pickle
import lmdb
import requests
import json
import datetime
import concurrent.futures

# Insert path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import crawl

class TestCrawlCoverageBoost(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Save original global state
        self.original_lmdb_storage_path = crawl.lmdb_storage_path
        self.original_bookmarks_path = crawl.bookmarks_path
        self.original_failed_urls_path = crawl.failed_urls_path
        self.original_BACKUP_BASE_DIR = crawl.BACKUP_BASE_DIR
        self.original_lmdb_env = crawl.lmdb_env
        self.original_use_fallback = crawl.use_fallback
        self.original_shutdown_flag = crawl.shutdown_flag
        self.original_fallback_url_hashes = crawl.fallback_url_hashes
        self.original_fallback_content_hashes = crawl.fallback_content_hashes
        self.original_fallback_bookmarks = crawl.fallback_bookmarks
        self.original_fallback_failed_records = crawl.fallback_failed_records

        # Set test state
        crawl.lmdb_storage_path = os.path.join(self.test_dir, "test_lmdb")
        crawl.bookmarks_path = os.path.join(self.test_dir, "bookmarks.json")
        crawl.failed_urls_path = os.path.join(self.test_dir, "failed_urls.json")
        crawl.BACKUP_BASE_DIR = os.path.join(self.test_dir, "backups")

        crawl.lmdb_env = None
        crawl.use_fallback = False
        crawl.shutdown_flag = False
        crawl.fallback_url_hashes = set()
        crawl.fallback_content_hashes = set()
        crawl.fallback_bookmarks = []
        crawl.fallback_failed_records = []

    def tearDown(self):
        if crawl.lmdb_env:
            try:
                crawl.lmdb_env.close()
            except Exception:
                pass

        # Restore global state
        crawl.lmdb_storage_path = self.original_lmdb_storage_path
        crawl.bookmarks_path = self.original_bookmarks_path
        crawl.failed_urls_path = self.original_failed_urls_path
        crawl.BACKUP_BASE_DIR = self.original_BACKUP_BASE_DIR
        crawl.lmdb_env = self.original_lmdb_env
        crawl.use_fallback = self.original_use_fallback
        crawl.shutdown_flag = self.original_shutdown_flag
        crawl.fallback_url_hashes = self.original_fallback_url_hashes
        crawl.fallback_content_hashes = self.original_fallback_content_hashes
        crawl.fallback_bookmarks = self.original_fallback_bookmarks
        crawl.fallback_failed_records = self.original_fallback_failed_records

        shutil.rmtree(self.test_dir)

    def test_sanitize_bookmark_recursion_handling(self):
        real_sanitize = crawl.sanitize_bookmark
        def side_effect(bookmark, depth=0, seen=None):
            if depth > 0:
                raise RecursionError("Forced error")
            return real_sanitize(bookmark, depth, seen)
        with patch('crawl.sanitize_bookmark', side_effect=side_effect):
            data = {'key': {'nested': 1}}
            result = crawl.sanitize_bookmark(data)
            self.assertEqual(result, {})

    def test_create_lmdb_backup_lock_failure_windows(self):
        mock_msvcrt = MagicMock()
        mock_msvcrt.locking.side_effect = OSError("Lock fail")
        mock_msvcrt.LK_NBLCK = 1
        with patch.dict('sys.modules', {'msvcrt': mock_msvcrt}):
            with patch('crawl.HAS_FCNTL', False), \
                 patch('crawl.HAS_MSVC', True, create=True), \
                 patch('crawl.check_lmdb_database_exists_and_has_data', return_value=(True, True, 10)), \
                 patch('builtins.open', mock_open()) as mock_file, \
                 patch('os.makedirs'), \
                 patch('crawl.msvcrt', mock_msvcrt, create=True):
                     success, path = crawl.create_lmdb_backup("test")
                     self.assertTrue(success)
                     self.assertIsNone(path)

    def test_resize_lmdb_database_all_attempts_fail(self):
        crawl.current_lmdb_map_size = 1024
        with patch('lmdb.open', side_effect=lmdb.Error("Fail")):
             success, new_size = crawl.resize_lmdb_database(1024, max_attempts=2)
        self.assertFalse(success)
        self.assertEqual(new_size, 1024)
        self.assertTrue(crawl.use_fallback)

    def test_load_custom_parsers_filter_missing(self):
        with patch('crawl.get_custom_parsers_dir', return_value=self.test_dir):
            with open(os.path.join(self.test_dir, "dummy.py"), "w") as f:
                f.write("def main(b): return b")
            with patch('builtins.print'):
                parsers = crawl.load_custom_parsers(parser_filter=['dummy', 'missing'])
                self.assertEqual(len(parsers), 1)

    def test_fix_encoding_optimization(self):
        self.assertEqual(crawl.fix_encoding("short"), "short")
        self.assertEqual(crawl.fix_encoding("Hello World " * 10), "Hello World " * 10)
        text = "a" + "\x80" + "b" * 20 + "\x80"
        self.assertEqual(crawl.fix_encoding(text), text)

    def test_fix_encoding_complex(self):
         text = ("a" * 8 + "\x80" * 2) * 20
         self.assertEqual(crawl.fix_encoding(text), text)
         text = "\x80" * 20
         with patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.9}):
             self.assertEqual(crawl.fix_encoding(text), text)

    def test_safe_lmdb_operation_map_full(self):
        mock_func = MagicMock()
        mock_func.side_effect = [lmdb.MapFullError("Full"), "Success"]
        crawl.lmdb_env = MagicMock()
        crawl.current_lmdb_map_size = 100
        crawl.lmdb_growth_factor = 2.0
        with patch('crawl.resize_lmdb_database', return_value=(True, 200)) as mock_resize:
            result = crawl.safe_lmdb_operation(mock_func, operation_name="test")
            self.assertEqual(result, "Success")
            self.assertEqual(crawl.current_lmdb_map_size, 200)

    def test_safe_lmdb_operation_map_full_resize_fail(self):
        mock_func = MagicMock(side_effect=lmdb.MapFullError("Full"))
        crawl.lmdb_env = MagicMock()
        crawl.current_lmdb_map_size = 100
        with patch('crawl.resize_lmdb_database', return_value=(False, 100)):
            result = crawl.safe_lmdb_operation(mock_func, operation_name="test")
            self.assertIsNone(result)
            self.assertTrue(crawl.use_fallback)

    def test_safe_lmdb_operation_fallback_fail(self):
        crawl.use_fallback = True
        fallback_func = MagicMock(side_effect=Exception("Fallback Fail"))
        result = crawl.safe_lmdb_operation(None, fallback_func, "op")
        self.assertIsNone(result)
        crawl.use_fallback = False
        mock_func = MagicMock(side_effect=Exception("Op Fail"))
        result = crawl.safe_lmdb_operation(mock_func, fallback_func, "op")
        self.assertIsNone(result)

    def test_call_qwen_api_exceptions(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.QWEN
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_response.text = "Error detail"
        with patch('requests.post', return_value=mock_response):
            with self.assertRaises(Exception) as cm:
                crawl.call_qwen_api("prompt", config)
            self.assertIn("API call failed", str(cm.exception))
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Bad JSON")
        mock_response.text = "Bad JSON text"
        with patch('requests.post', return_value=mock_response):
            with self.assertRaises(Exception) as cm:
                crawl.call_qwen_api("prompt", config)
            self.assertIn("Response parsing failed", str(cm.exception))

    def test_call_ollama_api_exceptions(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.OLLAMA
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_response.text = "Error detail"
        with patch('requests.post', return_value=mock_response):
            with self.assertRaises(Exception) as cm:
                crawl.call_ollama_api("prompt", config)
            self.assertIn("API call failed", str(cm.exception))

    def test_call_deepseek_api_exceptions(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.DEEPSEEK
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_response.text = "Error detail"
        with patch('requests.post', return_value=mock_response):
            with self.assertRaises(Exception) as cm:
                crawl.call_deepseek_api("prompt", config)
            self.assertIn("API call failed", str(cm.exception))

    def test_check_disk_space_fail(self):
        with patch('shutil.disk_usage', return_value=MagicMock(free=0)):
             self.assertFalse(crawl.check_disk_space(min_space_mb=100))
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs', side_effect=OSError("Fail")):
             self.assertFalse(crawl.check_disk_space())

    def test_signal_handler(self):
        with patch('crawl.cleanup_lmdb') as mock_cleanup:
            crawl.signal_handler(None, None)
            self.assertTrue(crawl.shutdown_flag)
            mock_cleanup.assert_called_once()

    def test_load_config_missing(self):
         with patch('builtins.open', side_effect=FileNotFoundError):
             config = crawl.load_config("missing.toml")
             self.assertEqual(config, {})
         with patch('builtins.open', side_effect=Exception("Error")):
             config = crawl.load_config("bad.toml")
             self.assertEqual(config, {})

    def test_main_execution(self):
        test_args = ['crawl.py', '--limit', '1', '--no-summary', '--workers', '1', '--config', 'test_config.toml']
        with patch('sys.argv', test_args), \
             patch('crawl.load_config', return_value={}), \
             patch('crawl.get_bookmarks', return_value=[{'url': 'http://example.com', 'name': 'Ex', 'type': 'url'}]), \
             patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0, 0)), \
             patch('crawl.init_lmdb'), \
             patch('crawl.prepare_webdriver'), \
             patch('crawl.cleanup_lmdb'), \
             patch('builtins.print'), \
             patch('json.dump'), \
             patch('crawl.safe_lmdb_operation', return_value=[]):
             crawl.main()
             crawl.init_lmdb.assert_called()

    def test_main_rebuild(self):
        test_args = ['crawl.py', '--rebuild', '--no-summary']
        mock_lmdb_env = MagicMock()
        crawl.lmdb_env = mock_lmdb_env
        with patch('sys.argv', test_args), \
             patch('crawl.load_config', return_value={}), \
             patch('crawl.get_bookmarks', return_value=[]), \
             patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0, 0)), \
             patch('crawl.init_lmdb'), \
             patch('crawl.prepare_webdriver'), \
             patch('crawl.cleanup_lmdb'), \
             patch('builtins.print'), \
             patch('json.dump'):
             crawl.main()

    def test_main_from_json(self):
        test_args = ['crawl.py', '--from-json', '--config', 'test.toml']
        with patch('sys.argv', test_args), \
             patch('crawl.load_config', return_value={'model': {'model_type': 'openai'}}), \
             patch('crawl.init_lmdb'), \
             patch('crawl.prepare_webdriver'), \
             patch('crawl.cleanup_lmdb'), \
             patch('builtins.print'), \
             patch('crawl.safe_lmdb_operation', return_value=[{'url': 'u', 'content': 'c'}]), \
             patch('crawl.test_api_connection', return_value=True), \
             patch('crawl.generate_summaries_for_bookmarks', return_value=[]) as mock_gen:
             crawl.main()
             mock_gen.assert_called()

    def test_main_backup_args(self):
         test_args = ['crawl.py', '--enable-backup', '--backup-on-failure-stop', '--no-summary']
         with patch('sys.argv', test_args), \
              patch('crawl.init_lmdb'), \
              patch('crawl.prepare_webdriver'), \
              patch('crawl.cleanup_lmdb'), \
              patch('crawl.get_bookmarks', return_value=[]), \
              patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0, 0)), \
              patch('crawl.load_config', return_value={}), \
              patch('json.dump'):
              crawl.main()

    def test_parallel_fetch_flush_error(self):
        bookmarks = [{'url': 'u1', 'name': 't1', 'type': 'url'}]
        crawl.lmdb_env = MagicMock()
        with patch('crawl.fetch_webpage_content', return_value=({'url': 'u1', 'content': 'c'}, None)):
            mock_txn = MagicMock()
            crawl.lmdb_env.begin.return_value.__enter__.return_value = mock_txn
            mock_txn.cursor.side_effect = Exception("Flush Error")
            mock_txn.get.return_value = None
            results = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1, flush_interval=0)
            if len(results) == 4: results = results[0] # Handle unpacked return
            self.assertEqual(len(results), 1)

    def test_parallel_fetch_worker_dedup_error(self):
        bookmarks = [{'url': 'u1', 'name': 't1', 'type': 'url'}]
        crawl.lmdb_env = MagicMock()
        mock_txn = MagicMock()
        crawl.lmdb_env.begin.return_value.__enter__.return_value = mock_txn
        mock_txn.get.side_effect = Exception("Dedup Error")
        results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['reason'], "Deduplication check failed")

    def test_parallel_fetch_worker_shutdown(self):
        crawl.shutdown_flag = True
        bookmarks = [{'url': 'u1', 'name': 't1', 'type': 'url'}]
        results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(failed), 0)
        crawl.shutdown_flag = False

    def test_parallel_fetch_future_exception(self):
        bookmarks = [{'url': 'u1', 'name': 't1', 'type': 'url'}]
        crawl.lmdb_env = MagicMock()
        mock_txn = MagicMock()
        crawl.lmdb_env.begin.return_value.__enter__.return_value = mock_txn
        mock_txn.get.return_value = None
        with patch('crawl.fetch_webpage_content', side_effect=Exception("Future Error")):
             results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1)
        self.assertEqual(len(results), 0)

    def test_parallel_fetch_limit_reached(self):
         bookmarks = [{'url': f'u{i}', 'name': f't{i}', 'type': 'url'} for i in range(5)]
         crawl.lmdb_env = MagicMock()
         mock_txn = MagicMock()
         crawl.lmdb_env.begin.return_value.__enter__.return_value = mock_txn
         mock_txn.get.return_value = None
         with patch('crawl.fetch_webpage_content', return_value=({'url': 'u', 'content': 'c'}, None)):
             results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1, limit=2)
         self.assertLessEqual(count, 3)

    def test_extract_domain_error(self):
        with patch('urllib.parse.urlparse', side_effect=Exception("Parse Error")):
            self.assertEqual(crawl.extract_domain("http://google.com"), "")

    def test_extract_date_error(self):
        bookmark = {'date_added': 'invalid'}
        self.assertEqual(crawl.extract_date(bookmark), datetime.datetime.now().strftime('%Y-%m-%d'))
        bookmark = {'crawl_time': 'invalid'}
        self.assertEqual(crawl.extract_date(bookmark), datetime.datetime.now().strftime('%Y-%m-%d'))

    def test_check_lmdb_exceptions(self):
        with patch('os.path.exists', side_effect=Exception("Outer Error")):
            exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
            self.assertFalse(exists)
        with patch('os.path.exists', return_value=True):
             with patch('os.path.join', return_value="path"):
                 with patch('lmdb.open', side_effect=Exception("Open Error")):
                     exists, has_data, count = crawl.check_lmdb_database_exists_and_has_data()
                     self.assertTrue(exists)
                     self.assertFalse(has_data)

    def test_create_lmdb_backup_reopen_failure(self):
        crawl.lmdb_env = MagicMock()
        with patch('crawl.check_lmdb_database_exists_and_has_data', return_value=(True, True, 10)), \
             patch('builtins.open', mock_open()), \
             patch('os.makedirs'), \
             patch('glob.glob', return_value=[]), \
             patch('shutil.copy2'), \
             patch('crawl.HAS_FCNTL', True), \
             patch('crawl.fcntl', create=True) as mock_fcntl, \
             patch('crawl.init_lmdb', side_effect=Exception("Reopen Fail")):
                 success, path = crawl.create_lmdb_backup("test")
                 self.assertTrue(success)

    def test_safe_lmdb_operation_exceptions(self):
        exceptions = [
            lmdb.MapResizedError("Resized"),
            lmdb.DiskError("Disk"),
            lmdb.InvalidError("Invalid"),
            lmdb.BadTxnError("BadTxn"),
            lmdb.BadRslotError("BadRslot"),
            lmdb.BadValsizeError("BadValsize")
        ]
        for exc in exceptions:
            mock_func = MagicMock(side_effect=exc)
            crawl.lmdb_env = MagicMock()
            crawl.use_fallback = False
            result = crawl.safe_lmdb_operation(mock_func, operation_name="test")
            self.assertIsNone(result)
            self.assertTrue(crawl.use_fallback)

    def test_safe_lmdb_operation_retry_fail(self):
        mock_func = MagicMock(side_effect=[lmdb.MapFullError("Full"), Exception("Retry Fail")])
        crawl.lmdb_env = MagicMock()
        crawl.current_lmdb_map_size = 100
        with patch('crawl.resize_lmdb_database', return_value=(True, 200)):
            result = crawl.safe_lmdb_operation(mock_func, operation_name="test")
            self.assertIsNone(result)
            self.assertTrue(crawl.use_fallback)

    def test_safe_lmdb_operation_readonly_map_full(self):
        mock_func = MagicMock(side_effect=lmdb.MapFullError("Full"))
        crawl.lmdb_env = MagicMock()
        result = crawl.safe_lmdb_operation(mock_func, operation_name="test", readonly=True)
        self.assertIsNone(result)
        self.assertTrue(crawl.use_fallback)

    def test_fetch_webpage_content_non_text(self):
        bookmark = {'url': 'http://image.png', 'name': 'img', 'type': 'url'}
        mock_response = MagicMock()
        mock_response.headers = {'Content-Type': 'image/png'}
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'pngdata'
        with patch('crawl.create_session') as mock_session:
            mock_session.return_value.get.return_value = mock_response
            result, failed = crawl.fetch_webpage_content(bookmark)
            self.assertIsNone(result)
            self.assertIn("Non-text content", failed['reason'])

    def test_call_deepseek_api_parsing(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.DEEPSEEK
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch('requests.post', return_value=mock_response):
            mock_response.json.return_value = {"choices": {"message": {"content": "DeepSeek Content"}}}
            result = crawl.call_deepseek_api("prompt", config)
            self.assertEqual(result, "DeepSeek Content")
            mock_response.json.return_value = {"choices": {"text": "DeepSeek Text"}}
            result = crawl.call_deepseek_api("prompt", config)
            self.assertEqual(result, "DeepSeek Text")

    def test_call_ollama_api_parsing(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.OLLAMA
        config.use_chat_api = True
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        with patch('requests.post', return_value=mock_response):
            mock_response.json.return_value = {"message": {"content": "Ollama Chat"}}
            result = crawl.call_ollama_api("prompt", config)
            self.assertEqual(result, "Ollama Chat")
            mock_response.json.return_value = {"response": "Ollama Response"}
            result = crawl.call_ollama_api("prompt", config)
            self.assertEqual(result, "Ollama Response")
            config.use_chat_api = False
            mock_response.json.return_value = {"response": "Ollama Generate"}
            result = crawl.call_ollama_api("prompt", config)
            self.assertEqual(result, "Ollama Generate")

    def test_generate_summary_dispatch_fixed(self):
        config = crawl.ModelConfig()
        with patch('crawl.call_ollama_api', return_value="O") as mock_ollama, \
             patch('crawl.call_qwen_api', return_value="Q") as mock_qwen, \
             patch('crawl.call_deepseek_api', return_value="D") as mock_deepseek:
             config.model_type = crawl.ModelConfig.OLLAMA
             self.assertEqual(crawl.generate_summary("t", "c", "u", config), "O")
             config.model_type = crawl.ModelConfig.QWEN
             self.assertEqual(crawl.generate_summary("t", "c", "u", config), "Q")
             config.model_type = crawl.ModelConfig.DEEPSEEK
             self.assertEqual(crawl.generate_summary("t", "c", "u", config), "D")
             config.model_type = "unknown"
             result = crawl.generate_summary("t", "c", "u", config)
             self.assertIn("Summary generation failed", result)
             self.assertIn("Unsupported model type", result)

    def test_main_parsers_arg(self):
        test_args = ['crawl.py', '--parsers', 'dummy', '--no-summary']
        with patch('sys.argv', test_args), \
             patch('crawl.load_custom_parsers') as mock_load, \
             patch('crawl.init_lmdb'), \
             patch('crawl.prepare_webdriver'), \
             patch('crawl.cleanup_lmdb'), \
             patch('crawl.get_bookmarks', return_value=[]), \
             patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0, 0)), \
             patch('crawl.load_config', return_value={}), \
             patch('json.dump'):
             crawl.main()
             mock_load.assert_called_with(parser_filter=['dummy'])

    def test_fetch_webpage_content_unicode_error(self):
        bookmark = {'url': 'http://example.com', 'name': 'Ex', 'type': 'url'}
        with patch('crawl.create_session', side_effect=Exception("Req Fail")), \
             patch('builtins.print', side_effect=[None, UnicodeEncodeError('utf-8', 'obj', 0, 1, 'bad'), None]):
             crawl.fetch_webpage_content(bookmark)

    def test_main_env_vars(self):
        test_args = ['crawl.py', '--no-summary']
        with patch('sys.argv', test_args), \
             patch.dict(os.environ, {'LMDB_MAP_SIZE': '123', 'LMDB_MAX_DBS': '5', 'LMDB_READONLY': 'True'}), \
             patch('crawl.init_lmdb') as mock_init, \
             patch('crawl.prepare_webdriver'), \
             patch('crawl.cleanup_lmdb'), \
             patch('crawl.get_bookmarks', return_value=[]), \
             patch('crawl.parallel_fetch_bookmarks', return_value=([], [], 0, 0)), \
             patch('crawl.load_config', return_value={}), \
             patch('json.dump'):
             crawl.main()
             mock_init.assert_called_with(map_size=123, max_dbs=5, readonly=True, resize_threshold=0.8, growth_factor=2.0)

    def test_prepare_webdriver_frozen_win32(self):
        with patch.object(sys, 'frozen', True, create=True), \
             patch('sys.platform', 'win32'), \
             patch('webdriver_manager.chrome.ChromeDriverManager.install', return_value='/path/to/driver'), \
             patch('os.path.exists', return_value=True):
             crawl.prepare_webdriver()
             self.assertTrue(crawl.webdriver_path.endswith('chromedriver.exe'))

    def test_fetch_webpage_content_with_delay(self):
        bookmark = {'url': 'http://example.com', 'name': 'Ex', 'type': 'url'}
        with patch('crawl.create_session') as mock_session, \
             patch('time.sleep') as mock_sleep:
            mock_session.return_value.get.return_value = MagicMock(content=b'html', text='html')
            crawl.fetch_webpage_content(bookmark, min_delay=0.1, max_delay=0.2)
            mock_sleep.assert_called()

    def test_fetch_with_selenium_zhihu(self):
        with patch('crawl.init_webdriver') as mock_driver:
            driver = mock_driver.return_value
            driver.page_source = "<html><body>Zhihu Content</body></html>"
            driver.find_element.side_effect = Exception("No button")
            content = crawl.fetch_with_selenium("http://zhihu.com/question/123")
            self.assertIn("Zhihu Content", content)

    def test_apply_custom_parsers_fail(self):
        fail_parser = MagicMock(side_effect=Exception("Parser Fail"))
        b = {'url': 'u'}
        res = crawl.apply_custom_parsers(b, [fail_parser])
        self.assertEqual(res, b)

    def test_fetch_webpage_content_shutdown(self):
        crawl.shutdown_flag = True
        b = {'url': 'u'}
        res, fail = crawl.fetch_webpage_content(b)
        self.assertIsNone(res)
        self.assertIsNone(fail)
        crawl.shutdown_flag = False

    def test_parallel_fetch_skip_unreachable_permutation(self):
        bookmarks = [{'url': 'u1', 'name': 't1', 'type': 'url'}]
        crawl.lmdb_env = MagicMock()
        mock_txn = MagicMock()
        crawl.lmdb_env.begin.return_value.__enter__.return_value = mock_txn
        mock_txn.get.return_value = None
        with patch('crawl.fetch_webpage_content', return_value=(None, {'reason': 'fail'})):
             results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1, skip_unreachable=True)
             self.assertEqual(len(results), 0)
             self.assertEqual(len(failed), 1)
             results, failed, count, _ = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1, skip_unreachable=False)
             self.assertEqual(len(results), 1)
             self.assertEqual(len(failed), 1)

    def test_get_bookmarks_function(self):
        with patch('crawl.Chrome') as mock_chrome:
            mock_chrome.return_value.fetch_bookmarks.return_value.bookmarks = [
                (datetime.datetime.now(), 'u', 't', 'f')
            ]
            bookmarks = crawl.get_bookmarks(browser='chrome')
            self.assertEqual(len(bookmarks), 1)
        with self.assertRaises(ValueError):
            crawl.get_bookmarks(browser='unknown')

    def test_api_connection_func(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.QWEN
        with patch('crawl.call_qwen_api', return_value="Hello"):
            self.assertTrue(crawl.test_api_connection(config))
        with patch('crawl.call_qwen_api', return_value=""):
            self.assertFalse(crawl.test_api_connection(config))

    def test_call_deepseek_api_empty_choices(self):
        config = crawl.ModelConfig()
        config.model_type = crawl.ModelConfig.DEEPSEEK
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"choices": []}
        with patch('requests.post', return_value=mock_response):
            result = crawl.call_deepseek_api("prompt", config)
            self.assertIn("[]", result)
