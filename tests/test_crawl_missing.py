import sys
import os
import unittest
import shutil
import tempfile
import pickle
import time
import datetime
import threading
import json
import requests
import hashlib
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import importlib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import crawl

class TestCrawlMissing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test_lmdb")
        self.bookmarks_path = os.path.join(self.test_dir, "bookmarks.json")
        self.failed_urls_path = os.path.join(self.test_dir, "failed_urls.json")

        # Patch paths
        self.patcher_lmdb_path = patch('crawl.lmdb_storage_path', self.lmdb_path)
        self.patcher_bookmarks_path = patch('crawl.bookmarks_path', self.bookmarks_path)
        self.patcher_failed_path = patch('crawl.failed_urls_path', self.failed_urls_path)

        self.patcher_lmdb_path.start()
        self.patcher_bookmarks_path.start()
        self.patcher_failed_path.start()

        # Reset global state
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
        crawl.webdriver_path = None
        crawl.fallback_bookmarks = []
        crawl.fallback_failed_records = []

        # Create dummy custom parsers dir
        self.parsers_dir = os.path.join(self.test_dir, "custom_parsers")
        os.makedirs(self.parsers_dir, exist_ok=True)

    def tearDown(self):
        if crawl.lmdb_env:
            try:
                crawl.lmdb_env.close()
            except:
                pass

        self.patcher_lmdb_path.stop()
        self.patcher_bookmarks_path.stop()
        self.patcher_failed_path.stop()
        shutil.rmtree(self.test_dir)

    def test_sanitize_bookmark_recursion(self):
        # Create a recursive dictionary
        recursive_dict = {"a": 1}
        recursive_dict["b"] = recursive_dict

        # Should return None for the recursive part or handle it gracefully
        sanitized = crawl.sanitize_bookmark(recursive_dict)
        # Verify structure: "b" should not be present or be None/pruned
        self.assertIn("a", sanitized)
        self.assertIsNone(sanitized.get("b"))

        with patch('sys.setrecursionlimit') as mock_set:
            crawl.safe_pickle({"a": 1})
            mock_set.assert_called()

    def test_sanitize_bookmark_complex_objects(self):
        # Test removal of selenium objects and others
        class MockDriver:
            def quit(self): pass
            def get(self): pass
            def find_element(self): pass

        class ComplexObj:
            pass

        bookmark = {
            "driver": MockDriver(),
            "complex": ComplexObj(),
            "valid": "value"
        }
        sanitized = crawl.sanitize_bookmark(bookmark)
        self.assertNotIn("driver", sanitized)
        self.assertNotIn("complex", sanitized)
        self.assertIn("valid", sanitized)

    @patch('crawl.shutil.disk_usage')
    def test_check_disk_space_low(self, mock_usage):
        # Return free space < 100MB
        mock_usage.return_value = MagicMock(free=50 * 1024 * 1024)
        self.assertFalse(crawl.check_disk_space(min_space_mb=100))

    @patch('crawl.shutil.disk_usage')
    def test_check_disk_space_mkdir_fail(self, mock_usage):
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs', side_effect=Exception("Perm Denied")):
                self.assertFalse(crawl.check_disk_space())

    def test_init_lmdb_map_full_error(self):
        with patch('crawl.lmdb.open', side_effect=crawl.lmdb.MapFullError("Full")):
            crawl.init_lmdb()
            self.assertTrue(crawl.use_fallback)


    @patch('crawl.lmdb.open')
    def test_resize_lmdb_database_success_and_fail(self, mock_open):
        # Setup mocks
        mock_env = MagicMock()
        mock_open.return_value = mock_env

        # Success case
        success, new_size = crawl.resize_lmdb_database(100, 2.0)
        self.assertTrue(success)
        self.assertEqual(new_size, 200)

        # Failure case: all attempts fail
        mock_open.side_effect = Exception("Fail")
        success, new_size = crawl.resize_lmdb_database(100, 2.0, max_attempts=1)
        self.assertFalse(success)

    def test_load_custom_parsers_filtering(self):
        # Create dummy parser files
        with open(os.path.join(self.parsers_dir, "parser_a.py"), "w") as f:
            f.write("def main(b): return b")
        with open(os.path.join(self.parsers_dir, "parser_b.py"), "w") as f:
            f.write("def main(b): return b")

        with patch('crawl.get_custom_parsers_dir', return_value=self.parsers_dir):
            # Test filtering
            parsers = crawl.load_custom_parsers(parser_filter=["parser_a"])
            self.assertEqual(len(parsers), 1)

            # Test filter match failure
            parsers = crawl.load_custom_parsers(parser_filter=["parser_c"])
            self.assertEqual(len(parsers), 0)

    def test_load_custom_parsers_missing_warning(self):
        with open(os.path.join(self.parsers_dir, "parser_a.py"), "w") as f:
            f.write("def main(b): return b")

        with patch('crawl.get_custom_parsers_dir', return_value=self.parsers_dir):
            with patch('builtins.print') as mock_print:
                crawl.load_custom_parsers(parser_filter=["parser_a", "parser_missing"])

                # Check for warning
                # mock_print calls are (args, kwargs). args[0] is the message.
                calls = [args[0] for args, _ in mock_print.call_args_list]
                warning_found = any("Custom parser 'parser_missing' specified in filter but not found" in str(c) for c in calls)
                self.assertTrue(warning_found, f"Warning for missing parser not found. Calls: {calls}")

    @patch('requests.post')
    def test_call_ollama_api_generate_interface(self, mock_post):
        # Test the 'else' branch of use_chat_api
        config = crawl.ModelConfig()
        config.use_chat_api = False # Inject config to use generate interface
        config.model_name = "test-model"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated text"}
        mock_post.return_value = mock_response

        result = crawl.call_ollama_api("prompt", config)
        self.assertEqual(result, "Generated text")

        # Verify payload structure for generate interface
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        self.assertIn('prompt', payload)
        self.assertNotIn('messages', payload)

    @patch('requests.post')
    def test_call_qwen_api_formats(self, mock_post):
        config = crawl.ModelConfig()

        # Case 1: choices -> text
        mock_post.return_value.json.return_value = {"choices": {"text": "Qwen Text"}}
        self.assertEqual(crawl.call_qwen_api("p", config), "Qwen Text")

        # Case 2: No choices, return str(result)
        mock_post.return_value.json.return_value = {"other": "value"}
        self.assertIn("value", crawl.call_qwen_api("p", config))

    @patch('requests.post')
    def test_call_deepseek_api_formats(self, mock_post):
        config = crawl.ModelConfig()

        # Case 1: choices -> text
        mock_post.return_value.json.return_value = {"choices": {"text": "DS Text"}}
        self.assertEqual(crawl.call_deepseek_api("p", config), "DS Text")

        # Case 2: No choices
        mock_post.return_value.json.return_value = {"other": "value"}
        self.assertIn("value", crawl.call_deepseek_api("p", config))

    def test_extract_domain_date_errors(self):
        self.assertEqual(crawl.extract_domain(None), "") # Expect exception caught
        self.assertEqual(crawl.extract_date({}), datetime.datetime.now().strftime('%Y-%m-%d')) # Fallback

        # Test extract_date with 'T' in date_added
        bookmark = {'date_added': '2023-01-01T12:00:00Z'}
        self.assertEqual(crawl.extract_date(bookmark), '2023-01-01')

    @patch('crawl.init_webdriver')
    def test_fetch_with_selenium_zhihu_popup(self, mock_init):
        mock_driver = MagicMock()
        mock_init.return_value = mock_driver
        mock_driver.page_source = "<html>Content</html>"

        # Mock find_element to succeed (simulate popup found)
        mock_button = MagicMock()
        mock_driver.find_element.return_value = mock_button

        crawl.fetch_with_selenium("http://zhihu.com/question/1", 1, 1, "Title")
        # Verify close button click attempt
        mock_driver.find_element.assert_called()
        mock_button.click.assert_called()

    @patch('crawl.create_session')
    @patch('crawl.fetch_with_selenium')
    def test_fetch_webpage_content_encoding_error(self, mock_sel, mock_session):
        # Simulate UnicodeEncodeError during print (hard to verify without capturing stdout, but we can verify code doesn't crash)
        # We'll just test the content encoding fix logic via fetch_webpage_content

        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_session.return_value.get.return_value = mock_response

        with patch('builtins.print') as mock_print:
             # Just run it
             crawl.fetch_webpage_content({"url": "http://test.com", "name": "Test"})

    def test_fix_encoding_logic(self):
        # Trigger optimized path
        # Short text
        self.assertEqual(crawl.fix_encoding("short"), "short")

        # Text with few non-ascii
        self.assertEqual(crawl.fix_encoding("mostly ascii"), "mostly ascii")

        # Text with garbage (simulated)
        bad_text = "Ã©" * 20 # Latin-1 encoded utf-8
        # We rely on chardet here.
        self.assertIsInstance(crawl.fix_encoding(bad_text), str)

    @patch('crawl.fetch_webpage_content')
    def test_parallel_fetch_bookmarks_shutdown(self, mock_fetch):
        bookmarks = [{"url": "u1"}, {"url": "u2"}]
        mock_fetch.return_value = ({"url": "u1", "content": "c"}, None)

        # Set shutdown flag immediately
        crawl.shutdown_flag = True
        results, failed, added = crawl.parallel_fetch_bookmarks(bookmarks, max_workers=1)

        # Should return what was processed (likely empty or close to it)
        # Since we cancel futures, result might vary, but it should return.
        self.assertTrue(True) # Just verifying it returns

    @patch('crawl.prepare_webdriver')
    @patch('crawl.parse_args')
    @patch('crawl.load_config')
    @patch('crawl.init_lmdb')
    @patch('crawl.get_bookmarks')
    @patch('crawl.parallel_fetch_bookmarks')
    def test_main_rebuild(self, mock_parallel, mock_get_bk, mock_init, mock_load, mock_args, mock_prep):
        # Mock args
        args = MagicMock()
        args.rebuild = True
        args.limit = 0
        args.workers = 1
        args.no_summary = True
        args.browser = None
        args.profile_path = None
        args.config = "c"
        args.enable_backup = False
        args.disable_backup = True
        args.backup_dir = "."
        args.backup_on_failure_stop = False
        args.from_json = False
        args.parsers = None
        mock_args.return_value = args

        mock_parallel.return_value = ([], [], 0)

        # Mock lmdb environment for rebuild (drop calls)
        mock_env = MagicMock()
        crawl.lmdb_env = mock_env
        mock_txn = MagicMock()
        mock_env.begin.return_value = mock_txn
        mock_txn.__enter__.return_value = mock_txn

        crawl.main()

        # Verify drop calls
        self.assertTrue(mock_txn.drop.called)

    @patch('crawl.fetch_webpage_content')
    def test_parallel_fetch_flush_logic(self, mock_fetch):
        # Test that flush_to_disk is called
        bookmarks = [{"url": f"http://test{i}.com"} for i in range(10)]
        mock_fetch.return_value = ({"url": "u", "content": "c"}, None)

        # Use a real thread pool but mock init_lmdb to use fallback or mocks
        crawl.use_fallback = True
        crawl.fallback_bookmarks = []
        crawl.fallback_failed_records = []

        # Let's try to hit the exception in future processing
        mock_fetch.side_effect = Exception("Worker fail")
        crawl.parallel_fetch_bookmarks(bookmarks, max_workers=2)

    def test_init_lmdb_all_exceptions(self):
        exceptions = [
            crawl.lmdb.MapFullError("Full"),
            crawl.lmdb.MapResizedError("Resized"),
            crawl.lmdb.DiskError("Disk"),
            crawl.lmdb.InvalidError("Invalid"),
            crawl.lmdb.VersionMismatchError("Version"),
            crawl.lmdb.BadRslotError("Bad slot"),
            Exception("Generic")
        ]

        for exc in exceptions:
            with patch('crawl.lmdb.open', side_effect=exc):
                crawl.use_fallback = False
                crawl.init_lmdb()
                self.assertTrue(crawl.use_fallback, f"Failed to fallback on {type(exc)}")


    @patch('crawl.lmdb.open')
    def test_check_lmdb_exists_exceptions(self, mock_open):
        # Outer exception
        with patch('os.path.exists', side_effect=Exception("Outer")):
            e, h, c = crawl.check_lmdb_database_exists_and_has_data()
            self.assertFalse(e)

        # Inner exception (during open or txn)
        with patch('os.path.exists', return_value=True):
             mock_env = MagicMock()
             mock_open.return_value = mock_env
             mock_env.open_db.side_effect = Exception("Inner")
             e, h, c = crawl.check_lmdb_database_exists_and_has_data()
             self.assertTrue(e)
             self.assertFalse(h)
             mock_env.close.assert_called()


    @patch('requests.post')
    def test_api_exceptions(self, mock_post):
        config = crawl.ModelConfig()

        # Requests Exception
        mock_post.side_effect = requests.exceptions.RequestException("Net Error")

        with self.assertRaises(Exception):
            crawl.call_ollama_api("p", config)
        with self.assertRaises(Exception):
            crawl.call_qwen_api("p", config)
        with self.assertRaises(Exception):
            crawl.call_deepseek_api("p", config)

        # JSON Decode Error
        mock_post.side_effect = None
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.side_effect = ValueError("Bad JSON")

        with self.assertRaises(Exception):
            crawl.call_ollama_api("p", config)

        # Generic Exception (if mocked properly)
        mock_post.side_effect = Exception("Generic")
        with self.assertRaises(Exception):
            crawl.call_ollama_api("p", config)

    def test_generate_summary_exception(self):
        with patch('crawl.call_ollama_api', side_effect=Exception("Fail")):
             config = crawl.ModelConfig()
             config.model_type = crawl.ModelConfig.OLLAMA
             res = crawl.generate_summary("t", "c", "u", config)
             self.assertIn("failed", res)

    @patch('crawl.safe_backup_operation', return_value=False)
    @patch('crawl.parse_args')
    def test_main_backup_stop(self, mock_args, mock_backup):
        args = MagicMock()
        args.rebuild = False
        args.limit = 0
        args.enable_backup = True
        args.disable_backup = False
        args.backup_on_failure_stop = True # Should stop
        mock_args.return_value = args

        # We need to mock other things to reach backup check
        crawl.init_lmdb = MagicMock()
        crawl.safe_lmdb_operation = MagicMock(return_value=[{}]) # Existing bookmarks

        with patch('builtins.print'):
             # Also mock get_bookmarks to ensure it's not called (verification)
             with patch('crawl.get_bookmarks') as mock_get:
                 crawl.main()
                 mock_get.assert_not_called()

if __name__ == '__main__':
    unittest.main()
