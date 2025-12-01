import unittest
from unittest.mock import patch, MagicMock, call, mock_open
import sys
import os
import shutil
import tempfile
import lmdb
import pickle
from fastapi.testclient import TestClient
from fastapi import HTTPException
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fuzzy_bookmark_search
from fuzzy_bookmark_search import (
    FuzzyBookmarkSearch, search_bookmarks, format_search_time,
    main, index_bookmarks, get_or_create_index, create_schema
)

class TestFuzzyCoverage(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test.lmdb")
        self.index_dir = os.path.join(self.test_dir, "whoosh_index")
        self.searcher = FuzzyBookmarkSearch(self.lmdb_path)

    def tearDown(self):
        try:
            if self.searcher.lmdb_env:
                # If it's a mock, reset side effect to avoid error during close
                if isinstance(self.searcher.lmdb_env, MagicMock):
                    self.searcher.lmdb_env.close.side_effect = None
                self.searcher.lmdb_env.close()
        except Exception:
            pass
        shutil.rmtree(self.test_dir)

    def test_lmdb_open_missing_path_exit(self):
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(SystemExit):
                self.searcher.lmdb_open(no_update=False)

    def test_lmdb_open_missing_path_fallback(self):
        with patch('os.path.exists', return_value=False):
            self.searcher.lmdb_open(no_update=True)
            self.assertTrue(self.searcher.use_fallback)

    def test_lmdb_open_exception(self):
        with patch('os.path.exists', return_value=True):
            with patch('lmdb.open', side_effect=Exception("LMDB Error")):
                self.searcher.lmdb_open()
                self.assertTrue(self.searcher.use_fallback)

    def test_lmdb_open_cleanup_exception(self):
        # Test exception during cleanup in the exception handler of lmdb_open
        with patch('os.path.exists', return_value=True):
            mock_env = MagicMock()
            mock_env.close.side_effect = Exception("Cleanup Error")
            with patch('lmdb.open', return_value=mock_env):
                # Force an error after open to trigger cleanup
                with patch.object(mock_env, 'open_db', side_effect=Exception("Open DB Error")):
                    self.searcher.lmdb_open()
                    self.assertTrue(self.searcher.use_fallback)

    def test_safe_lmdb_operation_exceptions(self):
        self.searcher.lmdb_env = MagicMock()

        exceptions = [
            lmdb.DiskError("Disk"),
            lmdb.InvalidError("Invalid"),
            lmdb.BadTxnError("BadTxn"),
            lmdb.BadRslotError("BadRslot"),
            lmdb.BadValsizeError("BadValsize"),
            Exception("Generic")
        ]

        for exc in exceptions:
            self.searcher.use_fallback = False # Reset
            mock_op = MagicMock(side_effect=exc)
            self.searcher.safe_lmdb_operation(mock_op, operation_name="test")
            self.assertTrue(self.searcher.use_fallback, f"Failed to set fallback for {type(exc)}")

    def test_safe_lmdb_operation_fallback_execution_failure(self):
        self.searcher.use_fallback = True
        mock_fallback = MagicMock(side_effect=Exception("Fallback Fail"))
        result = self.searcher.safe_lmdb_operation(MagicMock(), fallback_func=mock_fallback)
        self.assertIsNone(result)

    def test_safe_lmdb_operation_fallback_after_exception_failure(self):
        self.searcher.lmdb_env = MagicMock()
        mock_op = MagicMock(side_effect=Exception("Op Fail"))
        mock_fallback = MagicMock(side_effect=Exception("Fallback Fail"))

        result = self.searcher.safe_lmdb_operation(mock_op, fallback_func=mock_fallback)
        self.assertIsNone(result)

    def test_cleanup_lmdb_exception(self):
        self.searcher.lmdb_env = MagicMock()
        self.searcher.lmdb_env.close.side_effect = Exception("Close Fail")
        # Should not raise
        self.searcher.cleanup_lmdb()

    def test_load_bookmarks_data_none_list(self):
        with patch.object(self.searcher, 'load_bookmarks_from_lmdb', return_value=None):
            gen = self.searcher.load_bookmarks_data()
            self.assertEqual(list(gen), [])

    def test_load_bookmarks_data_empty_list(self):
        with patch.object(self.searcher, 'load_bookmarks_from_lmdb', return_value=[]):
            gen = self.searcher.load_bookmarks_data()
            self.assertEqual(list(gen), [])

    def test_load_bookmarks_data_truncation_and_processing(self):
        long_content = "a" * 10005
        bookmarks = [{
            'guid': 'guid1',
            'title': 'Title',
            'url': 'url',
            'content': long_content,
            'summary': 'summary'
        }]

        with patch.object(self.searcher, 'load_bookmarks_from_lmdb', return_value=bookmarks):
            gen = self.searcher.load_bookmarks_data()
            results = list(gen)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]['content'].endswith('...'))
            self.assertEqual(len(results[0]['content']), 10003) # 10000 + '...'

    def test_load_bookmarks_from_lmdb_corrupted(self):
        self.searcher.lmdb_env = MagicMock()
        self.searcher.bookmarks_db = MagicMock()

        mock_txn = MagicMock()
        mock_cursor = MagicMock()
        # Yield one good, one bad (pickle error), one good
        # Note: MagicMock iteration is weird, list works better
        mock_cursor.__iter__.return_value = [
            (b'key1', pickle.dumps({'title': 'good1'})),
            (b'key2', b'bad pickle data'),
            (b'key3', pickle.dumps({'title': 'good2'}))
        ]
        mock_txn.cursor.return_value = mock_cursor

        def mock_begin(write=False):
            ctx = MagicMock()
            ctx.__enter__.return_value = mock_txn
            ctx.__exit__.return_value = None
            return ctx

        self.searcher.lmdb_env.begin = mock_begin

        result = self.searcher.load_bookmarks_from_lmdb()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'good1')
        self.assertEqual(result[1]['title'], 'good2')

    def test_query_bookmarks_empty(self):
        # Test query return empty if keys not found
        self.searcher.lmdb_env = MagicMock()
        self.searcher.domain_index_db = MagicMock()
        self.searcher.date_index_db = MagicMock()

        mock_txn = MagicMock()
        mock_txn.get.return_value = None # No keys

        def mock_begin(write=False):
            ctx = MagicMock()
            ctx.__enter__.return_value = mock_txn
            ctx.__exit__.return_value = None
            return ctx
        self.searcher.lmdb_env.begin = mock_begin

        res_domain = self.searcher.query_bookmarks_by_domain("example.com")
        self.assertEqual(res_domain, [])

        res_date = self.searcher.query_bookmarks_by_date("2023-01-01")
        self.assertEqual(res_date, [])

    def test_create_app_api_search_errors(self):
        app = self.searcher.create_app()
        client = TestClient(app)

        # Test invalid page
        response = client.post("/api/search", json={"query": "test", "page": 0})
        self.assertEqual(response.status_code, 400)

        # Test invalid page_size
        response = client.post("/api/search", json={"query": "test", "page_size": 101})
        self.assertEqual(response.status_code, 400)

        # Test generic exception during search
        with patch('fuzzy_bookmark_search.search_bookmarks', side_effect=Exception("Search Fail")):
            response = client.post("/api/search", json={"query": "test"})
            self.assertEqual(response.status_code, 500)

    def test_index_bookmarks_batch_and_update(self):
        # Generate many bookmarks
        bookmarks = []
        for i in range(2005): # > 2000 batch size
            bookmarks.append({
                'key': f'key{i}',
                'title': f'title{i}',
                'url': f'url{i}',
                'content': f'content{i}',
                'summary': f'summary{i}',
                'total_records': 2005
            })

        mock_ix = MagicMock()
        mock_writer = MagicMock()
        mock_ix.writer.return_value = mock_writer

        with patch('fuzzy_bookmark_search.get_or_create_index', return_value=mock_ix):
            # Test update=False
            index_bookmarks(iter(bookmarks), self.index_dir, update=False)

            # Check add_document calls
            self.assertEqual(mock_writer.add_document.call_count, 2005)
            mock_writer.commit.assert_called_once()

    def test_index_bookmarks_update_duplicates(self):
        bookmarks = [
            {'key': 'key1', 'title': 't1', 'url': 'u1', 'content': 'c1', 'summary': 's1', 'total_records': 2},
            {'key': 'key2', 'title': 't2', 'url': 'u2', 'content': 'c2', 'summary': 's2', 'total_records': 2}
        ]

        mock_ix = MagicMock()
        mock_writer = MagicMock()
        mock_ix.writer.return_value = mock_writer
        mock_searcher = MagicMock()
        mock_ix.searcher.return_value.__enter__.return_value = mock_searcher
        mock_ix.searcher.return_value.__exit__.return_value = None

        # Mock existing documents
        mock_searcher.documents.return_value = [{'key': 'key1'}]

        with patch('fuzzy_bookmark_search.get_or_create_index', return_value=mock_ix):
            with patch('whoosh.index.exists_in', return_value=True):
                index_bookmarks(iter(bookmarks), self.index_dir, update=True)

                # key1 should be skipped, key2 added
                self.assertEqual(mock_writer.add_document.call_count, 1)
                args, _ = mock_writer.add_document.call_args
                # Verify we added key2 (or args from kwargs)
                # kwargs are used in call
                _, kwargs = mock_writer.add_document.call_args
                self.assertEqual(kwargs['key'], 'key2')

    def test_format_search_time(self):
        # Coverage for the format_search_time function
        self.assertEqual(format_search_time(1.5), ".2f")
        self.assertEqual(format_search_time(0.5), ".0f")

    def test_main(self):
        # Mock sys.argv
        with patch.object(sys, 'argv', ['fuzzy_bookmark_search.py', '--port', '9000', '--no-update']):
            with patch('fuzzy_bookmark_search.FuzzyBookmarkSearch') as MockSearch:
                with patch('uvicorn.run') as mock_uvicorn:
                    with patch('fuzzy_bookmark_search.index.exists_in', return_value=True):
                         with patch('fuzzy_bookmark_search.index_bookmarks') as mock_index:
                             # Mock open_dir for total count
                             mock_ix = MagicMock()
                             mock_ix.searcher.return_value.__enter__.return_value.doc_count_all.return_value = 100
                             with patch('fuzzy_bookmark_search.index.open_dir', return_value=mock_ix):
                                 main()

                                 mock_uvicorn.assert_called_once()
                                 MockSearch.return_value.lmdb_open.assert_called_once()
                                 # no-update passed, and index exists, so index_bookmarks should NOT be called?
                                 # Logic: if not index.exists or not args.no_update: index...
                                 # here index exists, no_update is true. So it enters "else: Index already exists".
                                 mock_index.assert_not_called()

    def test_main_indexing_needed(self):
         with patch.object(sys, 'argv', ['fuzzy_bookmark_search.py']): # default no-update is False
            with patch('fuzzy_bookmark_search.FuzzyBookmarkSearch') as MockSearch:
                with patch('uvicorn.run'):
                    with patch('fuzzy_bookmark_search.index.exists_in', return_value=False):
                         with patch('fuzzy_bookmark_search.index_bookmarks') as mock_index:
                             # Mock generator
                             MockSearch.return_value.load_bookmarks_data.return_value = iter([])

                             with patch('fuzzy_bookmark_search.index.open_dir'):
                                 main()
                                 mock_index.assert_called_once()

    def test_main_error_handling(self):
        # Test error during indexing
        with patch.object(sys, 'argv', ['fuzzy_bookmark_search.py']):
            with patch('fuzzy_bookmark_search.FuzzyBookmarkSearch') as MockSearch:
                with patch('uvicorn.run'):
                    with patch('fuzzy_bookmark_search.index.exists_in', side_effect=Exception("Index Error")):
                        main()
                        # Should continue to server startup even if indexing fails
                        MockSearch.return_value.create_app.assert_called_once()

        # Test error during count
        with patch.object(sys, 'argv', ['fuzzy_bookmark_search.py', '--no-update']):
             with patch('fuzzy_bookmark_search.FuzzyBookmarkSearch') as MockSearch:
                with patch('uvicorn.run'):
                    with patch('fuzzy_bookmark_search.index.exists_in', return_value=True):
                        with patch('fuzzy_bookmark_search.index.open_dir', side_effect=Exception("Open Error")):
                             main()
                             # Should print error but continue
                             MockSearch.return_value.create_app.assert_called_once()

    def test_get_or_create_index_exists(self):
        with patch('os.path.exists', return_value=True):
            with patch('whoosh.index.exists_in', return_value=True):
                with patch('whoosh.index.open_dir') as mock_open:
                    get_or_create_index(self.index_dir)
                    mock_open.assert_called_once()

    def test_get_or_create_index_create(self):
        with patch('os.path.exists', return_value=False): # Dir doesn't exist
            with patch('os.makedirs') as mock_makedirs:
                with patch('whoosh.index.exists_in', return_value=False):
                    with patch('whoosh.index.create_in') as mock_create:
                        get_or_create_index(self.index_dir)
                        mock_makedirs.assert_called_once()
                        mock_create.assert_called_once()

    def test_search_bookmarks_pagination(self):
        # Mock index and searcher
        mock_ix = MagicMock()
        mock_searcher = MagicMock()
        mock_ix.searcher.return_value.__enter__.return_value = mock_searcher
        mock_ix.searcher.return_value.__exit__.return_value = None

        # Mock search results
        mock_hit = MagicMock()
        mock_hit.score = 1.0
        mock_hit.__getitem__.side_effect = lambda k: "val"
        mock_hit.highlights.return_value = "snippet"
        mock_hit.fields.return_value = {}

        mock_searcher.search_page.return_value = [mock_hit]
        mock_searcher.search.return_value.estimated_length.return_value = 1

        with patch('whoosh.index.open_dir', return_value=mock_ix):
            with patch('whoosh.qparser.QueryParser.parse'):
                res = search_bookmarks("query", index_dir=self.index_dir, page=1, page_size=10)

                self.assertEqual(len(res['results']), 1)
                self.assertEqual(res['pagination']['total_results'], 1)

    def test_search_bookmarks_snippet_fallback(self):
        # Test when highlights returns None
        mock_ix = MagicMock()
        mock_searcher = MagicMock()
        mock_ix.searcher.return_value.__enter__.return_value = mock_searcher
        mock_ix.searcher.return_value.__exit__.return_value = None

        mock_hit = MagicMock()
        mock_hit.highlights.return_value = None
        mock_hit.__getitem__.side_effect = lambda k: "Content " * 50 # Long content
        mock_hit.fields.return_value = {}

        mock_searcher.search_page.return_value = [mock_hit]
        mock_searcher.search.return_value.estimated_length.return_value = 1

        with patch('whoosh.index.open_dir', return_value=mock_ix):
            with patch('whoosh.qparser.QueryParser.parse'):
                res = search_bookmarks("query", index_dir=self.index_dir)
                snippet = res['results'][0]['snippet']
                self.assertTrue(snippet.endswith("..."))

    def test_compatibility_functions(self):
        with patch('fuzzy_bookmark_search._default_search') as mock_def:
            fuzzy_bookmark_search.lmdb_open(True)
            mock_def.lmdb_open.assert_called_with(True)

            fuzzy_bookmark_search.load_bookmarks_data()
            mock_def.load_bookmarks_data.assert_called()

            fuzzy_bookmark_search.cleanup_lmdb()
            mock_def.cleanup_lmdb.assert_called()

            fuzzy_bookmark_search.query_bookmarks_by_domain("d")
            mock_def.query_bookmarks_by_domain.assert_called_with("d", 50)

            fuzzy_bookmark_search.query_bookmarks_by_date("d")
            mock_def.query_bookmarks_by_date.assert_called_with("d", 50)

            fuzzy_bookmark_search.get_domain_stats()
            mock_def.get_domain_stats.assert_called()

            fuzzy_bookmark_search.get_date_stats()
            mock_def.get_date_stats.assert_called()

if __name__ == '__main__':
    unittest.main()
