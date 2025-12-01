
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import lmdb
import pickle
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fuzzy_bookmark_search
from fuzzy_bookmark_search import FuzzyBookmarkSearch, search_bookmarks

class TestFuzzyBookmarkSearch(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.test_dir, "test_fuzzy.lmdb")
        self.index_dir = os.path.join(self.test_dir, "whoosh_index")

        # Create a dummy LMDB with data
        self.env = lmdb.open(self.lmdb_path, map_size=1024*1024, max_dbs=5)
        self.bookmarks_db = self.env.open_db(b'bookmarks')
        self.domain_index_db = self.env.open_db(b'domain_index')
        self.date_index_db = self.env.open_db(b'date_index')

        self.bookmarks = [
            {'url': 'https://example.com', 'title': 'Example', 'content': 'Content of example', 'summary': 'Summary of example', 'key': 'key1'},
            {'url': 'https://google.com', 'title': 'Google', 'content': 'Search engine', 'summary': 'Google search', 'key': 'key2'},
             {'url': 'https://python.org', 'title': 'Python', 'content': 'Python programming', 'summary': 'Python lang', 'key': 'key3'}
        ]

        with self.env.begin(write=True) as txn:
            for i, b in enumerate(self.bookmarks):
                key_bytes = b['key'].encode('utf-8')
                txn.put(key_bytes, pickle.dumps(b), db=self.bookmarks_db)

                # Mock domain index
                domain = b['url'].split('//')[1].encode('utf-8')

                # We need to accumulate keys for domain index if multiple bookmarks have same domain
                # But here they are unique
                # However, the previous implementation was overwriting instead of accumulating which is wrong if setup was meant to mimic real behavior.
                # Let's fix setup to accumulate
                existing = txn.get(domain, db=self.domain_index_db)
                if existing:
                    keys = pickle.loads(existing)
                    keys.add(key_bytes)
                else:
                    keys = {key_bytes}
                txn.put(domain, pickle.dumps(keys), db=self.domain_index_db)

                # Mock date index
                date = b'2023-01-01'
                existing_date = txn.get(date, db=self.date_index_db)
                if existing_date:
                    keys = pickle.loads(existing_date)
                    keys.add(key_bytes)
                else:
                    keys = {key_bytes}
                txn.put(date, pickle.dumps(keys), db=self.date_index_db)

        self.env.close()

        self.searcher = FuzzyBookmarkSearch(self.lmdb_path)

    def tearDown(self):
        self.searcher.cleanup_lmdb()
        shutil.rmtree(self.test_dir)

    def test_lmdb_open(self):
        self.searcher.lmdb_open()
        self.assertIsNotNone(self.searcher.lmdb_env)
        self.assertIsNotNone(self.searcher.bookmarks_db)

    def test_load_bookmarks_data(self):
        self.searcher.lmdb_open()
        generator = self.searcher.load_bookmarks_data()
        bookmarks = list(generator)
        self.assertEqual(len(bookmarks), 3)
        self.assertEqual(bookmarks[0]['title'], 'Example')

    def test_query_bookmarks_by_domain(self):
        self.searcher.lmdb_open()
        results = self.searcher.query_bookmarks_by_domain('example.com')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Example')

    def test_query_bookmarks_by_date(self):
        self.searcher.lmdb_open()
        results = self.searcher.query_bookmarks_by_date('2023-01-01')
        self.assertEqual(len(results), 3) # Logic in setup put all 3 in same date

    def test_get_domain_stats(self):
        self.searcher.lmdb_open()
        stats = self.searcher.get_domain_stats()
        self.assertEqual(len(stats), 3)
        self.assertIn('example.com', stats)

    def test_get_date_stats(self):
        self.searcher.lmdb_open()
        stats = self.searcher.get_date_stats()
        self.assertEqual(len(stats), 1)
        self.assertIn('2023-01-01', stats)

    def test_create_app(self):
        app = self.searcher.create_app()
        client = TestClient(app)

        # Test UI endpoint
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fuzzy Bookmark Search", response.text)

    @patch('fuzzy_bookmark_search.search_bookmarks')
    def test_api_search(self, mock_search):
        app = self.searcher.create_app()
        client = TestClient(app)

        mock_search.return_value = {
            'results': [],
            'pagination': {},
            'search_time': 0.1,
            'query': 'test'
        }

        response = client.post("/api/search", json={"query": "test"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['query'], 'test')

        # Test invalid query
        response = client.post("/api/search", json={"query": ""})
        self.assertEqual(response.status_code, 400)

    def test_indexing_and_searching(self):
        # Test the whole flow including indexing
        self.searcher.lmdb_open()

        # 1. Indexing
        # We need to call index_bookmarks. It uses a generator.
        bookmarks_gen = self.searcher.load_bookmarks_data()
        fuzzy_bookmark_search.index_bookmarks(bookmarks_gen, index_dir=self.index_dir)

        self.assertTrue(os.path.exists(self.index_dir))

        # 2. Searching
        result = search_bookmarks("Example", index_dir=self.index_dir)
        self.assertGreater(len(result['results']), 0)
        self.assertEqual(result['results'][0]['title'], 'Example')

        # Fuzzy search
        # Note: "Exampel" might not match "Example" with default fuzzy distance if word is short or depending on config.
        # But "Python" ~ "Pytho" should match.
        result = search_bookmarks("Python~1", index_dir=self.index_dir)
        self.assertGreater(len(result['results']), 0)

        # Pagination
        result = search_bookmarks("Python", index_dir=self.index_dir, page=1, page_size=1)
        self.assertEqual(len(result['results']), 1)
        # 3 items total, only 1 matches Python. So has_next should be False.
        self.assertFalse(result['pagination']['has_next'])

        # Check total results count
        # In setup we have 1 Python bookmark.
        self.assertEqual(result['pagination']['total_results'], 1)

    def test_safe_lmdb_operation_fallback(self):
        # Trigger an error to test fallback
        def op(txn):
            raise lmdb.Error("Fail")

        fallback = lambda: "Fallback"
        result = self.searcher.safe_lmdb_operation(op, fallback)
        self.assertEqual(result, "Fallback")
        self.assertTrue(self.searcher.use_fallback)

if __name__ == '__main__':
    unittest.main()
