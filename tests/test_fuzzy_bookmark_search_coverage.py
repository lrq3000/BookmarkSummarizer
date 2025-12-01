import sys
import os
import unittest
from unittest.mock import MagicMock, patch, ANY
import lmdb
import tempfile
import shutil
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzy_bookmark_search import FuzzyBookmarkSearch, index_bookmarks, main

class TestFuzzyBookmarkSearchCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.tmp_dir, "test.lmdb")
        self.index_dir = os.path.join(self.tmp_dir, "test_index")
        self.searcher = FuzzyBookmarkSearch(self.lmdb_path)

    def tearDown(self):
        self.searcher.cleanup_lmdb()
        shutil.rmtree(self.tmp_dir)

    def test_safe_lmdb_operation_fallback_failure(self):
        """Test safe_lmdb_operation returns None when fallback fails or is missing."""
        self.searcher.use_fallback = True

        # Case 1: No fallback function (Line 120 coverage)
        result = self.searcher.safe_lmdb_operation(lambda txn: "success", fallback_func=None)
        self.assertIsNone(result)

        # Case 2: Fallback function raises exception (Line 118-119 coverage)
        def failing_fallback():
            raise ValueError("Fallback failed")

        result = self.searcher.safe_lmdb_operation(lambda txn: "success", fallback_func=failing_fallback)
        self.assertIsNone(result)

    def test_index_bookmarks_empty(self):
        """Test index_bookmarks with empty generator."""
        # This covers the 'total_records = 0' branch (line 541 in original, now shifted)
        empty_gen = []

        # We need to mock get_or_create_index to avoid filesystem issues if any
        with patch('fuzzy_bookmark_search.get_or_create_index') as mock_get_index:
             with patch('fuzzy_bookmark_search.create_schema'):
                mock_ix = MagicMock()
                mock_get_index.return_value = mock_ix
                mock_writer = MagicMock()
                mock_ix.writer.return_value = mock_writer

                # Call with empty list
                index_bookmarks(iter(empty_gen), index_dir=self.index_dir, update=False)

                # Verify writer.commit() was called (empty index committed)
                mock_writer.commit.assert_called_once()

    @patch('sys.argv', ['fuzzy_bookmark_search.py'])
    @patch('fuzzy_bookmark_search.uvicorn.run')
    @patch('fuzzy_bookmark_search.FuzzyBookmarkSearch')
    @patch('fuzzy_bookmark_search.index.exists_in')
    @patch('fuzzy_bookmark_search.index_bookmarks')
    def test_main_update_existing_index(self, mock_index_bookmarks, mock_exists_in, mock_cls, mock_run):
        """Test main function where index already exists and we are updating."""
        # Simulate index exists
        mock_exists_in.return_value = True

        mock_instance = mock_cls.return_value
        mock_instance.load_bookmarks_data.return_value = iter([])

        # Run main
        main()

        # Check that index_bookmarks was called with update=True
        # args.no_update is False -> update=True
        mock_index_bookmarks.assert_called_with(ANY, ANY, update=True)

    def test_safe_lmdb_operation_exceptions(self):
         """Test specific exception handling in safe_lmdb_operation."""
         self.searcher.use_fallback = False
         self.searcher.lmdb_env = MagicMock()

         # Mock .begin() to raise DiskError
         self.searcher.lmdb_env.begin.side_effect = lmdb.DiskError("Disk full")

         result = self.searcher.safe_lmdb_operation(lambda txn: "ok")
         self.assertIsNone(result)
         self.assertTrue(self.searcher.use_fallback)

if __name__ == '__main__':
    unittest.main()
