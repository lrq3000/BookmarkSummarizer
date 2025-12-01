
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import importlib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import parser
# Since custom_parsers is a module, we can import it.
from custom_parsers import zhihu

class TestZhihuParser(unittest.TestCase):

    def test_non_zhihu_url(self):
        bookmark = {'url': 'https://example.com', 'name': 'Test'}
        result = zhihu.main(bookmark)
        self.assertEqual(result, bookmark)
        self.assertNotIn('content', result)

    @patch('custom_parsers.zhihu.webdriver.Chrome')
    @patch('custom_parsers.zhihu.ChromeDriverManager')
    @patch('custom_parsers.zhihu.Service')
    def test_zhihu_url_success(self, mock_service, mock_manager, mock_driver_cls):
        bookmark = {'url': 'https://www.zhihu.com/question/123', 'name': 'Zhihu Question'}

        mock_driver = MagicMock()
        mock_driver_cls.return_value = mock_driver
        mock_driver.page_source = '<html><body><div class="Post-RichText">Content</div></body></html>'

        # Mock finding close button
        mock_driver.find_element.return_value = MagicMock()

        result = zhihu.main(bookmark)
        self.assertIn('content', result)
        self.assertEqual(result['content'], 'Content')
        self.assertTrue(mock_driver.quit.called)

    @patch('custom_parsers.zhihu.webdriver.Chrome')
    @patch('custom_parsers.zhihu.ChromeDriverManager')
    @patch('custom_parsers.zhihu.Service')
    def test_zhihu_url_exception(self, mock_service, mock_manager, mock_driver_cls):
        bookmark = {'url': 'https://www.zhihu.com/question/123', 'name': 'Zhihu Question'}

        mock_driver = MagicMock()
        mock_driver_cls.return_value = mock_driver
        mock_driver.get.side_effect = Exception("Failed to load")

        result = zhihu.main(bookmark)
        self.assertEqual(result, bookmark)
        self.assertNotIn('content', result)
        self.assertTrue(mock_driver.quit.called)

if __name__ == '__main__':
    unittest.main()
