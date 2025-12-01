
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import shutil
import tempfile
import custom_parsers.youtube as parser

class MockTranscriptItem:
    def __init__(self, text):
        self.text = text

class TestYoutubeParser(unittest.TestCase):

    def test_non_youtube_url(self):
        bookmark = {'url': 'https://example.com', 'name': 'Example'}
        result = parser.main(bookmark)
        self.assertEqual(result, bookmark)

    @patch('custom_parsers.youtube.requests.Session')
    @patch('custom_parsers.youtube.YouTubeTranscriptApi')
    def test_youtube_success(self, mock_api_cls, mock_session_cls):
        # Use 11-char video ID to match regex
        video_id = 'VIDEO_ID_11'
        bookmark = {'url': f'https://www.youtube.com/watch?v={video_id}', 'name': 'Video'}

        # Mock Session
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        # Mock oEmbed response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'title': 'Video Title',
            'author_name': 'Channel Name',
            'description': 'Video Description'
        }
        mock_session.get.return_value = mock_response

        # Mock Transcript
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        # The fetch method returns list of objects with .text attribute for TextFormatter
        # In reality, youtube_transcript_api returns dicts, but TextFormatter seems to expect objects in this environment?
        # Or the version installed has different TextFormatter.
        # Based on error "AttributeError: 'dict' object has no attribute 'text'", we must provide objects.
        mock_api.fetch.return_value = [MockTranscriptItem('Hello')]

        result = parser.main(bookmark)
        # The parser updates 'description' field in bookmark
        self.assertIn('description', result)
        self.assertIn('Video Description', result['description'])
        self.assertIn('Hello', result['description'])

    @patch('custom_parsers.youtube.requests.Session')
    @patch('custom_parsers.youtube.YouTubeTranscriptApi')
    def test_youtube_no_transcript(self, mock_api_cls, mock_session_cls):
        video_id = 'VIDEO_ID_11'
        bookmark = {'url': f'https://youtu.be/{video_id}', 'name': 'Video'}

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {'title': 'T', 'description': 'Desc'}
        mock_session.get.return_value = mock_response

        # Mock Transcript fetch failure
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.fetch.side_effect = Exception("No transcript")

        result = parser.main(bookmark)
        self.assertIn('description', result)
        self.assertEqual(result['description'], 'Desc')

    @patch('custom_parsers.youtube.requests.Session')
    def test_youtube_metadata_fail(self, mock_session_cls):
        video_id = 'VIDEO_ID_11'
        bookmark = {'url': f'https://www.youtube.com/watch?v={video_id}'}

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.get.side_effect = Exception("Network error")

        result = parser.main(bookmark)
        self.assertEqual(result, bookmark) # Should return original

if __name__ == '__main__':
    unittest.main()
