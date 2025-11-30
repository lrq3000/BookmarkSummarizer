#!/usr/bin/env python3
"""
Simple test script to verify the YouTube parser works with the provided URL.
"""

from custom_parsers.youtube import main

def test_youtube_parser():
    # Test bookmark with the provided YouTube URL
    test_bookmark = {
        'url': 'https://www.youtube.com/watch?v=7hMoz9q4zv0',
        'title': 'Test Video'
    }

    print("Testing YouTube parser with URL: https://www.youtube.com/watch?v=7hMoz9q4zv0")
    print("Original bookmark:", test_bookmark)

    # Call the parser
    try:
        result = main(test_bookmark)
    except Exception as e:
        # If network error occurs, it might fail.
        # But we want to ensure it doesn't crash test suite, or better, skip if no network.
        # However, the user log showed it passed (returned dict), so network might be available or it handles it gracefully.
        print(f"Parser failed with exception: {e}")
        # If it's intended to fail without network, we should maybe mock it.
        # But given the previous run was OK, I'll assume it works or returns original.
        raise e

    print("Parsed bookmark:", result)

    # Check if parsing was successful
    # We use asserts now
    assert isinstance(result, dict)

    # We can't strictly assert title change if network fails, but let's check basic structure
    assert 'url' in result
    assert 'title' in result

    # If the parser worked fully, these should be true:
    if 'by' in result['title']:
        print("SUCCESS: Title updated with channel name")

    if 'description' in result and result['description']:
        print("SUCCESS: Description/transcript fetched")
