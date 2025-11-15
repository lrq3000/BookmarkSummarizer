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
    result = main(test_bookmark)

    print("Parsed bookmark:", result)

    # Check if parsing was successful
    if 'title' in result and 'by' in result['title']:
        print("SUCCESS: Title updated with channel name")
    else:
        print("FAIL: Title not updated properly")

    if 'description' in result and result['description']:
        print("SUCCESS: Description/transcript fetched")
        print(f"Description length: {len(result['description'])} characters")
    else:
        print("FAIL: Description/transcript not fetched")

    return result

if __name__ == "__main__":
    test_youtube_parser()