import urllib.parse

def main(bookmark: dict) -> dict:
    """
    Parses suspended tabs from Chrome extensions by decoding the 'url' parameter
    from the chrome-extension:// URL query string.

    Args:
        bookmark (dict): A bookmark dictionary containing at least a 'url' key.

    Returns:
        dict: The modified bookmark with the decoded URL if successful,
              otherwise the original bookmark unchanged.
    """
    try:
        url = bookmark.get('url', '')
        if not url.startswith('chrome-extension://'):
            return bookmark

        # Parse the URL to extract query parameters
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Check if 'url' parameter exists
        if 'url' not in query_params or not query_params['url']:
            return bookmark

        # Get the encoded URL (assuming single value)
        encoded_url = query_params['url'][0]
        current_url = encoded_url

        # Iterative decoding loop (max 5 iterations)
        for _ in range(5):
            if '://' in current_url:
                break
            current_url = urllib.parse.unquote(current_url)

        # Final check: if decoded URL lacks protocol, revert
        if '://' not in current_url:
            print(f"Error: Unable to decode URL properly for bookmark: {bookmark}")
            return bookmark

        # Update the bookmark with the decoded URL
        bookmark['url'] = current_url
        return bookmark

    except Exception as e:
        # Handle any unexpected errors (e.g., malformed URLs)
        print(f"Error processing bookmark: {e}")
        return bookmark