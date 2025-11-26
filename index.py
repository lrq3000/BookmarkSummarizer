# Copyright 2024 wyj
# Copyright 2025 Stephen Karl Larroque <lrq3000>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
import multiprocessing
import inspect
import browser_history.browsers as browsers_module

# Path to save to JSON file
output_path = os.path.expanduser("./bookmarks.json")

def get_bookmarks():
    """
    Fetch bookmarks from all installed browsers using browser_history module.
    Returns a list of bookmark dictionaries compatible with the existing script format.
    """
    # Fetch bookmarks from all browsers -- normal method that should work but currently fails on Firefox because of issue https://github.com/browser-history/browser-history/issues/286
    #outputs = browser_history.get_bookmarks()
    #bookmarks_data = outputs.bookmarks

    # Fetch bookmarks from all browsers manually (bypasses sorting in browser-history and hence the Firefox bug)
    # Dynamically retrieve list of all supported browser classes from browser_history module
    browser_classes = [
        getattr(browsers_module, name)
        for name in dir(browsers_module)
        if inspect.isclass(getattr(browsers_module, name)) and
           issubclass(getattr(browsers_module, name), browsers_module.Browser) and
           getattr(browsers_module, name) not in (browsers_module.Browser, browsers_module.ChromiumBasedBrowser)
    ]
    bookmarks_data = []
    for browser_class in browser_classes:
        try:
            b = browser_class()
            b.sort_bookmarks_descending = False  # Disable internal sorting to avoid None comparison errors
            outputs = b.fetch_bookmarks(sort=False)  # Disable sorting to prevent TypeError with None values
            bookmarks_data.extend(outputs.bookmarks)
        except Exception as e:
            print(f"Failed to fetch from {browser_class.__name__}: {e}")
    # Sort the combined bookmarks with custom key to handle None values
    bookmarks_data.sort(key=lambda x: (x[3] or "", x[2] or ""), reverse=True)

    bookmarks = []
    for dt, url, title, folder in bookmarks_data:
        # Handle None values in title and folder to prevent sorting errors
        title = title or ""
        folder = folder or ""
        # Map the tuple (datetime, url, title, folder) to the expected dictionary format
        bookmark_info = {
            "date_added": dt.timestamp() if dt else "N/A",  # Convert datetime to timestamp for compatibility
            "date_last_used": "N/A",  # Not available from browser_history
            "guid": "N/A",  # Not available from browser_history
            "id": "N/A",  # Not available from browser_history
            "name": title,  # Title of the bookmark
            "type": "url",  # All entries are URLs
            "url": url,  # URL of the bookmark
            "folder": folder,  # Folder information for filtering
        }
        bookmarks.append(bookmark_info)

    return bookmarks

def main():
    # Parse bookmarks from all browsers
    bookmarks = get_bookmarks()

    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as output_file:
        # Remove data with empty URLs, non-URL types, and 'Extensions' folder
        bookmarks = [bookmark for bookmark in bookmarks if bookmark["url"] and bookmark["type"] == "url" and bookmark["folder"] != "Extensions"]
        json.dump(bookmarks, output_file, ensure_ascii=False, indent=4)

    print(f"Extracted {len(bookmarks)} bookmarks in total, saved to {output_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
