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
import browser_history

# Path to save to JSON file
output_path = os.path.expanduser("./bookmarks.json")

def get_bookmarks():
    """
    Fetch bookmarks from all installed browsers using browser_history module.
    Returns a list of bookmark dictionaries compatible with the existing script format.
    """
    # Fetch bookmarks from all browsers
    outputs = browser_history.get_bookmarks()
    bookmarks_data = outputs.bookmarks

    bookmarks = []
    for dt, url, title, folder in bookmarks_data:
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
