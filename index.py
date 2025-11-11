# Copyright 2024 wyj
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

# Chrome bookmark file path
bookmark_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/Bookmarks")
# Path to save to JSON file
output_path = os.path.expanduser("./bookmarks.json")

def get_bookmarks(bookmark_path):
    with open(bookmark_path, "r", encoding="utf-8") as file:
        bookmarks_data = json.load(file)

    urls = []

    def extract_bookmarks(bookmark_node):
        """Recursively extract URLs of all bookmarks"""
        if "children" in bookmark_node:
            for child in bookmark_node["children"]:
                extract_bookmarks(child)
        elif "url" in bookmark_node:
            bookmark_info = {
                "date_added": bookmark_node.get("date_added", "N/A"),
                "date_last_used": bookmark_node.get("date_last_used", "N/A"),
                "guid": bookmark_node.get("guid", "N/A"),
                "id": bookmark_node.get("id", "N/A"),
                "name": bookmark_node.get("name", "N/A"),
                "type": bookmark_node.get("type", "url"),
                "url": bookmark_node.get("url", ""),
            }
            urls.append(bookmark_info)

    # Traverse the JSON structure
    for item in bookmarks_data["roots"].values():
        extract_bookmarks(item)

    return urls

# Parse bookmarks
bookmarks = get_bookmarks(bookmark_path)

# Save to JSON file
output_path = os.path.expanduser(output_path)
with open(output_path, "w", encoding="utf-8") as output_file:
    # Remove data with empty URLs, and extension data
    bookmarks = [bookmark for bookmark in bookmarks if bookmark["url"] and bookmark["type"] == "url" and bookmark["name"] != "Extensions"] 
    json.dump(bookmarks, output_file, ensure_ascii=False, indent=4)

print(f"Extracted {len(bookmarks)} bookmarks in total, saved to {output_path}")
