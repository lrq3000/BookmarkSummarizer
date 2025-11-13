import os
import shutil
import time
import sys
sys.path.append('..')
from fuzzy_bookmark_search import load_bookmarks_data, index_bookmarks

# Remove existing whoosh_index directory if it exists
if os.path.exists('../whoosh_index'):
    shutil.rmtree('../whoosh_index')
    print("Removed existing whoosh_index directory.")

# Load bookmarks data
print("Loading bookmarks data...")
bookmarks_gen = load_bookmarks_data()

# Measure indexing time
print("Starting indexing...")
start_time = time.time()
index_bookmarks(bookmarks_gen)
end_time = time.time()

indexing_time = end_time - start_time
print(f"Indexing time: {indexing_time:.2f} seconds")