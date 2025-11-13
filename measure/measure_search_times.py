import time
import sys
sys.path.append('..')
from fuzzy_bookmark_search import search_bookmarks

# Define sample queries for fuzzy searches
queries = ['python', 'machine learning~1', 'web development']

# Measure query times for each search
for query in queries:
    start_time = time.time()
    results = search_bookmarks(query)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Query '{query}': {elapsed_time:.4f} seconds")