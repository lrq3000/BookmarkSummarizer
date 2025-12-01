import os
import shutil
import time
import psutil
import sys
sys.path.append('..')
from fuzzy_bookmark_search import load_bookmarks_data, create_schema, get_or_create_index

def index_bookmarks_with_memory_tracking(bookmarks_generator, index_dir='./whoosh_index'):
    """
    Index bookmark data into Whoosh index with memory usage tracking.

    This function processes bookmarks in batches and tracks peak memory usage during indexing.
    It creates a composite text field by concatenating title, content, and summary for cross-field fuzzy searching.
    The index is committed after all documents are added.

    Args:
        bookmarks_generator: Generator yielding preprocessed bookmark dictionaries.
        index_dir (str): Directory for the index.

    Returns:
        float: Peak memory usage in MB during indexing.
    """
    schema = create_schema()
    ix = get_or_create_index(index_dir, schema)

    writer = ix.writer()

    batch_size = 1000  # Process in batches to manage memory
    batch = []
    peak_mem = 0

    process = psutil.Process()

    for bookmark in bookmarks_generator:
        # Combine text fields for composite search
        composite_text = f"{bookmark['title']} {bookmark['content']} {bookmark['summary']}"

        # Prepare document for indexing
        doc = {
            'title': bookmark['title'],
            'url': bookmark['url'],
            'content': bookmark['content'],
            'summary': bookmark['summary'],
            'composite_text': composite_text,
            'key': bookmark['key']
        }

        batch.append(doc)

        # Write batch when it reaches the limit and track memory
        if len(batch) >= batch_size:
            for d in batch:
                writer.add_document(**d)
            batch = []
            current_mem = process.memory_info().rss / 1024 / 1024
            if current_mem > peak_mem:
                peak_mem = current_mem

    # Write remaining documents
    for d in batch:
        writer.add_document(**d)

    writer.commit()

    # Final memory check
    final_mem = process.memory_info().rss / 1024 / 1024
    if final_mem > peak_mem:
        peak_mem = final_mem

    return peak_mem

# Clean existing index
if os.path.exists('../whoosh_index'):
    shutil.rmtree('../whoosh_index')
    print("Removed existing whoosh_index directory.")

process = psutil.Process()

# Initial memory measurement
initial_mem = process.memory_info().rss / 1024 / 1024
print(f"Initial memory: {initial_mem:.2f} MB")

# Load bookmarks data
print("Loading bookmarks data...")
bookmarks_gen = load_bookmarks_data()
bookmarks_list = list(bookmarks_gen)  # Convert to list to count and reuse
num_bookmarks = len(bookmarks_list)
print(f"Loaded {num_bookmarks} bookmarks")

# Memory after loading
after_load_mem = process.memory_info().rss / 1024 / 1024
print(f"Memory after loading: {after_load_mem:.2f} MB")

# Index bookmarks with memory tracking
print("Starting indexing...")
start_time = time.time()
peak_mem_during_indexing = index_bookmarks_with_memory_tracking(bookmarks_list)
end_time = time.time()
indexing_time = end_time - start_time
print(f"Indexing time: {indexing_time:.2f} seconds")
print(f"Peak memory during indexing: {peak_mem_during_indexing:.2f} MB")

# Measure memory used by the index after loading
# Open the index and create a searcher to load it into memory context
before_open_mem = process.memory_info().rss / 1024 / 1024
ix = get_or_create_index()
with ix.searcher() as searcher:
    after_open_mem = process.memory_info().rss / 1024 / 1024
    index_mem_usage = after_open_mem - before_open_mem
    print(f"Memory after opening index: {after_open_mem:.2f} MB")
    print(f"Estimated memory used by the index after loading: {index_mem_usage:.2f} MB")