import os
import time
import psutil
import json
import sys
import threading
import hashlib

# Import ZODB components
import ZODB
from ZODB.FileStorage import FileStorage
from ZODB.DB import DB
import BTrees.OOBTree as OOBTree
import BTrees.IOBTree as IOBTree
import persistent
import transaction

# Import crawl functions and global variables
from crawl import init_zodb, cleanup_zodb, fetch_webpage_content, url_hashes_tree, content_hashes_tree, bookmarks_tree

def create_test_bookmarks():
    """Create a small set of test bookmarks for memory testing"""
    return [
        {
            "date_added": "2024-01-01T00:00:00",
            "date_last_used": "N/A",
            "guid": "test1",
            "id": "1",
            "name": "Python Official Documentation",
            "type": "url",
            "url": "https://docs.python.org/3/"
        },
        {
            "date_added": "2024-01-02T00:00:00",
            "date_last_used": "N/A",
            "guid": "test2",
            "id": "2",
            "name": "GitHub",
            "type": "url",
            "url": "https://github.com"
        },
        {
            "date_added": "2024-01-03T00:00:00",
            "date_last_used": "N/A",
            "guid": "test3",
            "id": "3",
            "name": "Stack Overflow",
            "type": "url",
            "url": "https://stackoverflow.com"
        }
    ]

def test_zodb_memory_usage():
    """
    Test ZODB memory usage with direct function calls to measure memory improvements.
    """
    process = psutil.Process()
    initial_mem = process.memory_info().rss / 1024 / 1024

    print(f"Initial memory: {initial_mem:.2f} MB")

    # Clean up any existing ZODB files
    zodb_path = "./bookmark_index.fs"
    if os.path.exists(zodb_path):
        os.remove(zodb_path)
        print("Cleaned up existing ZODB file")

    # Initialize ZODB
    init_zodb()

    # Start memory monitoring
    memory_samples = []
    stop_monitoring = threading.Event()

    def monitor_memory():
        while not stop_monitoring.is_set():
            current_mem = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_mem)
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    # Get test bookmarks
    test_bookmarks = create_test_bookmarks()
    print(f"Processing {len(test_bookmarks)} test bookmarks")

    # Import global variables after init_zodb
    from crawl import url_hashes_tree, content_hashes_tree, bookmarks_tree

    successful_crawls = 0
    failed_crawls = 0

    # Process each bookmark
    for idx, bookmark in enumerate(test_bookmarks):
        print(f"Processing bookmark {idx+1}/{len(test_bookmarks)}: {bookmark['name']}")

        # Check URL deduplication
        url_hash = hashlib.sha256(bookmark['url'].encode('utf-8')).hexdigest()
        if url_hash in url_hashes_tree:
            print(f"  Skipping duplicate URL: {bookmark['url']}")
            continue

        url_hashes_tree[url_hash] = True
        transaction.commit()

        # Fetch content
        result, failed_info = fetch_webpage_content(bookmark, idx+1, len(test_bookmarks))

        if result:
            # Check content deduplication
            content_hash = hashlib.sha256(result['content'].encode('utf-8')).hexdigest()
            if content_hash in content_hashes_tree:
                print(f"  Skipping duplicate content: {bookmark['url']}")
            else:
                content_hashes_tree[content_hash] = True

                # Store in ZODB
                next_key = max(bookmarks_tree.keys()) + 1 if bookmarks_tree else 1
                bookmarks_tree[next_key] = result
                transaction.commit()

                successful_crawls += 1
                print(f"  Successfully stored bookmark: {result['title']} ({len(result['content'])} chars)")
        else:
            failed_crawls += 1
            print(f"  Failed to crawl: {bookmark['url']} - {failed_info['reason'] if failed_info else 'Unknown error'}")

    end_time = time.time()
    execution_time = end_time - start_time

    # Stop memory monitoring
    stop_monitoring.set()
    monitor_thread.join(timeout=1)

    # Calculate memory statistics
    if memory_samples:
        peak_mem = max(memory_samples)
        avg_mem = sum(memory_samples) / len(memory_samples)
        min_mem = min(memory_samples)
    else:
        peak_mem = avg_mem = min_mem = initial_mem

    # Check ZODB file size
    zodb_size = os.path.getsize(zodb_path) if os.path.exists(zodb_path) else 0

    # Count items in ZODB
    bookmarks_count = len(bookmarks_tree)
    url_hashes_count = len(url_hashes_tree)
    content_hashes_count = len(content_hashes_tree)

    # Cleanup
    cleanup_zodb()

    metrics = {
        'execution_time': execution_time,
        'initial_memory': initial_mem,
        'peak_memory': peak_mem,
        'average_memory': avg_mem,
        'memory_increase': peak_mem - initial_mem,
        'min_memory': min_mem,
        'memory_samples_count': len(memory_samples),
        'successful_crawls': successful_crawls,
        'failed_crawls': failed_crawls,
        'zodb_file_size': zodb_size,
        'bookmarks_in_zodb': bookmarks_count,
        'url_hashes_in_zodb': url_hashes_count,
        'content_hashes_in_zodb': content_hashes_count
    }

    return metrics

if __name__ == "__main__":
    print("Testing ZODB memory usage with direct function calls...")

    metrics = test_zodb_memory_usage()

    print("\n" + "="*60)
    print("ZODB MEMORY USAGE TEST RESULTS")
    print("="*60)

    print(f"Execution Time: {metrics['execution_time']:.2f} seconds")
    print(f"Initial Memory: {metrics['initial_memory']:.2f} MB")
    print(f"Peak Memory: {metrics['peak_memory']:.2f} MB")
    print(f"Average Memory: {metrics['average_memory']:.2f} MB")
    print(f"Memory Increase: {metrics['memory_increase']:.2f} MB")
    print(f"Min Memory: {metrics['min_memory']:.2f} MB")
    print(f"Memory Samples: {metrics['memory_samples_count']}")
    print()
    print(f"Successful Crawls: {metrics['successful_crawls']}")
    print(f"Failed Crawls: {metrics['failed_crawls']}")
    print()
    print(f"ZODB File Size: {metrics['zodb_file_size']} bytes")
    print(f"Bookmarks in ZODB: {metrics['bookmarks_in_zodb']}")
    print(f"URL Hashes in ZODB: {metrics['url_hashes_in_zodb']}")
    print(f"Content Hashes in ZODB: {metrics['content_hashes_in_zodb']}")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Analyze memory efficiency
    memory_per_bookmark = metrics['memory_increase'] / max(metrics['successful_crawls'], 1)
    print(f"Memory increase per bookmark: {memory_per_bookmark:.2f} MB")

    if metrics['successful_crawls'] > 0:
        avg_processing_time = metrics['execution_time'] / metrics['successful_crawls']
        print(f"Average processing time per bookmark: {avg_processing_time:.2f} seconds")

    # Check data persistence
    persistence_ok = metrics['bookmarks_in_zodb'] == metrics['successful_crawls']
    print(f"Data persistence check: {'PASS' if persistence_ok else 'FAIL'}")
    print(f"  Expected bookmarks: {metrics['successful_crawls']}, Found: {metrics['bookmarks_in_zodb']}")

    print("\nTest completed successfully!")