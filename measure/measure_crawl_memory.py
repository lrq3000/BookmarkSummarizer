import os
import time
import psutil
import json
import sys
import subprocess
import signal
import threading

def measure_memory_usage_during_crawl(bookmarks_file='test_bookmarks.json', limit=5, workers=2):
    """
    Measure memory usage during crawl.py execution with a test dataset.

    This function runs crawl.py with memory monitoring and collects metrics
    on memory usage, execution time, and data persistence verification.

    Args:
        bookmarks_file (str): Path to test bookmarks JSON file
        limit (int): Number of bookmarks to process
        workers (int): Number of worker threads

    Returns:
        dict: Dictionary containing memory metrics and performance data
    """
    process = psutil.Process()
    initial_mem = process.memory_info().rss / 1024 / 1024

    print(f"Initial memory: {initial_mem:.2f} MB")

    # Clean up any existing ZODB files
    zodb_path = "./bookmark_index.fs"
    if os.path.exists(zodb_path):
        os.remove(zodb_path)
        print("Cleaned up existing ZODB file")

    # Start memory monitoring in a separate thread
    memory_samples = []
    stop_monitoring = threading.Event()

    def monitor_memory():
        while not stop_monitoring.is_set():
            current_mem = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_mem)
            time.sleep(0.1)  # Sample every 100ms

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    # Run crawl.py with test parameters
    start_time = time.time()

    cmd = [
        sys.executable, 'crawl.py',
        '--limit', str(limit),
        '--workers', str(workers),
        '--no-summary',  # Skip summary generation for faster testing
        '--rebuild'  # Start fresh
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout

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

        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Peak memory: {peak_mem:.2f} MB")
        print(f"Average memory: {avg_mem:.2f} MB")
        print(f"Memory increase: {peak_mem - initial_mem:.2f} MB")

        # Check if ZODB file was created and has data
        zodb_exists = os.path.exists(zodb_path)
        zodb_size = os.path.getsize(zodb_path) if zodb_exists else 0

        print(f"ZODB file exists: {zodb_exists}")
        print(f"ZODB file size: {zodb_size} bytes")

        # Check for bookmarks.json output
        bookmarks_output_exists = os.path.exists('./bookmarks.json')
        bookmarks_with_content_exists = os.path.exists('./bookmarks_with_content.json')

        print(f"bookmarks.json exists: {bookmarks_output_exists}")
        print(f"bookmarks_with_content.json exists: {bookmarks_with_content_exists}")

        # Try to verify ZODB content (basic check)
        zodb_content_count = 0
        if zodb_exists:
            try:
                import ZODB
                from ZODB.FileStorage import FileStorage
                from ZODB.DB import DB

                storage = FileStorage(zodb_path)
                db = DB(storage)
                connection = db.open()
                root = connection.root()

                if 'bookmarks' in root:
                    zodb_content_count = len(root['bookmarks'])

                connection.close()
                db.close()

                print(f"ZODB contains {zodb_content_count} bookmarks")
            except Exception as e:
                print(f"Error reading ZODB: {e}")
                zodb_content_count = -1

        # Collect stdout/stderr for analysis
        stdout_lines = result.stdout.split('\n') if result.stdout else []
        stderr_lines = result.stderr.split('\n') if result.stderr else []

        # Check for errors
        has_errors = result.returncode != 0
        error_lines = [line for line in stderr_lines if 'error' in line.lower() or 'exception' in line.lower()]

        metrics = {
            'execution_time': execution_time,
            'initial_memory': initial_mem,
            'peak_memory': peak_mem,
            'average_memory': avg_mem,
            'memory_increase': peak_mem - initial_mem,
            'min_memory': min_mem,
            'memory_samples_count': len(memory_samples),
            'zodb_file_exists': zodb_exists,
            'zodb_file_size': zodb_size,
            'zodb_content_count': zodb_content_count,
            'bookmarks_output_exists': bookmarks_output_exists,
            'bookmarks_with_content_exists': bookmarks_with_content_exists,
            'return_code': result.returncode,
            'has_errors': has_errors,
            'error_lines': error_lines[:10],  # First 10 error lines
            'stdout_lines_count': len(stdout_lines),
            'stderr_lines_count': len(stderr_lines)
        }

        return metrics, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        stop_monitoring.set()
        monitor_thread.join(timeout=1)
        print("Crawl process timed out")
        return {'error': 'timeout'}, "", ""

    except Exception as e:
        stop_monitoring.set()
        monitor_thread.join(timeout=1)
        print(f"Error during crawl execution: {e}")
        return {'error': str(e)}, "", ""

if __name__ == "__main__":
    print("Starting memory measurement for crawl.py with ZODB indexing...")

    metrics, stdout, stderr = measure_memory_usage_during_crawl()

    print("\n" + "="*50)
    print("MEMORY MEASUREMENT RESULTS")
    print("="*50)

    if 'error' in metrics:
        print(f"ERROR: {metrics['error']}")
    else:
        print(f"Execution Time: {metrics['execution_time']:.2f} seconds")
        print(f"Initial Memory: {metrics['initial_memory']:.2f} MB")
        print(f"Peak Memory: {metrics['peak_memory']:.2f} MB")
        print(f"Average Memory: {metrics['average_memory']:.2f} MB")
        print(f"Memory Increase: {metrics['memory_increase']:.2f} MB")
        print(f"Min Memory: {metrics['min_memory']:.2f} MB")
        print(f"Memory Samples: {metrics['memory_samples_count']}")
        print(f"ZODB File Exists: {metrics['zodb_file_exists']}")
        print(f"ZODB File Size: {metrics['zodb_file_size']} bytes")
        print(f"ZODB Content Count: {metrics['zodb_content_count']}")
        print(f"Return Code: {metrics['return_code']}")
        print(f"Has Errors: {metrics['has_errors']}")

        if metrics['error_lines']:
            print(f"Error Lines ({len(metrics['error_lines'])}):")
            for line in metrics['error_lines']:
                print(f"  {line}")

    print("\n" + "="*50)
    print("STDOUT SUMMARY")
    print("="*50)
    # Print last 20 lines of stdout
    stdout_lines = stdout.split('\n')[-20:] if stdout else []
    for line in stdout_lines:
        if line.strip():
            print(line)

    if stderr:
        print("\n" + "="*50)
        print("STDERR")
        print("="*50)
        print(stderr)