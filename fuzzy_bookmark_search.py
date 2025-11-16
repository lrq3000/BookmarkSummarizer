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

import os
import zipfile
import argparse
import time
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, FuzzyTermPlugin
from whoosh import scoring
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tqdm import tqdm

# LMDB imports for persistent storage
import lmdb
import json
import pickle

# LMDB database and persistent structures for on-disk indexing
# LMDB provides memory-mapped database with efficient key-value storage
# This replaces in-memory sets with disk-based storage for scalability
lmdb_path = os.path.expanduser("./bookmark_index.lmdb")
lmdb_env = None
bookmarks_db = None  # LMDB database for storing bookmarks

# In-memory fallback structures for graceful degradation
fallback_bookmarks = []
use_fallback = False

# Check disk space before LMDB operations
def check_disk_space(min_space_mb=100):
    """
    Check if there's sufficient disk space for LMDB operations.

    Parameters:
        min_space_mb (int): Minimum required disk space in MB

    Returns:
        bool: True if sufficient space, False otherwise
    """
    try:
        # Get the directory containing the LMDB database
        db_dir = os.path.dirname(os.path.abspath(lmdb_path))
        if not os.path.exists(db_dir):
            # If directory doesn't exist, try to create it
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                print(f"Cannot create database directory {db_dir}: {e}")
                return False
        import shutil
        stat = shutil.disk_usage(db_dir)
        free_space_mb = stat.free / (1024 * 1024)
        if free_space_mb < min_space_mb:
            print(f"Insufficient disk space: {free_space_mb:.2f} MB free, {min_space_mb} MB required")
            return False
        return True
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return False

# Initialize LMDB database and persistent structures for on-disk indexing
# LMDB uses memory-mapped database for efficient key-value storage
def init_lmdb():
    """
    Initialize LMDB database for deduplication and storage.

    This function sets up the LMDB environment and opens the bookmarks database.
    All operations are transactional for data integrity.
    Includes comprehensive error handling with fallback to in-memory structures.
    """
    global lmdb_env, bookmarks_db, use_fallback

    # Check disk space first
    if not check_disk_space():
        print("Insufficient disk space for LMDB initialization. Falling back to in-memory structures.")
        use_fallback = True
        return

    try:
        # Create LMDB environment
        lmdb_env = lmdb.open(lmdb_path, map_size=1024*1024*1024, max_dbs=1)  # 1GB map size

        # Open bookmarks database
        bookmarks_db = lmdb_env.open_db(b'bookmarks')

        print(f"Initialized LMDB database at {lmdb_path}")

    except Exception as e:
        print(f"Error initializing LMDB: {e}")
        use_fallback = True

        # Cleanup on failure
        try:
            if lmdb_env:
                lmdb_env.close()
        except Exception as cleanup_e:
            print(f"Error during LMDB cleanup: {cleanup_e}")

        print("Falling back to in-memory structures for data integrity")

# Safe LMDB operations with error handling
def safe_lmdb_operation(operation_func, fallback_func=None, operation_name="LMDB operation"):
    """
    Perform an LMDB operation with error handling and fallback support.

    Parameters:
        operation_func (callable): Function performing the LMDB operation
        fallback_func (callable, optional): Fallback function if LMDB fails
        operation_name (str): Name of the operation for logging

    Returns:
        Any: Result of the operation or fallback
    """
    global use_fallback

    if use_fallback:
        if fallback_func:
            try:
                return fallback_func()
            except Exception as e:
                print(f"Fallback {operation_name} failed: {e}")
                return None
        return None

    try:
        return operation_func()
    except Exception as e:
        print(f"{operation_name} failed: {e}")
        use_fallback = True
        if fallback_func:
            try:
                print(f"Attempting fallback for {operation_name}")
                return fallback_func()
            except Exception as fallback_e:
                print(f"Fallback {operation_name} failed: {fallback_e}")
        return None

# Cleanup LMDB resources
def cleanup_lmdb():
    """
    Properly close LMDB environment to ensure data integrity.
    """
    global lmdb_env
    try:
        if lmdb_env:
            lmdb_env.close()
        print("LMDB cleanup completed")
    except Exception as e:
        print(f"Error during LMDB cleanup: {e}")


def create_schema():
    """
    Define the Whoosh schema for bookmark indexing.

    This schema includes separate fields for title, url, content, summary, and a composite_text field
    that combines all text fields for cross-field fuzzy searching. The key field is used for unique
    identification and deduplication. All text fields are stored for retrieval and use TEXT type
    for full-text indexing with standard analysis.

    Returns:
        Schema: Whoosh schema object.
    """
    return Schema(
        title=TEXT(stored=True),
        url=TEXT(stored=True),
        content=TEXT(stored=True),
        summary=TEXT(stored=True),
        composite_text=TEXT(stored=True),  # Combined field for multi-field search
        key=ID(stored=True, unique=True)
    )


def get_or_create_index(index_dir='./whoosh_index', schema=None):
    """
    Get or create a Whoosh index in the specified directory.

    This function checks if an index already exists in the directory. If it does, it opens it;
    otherwise, it creates a new index using the provided schema. This allows for incremental
    indexing without rebuilding from scratch.

    Args:
        index_dir (str): Directory to store the index.
        schema (Schema): Whoosh schema to use for the index. If None, uses create_schema().

    Returns:
        Index: Whoosh index object.
    """
    if schema is None:
        schema = create_schema()

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    if index.exists_in(index_dir):
        return index.open_dir(index_dir)
    else:
        return index.create_in(index_dir, schema)


def load_bookmarks_data(lmdb_path='bookmark_index.lmdb'):
    """
    Load bookmark data from an LMDB database.

    This function loads bookmark data from the LMDB bookmarks database, handling cases where
    the database doesn't exist or is corrupted. It yields each bookmark as a dict,
    with preprocessing to generate a unique key and normalize text fields.

    Args:
        lmdb_path (str): Path to the LMDB database directory.

    Yields:
        dict: Preprocessed bookmark dictionary with fields like title, url, content, summary, key.
    """
    global bookmarks_db, use_fallback

    # Try to load from LMDB first
    bookmarks_list = safe_lmdb_operation(
        lambda: load_bookmarks_from_lmdb(),
        lambda: fallback_bookmarks.copy(),
        "loading bookmarks from LMDB"
    )

    if bookmarks_list is None:
        bookmarks_list = []

    # Perform preliminary pass to count total records for progress tracking
    total_records = len(bookmarks_list)

    if total_records == 0:
        print("Warning: No bookmarks found in LMDB database. Make sure to run crawl.py first to populate the database.")
        return

    for bookmark in bookmarks_list:
        # Preprocess: generate key, normalize text
        guid = bookmark.get('guid', '')
        id_val = bookmark.get('id', '')
        url = bookmark.get('url', '').strip()
        # Treat 'N/A' as missing value for key generation
        key = (guid if guid != 'N/A' else '') or (id_val if id_val != 'N/A' else '') or url
        title = (bookmark.get('title') or bookmark.get('name', '')).strip()
        content = (bookmark.get('content', '')).strip()
        summary = (bookmark.get('summary', '')).strip()

        # Limit content length to prevent index bloat
        if len(content) > 10000:
            content = content[:10000] + '...'

        yield {
            'key': key,
            'title': title,
            'url': url,
            'content': content,
            'summary': summary,
            'total_records': total_records  # Include total count for progress tracking
        }


def load_bookmarks_from_lmdb():
    """
    Helper function to load bookmarks from LMDB within a transaction.
    """
    bookmarks = []
    with lmdb_env.begin(bookmarks_db) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            bookmark = pickle.loads(value)
            bookmarks.append(bookmark)
    return bookmarks

def index_bookmarks(bookmarks_generator, index_dir='./whoosh_index', update=False):
    """
    Index bookmark data into Whoosh index with batch processing and progress tracking.

    This function processes bookmarks in batches to manage memory usage during indexing,
    especially important for large datasets with millions of entries. It creates a composite
    text field by concatenating title, content, and summary for cross-field fuzzy searching.
    The index is committed after all documents are added, optimizing for disk-based storage.
    If update=True, it updates the existing index without rebuilding from scratch, deduplicating
    based on the 'key' field (URL or GUID).

    Progress bars are implemented using tqdm to provide visual feedback during indexing.
    For initial indexing, the total count is obtained from the generator's total_records field.
    For updates, the progress bar shows the number of new records being processed, with
    the total being the number of bookmarks that pass the deduplication check.

    Args:
        bookmarks_generator: Generator yielding preprocessed bookmark dictionaries.
        index_dir (str): Directory for the index.
        update (bool): If True, update existing index instead of rebuilding.
    """
    schema = create_schema()
    ix = get_or_create_index(index_dir, schema)

    writer = ix.writer()

    batch_size = 2000  # Process in batches to manage memory
    batch = []
    processed_keys = set()
    skipped_count = 0

    # If updating, load existing keys to avoid duplicates and count existing bookmarks
    existing_count = 0
    if update and index.exists_in(index_dir):
        with ix.searcher() as searcher:
            for doc in searcher.documents():
                processed_keys.add(doc['key'])
                existing_count += 1
        print(f"Existing index contains {existing_count} bookmarks.")

    # Initialize progress tracking variables
    total_records = None
    processed_count = 0
    new_records_count = 0

    # First pass to determine total for progress bar (only for initial indexing)
    if not update:
        # Peek at the first item to get total_records
        bookmarks_list = list(bookmarks_generator)
        if bookmarks_list:
            total_records = bookmarks_list[0].get('total_records', len(bookmarks_list))
        else:
            total_records = 0
        bookmarks_generator = iter(bookmarks_list)  # Reset generator

    # Create progress bar
    # For updates, we use a dynamic total since we don't know how many will be new
    # For initial indexing, we use the accurate total from the preliminary count
    if update:
        pbar = tqdm(desc="Indexing bookmarks (update mode)", unit="records")
    else:
        pbar = tqdm(total=total_records, desc="Indexing bookmarks", unit="records")

    for bookmark in bookmarks_generator:
        key = bookmark['key']
        processed_count += 1

        if key in processed_keys:
            skipped_count += 1
            continue

        # Combine text fields for composite search
        composite_text = f"{bookmark['title']} {bookmark['content']} {bookmark['summary']}"

        # Prepare document for indexing
        doc = {
            'title': bookmark['title'],
            'url': bookmark['url'],
            'content': bookmark['content'],
            'summary': bookmark['summary'],
            'composite_text': composite_text,
            'key': key
        }

        batch.append(doc)
        new_records_count += 1
        processed_keys.add(key)

        # Update progress bar
        if update:
            pbar.update(1)  # In update mode, update by 1 each time
        else:
            pbar.n = processed_count  # In initial mode, set exact position
            pbar.refresh()

        # Write batch when it reaches the limit
        if len(batch) >= batch_size:
            for d in batch:
                writer.add_document(**d)
            batch = []

    # Write remaining documents
    for d in batch:
        writer.add_document(**d)

    writer.commit()

    # Close progress bar and show final summary
    pbar.close()
    print(f"Records parsed from the LMDB database: {processed_count}")
    print(f"Records skipped as duplicates: {skipped_count}")
    print(f"Total bookmarks remaining in index: {existing_count + new_records_count}")

def get_total_results(query, searcher):
    """
    Get the total number of results for a query without loading all documents.

    This helper function efficiently counts the total matching results for pagination
    metadata without retrieving all result documents, which would be memory-intensive
    for large result sets.

    Args:
        query: Parsed Whoosh query object.
        searcher: Whoosh searcher instance.

    Returns:
        int: Total number of matching results.
    """
    return searcher.search(query, limit=None).estimated_length()


def format_search_time(seconds):
    """
    Format search execution time into a human-readable string.

    Converts raw seconds into appropriate time units (seconds, milliseconds) with
    proper formatting for display in search results metadata.

    Args:
        seconds (float): Search time in seconds.

    Returns:
        str: Formatted time string (e.g., "0.12 seconds", "45 ms").
    """
    if seconds >= 1.0:
        return ".2f"
    else:
        return ".0f"


def search_bookmarks(query_str, index_dir='./whoosh_index', limit=10, page=1, page_size=20):
    """
    Perform fuzzy search on indexed bookmarks across all fields with pagination support.

    This function enables fuzzy string matching using Whoosh's FuzzyTermPlugin, which supports
    edit distance-based queries (e.g., 'term~2' for 2-character edits). It searches the composite_text
    field, which combines title, content, and summary, allowing cross-field fuzzy matching. Results
    include BM25 scores, highlighted snippets, and metadata for display.

    Pagination is implemented using Whoosh's search_page method for efficient memory usage,
    loading only the current page's results instead of all matching documents. This prevents
    memory overload when dealing with large result sets.

    Args:
        query_str (str): Search query string (supports fuzzy syntax like 'python~1').
        index_dir (str): Directory of the index.
        limit (int): Maximum number of results to return (deprecated, use page_size).
        page (int): Page number for pagination (1-based, default 1).
        page_size (int): Number of results per page (default 20, max 100).

    Returns:
        dict: Dictionary containing:
            - results: List of search results, each a dict with title, url, score, snippet, key.
            - pagination: Dict with page, page_size, total_results, total_pages, has_next, has_prev.
            - search_time: Execution time in seconds.
            - query: Original query string.
    """
    start_time = time.time()

    ix = index.open_dir(index_dir)

    # Create query parser with fuzzy term plugin for fuzzy matching
    parser = QueryParser("composite_text", ix.schema)
    parser.add_plugin(FuzzyTermPlugin())

    # Parse the query
    query = parser.parse(query_str)

    # Validate pagination parameters
    page = max(1, page)  # Ensure page is at least 1
    page_size = min(max(1, page_size), 100)  # Clamp page_size between 1 and 100

    # Calculate pagination offset
    # Offset is zero-based, page is 1-based, so (page-1) * page_size
    offset = (page - 1) * page_size

    # Perform search with BM25 scoring for relevance
    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        # Get total results count for pagination metadata
        total_results = get_total_results(query, searcher)

        # Use search_page for efficient pagination - only loads current page
        page_results = searcher.search_page(query, page, pagelen=page_size)

        # Calculate pagination metadata
        total_pages = (total_results + page_size - 1) // page_size  # Ceiling division
        has_next = page < total_pages
        has_prev = page > 1

        # Prepare results with snippets
        search_results = []
        for hit in page_results:
            # Generate snippet from composite_text with highlights or truncation
            snippet = hit.highlights("composite_text") or hit["composite_text"][:200] + "..."

            search_results.append({
                'title': hit['title'],
                'url': hit['url'],
                'score': hit.score,
                'snippet': snippet,
                'key': hit['key'],
                'summary': hit['summary'],
                'content': hit['content'],
                'full_record': {field: hit[field] for field in hit.fields()}
            })

        # Calculate search execution time
        search_time = time.time() - start_time

        return {
            'results': search_results,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_results': total_results,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev
            },
            'search_time': search_time,
            'query': query_str
        }


# FastAPI application setup for web interface
# This section integrates a web server into the fuzzy bookmark search module,
# allowing users to interact with the search functionality through a browser.
# The app serves an embedded HTML/JS frontend and provides API endpoints for search operations.

# Initialize FastAPI app instance
app = FastAPI(title="Fuzzy Bookmark Search", description="Web interface for fuzzy bookmark searching")

# Add CORS middleware to allow local access from browser
# CORS (Cross-Origin Resource Sharing) is necessary for web applications running locally
# to make requests to the same server, enabling frontend-backend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Embedded HTML/JS UI as a string
# This enhanced web interface provides a search input field, displays results with pagination,
# and includes metadata like result count and search time. The UI is embedded directly in the
# Python file to create a single-file application. It uses vanilla JavaScript for simplicity
# and offline operation, with added pagination controls and keyboard navigation.
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fuzzy Bookmark Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .search-container { margin-bottom: 20px; }
        input[type="text"] { width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .results { margin-top: 20px; }
        .result-item { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 4px; background-color: #fafafa; }
        .result-title { font-weight: bold; color: #007bff; text-decoration: none; }
        .result-title:hover { text-decoration: underline; }
        .result-url { color: #666; font-size: 14px; margin: 5px 0; }
        .result-snippet { color: #333; margin: 5px 0; }
        .result-score { color: #888; font-size: 12px; }
        .loading { text-align: center; color: #666; }
        .error { color: red; text-align: center; }
        .results-info { margin-bottom: 15px; color: #666; font-size: 14px; }
        .pagination { display: flex; justify-content: center; align-items: center; margin-top: 20px; gap: 10px; }
        .pagination button { padding: 8px 12px; font-size: 14px; }
        .pagination .page-info { font-size: 14px; color: #666; }
        .page-numbers { display: flex; gap: 5px; }
        .page-numbers button { min-width: 35px; }
        .page-numbers button.active { background-color: #0056b3; }
        .result-buttons { margin-top: 10px; }
        .result-buttons button { margin-right: 10px; padding: 5px 10px; font-size: 14px; }
        .collapsible-section { margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #007bff; }
        .collapsible-section h4 { margin: 0 0 10px 0; color: #007bff; }
        .collapsible-section p { margin: 0; white-space: pre-wrap; }
        .collapsible-section pre { margin: 0; white-space: pre-wrap; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fuzzy Bookmark Search</h1>
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Enter search query (supports fuzzy matching like 'python~1')" />
            <button onclick="performSearch()">Search</button>
        </div>
        <div id="results" class="results"></div>
    </div>

    <script>
        let currentPage = 1;
        let currentQuery = '';
        let totalPages = 1;

        async function performSearch(page = 1) {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;

            currentQuery = query;
            currentPage = page;

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, page: page, page_size: 20 })
                });

                if (!response.ok) throw new Error('Search failed');

                const data = await response.json();
                totalPages = data.pagination.total_pages;
                displayResults(data);
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            }
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            const results = data.results;
            const pagination = data.pagination;
            const searchTime = data.search_time;

            // Results info with count and time
            const resultsInfo = `<div class="results-info">
                About ${pagination.total_results.toLocaleString()} results (${formatSearchTime(searchTime)})
            </div>`;

            if (results.length === 0) {
                resultsDiv.innerHTML = resultsInfo + '<div>No results found.</div>';
                return;
            }

            // Results list
            const resultsHtml = results.map((result, index) => `
                <div class="result-item">
                    <a href="${result.url}" class="result-title" target="_blank">${result.title}</a>
                    <div class="result-url">${result.url}</div>
                    <div class="result-snippet">${result.snippet}</div>
                    <div class="result-score">Score: ${result.score.toFixed(2)}</div>
                    <div class="result-buttons">
                        <button onclick="toggleSection(${index}, 'summary')">Show Summary</button>
                        <button onclick="toggleSection(${index}, 'content')">Show Content</button>
                        <button onclick="toggleSection(${index}, 'full_record')">Show Full Record</button>
                    </div>
                    <div id="summary-${index}" class="collapsible-section" style="display: none;">
                        <h4>Summary</h4>
                        <p>${result.summary}</p>
                    </div>
                    <div id="content-${index}" class="collapsible-section" style="display: none;">
                        <h4>Content</h4>
                        <p>${result.content}</p>
                    </div>
                    <div id="full_record-${index}" class="collapsible-section" style="display: none;">
                        <h4>Full Record</h4>
                        <pre>${JSON.stringify(result.full_record, null, 2)}</pre>
                    </div>
                </div>
            `).join('');

            // Pagination controls
            const paginationHtml = createPaginationControls(pagination);

            resultsDiv.innerHTML = resultsInfo + resultsHtml + paginationHtml;
        }

        function createPaginationControls(pagination) {
            const { page, total_pages, has_prev, has_next } = pagination;

            let controls = '<div class="pagination">';

            // Previous button
            controls += `<button onclick="changePage(${page - 1})" ${!has_prev ? 'disabled' : ''}>Previous</button>`;

            // Page numbers
            controls += '<div class="page-numbers">';

            // Calculate page range to show (current ±2, with ellipsis)
            const startPage = Math.max(1, page - 2);
            const endPage = Math.min(total_pages, page + 2);

            // First page and ellipsis if needed
            if (startPage > 1) {
                controls += `<button onclick="changePage(1)">1</button>`;
                if (startPage > 2) {
                    controls += '<span>...</span>';
                }
            }

            // Page numbers
            for (let i = startPage; i <= endPage; i++) {
                const activeClass = i === page ? 'active' : '';
                controls += `<button onclick="changePage(${i})" class="${activeClass}">${i}</button>`;
            }

            // Last page and ellipsis if needed
            if (endPage < total_pages) {
                if (endPage < total_pages - 1) {
                    controls += '<span>...</span>';
                }
                controls += `<button onclick="changePage(${total_pages})">${total_pages}</button>`;
            }

            controls += '</div>';

            // Next button
            controls += `<button onclick="changePage(${page + 1})" ${!has_next ? 'disabled' : ''}>Next</button>`;

            // Page info
            controls += `<div class="page-info">Page ${page} of ${total_pages}</div>`;

            controls += '</div>';

            return controls;
        }

        function changePage(page) {
            if (page >= 1 && page <= totalPages) {
                performSearch(page);
            }
        }

        function formatSearchTime(seconds) {
            if (seconds >= 1.0) {
                return seconds.toFixed(2) + ' seconds';
            } else {
                return Math.round(seconds * 1000) + ' ms';
            }
        }

        // Allow search on Enter key press
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') performSearch();
        });

        function toggleSection(index, section) {
            const element = document.getElementById(`${section}-${index}`);
            const button = event.target;
            if (element.style.display === 'none') {
                element.style.display = 'block';
                button.textContent = button.textContent.replace('Show', 'Hide');
            } else {
                element.style.display = 'none';
                button.textContent = button.textContent.replace('Hide', 'Show');
            }
        }

        // Keyboard navigation for pagination (left/right arrows)
        document.addEventListener('keydown', function(e) {
            if (currentQuery && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
                e.preventDefault();
                if (e.key === 'ArrowLeft' && currentPage > 1) {
                    changePage(currentPage - 1);
                } else if (e.key === 'ArrowRight' && currentPage < totalPages) {
                    changePage(currentPage + 1);
                }
            }
        });
    </script>
</body>
</html>
"""

# FastAPI route to serve the HTML UI
# This endpoint serves the embedded HTML interface when users access the root URL.
# It returns the HTML content with proper content type for browser rendering.
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serve the embedded HTML user interface.

    This route provides the web frontend for the fuzzy bookmark search application.
    The HTML includes a search input field and JavaScript for making API calls to perform searches.
    """
    return HTML_UI

# FastAPI route for search API
# This POST endpoint accepts search queries and returns JSON results with pagination support.
# It integrates the updated search_bookmarks function into the web API with pagination parameters.
@app.post("/api/search")
async def api_search(request: Request):
    """
    API endpoint for performing fuzzy bookmark searches with pagination support.

    Accepts a JSON payload with 'query', 'page', and 'page_size' fields.
    Returns a JSON response with paginated search results, pagination metadata,
    search execution time, and the original query.

    Pagination parameters:
    - page: Integer, 1-based page number (default: 1, min: 1)
    - page_size: Integer, results per page (default: 20, max: 100)

    Args:
        request (Request): FastAPI request object containing JSON payload.

    Returns:
        dict: JSON response with results, pagination metadata, search_time, and query.

    Raises:
        HTTPException: If query is missing, pagination parameters are invalid, or search fails.
    """
    try:
        data = await request.json()
        query = data.get('query', '').strip()
        page = data.get('page', 1)
        page_size = data.get('page_size', 20)

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        # Validate pagination parameters
        try:
            page = int(page)
            page_size = int(page_size)
            if page < 1 or page_size < 1 or page_size > 100:
                raise ValueError("Invalid pagination parameters")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid pagination parameters: page >= 1, 1 <= page_size <= 100")

        # Perform the search with pagination
        search_data = search_bookmarks(query, page=page, page_size=page_size)

        return search_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

def main():
    """
    Main function to run the fuzzy bookmark search application.

    This function parses command-line arguments and launches the FastAPI server.
    It supports options for setting the port, updating the index, and other configurations.
    """
    parser = argparse.ArgumentParser(description="Fuzzy Bookmark Search Engine")
    parser.add_argument('--port', type=int, default=8132,
                         help='Port to run the server on (default: 8132)')
    parser.add_argument('--no-update', action='store_true',
                         help='Skip updating the index')
    parser.add_argument('--index-dir', type=str, default='./whoosh_index',
                         help='Directory for the Whoosh index (default: ./whoosh_index)')
    parser.add_argument('--lmdb-path', type=str, default='bookmark_index.lmdb',
                          help='Path to the LMDB database directory (default: bookmark_index.lmdb)')

    args = parser.parse_args()

    # Initialize LMDB database
    print("Initializing LMDB database...")
    init_lmdb()

    # Ensure bookmarks are indexed before starting the server
    # This step is necessary for the search functionality to work.
    print("Checking and indexing bookmarks if necessary...")
    try:
        # Attempt to open the index; if it doesn't exist or no-update is not requested, create/update it
        if not index.exists_in(args.index_dir) or not args.no_update:
            if not args.no_update and index.exists_in(args.index_dir):
                print("Updating existing index...")
            else:
                print("Creating new index...")
            bookmarks_gen = load_bookmarks_data(args.lmdb_path)
            index_bookmarks(bookmarks_gen, args.index_dir, update=not args.no_update)
            print("Indexing complete.")
        else:
            print("Index already exists. Skipping indexing.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        print("Continuing with server startup...")

    # Always print the total number of entries in the index
    try:
        ix = index.open_dir(args.index_dir)
        with ix.searcher() as searcher:
            total_entries = searcher.doc_count_all()
        print(f"Total bookmarks in index: {total_entries}")
    except Exception as e:
        print(f"Error accessing index for count: {e}")

    # Cleanup LMDB resources before starting server
    cleanup_lmdb()

    # Launch the FastAPI server using uvicorn
    # The server will be accessible at the specified port
    # This provides both the web UI and API endpoints for bookmark searching.
    print(f"Starting FastAPI server on http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()