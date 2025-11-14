import os
import json
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


def load_bookmarks_data(json_path='bookmarks_with_content.json'):
    """
    Load bookmark data from JSON file, handling zipped files if necessary.

    This function first checks if the JSON file exists directly. If not, it looks for a zipped version
    (bookmarks_with_content.zip) and extracts it. It then loads the JSON data as a generator to handle
    large files without loading everything into memory at once. Each bookmark is yielded as a dict,
    with preprocessing to generate a unique key and normalize text fields.

    Args:
        json_path (str): Path to the JSON file or zip file.

    Yields:
        dict: Preprocessed bookmark dictionary with fields like title, url, content, summary, key.
    """
    # Check if JSON exists, otherwise extract from zip
    if not os.path.exists(json_path):
        zip_path = json_path.replace('.json', '.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
        else:
            raise FileNotFoundError(f"Neither {json_path} nor {zip_path} found.")

    # Load JSON as generator for memory efficiency
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load entire JSON; for very large files, consider streaming

    # Perform preliminary pass to count total records for progress tracking
    # This allows tqdm to display accurate progress bars with total counts
    total_records = len(data)

    for bookmark in data:
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

    batch_size = 1000  # Process in batches to manage memory
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
    print(f"Records parsed from JSON file: {processed_count}")
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
                'key': hit['key']
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
            const resultsHtml = results.map(result => `
                <div class="result-item">
                    <a href="${result.url}" class="result-title" target="_blank">${result.title}</a>
                    <div class="result-url">${result.url}</div>
                    <div class="result-snippet">${result.snippet}</div>
                    <div class="result-score">Score: ${result.score.toFixed(2)}</div>
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
    parser.add_argument('--json-path', type=str, default='bookmarks_with_content.json',
                        help='Path to the bookmarks JSON file (default: bookmarks_with_content.json)')

    args = parser.parse_args()

    # Ensure bookmarks are indexed before starting the server
    # This step is necessary for the search functionality to work.
    print("Checking and indexing bookmarks if necessary...")
    try:
        # Attempt to open the index; if it doesn't exist or no-update is not requested, create/update it
        if not index.exists_in(args.index_dir) or not args.no_update:
            if not args.no_update and index.exists_in(args.index_dir):
                print("Updating existing index...")
            else:
                print("Index not found. Loading and indexing bookmarks...")
            bookmarks_gen = load_bookmarks_data(args.json_path)
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

    # Launch the FastAPI server using uvicorn
    # The server will be accessible at the specified port
    # This provides both the web UI and API endpoints for bookmark searching.
    print(f"Starting FastAPI server on http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()