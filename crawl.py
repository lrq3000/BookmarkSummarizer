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

import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chardet
from tqdm import tqdm
import traceback
from browser_history.browsers import *
import hashlib
import threading
import importlib.util
import sys
import signal
import logging
import shutil
import contextlib
import multiprocessing

# Platform-specific imports for file locking
try:
    import fcntl  # Unix-like systems
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import msvcrt  # Windows
        HAS_MSVC = True
    except ImportError:
        HAS_MSVC = False

# TOML parsing imports with fallback for older Python versions
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older versions

# LMDB imports for persistent storage
import lmdb
import pickle
import sys

def sanitize_bookmark(bookmark, depth=0, seen=None):
    """
    Sanitize bookmark dictionary by removing non-serializable objects like selenium webdriver instances.
    Recursively processes nested dictionaries and lists with cycle detection.
    """
    if seen is None:
        seen = set()

    # Prevent infinite recursion by detecting cycles
    if id(bookmark) in seen:
        return None

    # Add current object to seen set
    seen.add(id(bookmark))

    try:
        if not isinstance(bookmark, dict):
            return bookmark
        sanitized = {}
        for key, value in bookmark.items():
            try:
                if isinstance(value, dict):
                    sanitized[key] = sanitize_bookmark(value, depth + 1, seen)
                elif isinstance(value, list):
                    sanitized[key] = [sanitize_bookmark(item, depth + 1, seen) if isinstance(item, dict) else item for item in value]
                else:
                    # Check if it's a selenium webdriver instance
                    if hasattr(value, 'quit') and hasattr(value, 'get') and hasattr(value, 'find_element'):
                        continue
                    # Check for other complex objects
                    if hasattr(value, '__dict__') or hasattr(value, '__slots__'):
                        continue
                    sanitized[key] = value
            except RecursionError:
                continue
        return sanitized
    finally:
        # Remove from seen set when done processing this object
        seen.discard(id(bookmark))

def safe_pickle(obj):
    """
    Safely pickle an object with increased recursion limit and sanitization.
    """
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(20000)
    try:
        sanitized = sanitize_bookmark(obj)
        return pickle.dumps(sanitized)
    finally:
        sys.setrecursionlimit(old_limit)

# --- Browser Profile Configuration ---
# Default profile paths are handled by browser_history module
# Users can specify custom paths via --profile-path argument
# ----------------------------------------------------

bookmarks_path = os.path.expanduser("./bookmarks.json")
failed_urls_path = os.path.expanduser("./failed_urls.json")

# LMDB database and persistent structures for on-disk indexing
# LMDB provides persistent key-value storage for efficient O(1) lookups
# This replaces in-memory sets with disk-based storage for scalability
lmdb_storage_path = os.path.expanduser("./bookmark_index.lmdb")
lmdb_env = None
url_hashes_db = None  # LMDB database for URL hash deduplication
content_hashes_db = None  # LMDB database for content hash deduplication
bookmarks_db = None  # LMDB database for storing bookmarks with integer keys
failed_records_db = None  # LMDB database for storing failed records
url_to_key_db = None  # LMDB database for URL to key mapping (O(1) lookups for flushing)
domain_index_db = None  # LMDB database for domain-based secondary indexing (stores only keys)
date_index_db = None  # LMDB database for date-based secondary indexing (stores only keys)

# LMDB configuration defaults
DEFAULT_LMDB_MAP_SIZE = 10 * 1024 * 1024  # 10MB - reduced for dynamic resizing
DEFAULT_LMDB_MAX_DBS = 7

# Backup configuration defaults
BACKUP_BASE_DIR = os.path.expanduser("./backups")
BACKUP_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

content_lock = threading.Lock()

# Global flag for graceful shutdown
shutdown_flag = False

# Global variable to store the WebDriver path
webdriver_path = None


# Custom parsers list - dynamically loaded from custom_parsers/ directory
custom_parsers = []

# Setup logging for comprehensive error handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# In-memory fallback structures for graceful degradation
fallback_url_hashes = set()
fallback_content_hashes = set()
fallback_bookmarks = []
fallback_failed_records = []
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
        # Get the directory containing the storage file
        storage_dir = os.path.dirname(os.path.abspath(lmdb_storage_path))
        if not os.path.exists(storage_dir):
            # If directory doesn't exist, try to create it
            try:
                os.makedirs(storage_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create storage directory {storage_dir}: {e}")
                return False
        stat = shutil.disk_usage(storage_dir)
        free_space_mb = stat.free / (1024 * 1024)
        if free_space_mb < min_space_mb:
            logger.error(f"Insufficient disk space: {free_space_mb:.2f} MB free, {min_space_mb} MB required")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        return False

# Check if LMDB database exists and contains data
def check_lmdb_database_exists_and_has_data():
    """
    Check if the LMDB database file exists and contains any data.

    Returns:
        tuple: (exists, has_data, data_count)
            - exists (bool): True if database file exists
            - has_data (bool): True if database contains any bookmarks
            - data_count (int): Number of bookmarks in database (0 if no data)
    """
    try:
        # Check if the LMDB directory exists
        if not os.path.exists(lmdb_storage_path):
            logger.info(f"LMDB database directory does not exist: {lmdb_storage_path}")
            return False, False, 0

        # Check if data.mdb file exists (main LMDB data file)
        data_file = os.path.join(lmdb_storage_path, 'data.mdb')
        if not os.path.exists(data_file):
            logger.info(f"LMDB data file does not exist: {data_file}")
            return False, False, 0

        # Try to open database in read-only mode to check for data
        try:
            env = lmdb.open(lmdb_storage_path, readonly=True, max_dbs=5)
            try:
                # Check bookmarks database specifically
                bookmarks_db_check = env.open_db(b'bookmarks')
                with env.begin() as txn:
                    cursor = txn.cursor(bookmarks_db_check)
                    count = sum(1 for _ in cursor)
                env.close()
                has_data = count > 0
                logger.info(f"LMDB database exists with {count} bookmarks")
                return True, has_data, count
            except Exception as e:
                logger.warning(f"Error checking LMDB data: {e}")
                env.close()
                return True, False, 0
        except Exception as e:
            logger.warning(f"Error opening LMDB database for check: {e}")
            return True, False, 0

    except Exception as e:
        logger.error(f"Error checking LMDB database existence: {e}")
        return False, False, 0

# Create timestamped backup of LMDB database
def create_lmdb_backup(operation_name="pre_write_backup"):
    """
    Create a timestamped backup of the LMDB database before write operations.

    This function creates a backup in a separate directory with clear naming convention:
    backups/lmdb_backup_YYYYMMDD_HHMMSS_<operation_name>/

    Parameters:
        operation_name (str): Descriptive name for the operation triggering the backup

    Returns:
        tuple: (success, backup_path)
            - success (bool): True if backup was created successfully
            - backup_path (str): Path to the backup directory, or None if failed
    """
    try:
        # Check if database exists and has data
        exists, has_data, data_count = check_lmdb_database_exists_and_has_data()
        if not exists or not has_data:
            logger.info(f"No backup needed: database exists={exists}, has_data={has_data}, data_count={data_count}")
            return True, None  # Not an error, just nothing to backup

        # Create backup directory structure
        timestamp = datetime.datetime.now().strftime(BACKUP_TIMESTAMP_FORMAT)
        backup_dir_name = f"lmdb_backup_{timestamp}_{operation_name}"
        backup_path = os.path.join(BACKUP_BASE_DIR, backup_dir_name)

        # Ensure backup base directory exists
        os.makedirs(backup_path, exist_ok=True)

        logger.info(f"Creating LMDB backup: {backup_path}")

        # For concurrent access safety, use platform-specific file locking during backup
        lock_file = os.path.join(lmdb_storage_path, 'backup.lock')

        # Close any existing LMDB environment to ensure clean copy
        global lmdb_env
        env_was_open = lmdb_env is not None
        if env_was_open:
            try:
                lmdb_env.close()
                lmdb_env = None
                logger.debug("Temporarily closed LMDB environment for backup")
            except Exception as e:
                logger.warning(f"Error closing LMDB environment for backup: {e}")

        try:
            # Acquire file lock to prevent concurrent access during backup
            with open(lock_file, 'w') as lock_f:
                lock_acquired = False
                try:
                    if HAS_FCNTL:
                        # Unix-like systems
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking exclusive lock
                        lock_acquired = True
                    elif HAS_MSVC:
                        # Windows systems
                        # For Windows, we'll use a simpler approach since msvcrt locking is more complex
                        # Just check if another process has the file open
                        try:
                            msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
                            lock_acquired = True
                        except OSError:
                            lock_acquired = False
                    else:
                        # Fallback: no locking available
                        logger.warning("No file locking mechanism available, proceeding without lock")
                        lock_acquired = True

                    if not lock_acquired:
                        logger.warning("Could not acquire backup lock (another backup in progress), skipping backup")
                        return True, None  # Not a failure, just concurrent access

                    logger.debug("Acquired backup lock for concurrent access safety")

                    # Copy all LMDB files
                    import glob
                    lmdb_files = glob.glob(os.path.join(lmdb_storage_path, "*"))
                    for src_file in lmdb_files:
                        if os.path.isfile(src_file) and not src_file.endswith('.lock'):  # Skip lock files
                            filename = os.path.basename(src_file)
                            dst_file = os.path.join(backup_path, filename)
                            shutil.copy2(src_file, dst_file)
                            logger.debug(f"Backed up file: {filename}")

                    # Verify backup integrity by checking file sizes
                    original_size = sum(os.path.getsize(f) for f in lmdb_files if os.path.isfile(f) and not f.endswith('.lock'))
                    backup_size = sum(os.path.getsize(os.path.join(backup_path, os.path.basename(f)))
                                    for f in lmdb_files if os.path.isfile(f) and not f.endswith('.lock'))

                    if backup_size != original_size:
                        logger.warning(f"Backup size mismatch: original={original_size}, backup={backup_size}")
                    else:
                        logger.info(f"Backup created successfully: {backup_path} ({backup_size} bytes)")

                    return True, backup_path

                finally:
                    # Release lock
                    try:
                        if lock_acquired:
                            if HAS_FCNTL:
                                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                            elif HAS_MSVC:
                                try:
                                    msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
                                except OSError:
                                    pass  # Ignore unlock errors
                            logger.debug("Released backup lock")
                    except Exception as e:
                        logger.warning(f"Error releasing backup lock: {e}")

        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except Exception as e:
                logger.warning(f"Error cleaning up lock file: {e}")

            # Re-open LMDB environment if it was previously open
            if env_was_open:
                try:
                    init_lmdb()
                    logger.debug("Re-opened LMDB environment after backup")
                except Exception as e:
                    logger.error(f"Error re-opening LMDB environment after backup: {e}")
                    # This is serious, but we'll continue with the backup success

    except Exception as e:
        logger.error(f"Error creating LMDB backup: {e}")
        return False, None

# Safe backup operation with graceful failure handling
def safe_backup_operation(operation_name="pre_write_backup", continue_on_failure=True):
    """
    Perform backup operation with graceful failure handling.

    Parameters:
        operation_name (str): Descriptive name for the operation
        continue_on_failure (bool): If True, continue execution even if backup fails

    Returns:
        bool: True if backup succeeded or was not needed, False only if backup failed critically
    """
    try:
        logger.info(f"Starting backup operation: {operation_name}")
        success, backup_path = create_lmdb_backup(operation_name)

        if success:
            if backup_path:
                logger.info(f"Backup completed successfully: {backup_path}")
            else:
                logger.info("Backup skipped (no data to backup)")
            return True
        else:
            logger.error(f"Backup operation '{operation_name}' failed")
            if continue_on_failure:
                logger.warning("Continuing execution despite backup failure")
                return True
            else:
                logger.error("Stopping execution due to backup failure")
                return False

    except Exception as e:
        logger.error(f"Unexpected error during backup operation '{operation_name}': {e}")
        if continue_on_failure:
            logger.warning("Continuing execution despite backup error")
            return True
        else:
            logger.error("Stopping execution due to backup error")
            return False

# LMDB operations are atomic and don't require retry mechanisms like ZODB
# This function is removed as LMDB handles transactions differently

# Global resize configuration and state tracking
lmdb_resize_threshold = 0.8  # Default threshold for triggering resize
lmdb_growth_factor = 2.0     # Default growth factor for resize
current_lmdb_map_size = None  # Track current map size for resize operations

# Initialize LMDB database and persistent structures for on-disk indexing
# LMDB uses key-value stores for efficient indexing and provides transactional persistence
def init_lmdb(map_size=None, max_dbs=None, readonly=False, resize_threshold=None, growth_factor=None):
    """
    Initialize LMDB database with persistent key-value stores for deduplication and storage.

    This function sets up the LMDB environment and creates database structures:
    - url_hashes_db: LMDB database for URL hash deduplication (O(1) lookups)
    - content_hashes_db: LMDB database for content hash deduplication (O(1) lookups)
    - bookmarks_db: LMDB database for storing bookmarks with integer keys
    - failed_records_db: LMDB database for storing failed records with integer keys
    - url_to_key_db: LMDB database for URL to key mapping (O(1) lookups for flushing)
    - domain_index_db: LMDB database for domain-based secondary indexing (stores only keys)
    - date_index_db: LMDB database for date-based secondary indexing (stores only keys)

    All operations are transactional for data integrity.
    Includes comprehensive error handling with fallback to in-memory structures.
    Supports dynamic resizing configuration for MapFullError handling.

    Parameters:
        map_size (int, optional): Size of the memory map in bytes. Defaults to 10MB.
        max_dbs (int, optional): Maximum number of named databases. Defaults to 7.
        readonly (bool, optional): Open database in read-only mode. Defaults to False.
        resize_threshold (float, optional): Threshold for triggering resize (0.0-1.0). Defaults to 0.8.
        growth_factor (float, optional): Growth factor for resize. Defaults to 2.0.
    """
    global lmdb_env, url_hashes_db, content_hashes_db, bookmarks_db, failed_records_db, url_to_key_db, domain_index_db, date_index_db, use_fallback
    global lmdb_resize_threshold, lmdb_growth_factor, current_lmdb_map_size

    # Use defaults if not specified
    if map_size is None:
        map_size = DEFAULT_LMDB_MAP_SIZE
    if max_dbs is None:
        max_dbs = DEFAULT_LMDB_MAX_DBS
    if resize_threshold is not None:
        lmdb_resize_threshold = resize_threshold
    if growth_factor is not None:
        lmdb_growth_factor = growth_factor

    # Track current map size for resize operations
    current_lmdb_map_size = map_size

    # Check disk space first (skip for readonly mode)
    if not readonly and not check_disk_space():
        logger.error("Insufficient disk space for LMDB initialization. Falling back to in-memory structures.")
        use_fallback = True
        return

    try:
        # Create LMDB environment with configurable size limits
        lmdb_env = lmdb.open(lmdb_storage_path, map_size=map_size, max_dbs=max_dbs, readonly=readonly)

        # Open named databases
        url_hashes_db = lmdb_env.open_db(b'url_hashes')
        content_hashes_db = lmdb_env.open_db(b'content_hashes')
        bookmarks_db = lmdb_env.open_db(b'bookmarks')
        failed_records_db = lmdb_env.open_db(b'failed_records')
        url_to_key_db = lmdb_env.open_db(b'url_to_key')
        domain_index_db = lmdb_env.open_db(b'domain_index')
        date_index_db = lmdb_env.open_db(b'date_index')

        logger.info(f"Initialized LMDB database at {lmdb_storage_path} (map_size={map_size}, max_dbs={max_dbs}, readonly={readonly}, resize_threshold={lmdb_resize_threshold}, growth_factor={lmdb_growth_factor})")

    except lmdb.MapFullError as e:
        logger.error(f"LMDB MapFullError: Database map size {map_size} is too small. Consider increasing map_size.")
        use_fallback = True
    except lmdb.MapResizedError as e:
        logger.error(f"LMDB MapResizedError: Database was resized by another process. Try reopening.")
        use_fallback = True
    except lmdb.DiskError as e:
        logger.error(f"LMDB DiskError: Disk I/O error occurred: {e}")
        use_fallback = True
    except lmdb.InvalidError as e:
        logger.error(f"LMDB InvalidError: Invalid parameter or corrupted database: {e}")
        use_fallback = True
    except lmdb.VersionMismatchError as e:
        logger.error(f"LMDB VersionMismatchError: LMDB version mismatch: {e}")
        use_fallback = True
    except lmdb.BadRslotError as e:
        logger.error(f"LMDB BadRslotError: Reader slot corruption detected: {e}")
        use_fallback = True
    except Exception as e:
        logger.error(f"Error initializing LMDB: {e}")
        use_fallback = True

        # Cleanup on failure
        try:
            if lmdb_env:
                lmdb_env.close()
        except Exception as cleanup_e:
            logger.error(f"Error during LMDB cleanup: {cleanup_e}")

        logger.info("Falling back to in-memory structures for data integrity")

# Safe LMDB operations with error handling and transaction management
def safe_lmdb_operation(operation_func, fallback_func=None, operation_name="LMDB operation", readonly=False):
    """
    Perform an LMDB operation with error handling, transaction management, and fallback support.

    Parameters:
        operation_func (callable): Function performing the LMDB operation
        fallback_func (callable, optional): Fallback function if LMDB fails
        operation_name (str): Name of the operation for logging
        readonly (bool): Whether this is a read-only operation

    Returns:
        Any: Result of the operation or fallback
    """
    global use_fallback, current_lmdb_map_size, lmdb_growth_factor

    if use_fallback:
        if fallback_func:
            try:
                return fallback_func()
            except Exception as e:
                logger.error(f"Fallback {operation_name} failed: {e}")
                return None
        return None

    try:
        # Execute operation with proper transaction scoping
        with lmdb_env.begin(write=not readonly) as txn:
            result = operation_func(txn)
        return result
    except lmdb.MapFullError as e:
        logger.warning(f"LMDB MapFullError during {operation_name}: Database map is full, attempting dynamic resize.")

        # Attempt dynamic resize if not in readonly mode
        if not readonly and current_lmdb_map_size is not None:
            resize_success, new_map_size = resize_lmdb_database(
                current_lmdb_map_size,
                lmdb_growth_factor
            )
            if resize_success:
                current_lmdb_map_size = new_map_size
                logger.info(f"Resize successful, retrying {operation_name}")
                # Retry the operation with new map size
                try:
                    with lmdb_env.begin(write=not readonly) as txn:
                        result = operation_func(txn)
                    return result
                except Exception as retry_e:
                    logger.error(f"Operation {operation_name} failed even after resize: {retry_e}")
                    use_fallback = True
            else:
                logger.error(f"Resize failed for {operation_name}, falling back to in-memory structures")
                use_fallback = True
        else:
            logger.error(f"MapFullError in readonly mode or no map size tracking for {operation_name}, falling back to in-memory structures")
            use_fallback = True
    except lmdb.MapResizedError as e:
        logger.error(f"LMDB MapResizedError during {operation_name}: Database was resized by another process.")
        use_fallback = True
    except lmdb.DiskError as e:
        logger.error(f"LMDB DiskError during {operation_name}: Disk I/O error: {e}")
        use_fallback = True
    except lmdb.InvalidError as e:
        logger.error(f"LMDB InvalidError during {operation_name}: Invalid parameter or corrupted data: {e}")
        use_fallback = True
    except lmdb.BadTxnError as e:
        logger.error(f"LMDB BadTxnError during {operation_name}: Transaction error: {e}")
        use_fallback = True
    except lmdb.BadRslotError as e:
        logger.error(f"LMDB BadRslotError during {operation_name}: Reader slot corruption: {e}")
        use_fallback = True
    except lmdb.BadValsizeError as e:
        logger.error(f"LMDB BadValsizeError during {operation_name}: Value too large: {e}")
        use_fallback = True
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        use_fallback = True

    # Attempt fallback if operation failed
    if fallback_func:
        try:
            logger.info(f"Attempting fallback for {operation_name}")
            return fallback_func()
        except Exception as fallback_e:
            logger.error(f"Fallback {operation_name} failed: {fallback_e}")
    return None

# Resize LMDB database dynamically when MapFullError occurs
def resize_lmdb_database(current_map_size, growth_factor=2.0, max_attempts=5):
    """
    Dynamically resize the LMDB database by increasing the map size.

    This function implements dynamic resizing logic that:
    1. Calculates new map size using growth factor
    2. Attempts to reopen the database with new size
    3. Handles multiple resize attempts if needed
    4. Provides detailed logging of resize operations

    Parameters:
        current_map_size (int): Current map size in bytes
        growth_factor (float): Factor by which to grow the map size (default: 2.0)
        max_attempts (int): Maximum number of resize attempts (default: 5)

    Returns:
        tuple: (success, new_map_size)
            - success (bool): True if resize succeeded
            - new_map_size (int): New map size in bytes, or current size if failed
    """
    global lmdb_env, lmdb_storage_path

    logger.info(f"Attempting to resize LMDB database from {current_map_size} bytes ({current_map_size/1024/1024:.1f} MB)")

    for attempt in range(max_attempts):
        try:
            # Calculate new map size
            new_map_size = int(current_map_size * growth_factor)
            logger.info(f"Resize attempt {attempt + 1}/{max_attempts}: trying new map size {new_map_size} bytes ({new_map_size/1024/1024:.1f} MB)")

            # Close current environment if open
            if lmdb_env:
                try:
                    lmdb_env.close()
                    lmdb_env = None
                    logger.debug("Closed existing LMDB environment for resize")
                except Exception as e:
                    logger.warning(f"Error closing LMDB environment during resize: {e}")

            # Attempt to reopen with new map size
            lmdb_env = lmdb.open(lmdb_storage_path, map_size=new_map_size, max_dbs=DEFAULT_LMDB_MAX_DBS)

            # Re-open databases
            global url_hashes_db, content_hashes_db, bookmarks_db, failed_records_db, url_to_key_db, domain_index_db, date_index_db
            url_hashes_db = lmdb_env.open_db(b'url_hashes')
            content_hashes_db = lmdb_env.open_db(b'content_hashes')
            bookmarks_db = lmdb_env.open_db(b'bookmarks')
            failed_records_db = lmdb_env.open_db(b'failed_records')
            url_to_key_db = lmdb_env.open_db(b'url_to_key')
            domain_index_db = lmdb_env.open_db(b'domain_index')
            date_index_db = lmdb_env.open_db(b'date_index')

            logger.info(f"Successfully resized LMDB database to {new_map_size} bytes ({new_map_size/1024/1024:.1f} MB)")
            return True, new_map_size

        except Exception as e:
            logger.warning(f"Resize attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                logger.error(f"All {max_attempts} resize attempts failed. Keeping current map size.")
                # Try to reopen with original size
                try:
                    if lmdb_env:
                        lmdb_env.close()
                    lmdb_env = lmdb.open(lmdb_storage_path, map_size=current_map_size, max_dbs=DEFAULT_LMDB_MAX_DBS)
                    # Re-open databases
                    url_hashes_db = lmdb_env.open_db(b'url_hashes')
                    content_hashes_db = lmdb_env.open_db(b'content_hashes')
                    bookmarks_db = lmdb_env.open_db(b'bookmarks')
                    failed_records_db = lmdb_env.open_db(b'failed_records')
                    url_to_key_db = lmdb_env.open_db(b'url_to_key')
                    domain_index_db = lmdb_env.open_db(b'domain_index')
                    date_index_db = lmdb_env.open_db(b'date_index')
                    logger.info("Reopened LMDB database with original map size after resize failure")
                except Exception as reopen_e:
                    logger.error(f"Failed to reopen LMDB database after resize failure: {reopen_e}")
                    global use_fallback
                    use_fallback = True
                return False, current_map_size

    return False, current_map_size

# Cleanup LMDB resources
def cleanup_lmdb():
    """
    Properly close LMDB environment to ensure data integrity.
    """
    global lmdb_env
    try:
        if lmdb_env:
            lmdb_env.close()
        logger.info("LMDB cleanup completed")
    except Exception as e:
        logger.error(f"Error during LMDB cleanup: {e}")

# Get path to custom parsers directory handling frozen environments
def get_custom_parsers_dir():
    """
    Get the directory containing custom parsers, handling both normal and frozen environments.

    In a frozen (PyInstaller) environment, resources are extracted to a temporary
    directory pointed to by sys._MEIPASS. In a normal Python environment,
    they are relative to the script's location.
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temporary bundle directory at sys._MEIPASS
        # This is where --add-data files are extracted
        base_dir = sys._MEIPASS
    else:
        # Standard development environment
        base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_dir, 'custom_parsers')

# Load custom parsers from custom_parsers/ directory
def load_custom_parsers(parser_filter=None):
    """
    Dynamically discover and load custom parsers from the custom_parsers/ directory.
    Each parser should be a Python module with a 'main(bookmark: dict) -> dict' function.

    Parameters:
        parser_filter (list): Optional list of parser names (without .py extension) to load.
                             If None, all parsers are loaded.

    Returns:
        list: List of callable parser functions, sorted alphabetically by filename.
    """
    parsers = []
    parsers_dir = get_custom_parsers_dir()

    if not os.path.exists(parsers_dir):
        print(f"custom_parsers/ directory not found at {parsers_dir}, skipping custom parsers")
        return parsers

    # Iterate through all .py files in custom_parsers/
    for filename in os.listdir(parsers_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            
            # Skip if parser_filter is specified and this parser is not in the list
            if parser_filter is not None and module_name not in parser_filter:
                print(f"Skipping custom parser (not in filter): {module_name}")
                continue
            
            module_path = os.path.join(parsers_dir, filename)

            try:
                # Load the module dynamically
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Check if the module has a 'main' function
                    if hasattr(module, 'main') and callable(module.main):
                        parsers.append((filename, module.main))
                        print(f"Loaded custom parser: {module_name}")
                    else:
                        print(f"Warning: {module_name} does not have a callable 'main' function, skipping")
                else:
                    print(f"Warning: Could not load module {module_name}")
            except Exception as e:
                print(f"Error loading custom parser {module_name}: {e}")

    # Sort parsers alphabetically by filename to ensure systematic execution order
    parsers.sort(key=lambda x: x[0])

    # Check for missing parsers in filter
    if parser_filter:
        found_parser_names = {filename[:-3] for filename, _ in parsers}
        for name in parser_filter:
            if name not in found_parser_names:
                print(f"Warning: Custom parser '{name}' specified in filter but not found.")

    print(f"Loaded {len(parsers)} custom parsers")
    return [parser for filename, parser in parsers]

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """
    Handle KeyboardInterrupt (CTRL-C) signal for graceful shutdown.
    Sets the global shutdown flag, cleans up LMDB resources, and prints a shutdown message.
    """
    global shutdown_flag
    print("\nReceived KeyboardInterrupt (CTRL-C). Initiating graceful shutdown...")
    shutdown_flag = True
    cleanup_lmdb()

# Load TOML configuration
def load_config(config_path="default_config.toml"):
    """
    Load configuration from TOML file.

    Parameters:
        config_path (str): Path to the TOML configuration file.

    Returns:
        dict: Configuration dictionary loaded from TOML file.
    """
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{config_path}' not found. Using default values.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading configuration from '{config_path}': {e}. Using default values.")
        return {}

# Configuration settings
class ModelConfig:
    # Supported model types
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"  # Added Ollama model type

    def __init__(self, config_data=None):
        """
        Initialize ModelConfig with TOML configuration data.

        Parameters:
            config_data (dict, optional): Configuration dictionary from TOML file.
                                         If None, uses default values.
        """
        if config_data is None:
            config_data = {}

        # Extract model section from config
        model_config = config_data.get("model", {})

        # Default configuration with TOML overrides
        self.model_type = model_config.get("model_type", self.OPENAI)
        self.api_key = model_config.get("api_key", "")
        self.api_base = model_config.get("api_base", "https://api.openai.com/v1")
        self.model_name = model_config.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = model_config.get("max_tokens", 1000)
        self.temperature = model_config.get("temperature", 0.3)

        # Extract crawl section from config
        crawl_config = config_data.get("crawl", {})
        self.max_input_content_length = crawl_config.get("max_input_content_length", 6000)
        self.generate_summary = crawl_config.get("generate_summary", True)

        # DeepSeek specific configuration (keeping defaults for backward compatibility)
        self.top_p = model_config.get("top_p", 0.7)
        self.top_k = model_config.get("top_k", 50)
        self.frequency_penalty = model_config.get("frequency_penalty", 0.5)
        self.system_prompt = model_config.get("system_prompt", "")
        self.use_tools = model_config.get("use_tools", False)

        # Qwen specific configuration
        self.qwen_api_version = model_config.get("qwen_api_version", "2023-12-01-preview")
        # Ollama specific configuration
        self.ollama_format = model_config.get("ollama_format", "text")  # Options: json, text

# Generate summary using a Large Language Model (LLM)
def generate_summary(title, content, url, config=None):
    """
    Generates a summary of the webpage content using an LLM.
    
    Parameters:
        title (str): Webpage title
        content (str): Webpage content
        url (str): Webpage URL
        config (ModelConfig, optional): Model configuration, defaults to environment variables.
        
    Returns:
        str: The generated summary.
    """
    if config is None:
        config = ModelConfig()
    
    try:
        # Limit content length to avoid exceeding token limits
        max_content_length = config.max_input_content_length
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Construct a more detailed prompt
        prompt = f"""Generate only a comprehensive, informative summary (approx. 500 words) for the following webpage content. Begin directly with the summary content, produce no introductory phrases or meta-statements of any kind, such as “Here is a summary of”, "**Summary:**" or any other variant.

Webpage Title: {title}
Webpage URL: {url}

Webpage Content:
{content}

Summary Requirements:
1. Automatically detect the language of the content and generate the summary in the same language.
2. Organize the content in a key-information-dense manner, ensuring the inclusion of important technical terms, entity names, and key concepts.
3. Use a clear paragraph structure, dividing information by topic, with each paragraph focusing on a core point.
4. Provide a concise introductory summary sentence at the beginning, briefly stating the main content and purpose of the document.
5. Use factual, specific statements, avoiding vague or general descriptions.
6. Retain important numbers, dates, names, technical terms, and unique identifiers from the original text.
7. For technical content, include specific technology names, version numbers, parameters, and method names.
8. For news events, clearly include the time, location, people, and key details of the event.
9. For tutorials or guides, list specific step names and critical operational points.
10. For products or services, include specific product names, features, and specifications.
11. Ensure high information density for easy vector retrieval matching.
12. Output only the summary text. Do not add any explanations, comments, or meta statements before or after the summary
13. Be concise, straight-to-the-point, and avoid unnecessary filler words, write in a note taking kind of way, without conjunctive words, only keep information dense words.
14. Never account nor mention that "JavaScript is disabled in your browser" in the summary, this is not a content but an error.

Please generate an information-dense, clearly structured summary, optimized for a text format suitable for vector retrieval, minimizing filler words, unnecessary repetition, and words like: 'okay', 'um', etc.
"""
        # Call the corresponding API based on the model type
        if config.model_type == ModelConfig.OLLAMA:
            return call_ollama_api(prompt, config)
        elif config.model_type == ModelConfig.QWEN:
            return call_qwen_api(prompt, config)
        elif config.model_type == ModelConfig.DEEPSEEK:
            return call_deepseek_api(prompt, config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    except Exception as e:
        print(f"Summary generation failed: {url} - {e}")
        return f"Summary generation failed: {str(e)}"

# Ollama API Call
def call_ollama_api(prompt, config=None):
    """
    API call specifically designed for models deployed with Ollama
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # Determine whether to use the chat or generate interface
    use_chat_api = getattr(config, 'use_chat_api', True)
    
    # API endpoint
    if use_chat_api:
        url = f"{config.api_base}/api/chat"
    else:
        url = f"{config.api_base}/api/generate"
    
    # Construct request payload
    if use_chat_api:
        # Use chat interface
        messages = [{"role": "user", "content": prompt}]
        
        # If there is a system prompt, add it to the messages
        if hasattr(config, 'system_prompt') and config.system_prompt:
            messages.insert(0, {"role": "system", "content": config.system_prompt})
        
        payload = {
            "model": config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens
            }
        }
    else:
        # Use generate interface
        system_prompt = config.system_prompt if hasattr(config, 'system_prompt') and config.system_prompt else ""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": config.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens
            }
        }
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send request
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120  # Increase timeout, local models may require longer processing time
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract generated text - Ollama API format
        if use_chat_api:
            if "message" in result:
                return result["message"]["content"]
            elif "response" in result:
                return result["response"]
        else:
            if "response" in result:
                return result["response"]
        
        # If the expected field is not found, return the entire response
        return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"Ollama API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"Ollama API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"Ollama API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise
      
# API call for Qwen (Tongyi Qianwen)
def call_qwen_api(prompt, config=None):
    """
    API call specifically designed for Qwen (Tongyi Qianwen) models.
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # API endpoint
    url = f"{config.api_base}/chat/completions"
    
    # Construct messages
    messages = [{"role": "user", "content": prompt}]
    
    # If there is a system prompt, add it to the messages
    if hasattr(config, 'system_prompt') and config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})
    
    # Construct request payload - Qwen 2.5 is usually compatible with OpenAI format
    payload = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "stream": False
    }
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # If there is an API key, add it to the headers
    if config.api_key and config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key}"
    
    try:
        # Send request
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract generated text - Qwen API usually follows OpenAI format
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"]:
                return result["choices"]["message"]["content"]
            elif "text" in result["choices"]:
                return result["choices"]["text"]
            else:
                # If the expected field is not found, return the entire choice object
                return str(result["choices"])
        else:
            # If the response does not contain the choices field, return the entire response
            return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"Qwen API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"Qwen API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"Qwen API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise

def call_deepseek_api(prompt, config=None):
    """
    API call specifically designed for DeepSeek R1.
    
    Parameters:
        prompt (str): The prompt text
        config (ModelConfig, optional): Model configuration
        
    Returns:
        str: The response text generated by the model
    """
    if config is None:
        config = ModelConfig()
    
    # API endpoint
    url = f"{config.api_base}/chat/completions"
    print(f"Calling DeepSeek API: {url}")
    print(f"Using model: {config.model_name}")
    print(f"API Key Length: {len(config.api_key) if config.api_key else 0}")
    
    # Construct messages
    messages = [{"role": "user", "content": prompt}]
    
    # If there is a system prompt, add it to the messages
    if hasattr(config, 'system_prompt') and config.system_prompt:
        messages.insert(0, {"role": "system", "content": config.system_prompt})
    
    # Construct request payload
    payload = {
        "model": config.model_name,  # e.g., "deepseek-ai/DeepSeek-R1"
        "messages": messages,
        "stream": False,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": getattr(config, 'top_p', 0.7),
        "top_k": getattr(config, 'top_k', 50),
        "frequency_penalty": getattr(config, 'frequency_penalty', 0.5),
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    # Print request configuration for debugging
    print(f"Request config: temperature={config.temperature}, max_tokens={config.max_tokens}")
    
    # Construct headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # If there is an API key, add it to the headers
    if config.api_key and config.api_key.strip():
        headers["Authorization"] = f"Bearer {config.api_key}"
        print("Authorization header added")
    else:
        print("API key not set, request does not include Authorization header")
    
    try:
        # Send request
        print("Sending request...")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60  # Increase timeout, as LLMs may require longer processing time
        )
        
        # Check response status
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        print(f"Successfully received response: {result.keys() if isinstance(result, dict) else 'Non-dictionary response'}")
        
        # Extract generated text
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"]:
                content = result["choices"]["message"]["content"]
                print(f"Successfully extracted content, length: {len(content)}")
                return content
            elif "text" in result["choices"]:
                text = result["choices"]["text"]
                print(f"Successfully extracted text, length: {len(text)}")
                return text
            else:
                # If the expected field is not found, return the entire choice object
                print(f"Content or text field not found, returning entire choice object: {result['choices']}")
                return str(result["choices"])
        else:
            # If the response does not contain the choices field, return the entire response
            print(f"Response does not contain choices field, returning entire response: {result}")
            return str(result)
            
    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API Request Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"API call failed: {str(e)}")
    except ValueError as e:
        print(f"DeepSeek API Response Parsing Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise Exception(f"Response parsing failed: {str(e)}")
    except Exception as e:
        print(f"DeepSeek API Unknown Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        raise

def test_api_connection(config=None):
    """Test if API connection is normal"""
    if config is None:
        config = ModelConfig()
    
    print(f"==== API Connection Test ====")
    print(f"Model Type: {config.model_type}")
    print(f"API Base URL: {config.api_base}")
    print(f"Model Name: {config.model_name}")
    print(f"API Key Length: {len(config.api_key) if config.api_key else 0}")
    
    try:
        # Simple test prompt
        test_prompt = "Who are you? Answer briefly."
        print(f"Test Prompt: '{test_prompt}'")

        print(f"Starting API connection test...")
        response = None

        # Call the corresponding API based on model type
        print(f'DEBUGLINE: config.model_type = {config.model_type}')
        if config.model_type == ModelConfig.OLLAMA:
            print(f"Using Ollama API")
            response = call_ollama_api(test_prompt, config)
        elif config.model_type == ModelConfig.QWEN:
            print(f"Using Qwen API")
            response = call_qwen_api(test_prompt, config)
        elif config.model_type == ModelConfig.DEEPSEEK:
            print(f"Using DeepSeek API")
            response = call_deepseek_api(test_prompt, config)
        else:
            # Handling for other model types...
            print(f"Unrecognized model type: {config.model_type}, attempting to use DeepSeek API")
            response = call_deepseek_api(test_prompt, config)

        # Check response
        if response and isinstance(response, str) and len(response) > 0:
            print("API connection test successful!")
            print(f"Model Response: {response[:100]}...")
            return True
        else:
            print(f"API returned empty or invalid response: {response}")
            return False

    except Exception as e:
        print(f"API connection test failed: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Detailed error information: {traceback_str}")
        return False

# Generate summaries for bookmarks stored in LMDB
def generate_summaries_for_bookmarks(bookmarks_with_content, model_config=None, force_recompute=False):
    """
    Generates summaries for bookmark content stored in LMDB.

    This function iterates through the LMDB bookmarks database and generates AI-powered summaries
    using the configured language model. By default, it skips bookmarks that already have a non-empty
    "summary" field to avoid redundant API calls and preserve existing summaries. The force_recompute
    parameter allows overriding this behavior to regenerate all summaries, which is useful for updating
    summaries with improved prompts or models.

    Parameters:
        bookmarks_with_content (list): List of bookmark dictionaries containing content to summarize.
        model_config (ModelConfig, optional): Configuration for the language model. Defaults to environment settings.
        force_recompute (bool): If True, recomputes summaries for all bookmarks regardless of existing summaries.
                                Defaults to False for efficiency.

    Returns:
        list: Updated list of bookmarks with generated summaries.
    """
    if model_config is None:
        model_config = ModelConfig()

    total_count = len(bookmarks_with_content)
    print('Generating summaries for bookmarks...')
    print(f"Using {model_config.model_type} model {model_config.model_name} to generate content summaries for {total_count} items...")
    if force_recompute:
        print("Force recompute mode enabled: regenerating all summaries regardless of existing ones.")

    # Create a map from URL to bookmark for quick lookup of existing summaries from LMDB
    existing_map = {}
    with lmdb_env.begin() as txn:
        cursor = txn.cursor(url_to_key_db)
        for url_bytes, key_bytes in cursor:
            url = url_bytes.decode('utf-8')
            key = int.from_bytes(key_bytes, 'big')
            bookmark_bytes = txn.get(key.to_bytes(4, 'big'), db=bookmarks_db)
            if bookmark_bytes:
                bookmark = pickle.loads(bookmark_bytes)
                existing_map[url] = bookmark

    success_count = 0
    skipped_count = 0
    for idx, bookmark in enumerate(tqdm(bookmarks_with_content, desc="Summary Generation Progress")):
        url = bookmark["url"]
        title = bookmark.get("title", bookmark.get("name", "No Title"))
        print(f"Generating summary [{idx+1}/{total_count}]: {title} - {url}")

        # Skip bookmarks that failed to crawl (no content) or have errors
        if "content" not in bookmark or "error" in bookmark:
            print(f"[{idx+1}/{total_count}] Skipping bookmark without content: {title} - {url}")
            skipped_count += 1
            continue

        # Check if already processed and has non-empty summary, unless force recompute is enabled
        # This optimization prevents redundant API calls and preserves existing summaries
        existing_summary = existing_map.get(url, {}).get("summary", "").strip()
        if not force_recompute and existing_summary:
            print(f"[{idx+1}/{total_count}] Skipping existing summary: {title} - {url}")
            success_count += 1
            skipped_count += 1
            continue

        progress_info = f"[{idx+1}/{total_count}]"
        print(f"{progress_info} Generating summary for the following link: {url}")

        # Generate summary
        summary = generate_summary(title, bookmark["content"], url, model_config)
        print(f"{progress_info} title: {title}")
        print(f"{progress_info} summary length: {len(summary)} characters")
        print(f"{progress_info} summary truncated: {summary[:200]}...")

        # Add summary to bookmark data
        bookmark["summary"] = summary
        bookmark["summary_model"] = model_config.model_name
        bookmark["summary_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if "Summary generation failed" not in summary:
            success_count += 1
            print(f"{progress_info} Summary generated successfully")

            # Update LMDB bookmark record transactionally
            try:
                with lmdb_env.begin(write=True) as txn:
                    # Find the key for this bookmark using O(1) lookup
                    key_bytes = txn.get(url.encode('utf-8'), db=url_to_key_db)

                    if key_bytes is not None:
                        # Update existing record
                        key = int.from_bytes(key_bytes, 'big')
                        bookmark_key = key.to_bytes(4, 'big')
                        txn.put(bookmark_key, safe_pickle(bookmark), db=bookmarks_db)
                        # Update secondary indexes for existing record
                        update_secondary_indexes(txn, bookmark_key, bookmark)
                    else:
                        # Add new record with next available key
                        cursor = txn.cursor(bookmarks_db)
                        if cursor.last():
                            next_key = int.from_bytes(cursor.key(), 'big') + 1
                        else:
                            next_key = 1
                        bookmark_key = next_key.to_bytes(4, 'big')
                        txn.put(bookmark_key, safe_pickle(bookmark), db=bookmarks_db)
                        # Update url_to_key_db for future O(1) lookups
                        txn.put(url.encode('utf-8'), bookmark_key, db=url_to_key_db)
                        # Update secondary indexes for new record
                        update_secondary_indexes(txn, bookmark_key, bookmark)

                print(f"{progress_info} Current progress saved to LMDB")
            except Exception as e:
                print(f"{progress_info} Error saving to LMDB: {str(e)}")
        else:
            print(f"{progress_info} Summary generation failed: {summary}")

        # Brief pause after each request to avoid API limits
        time.sleep(0.5)

    print(f"Summary generation complete! Success: {success_count}/{total_count}")
    if not force_recompute:
        print(f"Skipped {skipped_count} bookmarks with existing summaries.")

    # Return bookmarks as list from LMDB for compatibility
    bookmarks_list = []
    with lmdb_env.begin() as txn:
        cursor = txn.cursor(bookmarks_db)
        for key_bytes, bookmark_bytes in cursor:
            bookmark = pickle.loads(bookmark_bytes)
            bookmarks_list.append(bookmark)
    return bookmarks_list

# Fetch bookmarks using browser_history module
def get_bookmarks(browser=None, profile_path=None):
    """
    Fetches bookmarks from specified browser or all browsers if none specified.

    Parameters:
        browser (str, optional): Browser name (e.g., 'chrome', 'firefox'). If None, fetches from all browsers.
        profile_path (str, optional): Path to browser profile directory.

    Returns:
        list: List of bookmark dictionaries with url, name, date_added, etc.
    """
    urls = []

    # Map browser name to browser_history class
    browser_map = {
        'chrome': Chrome,
        'firefox': Firefox,
        'edge': Edge,
        'opera': Opera,
        'opera_gx': OperaGX,
        'safari': Safari,
        'vivaldi': Vivaldi,
        'brave': Brave,
    }

    try:
        if browser:
            if browser not in browser_map:
                raise ValueError(f"Unsupported browser: {browser}")

            browser_class = browser_map[browser]

            # Initialize browser instance
            if profile_path:
                browser_instance = browser_class(profile_path)
            else:
                browser_instance = browser_class()

            # Fetch bookmarks
            bookmarks_output = browser_instance.fetch_bookmarks()
            bookmarks = bookmarks_output.bookmarks
        else:
            # Fetch from all browsers individually to handle errors per browser
            bookmarks = []
            for browser_name, browser_class in browser_map.items():
                try:
                    browser_instance = browser_class()
                    bookmarks_output = browser_instance.fetch_bookmarks()
                    bookmarks.extend(bookmarks_output.bookmarks)
                except Exception as e:
                    print(f"Error fetching bookmarks from {browser_name}: {e}")
                    continue

        # Convert to the expected format, filtering out invalid bookmarks
        for bookmark in bookmarks:
            # browser_history returns tuples of (datetime, url, title, folder)
            timestamp, url, title, folder = bookmark

            # Skip bookmarks with missing URL or title
            if not url or not title:
                continue

            bookmark_info = {
                "date_added": timestamp.isoformat() if timestamp else "N/A",
                "date_last_used": "N/A",  # browser_history doesn't provide this
                "guid": "N/A",  # browser_history doesn't provide this
                "id": "N/A",  # browser_history doesn't provide this
                "name": title,
                "type": "url",
                "url": url,
            }
            urls.append(bookmark_info)

    except Exception as e:
        print(f"Error fetching bookmarks: {e}")
        # Fallback to empty list or raise error
        raise

    return urls

# Create a session with a retry mechanism
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Maximum 3 retries
        backoff_factor=0.5,  # Retry interval backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes that trigger a retry
        allowed_methods=["GET"]  # Only retry for GET requests
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Clean up text content
def clean_text(text):
    # Remove excessive blank lines and spaces
    lines = [line.strip() for line in text.split('\n')]
    # Filter out empty lines
    lines = [line for line in lines if line]
    # Join lines
    return '\n'.join(lines)

# Extract domain from URL for secondary indexing
def extract_domain(url):
    """
    Extract domain from URL for secondary indexing.

    Parameters:
        url (str): The URL to extract domain from

    Returns:
        str: The domain (e.g., 'example.com') or empty string if extraction fails
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

# Extract date from bookmark for secondary indexing
def extract_date(bookmark):
    """
    Extract date from bookmark for secondary indexing.

    Uses date_added field if available, otherwise falls back to crawl_time or current date.

    Parameters:
        bookmark (dict): The bookmark dictionary

    Returns:
        str: Date in YYYY-MM-DD format
    """
    try:
        # Try date_added first (from browser bookmarks)
        date_str = bookmark.get('date_added')
        if date_str and date_str != 'N/A':
            # Parse ISO format date
            if 'T' in date_str:
                date_obj = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%Y-%m-%d')
    except Exception:
        pass

    try:
        # Try crawl_time
        crawl_time = bookmark.get('crawl_time')
        if crawl_time:
            date_obj = datetime.datetime.strptime(crawl_time, '%Y-%m-%dT%H:%M:%S')
            return date_obj.strftime('%Y-%m-%d')
    except Exception:
        pass

    # Fallback to current date
    return datetime.datetime.now().strftime('%Y-%m-%d')

# Update secondary indexes for a bookmark
def update_secondary_indexes(txn, bookmark_key, bookmark):
    """
    Update secondary indexes (domain and date) for a bookmark.

    This function maintains the secondary indexes by storing bookmark keys
    under domain and date keys for efficient querying.

    Parameters:
        txn: LMDB transaction object
        bookmark_key (bytes): The primary key of the bookmark
        bookmark (dict): The bookmark dictionary
    """
    try:
        # Extract domain and date
        url = bookmark.get('url', '')
        domain = extract_domain(url)
        date = extract_date(bookmark)

        # Update domain index (domain -> list of bookmark keys)
        if domain:
            domain_key = domain.encode('utf-8')
            # Get existing keys for this domain
            existing_keys = txn.get(domain_key, db=domain_index_db)
            if existing_keys:
                # Deserialize existing keys, add new key, re-serialize
                keys_set = set(pickle.loads(existing_keys))
                keys_set.add(bookmark_key)
                txn.put(domain_key, pickle.dumps(keys_set), db=domain_index_db)
            else:
                # First key for this domain
                txn.put(domain_key, pickle.dumps({bookmark_key}), db=domain_index_db)

        # Update date index (date -> list of bookmark keys)
        if date:
            date_key = date.encode('utf-8')
            # Get existing keys for this date
            existing_keys = txn.get(date_key, db=date_index_db)
            if existing_keys:
                # Deserialize existing keys, add new key, re-serialize
                keys_set = set(pickle.loads(existing_keys))
                keys_set.add(bookmark_key)
                txn.put(date_key, pickle.dumps(keys_set), db=date_index_db)
            else:
                # First key for this date
                txn.put(date_key, pickle.dumps({bookmark_key}), db=date_index_db)

    except Exception as e:
        logger.warning(f"Failed to update secondary indexes for bookmark {bookmark_key}: {e}")
        # Don't fail the entire operation for index update issues

def prepare_webdriver():
    """Install and cache the WebDriver, and store the path in a global variable."""
    global webdriver_path
    try:
        if getattr(sys, 'frozen', False):
            # Handle frozen environment pathing.
            driver_path = ChromeDriverManager().install()
            if not driver_path.endswith('.exe') and sys.platform == 'win32':
                driver_dir = os.path.dirname(driver_path)
                exe_path = os.path.join(driver_dir, "chromedriver.exe")
                if os.path.exists(exe_path):
                    driver_path = exe_path
            webdriver_path = driver_path
        else:
            # Standard installation.
            webdriver_path = ChromeDriverManager().install()
        print(f"WebDriver installed at: {webdriver_path}")
    except Exception as e:
        logger.warning(f"WebDriver installation failed: {e}. Selenium will not be available.")

# Initialize Selenium WebDriver
def init_webdriver():
    """Initializes a new WebDriver instance using the pre-installed driver path."""
    if not webdriver_path:
        return None  # Return None if the driver was not prepared.

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Add more user agent information
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")

    # Disable image loading to improve speed
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    try:
        service = Service(webdriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logger.warning(f"Chrome/Chromium webdriver initialization failed: {e}. Skipping Selenium-based crawling.")
        return None

# Fetch dynamic content using Selenium
def fetch_with_selenium(url, current_idx=None, total_count=None, title="No Title", min_delay=None, max_delay=None):
    """Fetches webpage content using Selenium"""
    # Get worker thread ID for logging
    worker_id = threading.get_ident()
    progress_info = f"[{current_idx}/{total_count}]" if current_idx and total_count else ""
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add a more realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36')
    
    try:
        driver = init_webdriver()
        if driver is None:
            print(f"[{worker_id}] {progress_info} Selenium not available, skipping crawl for: {title} - {url}")
            return None

        print(f"[{worker_id}] {progress_info} Starting Selenium crawl for: {title} - {url}")
        driver.get(url)

        # Wait for page to load with semi-random delay (minimum 5 seconds)
        if min_delay is not None and max_delay is not None:
            selenium_min = max(5, min_delay)
            delay = random.uniform(selenium_min, max(selenium_min, max_delay))
            print(f"[{worker_id}] Waiting {delay:.2f} seconds before fetching {url}")
            time.sleep(delay)
            print(f"[{worker_id}] Starting to fetch {url}")
        else:
            time.sleep(5)
        
        # Special handling for Zhihu: attempt to close login pop-up if present
        if "zhihu.com" in url:
            try:
                # Attempt to click the close button (multiple possible selectors)
                selectors = ['.Modal-closeButton', '.Button.Modal-closeButton', 
                            'button.Button.Modal-closeButton', '.close']
                for selector in selectors:
                    try:
                        # Use a more robust locator strategy if possible, but stick to the original logic for now
                        close_button = driver.find_element("css selector", selector)
                        close_button.click()
                        print(f"[{worker_id}] {progress_info} Successfully closed Zhihu login pop-up - using selector: {selector}")
                        time.sleep(1)
                        break
                    except:
                        continue
            except Exception as e:
                print(f"[{worker_id}] {progress_info} Failed to handle Zhihu login pop-up: {title} - {str(e)}")
        
        # Get page content
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
        
        # Special handling for Zhihu: extract article content
        if "zhihu.com" in url:
            article = soup.select_one('.Post-RichText') or soup.select_one('.RichText') or soup.select_one('.AuthorInfo') or soup.select_one('article')
            if article:
                text_content = article.get_text(strip=True)
            else:
                text_content = soup.get_text(strip=True)
        else:
            # General webpage handling
            text_content = soup.get_text(strip=True)
        
        # Fix encoding issues
        text_content = fix_encoding(text_content)
        
        # Ensure text is not empty
        if not text_content or len(text_content.strip()) < 5:  # At least 5 characters for valid content
            print(f"[{worker_id}] {progress_info} Selenium crawl content is empty or too short: {title} - {url}")
            return None
            
        print(f"[{worker_id}] {progress_info} Selenium successfully crawled: {title} - {url}, content length: {len(text_content)} characters")
        return text_content
        
    except Exception as e:
        print(f"[{worker_id}] {progress_info} Selenium crawl failed: {title} - {url} - {str(e)}")
        return None
    finally:
        if 'driver' in locals() and driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

# Detect and fix encoding issues - Optimized encoding fix function
def fix_encoding(text):
    """
    Detects and fixes text encoding issues, optimized performance version.
    """
    if not text or len(text) < 20:  # Return directly for short text
        return text
    
    # Quick check if fixing is needed - only check a small sample of the text
    sample_size = min(1000, len(text))
    sample_text = text[:sample_size]
    
    # If the proportion of non-ASCII characters in the sample is low, return the original text directly
    non_ascii_count = sum(1 for c in sample_text if ord(c) > 127)
    if non_ascii_count < sample_size * 0.1:  # If non-ASCII characters are less than 10%
        return text
    
    # Check for obvious encoding issue characteristics (consecutive special characters)
    # Use a more efficient method instead of regex
    special_char_sequence = 0
    for c in sample_text:
        if ord(c) > 127:
            special_char_sequence += 1
            if special_char_sequence >= 10:  # Found 10 consecutive non-ASCII characters
                break
        else:
            special_char_sequence = 0
    
    # If there are no obvious encoding issue characteristics, return directly
    if special_char_sequence < 10:
        return text
    
    # Only perform encoding detection on potentially problematic text
    try:
        # Only detect encoding on the sample, not the entire text
        sample_bytes = sample_text.encode('latin-1', errors='ignore')
        detected = chardet.detect(sample_bytes)
        
        # If the detected encoding is different from the current one and confidence is high
        if detected['confidence'] > 0.8 and detected['encoding'] not in ('ascii', 'utf-8'):
            # Re-encode the entire text
            text_bytes = text.encode('latin-1', errors='ignore')
            return text_bytes.decode(detected['encoding'], errors='replace')
    except Exception as e:
        print(f"Encoding fix failed: {e}")
    
    return text

# Apply custom parsers to bookmark before fetching content
def apply_custom_parsers(bookmark, parsers):
    """
    Apply all custom parsers in sequence to the bookmark.

    Parameters:
        bookmark (dict): The bookmark dictionary to process
        parsers (list): List of parser functions to apply

    Returns:
        dict: The updated bookmark after applying all parsers
    """
    updated_bookmark = bookmark.copy()
    for parser in parsers:
        try:
            result = parser(updated_bookmark)
            if result and isinstance(result, dict):
                updated_bookmark = result
        except Exception as e:
            print(f"Custom parser {parser.__module__ if hasattr(parser, '__module__') else 'unknown'} failed: {e}")
            # Continue with next parser, don't fail the entire process
    return updated_bookmark

# Crawl webpage content
def fetch_webpage_content(bookmark, current_idx=None, total_count=None, min_delay=None, max_delay=None, no_fetch=False):
    """Crawls webpage content"""
    # Get worker thread ID for logging
    worker_id = threading.get_ident()

    # Check for shutdown signal at the beginning of processing
    global shutdown_flag
    if shutdown_flag:
        print(f"[{worker_id}] Shutdown signal received, skipping bookmark processing: {bookmark.get('name', 'No Title')}")
        return None, None

    url = bookmark["url"]
    bookmark_title = bookmark.get("name", "No Title")  # Preserve original bookmark title
    progress_info = f"[{current_idx}/{total_count}]" if current_idx and total_count else ""

    # Handle no-fetch mode
    if no_fetch:
        print(f"[{worker_id}] {progress_info} Skipping fetch (no-fetch mode): {bookmark_title} - {url}")
        bookmark_with_content = bookmark.copy()
        bookmark_with_content["title"] = bookmark_title
        bookmark_with_content["content"] = ""
        bookmark_with_content["content_length"] = 0
        bookmark_with_content["crawl_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        bookmark_with_content["crawl_method"] = "no-fetch"
        return bookmark_with_content, None

    # Apply custom parsers before fetching content
    global custom_parsers
    bookmark = apply_custom_parsers(bookmark, custom_parsers)

    # Re-read url and title as they might have been modified by custom parsers
    url = bookmark["url"]
    bookmark_title = bookmark.get("name", "No Title")

    # Initialize variables to prevent unassigned error
    content = None
    crawl_method = None
    
    # Use Selenium directly for Zhihu links
    if "zhihu.com" in url:
        print(f"[{worker_id}] {progress_info} Detected Zhihu link, using Selenium directly for crawl: {bookmark_title} - {url}")
        content = fetch_with_selenium(url, current_idx, total_count, bookmark_title, min_delay, max_delay)
        crawl_method = "selenium"

        # Record crawl result
        if content:
            print(f"[{worker_id}] {progress_info} Successfully crawled Zhihu content: {bookmark_title} - {url}, content length: {len(content)} characters")
        else:
            print(f"[{worker_id}] {progress_info} Failed to crawl Zhihu content: {bookmark_title} - {url}")
            return None, {"url": url, "title": bookmark_title, "reason": "Zhihu content crawl failed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    else:
        try:
            try:
                print(f"[{worker_id}] {progress_info} Starting crawl: {bookmark_title} - {url}")
            except UnicodeEncodeError:
                print(f"[{worker_id}] {progress_info} Starting crawl: {bookmark_title.encode('ascii', 'replace').decode('ascii')} - {url}")
            session = create_session()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7" # Changed to prioritize English
            }
            # Add semi-random delay before HTTP request to prevent detection
            if min_delay is not None and max_delay is not None:
                delay = random.uniform(min_delay, max_delay)
                print(f"[{worker_id}] Waiting {delay:.2f} seconds before fetching {url}")
                time.sleep(delay)
                print(f"[{worker_id}] Starting to fetch {url}")
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Detect response content encoding
            detected_encoding = chardet.detect(response.content)
            if detected_encoding['confidence'] > 0.7:
                response.encoding = detected_encoding['encoding']
            
            # Check content type to ensure it is HTML or text
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower() and 'text/plain' not in content_type.lower():
                error_msg = f"Non-text content (Content-Type: {content_type})"
                print(f"[{worker_id}] {progress_info} Skipping {error_msg}: {bookmark_title} - {url}")
                failed_info = {"url": url, "title": bookmark_title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                return None, failed_info
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract HTML page title
            if soup.title and soup.title.string and isinstance(soup.title.string, str):
                html_title = soup.title.get_text().strip()
            else:
                html_title = "No Title"
            
            # Remove unnecessary elements like scripts, styles, navigation, etc.
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'meta', 'link']):
                element.decompose()
            
            # Get the full text content of the page directly
            full_text = soup.get_text(separator='\n')
            if isinstance(full_text, str) and full_text.strip():
                # Clean up text
                content = clean_text(full_text)
            else:
                content = ""
            crawl_method = "requests"
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            try:
                print(f"[{worker_id}] {progress_info} {error_msg}: {bookmark_title} - {url}")
            except UnicodeEncodeError:
                print(f"[{worker_id}] {progress_info} {error_msg}: {bookmark_title.encode('ascii', 'replace').decode('ascii')} - {url}")
            failed_info = {"url": url, "title": bookmark_title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return None, failed_info
    
    # If content is empty after regular crawl or for special sites, try Selenium
    if content is None or (isinstance(content, str) and not content.strip()):
        print(f"[{worker_id}] {progress_info} Regular crawl content is empty, attempting Selenium: {bookmark_title} - {url}")
        content = fetch_with_selenium(url, current_idx, total_count, bookmark_title, min_delay, max_delay)
        crawl_method = "selenium"
        
        # Record Selenium crawl result
        if content:
            print(f"[{worker_id}] {progress_info} Selenium successfully crawled {url}, content length: {len(content)} characters")
        else:
            print(f"[{worker_id}] {progress_info} Selenium crawl failed or content is empty: {url}")
    
    # Fix possible encoding issues
    if html_title:
        html_title = fix_encoding(html_title)
    else:
        html_title = "No Title"
        
    if content and isinstance(content, str):
        content = fix_encoding(content)
    else:
        content = ""

    # Prepend HTML title to content with appropriate formatting
    if html_title and html_title != "No Title":
        content = f"<h1>{html_title}</h1>\n\n{content}"

    # Check if content is empty
    if not content or not content.strip():
        # If we have a valid HTML title, use it as content and log a warning
        if html_title and html_title != "No Title":
            content = html_title
            print(f"[{worker_id}] {progress_info} Warning: Using HTML title as content (no webpage content available): {bookmark_title} - {url}")
        else:
            error_msg = "Extracted content is empty and no HTML title available"
            print(f"[{worker_id}] {progress_info} {error_msg}: {bookmark_title} - {url}")
            failed_info = {"url": url, "title": bookmark_title, "reason": error_msg, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return None, failed_info
            
    # Check for content deduplication using LMDB (transactional)
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    print(f"[{worker_id}] {progress_info} DEBUG: Generated content hash: {content_hash[:16]}... for URL: {url}")
    with content_lock:
        def check_content_deduplication(txn):
            if txn.get(content_hash.encode('utf-8'), db=content_hashes_db):
                print(f"[{worker_id}] {progress_info} Skipping duplicate content: {bookmark_title} - {url} (hash: {content_hash[:16]}...)")
                return True  # Duplicate found
            print(f"[{worker_id}] {progress_info} DEBUG: Content hash not found in database, adding: {content_hash[:16]}...")
            txn.put(content_hash.encode('utf-8'), b'1', db=content_hashes_db)
            return False  # No duplicate

        is_duplicate = safe_lmdb_operation(
            check_content_deduplication,
            lambda: content_hash in fallback_content_hashes,
            "content deduplication check"
        )

        if is_duplicate:
            if use_fallback:
                # Use fallback in-memory check
                if content_hash in fallback_content_hashes:
                    print(f"[{worker_id}] {progress_info} Skipping duplicate content (fallback): {bookmark_title} - {url}")
                    return None, None
                fallback_content_hashes.add(content_hash)
            else:
                return None, None

    # Create a copy of the bookmark including the content
    bookmark_with_content = bookmark.copy()
    bookmark_with_content["title"] = bookmark_title  # Preserve original bookmark title
    bookmark_with_content["content"] = content
    bookmark_with_content["content_length"] = len(content)
    bookmark_with_content["crawl_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    bookmark_with_content["crawl_method"] = crawl_method

    try:
        print(f"[{worker_id}] {progress_info} Successfully crawled: {bookmark_title} - {url}, content length: {len(content)} characters")
    except UnicodeEncodeError:
        # Handle Unicode encoding issues on Windows console
        safe_title = bookmark_title.encode('utf-8', 'replace').decode('utf-8')
        print(f"[{worker_id}] {progress_info} Successfully crawled: {safe_title} - {url}, content length: {len(content)} characters")
    return bookmark_with_content, None

# Parallel crawl bookmark content
def parallel_fetch_bookmarks(bookmarks, max_workers=20, limit=None, flush_interval=60, skip_unreachable=False, min_delay=None, max_delay=None, no_fetch=False):
    from concurrent.futures import as_completed

    bookmarks_to_process = bookmarks
    all_bookmarks_with_content = []  # This will accumulate all results for bookmarks
    all_failed_records = []  # This will accumulate all failed records
    
    # These lists will be used as temporary buffers for periodic flushing
    bookmarks_batch = []
    failed_records_batch = []
    
    skipped_url_count = 0
    new_bookmarks_added = 0  # To track for the limit

    # Batch flushing variables for thread-safety
    bookmarks_lock = threading.Lock()
    last_flush_time = time.time()

    def flush_to_disk(current_bookmarks, current_failed):
        """Flushes a batch of bookmarks and failed records to the LMDB database."""
        if not current_bookmarks and not current_failed:
            return

        try:
            # Batch process bookmarks to LMDB
            if current_bookmarks:
                with lmdb_env.begin(write=True) as txn:
                    cursor_b = txn.cursor(bookmarks_db)
                    # Determine the next available key for new entries
                    next_key_b = int.from_bytes(cursor_b.key(), 'big') + 1 if cursor_b.last() else 1
                    # In this loop, we handle both new and existing bookmarks.
                    # If a bookmark's URL is already in the database, we update the existing record.
                    # Otherwise, we create a new one. This prevents data corruption and duplicates.
                    for bookmark in current_bookmarks:
                        url = bookmark.get('url')
                        if not url: continue

                        # Check if the bookmark URL already exists to decide whether to update or insert
                        key_bytes = txn.get(url.encode('utf-8'), db=url_to_key_db)

                        if key_bytes:
                            # Update existing bookmark
                            bookmark_key = key_bytes
                        else:
                            # Insert new bookmark
                            bookmark_key = next_key_b.to_bytes(4, 'big')
                            next_key_b += 1
                        
                        # Write to the database
                        txn.put(bookmark_key, safe_pickle(bookmark), db=bookmarks_db)
                        # Ensure the URL-to-key mapping is up-to-date
                        txn.put(url.encode('utf-8'), bookmark_key, db=url_to_key_db)
                        # Update secondary indexes
                        update_secondary_indexes(txn, bookmark_key, bookmark)

            # Batch process failed records to LMDB
            if current_failed:
                with lmdb_env.begin(write=True) as txn:
                    cursor_f = txn.cursor(failed_records_db)
                    next_key_f = int.from_bytes(cursor_f.key(), 'big') + 1 if cursor_f.last() else 1
                    for failed_record in current_failed:
                        failed_key = next_key_f.to_bytes(4, 'big')
                        txn.put(failed_key, safe_pickle(failed_record), db=failed_records_db)
                        next_key_f += 1
            
            print(f"Successfully flushed {len(current_bookmarks)} bookmarks and {len(current_failed)} failed records to LMDB.")
        except Exception as e:
            logger.error(f"Error during periodic flush: {e}")

    def _crawl_bookmark(args):
        """Wrapper function to perform URL deduplication and crawl in a single thread task."""
        bookmark, idx, total_count, min_delay, max_delay, no_fetch = args
        
        if shutdown_flag:
            return None, None

        url = bookmark['url']
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        
        try:
            with lmdb_env.begin(write=True) as txn:
                if txn.get(url_hash.encode('utf-8'), db=url_hashes_db):
                    worker_id = threading.get_ident()
                    print(f"[{worker_id}] Skipping duplicate URL [{idx+1}/{total_count}]: {bookmark.get('name', 'No Title')} - {url}")
                    return "skipped", None
                txn.put(url_hash.encode('utf-8'), b'1', db=url_hashes_db)
        except Exception as e:
            logger.error(f"Error during URL deduplication check in worker: {e}")
            return None, {"url": url, "title": bookmark.get("name", "No Title"), "reason": "Deduplication check failed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

        return fetch_webpage_content(bookmark, idx+1, total_count, min_delay, max_delay, no_fetch=no_fetch)


    start_time = time.time()
    total_count = len(bookmarks_to_process)
    print(f"Starting parallel crawl of bookmark content, max workers: {max_workers}, total: {total_count}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_crawl_bookmark, (bookmark, idx + 1, total_count, min_delay, max_delay, no_fetch)): bookmark for idx, bookmark in enumerate(bookmarks_to_process)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Crawl Progress"):
            if shutdown_flag or (limit and new_bookmarks_added >= limit):
                print("Limit reached or shutdown signal received, cancelling remaining tasks...")
                for f in futures: f.cancel()
                break
            
            bookmarks_to_flush = None
            failed_to_flush = None
            try:
                result, failed_info = future.result()

                if result == "skipped":
                    skipped_url_count += 1
                    continue
                
                with bookmarks_lock:
                    if result:
                        all_bookmarks_with_content.append(result)
                        bookmarks_batch.append(result)
                        new_bookmarks_added += 1
                    if failed_info:
                        if not skip_unreachable:
                            original_bookmark = futures[future]
                            error_bookmark = original_bookmark.copy()
                            error_bookmark["error"] = failed_info["reason"]
                            error_bookmark["crawl_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                            all_bookmarks_with_content.append(error_bookmark)
                            bookmarks_batch.append(error_bookmark)
                            new_bookmarks_added += 1
                        all_failed_records.append(failed_info)
                        failed_records_batch.append(failed_info)
                    
                    if time.time() - last_flush_time >= flush_interval:
                        print(f"Flush interval of {flush_interval} seconds reached. Flushing data to disk...")
                        bookmarks_to_flush = list(bookmarks_batch)
                        failed_to_flush = list(failed_records_batch)
                        bookmarks_batch.clear()
                        failed_records_batch.clear()
                        last_flush_time = time.time()

            except Exception as e:
                logger.error(f"An error occurred while processing a future: {e}")
            else:
                if bookmarks_to_flush is not None:
                    flush_to_disk(bookmarks_to_flush, failed_to_flush)

    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed_time = end_time - start_time
    print(f"Total time for parallel bookmark crawl: {elapsed_time:.2f} seconds")
    
    # Final flush for any remaining items in the batch
    final_bookmarks_to_flush = None
    final_failed_to_flush = None
    with bookmarks_lock:
        if bookmarks_batch or failed_records_batch:
            print("Performing final flush to LMDB...")
            final_bookmarks_to_flush = list(bookmarks_batch)
            final_failed_to_flush = list(failed_records_batch)
            # Clear temporary lists containing the current batch of records that have now been flushed to the db, so we clean to ensure we don't write the same data multiple times. The main result lists are preserved and returned correctly in all_bookmarks_with_content and all_failed_records.
            bookmarks_batch.clear()
            failed_records_batch.clear()

    if final_bookmarks_to_flush:
        flush_to_disk(final_bookmarks_to_flush, final_failed_to_flush)
    
    return all_bookmarks_with_content, all_failed_records, new_bookmarks_added

# Parse command-line arguments
def parse_args():
    # Create the argument parser with a clear description of the application's purpose.
    parser = argparse.ArgumentParser(description='Crawl browser bookmarks and build a knowledge base')

    # Argument for limiting the number of bookmarks to process.
    parser.add_argument('--limit', type=int, help='Limit the number of bookmarks to process (0 for no limit)')

    # Argument for setting the number of concurrent workers for parallel fetching.
    parser.add_argument('--workers', type=int, help='Number of worker threads for parallel fetching')

    # Flag to skip the summary generation step, useful for content fetching only.
    parser.add_argument('--no-summary', action='store_true', help='Skip the summary generation step')

    # Flag to skip fetching full-text content, only record titles and URLs.
    parser.add_argument('--no-fetch', action='store_true', help='Skip fetching full-text content, only record titles and URLs')

    # Flag to generate summaries from an existing content file, skipping the crawl.
    parser.add_argument('--from-json', action='store_true', help='Generate summaries from existing bookmarks_with_content.json')

    # Add optional command-line argument to specify a custom browser.
    # This allows the application to read bookmarks from a specific browser.
    parser.add_argument(
        '--browser',
        '-b',
        type=str,
        choices=['chrome', 'firefox', 'edge', 'opera', 'opera_gx', 'safari', 'vivaldi', 'brave'],
        help='Specify the browser to fetch bookmarks from. If not specified, fetches from all browsers.'
    )

    # Add optional command-line argument to specify a custom profile path.
    # This allows the application to read bookmarks from a specific profile directory.
    parser.add_argument(
        '--profile-path',
        type=str,
        help='Specify a custom path to the browser profile directory. Used in conjunction with --browser.'
    )

    # Add optional command-line argument to specify a custom config file path.
    parser.add_argument(
        '--config',
        type=str,
        default='default_config.toml',
        help='Path to the TOML configuration file (default: default_config.toml)'
    )

    # Add --rebuild argument to rebuild the entire index from scratch
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild the entire index from scratch instead of resuming from existing bookmarks_with_content.json'
    )

    # Add --flush-interval argument to control the interval for flushing to disk
    parser.add_argument(
        '--flush-interval',
        type=int,
        default=60,
        help='Interval in seconds for flushing to disk to save intermediate results (default: 60)'
    )

    # Add --force-recompute-summaries argument to force regeneration of all summaries
    parser.add_argument(
        '--force-recompute-summaries',
        action='store_true',
        help='Force recomputation of summaries for all bookmarks, overriding the default skip behavior for existing summaries'
    )

    # Add --skip-unreachable argument to control saving of unreachable bookmarks
    parser.add_argument(
        '--skip-unreachable',
        action='store_true',
        help='Skip saving unreachable bookmarks. When not provided, unreachable bookmarks are saved with an "error" field containing the error message.'
    )

    # Add LMDB configuration arguments
    parser.add_argument(
        '--lmdb-map-size',
        type=int,
        help=f'Size of LMDB memory map in bytes (default: {DEFAULT_LMDB_MAP_SIZE})'
    )
    parser.add_argument(
        '--lmdb-max-dbs',
        type=int,
        help=f'Maximum number of LMDB named databases (default: {DEFAULT_LMDB_MAX_DBS})'
    )
    parser.add_argument(
        '--lmdb-readonly',
        action='store_true',
        help='Open LMDB database in read-only mode for concurrent access'
    )
    parser.add_argument(
        '--lmdb-resize-threshold',
        type=float,
        default=0.8,
        help='Threshold for triggering LMDB resize (0.0-1.0, default: 0.8)'
    )
    parser.add_argument(
        '--lmdb-growth-factor',
        type=float,
        default=2.0,
        help='Growth factor for LMDB resize (default: 2.0)'
    )

    # Add backup control arguments
    parser.add_argument(
        '--enable-backup',
        action='store_true',
        help='Enable automatic LMDB database backup before write operations (default: enabled)'
    )
    parser.add_argument(
        '--disable-backup',
        action='store_true',
        help='Disable automatic LMDB database backup before write operations'
    )
    parser.add_argument(
        '--backup-dir',
        type=str,
        default=BACKUP_BASE_DIR,
        help=f'Directory for LMDB backups (default: {BACKUP_BASE_DIR})'
    )
    parser.add_argument(
        '--backup-on-failure-stop',
        action='store_true',
        help='Stop execution if backup fails instead of continuing (default: continue on failure)'
    )

    # Add delay control arguments
    parser.add_argument(
        '--min-delay',
        type=float,
        default=1.0,
        help='Minimum delay in seconds between requests (default: 1.0)'
    )
    parser.add_argument(
        '--max-delay',
        type=float,
        default=5.0,
        help='Maximum delay in seconds between requests (default: 5.0)'
    )

    # Add custom parsers filter argument
    # Get list of available parsers for help message
    available_parsers = []
    parsers_dir = get_custom_parsers_dir()
    if os.path.exists(parsers_dir):
        available_parsers = [f[:-3] for f in os.listdir(parsers_dir) 
                            if f.endswith('.py') and not f.startswith('__')]
        available_parsers.sort()
    
    parsers_help = 'Pipe-delimited list of custom parser filenames (without .py extension) to enable. If not specified, all parsers are loaded.'
    if available_parsers:
        parsers_help += f' Available parsers: {", ".join(available_parsers)}. Example: --parsers "youtube|zhihu"'
    
    parser.add_argument(
        '--parsers',
        type=str,
        help=parsers_help
    )

    return parser.parse_args()

# Main function to orchestrate the bookmark crawling and summarization process.
def main():
    # Declare global variables used in this function
    global use_fallback, lmdb_env, url_hashes_db, content_hashes_db, bookmarks_db, failed_records_db, url_to_key_db, domain_index_db, date_index_db

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Prepare the WebDriver in the main thread before starting parallel operations
    prepare_webdriver()

    # Parse command-line arguments
    args = parse_args()

    # Load custom parsers at startup
    global custom_parsers
    # Parse the parsers filter if provided
    parser_filter = None
    if args.parsers:
        parser_filter = [p.strip() for p in args.parsers.split('|') if p.strip()]
        print(f"Custom parser filter enabled: {parser_filter}")
    custom_parsers = load_custom_parsers(parser_filter=parser_filter)

    # Load TOML configuration
    config_data = load_config(args.config)

    # Read configuration from TOML file, command-line arguments take precedence
    bookmark_limit = args.limit if args.limit is not None else 0  # Default: no limit
    max_workers = args.workers if args.workers is not None else 20  # Default: 20 worker threads
    generate_summary_flag = not args.no_summary  # Command-line flag overrides config
    flush_interval = args.flush_interval  # Interval for flushing to disk
    min_delay = args.min_delay  # Minimum delay between requests
    max_delay = args.max_delay  # Maximum delay between requests

    # Initialize LMDB for persistent storage with configurable settings
    # Command-line arguments take precedence over environment variables
    lmdb_map_size = args.lmdb_map_size or int(os.environ.get('LMDB_MAP_SIZE', DEFAULT_LMDB_MAP_SIZE))
    lmdb_max_dbs = args.lmdb_max_dbs or int(os.environ.get('LMDB_MAX_DBS', DEFAULT_LMDB_MAX_DBS))
    lmdb_readonly = args.lmdb_readonly or bool(os.environ.get('LMDB_READONLY', False))
    lmdb_resize_threshold = args.lmdb_resize_threshold
    lmdb_growth_factor = args.lmdb_growth_factor

    # Configure backup settings
    global BACKUP_BASE_DIR
    BACKUP_BASE_DIR = args.backup_dir
    enable_backup = not args.disable_backup  # Default to enabled unless explicitly disabled
    if args.enable_backup:
        enable_backup = True  # Explicitly enabled
    backup_continue_on_failure = not args.backup_on_failure_stop

    init_lmdb(map_size=lmdb_map_size, max_dbs=lmdb_max_dbs, readonly=lmdb_readonly,
              resize_threshold=lmdb_resize_threshold, growth_factor=lmdb_growth_factor)

    # Load existing bookmarks from LMDB if not rebuilding from scratch
    existing_bookmarks = []
    if not args.rebuild:
        existing_bookmarks = safe_lmdb_operation(
            lambda txn: [pickle.loads(bookmark_bytes) for key_bytes, bookmark_bytes in txn.cursor(bookmarks_db)] if lmdb_env is not None else [],
            lambda: fallback_bookmarks.copy(),
            "loading existing bookmarks"
        )
        if existing_bookmarks is None:
            existing_bookmarks = []
        print(f"Loaded {len(existing_bookmarks)} existing bookmarks from LMDB")

        # Backup before any write operations if enabled
        if enable_backup and existing_bookmarks:
            if not safe_backup_operation("pre_crawl_backup", backup_continue_on_failure):
                print("Backup failed and configured to stop on failure. Exiting.")
                return

        # Populate LMDB deduplication databases with existing data
        try:
            with lmdb_env.begin(write=True) as txn:
                for bookmark in existing_bookmarks:
                    url = bookmark.get('url')
                    if url:
                        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                        txn.put(url_hash.encode('utf-8'), b'1', db=url_hashes_db)
                        # Populate URL to key mapping for O(1) flush lookups
                        # Note: We use the URL itself as key for simplicity, but could use hash if needed
                        # Find the key for this bookmark
                        cursor = txn.cursor(bookmarks_db)
                        for key_bytes, bookmark_bytes in cursor:
                            if pickle.loads(bookmark_bytes) == bookmark:
                                txn.put(url.encode('utf-8'), key_bytes, db=url_to_key_db)
                                break
                    content = bookmark.get('content')
                    if content:
                        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                        txn.put(content_hash.encode('utf-8'), b'1', db=content_hashes_db)
            print(f"Populated LMDB deduplication databases: URLs, content hashes, URL mappings")
        except Exception as e:
            logger.error(f"Error populating deduplication databases: {e}")
            use_fallback = True
            # Populate fallback structures
            for bookmark in existing_bookmarks:
                url = bookmark.get('url')
                if url:
                    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                    fallback_url_hashes.add(url_hash)
                content = bookmark.get('content')
                if content:
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    fallback_content_hashes.add(content_hash)
            print(f"Populated fallback deduplication structures: {len(fallback_url_hashes)} URLs, {len(fallback_content_hashes)} content hashes")
    else:
        print("Rebuilding from scratch (--rebuild flag used)")
        # Clear existing deduplication databases for rebuild
        try:
            with lmdb_env.begin(write=True) as txn:
                txn.drop(url_hashes_db)
                txn.drop(content_hashes_db)
                txn.drop(bookmarks_db)
                txn.drop(failed_records_db)
                txn.drop(url_to_key_db)
                txn.drop(domain_index_db)
                txn.drop(date_index_db)
            # Re-open databases after clearing
            url_hashes_db = lmdb_env.open_db(b'url_hashes')
            content_hashes_db = lmdb_env.open_db(b'content_hashes')
            bookmarks_db = lmdb_env.open_db(b'bookmarks')
            failed_records_db = lmdb_env.open_db(b'failed_records')
            url_to_key_db = lmdb_env.open_db(b'url_to_key')
            domain_index_db = lmdb_env.open_db(b'domain_index')
            date_index_db = lmdb_env.open_db(b'date_index')
            print("Cleared existing LMDB databases for rebuild")
        except Exception as e:
            logger.error(f"Error clearing LMDB databases for rebuild: {e}")
            use_fallback = True
            # Clear fallback structures
            fallback_url_hashes.clear()
            fallback_content_hashes.clear()
            fallback_bookmarks.clear()
            fallback_failed_records.clear()
            print("Cleared fallback structures for rebuild")

    # If the --from-json argument is used, read directly from LMDB and generate summaries
    if args.from_json:
        print("Generating summaries from existing bookmarks in LMDB...")

        # Backup before summary generation write operations if enabled
        if enable_backup:
            if not safe_backup_operation("pre_summary_generation_backup", backup_continue_on_failure):
                print("Backup failed and configured to stop on failure. Exiting.")
                return

        try:
            bookmarks_with_content = safe_lmdb_operation(
                lambda txn: [pickle.loads(bookmark_bytes) for key_bytes, bookmark_bytes in txn.cursor(bookmarks_db)] if lmdb_env is not None else [],
                lambda: fallback_bookmarks.copy(),
                "loading bookmarks for summary generation"
            )
            if bookmarks_with_content is None:
                bookmarks_with_content = []

            if not bookmarks_with_content:
                print("Error: LMDB bookmarks database is empty")
                return

            if bookmark_limit > 0:
                print(f"Processing only the first {bookmark_limit} bookmarks based on limit")
                bookmarks_with_content = bookmarks_with_content[:bookmark_limit]

            # Configure model and generate summaries
            model_config = ModelConfig(config_data)

            # Test API connection
            if not test_api_connection(model_config):
                print("LLM API connection failed, please check configuration and try again.", model_config.api_base, model_config.model_name, model_config.api_key, model_config.model_type)
                return

            # Generate summaries for content, respecting the force recompute flag
            bookmarks_with_content = generate_summaries_for_bookmarks(bookmarks_with_content, model_config, args.force_recompute_summaries)

            print(f"Summary generation complete, LMDB updated with {len(bookmarks_with_content)} bookmarks")
            return

        except Exception as e:
            print(f"Error processing LMDB data: {str(e)}")
            return

    # Original crawling logic
    print(f"Configuration:")
    print(f"  - Browser: {args.browser if args.browser else 'All browsers'}")
    print(f"  - Profile Path: {args.profile_path if args.profile_path else 'Default'}")
    print(f"  - Bookmark Limit: {bookmark_limit if bookmark_limit > 0 else 'No Limit'}")
    print(f"  - Parallel Workers: {max_workers}")
    print(f"  - Generate Summary: {'Yes' if generate_summary_flag else 'No'}")
    print(f"  - LMDB Backup: {'Enabled' if enable_backup else 'Disabled'}")
    if enable_backup:
        print(f"  - Backup Directory: {BACKUP_BASE_DIR}")
        print(f"  - Stop on Backup Failure: {'Yes' if not backup_continue_on_failure else 'No'}")

    # Get bookmark data
    bookmarks = get_bookmarks(browser=args.browser, profile_path=args.profile_path)

    # Filter bookmarks: remove empty URLs, 10.0. network URLs, and non-qualifying types
    filtered_bookmarks = []
    for bookmark in bookmarks:
        url = bookmark["url"]
        # Check for empty URL, URL type, not "Extension" name, and not 10.0. network URL
        if (url and
            bookmark["type"] == "url" and
            bookmark["name"] != "扩展程序" and # "扩展程序" is a folder name for extensions in Chinese Chrome
            not re.match(r"https?://10\.0\.", url)):
            # If not rebuilding, skip URLs already in LMDB bookmarks
            if not args.rebuild:
                url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
                try:
                    with lmdb_env.begin() as txn:
                        if txn.get(url_hash.encode('utf-8'), db=url_hashes_db):
                            try:
                                print(f"Skipping already indexed URL: {bookmark.get('name', 'No Title')} - {url}")
                            except UnicodeEncodeError:
                                print(f"Skipping already indexed URL: {bookmark.get('name', 'No Title').encode('ascii', 'replace').decode('ascii')} - {url}")
                            continue
                except Exception as e:
                    logger.error(f"Error checking URL deduplication: {e}")
                    if use_fallback and url_hash in fallback_url_hashes:
                        try:
                            print(f"Skipping already indexed URL (fallback): {bookmark.get('name', 'No Title')} - {url}")
                        except UnicodeEncodeError:
                            print(f"Skipping already indexed URL (fallback): {bookmark.get('name', 'No Title').encode('ascii', 'replace').decode('ascii')} - {url}")
                        continue
            filtered_bookmarks.append(bookmark)
    
    # Save filtered bookmark data
    with open(bookmarks_path, "w", encoding="utf-8") as output_file:
        json.dump(filtered_bookmarks, output_file, ensure_ascii=False, indent=4)
    
    # Parallel crawl bookmark content
    bookmarks_with_content, failed_records, skipped_url_count = parallel_fetch_bookmarks(
        filtered_bookmarks,
        max_workers=max_workers,
        limit=bookmark_limit if bookmark_limit > 0 else None,
        flush_interval=flush_interval,
        skip_unreachable=args.skip_unreachable,
        min_delay=min_delay,
        max_delay=max_delay,
        no_fetch=args.no_fetch
    )
    
    # Only execute the following code if summary generation is enabled
    if generate_summary_flag and bookmarks_with_content:
        # Configure model
        model_config = ModelConfig(config_data)

        # Test API connection
        if not test_api_connection(model_config):
            print("LLM API connection failed, please check configuration and try again.", model_config.api_base, model_config.model_name, model_config.api_key, model_config.model_type)
            print("Skipping summary generation step...")
        else:
            # Generate summaries for the crawled content, respecting the force recompute flag
            bookmarks_with_content = generate_summaries_for_bookmarks(bookmarks_with_content, model_config, args.force_recompute_summaries)
    elif not generate_summary_flag:
        print("Skipping summary generation step based on configuration...")

    # All bookmarks are already stored in LMDB via periodic flushes
    # Just ensure final consistency and provide summary
    try:
        bookmarks_with_content = safe_lmdb_operation(
            lambda txn: [pickle.loads(bookmark_bytes) for key_bytes, bookmark_bytes in txn.cursor(bookmarks_db)],
            lambda: fallback_bookmarks.copy(),
            "retrieving final bookmarks list"
        )
        if bookmarks_with_content is None:
            bookmarks_with_content = []
        print(f"LMDB contains {len(bookmarks_with_content)} total bookmarks")
    except Exception as e:
        logger.error(f"Error during final summary: {e}")
        bookmarks_with_content = fallback_bookmarks.copy() if use_fallback else []
        print(f"Fallback contains {len(bookmarks_with_content)} total bookmarks")

    # Save failed URLs and reasons (keeping JSON format for compatibility)
    try:
        failed_records_list = safe_lmdb_operation(
            lambda txn: [pickle.loads(record_bytes) for key_bytes, record_bytes in txn.cursor(failed_records_db)],
            lambda: fallback_failed_records.copy(),
            "retrieving failed records list"
        )
        if failed_records_list is None:
            failed_records_list = []
        with open(failed_urls_path, "w", encoding="utf-8") as f:
            json.dump(failed_records_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving failed URLs: {e}")
        # Try to save fallback data
        try:
            with open(failed_urls_path, "w", encoding="utf-8") as f:
                json.dump(fallback_failed_records, f, ensure_ascii=False, indent=4)
        except Exception as fallback_e:
            logger.error(f"Error saving fallback failed URLs: {fallback_e}")
    
    print(f"Extracted {len(filtered_bookmarks)} valid bookmarks, saved to {bookmarks_path}")
    print(f"Successfully crawled content for {len(bookmarks_with_content)} bookmarks, saved to {lmdb_storage_path}")
    print(f"Skipped {skipped_url_count} duplicate URLs during crawling")
    print(f"Failed to crawl {len(failed_records)} URLs, details saved to {failed_urls_path}")
    
    # Print list of failed URLs and titles for easy viewing
    if failed_records:
        print("\nFailed URLs and Titles:")
        for idx, record in enumerate(failed_records):
            print(f"{idx+1}. {record.get('title', 'No Title')} - {record['url']} - Reason: {record['reason']}")
    elif use_fallback and fallback_failed_records:
        print("\nFailed URLs and Titles (from fallback):")
        for idx, record in enumerate(fallback_failed_records):
            print(f"{idx+1}. {record.get('title', 'No Title')} - {record['url']} - Reason: {record['reason']}")
    
    # Display content length statistics from LMDB
    if bookmarks_with_content:
        try:
            total_length = sum(b.get("content_length", 0) for b in bookmarks_with_content)
            avg_length = total_length / len(bookmarks_with_content)
            print(f"Average crawled content length: {avg_length:.2f} characters")
            print(f"Longest content: {max(b.get('content_length', 0) for b in bookmarks_with_content)} characters")
            print(f"Shortest content: {min(b.get('content_length', 0) for b in bookmarks_with_content)} characters")

            # Statistics on crawl methods used
            selenium_count = sum(1 for b in bookmarks_with_content if b.get("crawl_method") == "selenium")
            requests_count = sum(1 for b in bookmarks_with_content if b.get("crawl_method") == "requests")
            print(f"Crawled using Selenium: {selenium_count} items")
            print(f"Crawled using Requests: {requests_count} items")
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
    elif use_fallback:
        print("Using fallback mode - statistics not available")

    # Cleanup LMDB resources
    cleanup_lmdb()

    # Log final status and backup summary
    if enable_backup:
        logger.info("LMDB backup functionality was enabled during this run")
        # Count existing backups
        try:
            if os.path.exists(BACKUP_BASE_DIR):
                backup_count = len([d for d in os.listdir(BACKUP_BASE_DIR) if os.path.isdir(os.path.join(BACKUP_BASE_DIR, d)) and d.startswith('lmdb_backup_')])
                logger.info(f"Total LMDB backups available: {backup_count}")
        except Exception as e:
            logger.warning(f"Could not count existing backups: {e}")
    else:
        logger.info("LMDB backup functionality was disabled for this run")

    if use_fallback:
        logger.warning("Script completed using fallback in-memory structures due to LMDB issues")
    else:
        logger.info("Script completed successfully with LMDB persistence")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()