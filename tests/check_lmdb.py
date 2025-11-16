#!/usr/bin/env python3
"""
Quick script to check LMDB database contents.
"""

import lmdb
import json

def check_lmdb():
    env = lmdb.open('./bookmark_index.lmdb', max_dbs=5)
    txn = env.begin()
    bookmarks_db = env.open_db(b'bookmarks')
    cursor = txn.cursor(bookmarks_db)
    count = 0
    sample_bookmark = None
    for key, value in cursor:
        count += 1
        if sample_bookmark is None:
            sample_bookmark = json.loads(value.decode('utf-8'))
    print(f'Found {count} bookmarks in LMDB')
    if sample_bookmark:
        print(f'Sample bookmark: {sample_bookmark.get("title", "No Title")}')
    env.close()

if __name__ == "__main__":
    check_lmdb()