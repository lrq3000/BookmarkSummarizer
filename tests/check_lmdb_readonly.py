#!/usr/bin/env python3
"""
Quick script to check LMDB database contents in readonly mode.
"""

import lmdb
import json

def check_lmdb():
    env = lmdb.open('./bookmark_index.lmdb', max_dbs=5, readonly=True)
    txn = env.begin()
    bookmarks_db = env.open_db(b'bookmarks')
    cursor = txn.cursor(bookmarks_db)
    count = 0
    for key, value in cursor:
        count += 1
    print(f'Found {count} bookmarks in LMDB')
    env.close()

if __name__ == "__main__":
    check_lmdb()