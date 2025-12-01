#!/usr/bin/env python3
"""
Quick script to check LMDB database contents.
"""

import lmdb
import pickle

def check_lmdb():
    try:
        env = lmdb.open('./bookmark_index.lmdb', max_dbs=7, readonly=True)
        print("LMDB env opened successfully")
        stat = env.stat()
        print(f"LMDB stat: {stat}")
        info = env.info()
        print(f"LMDB info: {info}")

        txn = env.begin()
        print("Transaction begun")

        # Try to open db
        try:
            bookmarks_db = env.open_db(b'bookmarks')
            print("Bookmarks db opened")
        except Exception as e:
            print(f"Failed to open bookmarks db: {e}")
            env.close()
            return

        cursor = txn.cursor(bookmarks_db)
        count = 0
        sample_bookmark = None
        corrupted = 0
        for key, value in cursor:
            count += 1
            try:
                bookmark = pickle.loads(value)
                if sample_bookmark is None:
                    sample_bookmark = bookmark
            except Exception as e:
                print(f"Corrupted entry at count {count}, key: {key[:20]}..., error: {e}")
                corrupted += 1
                if corrupted > 5:
                    break  # Stop after 5 corrupted
        print(f'Found {count} bookmarks in LMDB, {corrupted} corrupted')
        if sample_bookmark:
            print(f'Sample bookmark: {sample_bookmark.get("title", "No Title")}')
        env.close()
    except Exception as e:
        print(f"Error opening LMDB: {e}")

if __name__ == "__main__":
    check_lmdb()