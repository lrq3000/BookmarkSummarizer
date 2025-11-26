# TODO

* [ ] Finish implementing recommendations in docs/memory_test_report.md (for the moment we implemented 1. Fix content deduplication logic, 2. Resolve recursion errors in flush operations, 3. Implement comprehensive error handling).
* [ ] Easy multi platforms installers (maybe via pyInstaller?) to easy install for non Python developers.
* [ ] Make a user-friendly GUI. Maybe just a simple one that wraps around argparse and expose the flags as GUI widgets. For both crawl.py (the most complex) to fuzzy search py.
* [ ] Clean up the console output, make the verbosity hidden by default but displayable with an argparse flag.
* [ ] restore tqdm progress bar.
* [ ] Interleave multiple different websites during fetching to avoid querying the same website too fast, this is a much faster alternative to semi-random delay, but requires planification over the whole set of bookmarks. Maybe with a system of queue, but it would be very complex to set given there are parallel workers.


