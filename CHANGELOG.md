# Changelog

所有对 BookmarkSummarizer 项目的显著更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 0.4.1

Refactored crawl.py for parallel processing.

There was an intentionally sequential path that was triggered when a --limit was set, which was the primary cause of the non-parallel behavior. It was replaced with a single, unified parallel implementation that now correctly handles both limited and unlimited crawls.

*   **Parallel Bookmark Processing:** The processing logic now resides in the `_crawl_bookmark` worker function, which is called for every bookmark within the `ThreadPoolExecutor`. This ensures all bookmarks are processed concurrently.
*   **Partial Flushing:** The periodic flushing is handled within the main `for future in as_completed(futures):` loop. It checks the time elapsed since the last flush and writes the latest batch of results to disk, preserving the exact same data-saving functionality as before.

### 0.3.1

Big bundle of updates, with various new features and bugfixes:
* Translates the whole project from Chinese to English, including the summarization prompt, but language autodetection was added so that the summary is in the webpage's content language.
* Add support for other browsers, and in addition, bookmarks are by default imported from all installed browsers (hence we import from multiple browsers at once). A single browser can still be specified using an argument.
* Add a very fast fuzzy search engine with a GUI web app with pagination support. It is blazingly fast and scalable both for the indexing and lookup, it is intended to scale to millions of bookmarks, everything is stored on-disk so RAM is not an issue.
* Indexing resuming and deduplication (also implemented for summarization) and atomic intermediate flushing, so we can do incremental updates of the database or interrupt and continue. This is especially important for those with a LOT of bookmarks (like me! Because I use bookmarks as a past browsing sessions saver/dump).
* Pythonic packaging pyproject.toml, so this app can be published on pypi and easily installed through pip install.
* CLI entrypoints are created on pip install for the main scripts: index.py, crawl.py and fuzzy_bookmark_search.py.
* A LMDB database for the content crawling and the summaries, and a Whoosh database for fast fuzzy searching. Both databases scale dynamically along with the number of bookmarks (the crawling database is multiplied by 2 in size each time the bookmarks' content reach too close to the database total size). The LMDB is out-of-core, so it is extremely scalable as it can grow in size much beyond the current RAM available on the user's system, and only a fraction of RAM is necessary to create a view to access the LMDB, so the RAM footprint remains very minimal (a few dozens to hundreds of MB) even when the database is dozens of GB (and a few GB RAM to access a multi-TB database).
* Changed the default settings for the summaries to use ollama and qwen3:1.7b, it is very effective. Alternatively, qwen3:0.6b produces acceptable summaries too albeit less accurate and with a shorter context window.
* Modular architecture: custom parsers can be added without modifying the core logic by adding python files in custom_parsers. For example, custom parsers are provided to extract YouTube transcripts as content to summarize, and suspended tabs that got bookmarked are transparently unsuspended to fetch the true target page content.
* A lot of bugfixes here and there, and additional verbose outputs.

### 新增
- 初始版本开发
- 支持从 Chrome 书签提取 URL
- 多线程爬取书签内容
- 支持通过大语言模型生成摘要
- 支持 OpenAI、Deepseek、Qwen 和 Ollama 模型
- 断点续传功能
- 命令行参数支持
- 环境变量配置支持

### 修复
- 无

### 变更
- 无

### 已弃用
- 无

### 已移除
- 无

### 安全
- 无 