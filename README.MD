# BookmarkSummarizer

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LLM-enabled-green.svg" alt="LLM">

  [![PyPI-Status][1]][2] [![PyPI-Versions][3]][2] [![PyPi-License][4]][2] [![PyPI-Downloads][5]][2]

  [![Build-Status][7]][8] [![Coverage-Status][9]][10]
</p>


BookmarkSummarizer is a powerful tool that crawls your browsers' bookmarks, generates summaries using large language models, and turns them into a personal knowledge base. Easily search and utilize all your bookmarked web resources without manual organization. Supports all common desktop browsers (Chrome, Firefox, Edge, Safari) as well as uncommon ones (Chromium, Brave, Vivaldi, Opera, etc).

<p align="right"><a href="../README-CN.MD">中文文档</a></p>

## ✨ Key Features

- 🔍 **Smart Bookmark Crawling**: Automatically extract content from your browsers' bookmarks by fetching the bookmarks' URLs webpages content.
- 🤖 **AI Summary Generation**: Create high-quality summaries for each bookmark using large language models
- 🚀 **Blazingly fast and scalable full-text fuzzy search**: Rocket fast fuzzy search indexing and retrieval based on Whoosh, supporting millions of bookmarks, all offline!
- 🔄 **Parallel Processing**: Efficient multi-threaded crawling to significantly reduce processing time
- 🌐 **Multiple Model Support**: Compatible with OpenAI, Deepseek, Qwen, and Ollama offline models
- 💾 **Incremental Update And Checkpoint Recovery**: Update the database with new bookmarks or continue processing after interruptions without losing completed work
- 📊 **Detailed Logging**: Clear progress and status reports for monitoring and debugging
- **Made to scale**: Start small with hundreds of bookmarks in a <10MB LMDB database, and with incremental updates you can scale to thousands of bookmarks of a few GB using just a fraction of the RAM thanks to the out-of-core database saved on-disk, up to millions of bookmarks with a LMDB database of several TBs using only a few GBs of memory to load during crawling. The fuzzy search engine further improves scaling by building another fuzzy search Whoosh database much smaller in size, so that searching bookmarks content, URL, titles or summaries is blazingly fast with negligible RAM footprint. 
- **Modular architecture**: custom parsers can be added without modifying the core logic by adding python files in custom_parsers. For example, custom parsers are provided to extract YouTube transcripts as content to summarize, and suspended tabs that got bookmarked are transparently unsuspended to fetch the true target page content.

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- Chrome browser
- Internet connection
- Large language model API key (optional)

### Installation

#### Portable binaries

Head to the [GitHub Releases](https://github.com/lrq3000/BookmarkSummarizer/releases) and pick the latest release, you will find precompiled binaries for Windows, MacOS and Linux.

#### From source

1. Clone the repository:
```bash
git clone https://github.com/lrq3000/BookmarkSummarizer.git
cd BookmarkSummarizer
```

2. Install dependencies:
```bash
pip install -e .
```

3. Make a TOML configuration file to finetune behavior (create a `.toml` file):
```
model_type=ollama  # options: openai, deepseek, qwen, ollama
api_key=your_api_key_here
api_base=http://localhost:11434  # ollama local endpoint or other model api address
model_name=qwen3:1.7b  # or other supported model
max_tokens=1000
temperature=0.3
```

### Usage

#### Fetch Bookmarks from Browsers

**Fetch bookmarks from all browsers** (default):
```bash
python index.py
```
This fetches bookmarks from all installed browsers (Chrome, Firefox, Edge, Safari, Opera, Brave, Vivaldi, etc.) using the browser-history module and saves them to `bookmarks.json`.

**Fetch bookmarks from a specific browser**:
```bash
python index.py --browser chrome
```
Supported browsers: `chrome`, `firefox`, `edge`, `opera`, `opera_gx`, `safari`, `vivaldi`, `brave`.

**Fetch bookmarks from a custom profile path**:
```bash
python index.py --browser chrome --profile-path "C:\Users\Username\AppData\Local\Google\Chrome\User Data\Profile 1"
```
This is useful when you have multiple Chrome profiles or custom browser installations.

#### Crawl and Summarize Bookmarks

**Basic usage (crawl and summarize from all browsers)**:
```bash
python crawl.py
```
This fetches bookmarks from all browsers, crawls their content, generates AI summaries, and saves the results. Use the same command to update crawled bookmarks incrementally or resume after interruptions - already processed bookmarks will be skipped.

**Crawl from a specific browser**:
```bash
python crawl.py --browser firefox
```
Fetches and crawls bookmarks only from Firefox.

**Crawl from a custom profile path**:
```bash
python crawl.py --browser chrome --profile-path "/home/user/.config/google-chrome/Profile 1"
```
Combines browser selection with custom profile path.

**Limit the number of bookmarks**:
```bash
python crawl.py --limit 10
```
Processes only the first 10 bookmarks.

**Set the number of parallel processing threads**:
```bash
python crawl.py --workers 10
```
Uses 10 worker threads for parallel crawling (default: 20).

**Skip summary generation**:
```bash
python crawl.py --no-summary
```
Crawls content but skips AI summary generation.

**Generate summaries from already crawled content**:
```bash
python crawl.py --from-json
```
Generates summaries for existing `bookmarks_with_content.json` without re-crawling.

#### Search Through Bookmarks

Once your bookmarks are crawled, a `bookmarks_with_content.json` file will be present in the current folder. Then you can search through it with a fuzzy search engine:

**Launch the search interface with index updates**:
```bash
python fuzzy_bookmark_search.py --update-index
```
This launches a local web server with the search engine accessible through http://localhost:8132/ (the port can be changed via `--port xxx`). The search engine uses Whoosh to build a fast, on-disk, fuzzy searchable index.

**Launch the search interface without updating the index**:
```bash
python fuzzy_bookmark_search.py
```
Uses the existing index without rebuilding it.

#### Output Files

- `bookmarks.json`: Filtered bookmark list from browsers, it is just a compilation of all bookmarks fetched directly from the browsers.
- `bookmark_index.lmdb`: Folder of bookmark data with crawled content and AI-generated summaries stored in a LMDB.
- `failed_urls.json`: URLs that failed to crawl with reasons.
- `crawl_errors.log`: Errors log for the crawler, this logs all errors even if not related to the unreachability of bookmarks' contents (eg, this logs software logic bugs).
- `whoosh_index/`: Directory containing the Whoosh search index files for the seach engine.

## 📋 Detailed Features

### Bookmark Crawling

BookmarkSummarizer automatically reads all bookmarks from the Chrome bookmarks file and intelligently filters out ineligible URLs. It uses two strategies to crawl web content:

1. **Regular Crawling**: Uses the Requests library to capture content from most web pages
2. **Dynamic Content Crawling**: For dynamic webpages (such as Zhihu and other platforms), automatically switches to Selenium
3. **Modular architecture with custom parsers** : For specific websites or content such as YouTube, custom parsers / adapters can be implemented in `custom_parsers/` as separate `.py` modules that will be automatically called to filter and process every bookmarks. The custom parsers get a full copy of the bookmark's metadata and can choose to filter based on any criterion, not only the URL, but content based or title based, etc. For example, for YouTube, the transcript is downloaded to be the content for summarization.

### Summary Generation

BookmarkSummarizer uses advanced large language models to generate high-quality summaries for each bookmark content, including:

- Extracting key information and important concepts
- Preserving technical terms and key data
- Generating structured summaries for easier retrieval
- Supporting various mainstream large language models
- Supportign 100% offline generation via ollama for complete privacy

Tip: if ollama is used, it is advised to set the context window to 128k and use a model that supports such a wide context window such as qwen3:4b (supports 256k context!) or qwen3:1.7b or qwen3:0.6b (40k context) for less power machines, so that summaries are done on the whole bookmark's full-text content without truncation. `gemma3:1b` can also be interesting (32k context) but it has hallucination issues when there is not much full-text content.

### Checkpoint Recovery

- Saves progress immediately after processing each bookmark
- Automatically skips previously processed bookmarks when restarted
- Ensures data safety even when processing large numbers of bookmarks

## 📁 Output Files

- `bookmarks.json`: Filtered bookmark list
- `bookmarks_with_content.json`: Bookmark data with content and summaries
- `failed_urls.json`: Failed URLs and reasons

## 🔧 Custom Configuration

In addition to command-line parameters, you can set the following parameters through a `.toml` configuration file:

```
# model type settings
model_type=ollama  # openai, deepseek, qwen, ollama
api_key=your_api_key_here
api_base=http://localhost:11434
model_name=gemma3:1b

# content processing settings
max_tokens=1024  # maximum number of tokens for summary generation
max_input_content_length=6000  # maximum length of input content
temperature=0.3  # randomness of summary generation

# crawler settings
bookmark_limit=0  # no limit by default
max_workers=20  # number of parallel worker threads
generate_summary=true  # whether to generate summaries
```

## 🤝 Contributing

Pull Requests are welcome! For any issues or suggestions, please create an Issue.

## Author

Originally created by [wyj/sologuy](https://github.com/sologuy/BookmarkSummarizer/).

Development of new features and maintenance is done since Novembre 2025 by [Stephen Karl Larroque](https://github.com/lrq3000/BookmarkSummarizer/).

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## Suggested complementary 3rd-party bookmarks tools

Here is a non-exhaustive list of complementary **opensource** 3rd-party extensions or tools that can complement BookmarkSummarizer:
* [Search Bookmarks, History and Tabs](https://github.com/Fannon/search-bookmarks-history-and-tabs): Fast bookmarks fuzzy search engine on URL and bookmark's title (not the full-page content). Chrome extension.
* [Full text tabs forever (FTTF)](https://github.com/iansinnott/full-text-tabs-forever): Full-text search of historically visited pages. This has the advantage of causing no network overhead (no additional HTTP request is done, the pages you access are indexed on-the-fly), hence no risk of rate limiting/IP banning. Chrome extension.
* [Floccus](https://github.com/floccusaddon/floccus): Autosync bookmarks (and hence sessions if using InfiniTabs) between browsers (also works on mobile via native Floccus app on F-Droid or [Mises](https://github.com/mises-id/mises-browser-core) or [Cromite](https://github.com/uazo/cromite/)). Chrome extension.
* [TidyMark](https://github.com/PanHywel/TidyMark): Reorganize/group bookmarks (supports cloud or offline ollama). Chrome extension.
* [Wherewasi](https://github.com/Jay-Karia/wherewasi): Temporal and semantic tabs clustering into sessions using cloud Gemini AI. Chrome extension.
* LinkWarden or ArchiveBox: alternatives to BookmarkSummarizer to index/archive the full-text content pointed at by the bookmarks.


[1]: https://img.shields.io/pypi/v/BookmarkSummarizer.svg
[2]: https://pypi.org/project/BookmarkSummarizer
[3]: https://img.shields.io/pypi/pyversions/BookmarkSummarizer.svg?logo=python&logoColor=white
[4]: https://img.shields.io/pypi/l/BookmarkSummarizer.svg
[5]: https://img.shields.io/pypi/dm/BookmarkSummarizer.svg?label=pypi%20downloads&logo=python&logoColor=white
[7]: https://github.com/lrq3000/BookmarkSummarizer/actions/workflows/ci-build.yml/badge.svg?event=push
[8]: https://github.com/lrq3000/BookmarkSummarizer/actions/workflows/ci-build.yml
[9]: https://codecov.io/gh/lrq3000/BookmarkSummarizer/graph/badge.svg?token=NuNgXwZqAO
[10]: https://codecov.io/gh/lrq3000/BookmarkSummarizer
