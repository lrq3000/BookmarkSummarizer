# BookmarkSummarizer

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LLM-enabled-green.svg" alt="LLM">
</p>

BookmarkSummarizer is a powerful tool that crawls your browsers' bookmarks, generates summaries using large language models, and turns them into a personal knowledge base. Easily search and utilize all your bookmarked web resources without manual organization. Supports all common desktop browsers (Chrome, Firefox, Edge, Safari) as well as uncommon ones (Chromium, Brave, Vivaldi, Opera, etc).

<p align="right"><a href="../README-CN.MD">中文文档</a></p>

## ✨ Key Features

- 🔍 **Smart Bookmark Crawling**: Automatically extract content from Chrome bookmarks by fetching the bookmarks' URLs webpages content.
- 🤖 **AI Summary Generation**: Create high-quality summaries for each bookmark using large language models
- 🚀 **Blazingly fast and scalable full-text fuzzy search**: Rocket fast fuzzy search indexing and retrieval based on Whoosh, supporting millions of bookmarks, all offline!
- 🔄 **Parallel Processing**: Efficient multi-threaded crawling to significantly reduce processing time
- 🌐 **Multiple Model Support**: Compatible with OpenAI, Deepseek, Qwen, and Ollama offline models
- 💾 **Incremental Update And Checkpoint Recovery**: Update the database with new bookmarks or continue processing after interruptions without losing completed work
- 📊 **Detailed Logging**: Clear progress and status reports for monitoring and debugging

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- Chrome browser
- Internet connection
- Large language model API key (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/wyj/BookmarkSummarizer.git
cd BookmarkSummarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (create a `.env` file):
```
MODEL_TYPE=ollama  # Options: openai, deepseek, qwen, ollama
API_KEY=your_api_key_here
API_BASE=http://localhost:11434  # Ollama local endpoint or other model API address
MODEL_NAME=llama2  # Or other supported model
MAX_TOKENS=1000
TEMPERATURE=0.3
```

### Usage

#### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables** (optional, create a `.env` file):
   ```
   MODEL_TYPE=ollama  # Options: openai, deepseek, qwen, ollama
   API_KEY=your_api_key_here
   API_BASE=http://localhost:11434  # Ollama local endpoint or other model API address
   MODEL_NAME=llama2  # Or other supported model
   MAX_TOKENS=1000
   TEMPERATURE=0.3
   ```

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

**Use run_with_env.py to use a global env file for the current terminal session**:
```bash
python run_with_env.py --env ./env.env "python ./crawl.py --browser chrome --profile-path 'C:\Users\Username\AppData\Local\Chromium\User Data\Profile 1' --limit 10"
```

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

- `bookmarks.json`: Filtered bookmark list from browsers
- `bookmarks_with_content.json`: Bookmark data with crawled content and AI-generated summaries
- `failed_urls.json`: URLs that failed to crawl with reasons
- `whoosh_index/`: Directory containing the Whoosh search index files

## 📋 Detailed Features

### Bookmark Crawling

BookmarkSummarizer automatically reads all bookmarks from the Chrome bookmarks file and intelligently filters out ineligible URLs. It uses two strategies to crawl web content:

1. **Regular Crawling**: Uses the Requests library to capture content from most web pages
2. **Dynamic Content Crawling**: For dynamic webpages (such as Zhihu and other platforms), automatically switches to Selenium

### Summary Generation

BookmarkSummarizer uses advanced large language models to generate high-quality summaries for each bookmark content, including:

- Extracting key information and important concepts
- Preserving technical terms and key data
- Generating structured summaries for easier retrieval
- Supporting various mainstream large language models

### Checkpoint Recovery

- Saves progress immediately after processing each bookmark
- Automatically skips previously processed bookmarks when restarted
- Ensures data safety even when processing large numbers of bookmarks

## 📁 Output Files

- `bookmarks.json`: Filtered bookmark list
- `bookmarks_with_content.json`: Bookmark data with content and summaries
- `failed_urls.json`: Failed URLs and reasons

## 🔧 Custom Configuration

In addition to command-line parameters, you can set the following environment variables through the `.env` file:

```
# Model Type Settings
MODEL_TYPE=ollama  # openai, deepseek, qwen, ollama
API_KEY=your_api_key_here
API_BASE=http://localhost:11434
MODEL_NAME=gemma3:1b

# Content Processing Settings
MAX_TOKENS=1024  # Maximum number of tokens for summary generation
MAX_INPUT_CONTENT_LENGTH=6000  # Maximum length of input content
TEMPERATURE=0.3  # Randomness of summary generation

# Crawler Settings
BOOKMARK_LIMIT=0  # No limit by default
MAX_WORKERS=20  # Number of parallel worker threads
GENERATE_SUMMARY=true  # Whether to generate summaries
```

Note: Make sure there are NO comments in the file (especially inline comments), otherwise it may fail to be parsed correctly.

## 🤝 Contributing

Pull Requests are welcome! For any issues or suggestions, please create an Issue.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
