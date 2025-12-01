# BookmarkSummarizer (书签大脑)

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LLM-enabled-green.svg" alt="LLM">

  [![PyPI-Status][1]][2] [![PyPI-Versions][3]][2] [![PyPI-Downloads][5]][2]

  [![Build-Status][7]][8] [![Coverage-Status][9]][10]
</p>

BookmarkSummarizer 是一个强大的工具，它能够爬取您浏览器的书签内容，使用大语言模型生成摘要，并将它们转化为个人知识库。无需整理，轻松搜索和利用您收藏的所有网页资源。支持所有常见桌面浏览器（Chrome、Firefox、Edge、Safari）以及不常见的浏览器（Chromium、Brave、Vivaldi、Opera等）。

<p align="right"><a href="../README.MD">English Documentation</a></p>

## ✨ 主要功能

- 🔍 **智能书签爬取**：自动从浏览器书签中提取网页内容
- 🤖 **AI 摘要生成**：使用大语言模型为每个书签创建高质量摘要
- 🚀 **极速可扩展的全文模糊搜索**：基于 Whoosh 的超快速模糊搜索索引和检索，支持数百万书签，完全离线！
- 🔄 **并行处理**：高效的多线程爬取，显著减少处理时间
- 🌐 **多模型支持**：兼容 OpenAI、Deepseek、Qwen 和 Ollama 离线模型
- 💾 **增量更新与断点恢复**：更新数据库新书签或中断后继续处理，不会丢失已完成的工作
- 📊 **详细日志**：清晰的进度和状态报告，便于监控和调试
- **大规模扩展能力**：从几百个书签的<10MB LMDB数据库开始，通过增量更新可扩展到数千个书签的几GB数据库，仅使用少量RAM（得益于磁盘外存储数据库），最高可达数百万书签的数TB LMDB数据库，仅需几GB内存加载。模糊搜索引擎通过构建更小的 Whoosh 数据库进一步提升扩展性，使搜索书签内容、URL、标题或摘要极其快速，且内存占用极小。
- **模块化架构**：可通过在 custom_parsers 目录添加 Python 文件来添加自定义解析器，无需修改核心逻辑。例如，提供了自定义解析器来提取 YouTube 字幕作为内容进行摘要，以及透明地恢复被书签保存的挂起标签页以获取真实目标页面内容。

## 🚀 快速开始

### 前提条件

- Python 3.6+
- Chrome 浏览器
- 网络连接
- 大语言模型 API 密钥（可选）

### 安装

#### 便携式二进制文件

前往 [GitHub Releases](https://github.com/lrq3000/BookmarkSummarizer/releases) 并选择最新版本，您将找到 Windows、MacOS 和 Linux 的预编译二进制文件。

#### 从 PyPI 安装

如果您已安装 Python，只需执行：

```bash
pip install --upgrade bookmark-summarizer
```

#### 从源码安装

1. 克隆仓库：
```bash
git clone https://github.com/lrq3000/BookmarkSummarizer.git
cd BookmarkSummarizer
```

2. 安装依赖：
```bash
pip install -e .
```

3. 创建 TOML 配置文件以微调行为（创建 `.toml` 文件）：
```toml
model_type="ollama"  # 选项：openai, deepseek, qwen, ollama
api_key="your_api_key_here"
api_base="http://localhost:11434"  # ollama 本地端点或其他模型 API 地址
model_name="qwen3:1.7b"  # 或其他支持的模型
max_tokens=1000
temperature=0.3
```

### 使用方法

#### 从浏览器获取书签

**从所有浏览器获取书签**（默认）：
```bash
python index.py
```
这会从所有已安装的浏览器（Chrome、Firefox、Edge、Safari、Opera、Brave、Vivaldi等）获取书签，并保存到 `bookmarks.json`。

**从特定浏览器获取书签**：
```bash
python index.py --browser chrome
```
支持的浏览器：`chrome`、`firefox`、`edge`、`opera`、`opera_gx`、`safari`、`vivaldi`、`brave`。

**从自定义配置文件路径获取书签**：
```bash
python index.py --browser chrome --profile-path "C:\Users\Username\AppData\Local\Google\Chrome\User Data\Profile 1"
```
当您有多个 Chrome 配置文件或自定义浏览器安装时很有用。

#### 爬取和摘要书签

**基础用法（从所有浏览器爬取和摘要）**：
```bash
python crawl.py
```
这会从所有浏览器获取书签，爬取其内容，生成 AI 摘要并保存结果。使用相同命令可增量更新已爬取的书签或中断后恢复 - 已处理的书签将被跳过。

**从特定浏览器爬取**：
```bash
python crawl.py --browser firefox
```
仅从 Firefox 获取和爬取书签。

**从自定义配置文件路径爬取**：
```bash
python crawl.py --browser chrome --profile-path "/home/user/.config/google-chrome/Profile 1"
```
结合浏览器选择和自定义配置文件路径。

**限制书签数量**：
```bash
python crawl.py --limit 10
```
仅处理前 10 个书签。

**设置并行处理线程数**：
```bash
python crawl.py --workers 10
```
使用 10 个工作线程进行并行爬取（默认：20）。

**跳过摘要生成**：
```bash
python crawl.py --no-summary
```
爬取内容但跳过 AI 摘要生成。

**从已爬取的内容生成摘要**：
```bash
python crawl.py --from-json
```
为现有的 `bookmarks_with_content.json` 生成摘要，无需重新爬取。

#### 搜索书签

一旦您的书签被爬取，当前文件夹中将出现一个 `bookmarks_with_content.json` 文件。然后您可以使用模糊搜索引擎进行搜索：

**启动搜索界面但不重建索引**：
```bash
python fuzzy_bookmark_search.py --no-index
```
这会启动一个本地 Web 服务器，搜索引擎可通过 http://localhost:8132/ 访问（端口可通过 `--port xxx` 更改）。搜索引擎使用 Whoosh 构建快速的磁盘上模糊可搜索索引。

**启动搜索界面但不更新索引**：
```bash
python fuzzy_bookmark_search.py
```
使用现有索引而不重建。

#### 输出文件

- `bookmarks.json`：从浏览器过滤的书签列表，只是直接从浏览器获取的所有书签的汇编。
- `bookmark_index.lmdb`：包含爬取内容和 AI 生成摘要的书签数据文件夹，存储在 LMDB 中。
- `failed_urls.json`：爬取失败的 URL 及原因。
- `crawl_errors.log`：爬虫的错误日志，记录所有错误，即使与书签内容不可达性无关（例如，记录软件逻辑错误）。
- `whoosh_index/`：包含搜索引擎的 Whoosh 搜索索引文件的目录。

## 📋 功能详解

### 书签爬取

BookmarkSummarizer 会自动从 Chrome 书签文件中读取所有书签，并智能过滤掉不符合条件的 URL。它使用两种策略爬取网页内容：

1. **常规爬取**：使用 Requests 库抓取大多数网页内容
2. **动态内容爬取**：对于动态网页（如知乎等平台），自动切换到 Selenium
3. **模块化架构与自定义解析器**：对于特定网站或内容（如 YouTube），可以在 `custom_parsers/` 中实现自定义解析器/适配器作为独立的 `.py` 模块，它们将被自动调用以过滤和处理每个书签。自定义解析器获得书签元数据的完整副本，可以基于任何标准选择过滤，不仅是 URL，还可以基于内容或标题等。例如，对于 YouTube，会下载字幕作为内容进行摘要。

### 摘要生成

BookmarkSummarizer 使用先进的大语言模型为每个书签内容生成高质量摘要，包括：

- 提取关键信息和重要概念
- 保留专业术语和关键数据
- 生成结构化摘要，便于后续检索
- 支持多种主流大语言模型
- 通过 ollama 支持 100% 离线生成，完全保护隐私

**提示**：如果使用 ollama，建议将上下文窗口设置为 128k，并使用支持如此宽上下文窗口的模型，例如 qwen3:4b（支持 256k 上下文！）或 qwen3:1.7b 或 qwen3:0.6b（40k 上下文）用于性能较弱的机器，以便在整个书签的全文内容上完成摘要而无需截断。`gemma3:1b` 也可能很有趣（32k 上下文），但当全文内容不多时会出现幻觉问题。

### 断点续传

- 每处理完一个书签就立即保存进度
- 中断后重启时会自动跳过已处理的书签
- 即使在大量书签处理过程中，也能保证数据安全

## 📁 输出文件

- `bookmarks.json`：过滤后的书签列表
- `bookmarks_with_content.json`：带有内容和摘要的书签数据
- `failed_urls.json`：爬取失败的 URL 及原因

## 🔧 自定义配置

除了命令行参数外，您还可以通过 `.toml` 配置文件设置以下参数：

```toml
# 模型类型设置
model_type="ollama"  # openai, deepseek, qwen, ollama
api_key="your_api_key_here"
api_base="http://localhost:11434"
model_name="gemma3:1b"

# 内容处理设置
max_tokens=1024  # 生成摘要的最大令牌数
max_input_content_length=6000  # 输入内容的最大长度
temperature=0.3  # 生成摘要的随机性

# 爬虫设置
bookmark_limit=0  # 默认不限制
max_workers=20  # 并行工作线程数
generate_summary=true  # 是否生成摘要
```

## 🤝 贡献

欢迎提交 Pull Requests！有任何问题或建议，请创建 Issue。

## 作者

最初由 [wyj/sologuy](https://github.com/sologuy/BookmarkSummarizer/) 创建。

自 2025 年 11 月起，新功能开发和维护由 [Stephen Karl Larroque](https://github.com/lrq3000/BookmarkSummarizer/) 完成。

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 推荐的第三方书签工具

以下是可与 BookmarkSummarizer 互补的**开源**第三方扩展或工具的非详尽列表：
* [Search Bookmarks, History and Tabs](https://github.com/Fannon/search-bookmarks-history-and-tabs)：基于 URL 和书签标题（非全页内容）的快速书签模糊搜索引擎。Chrome 扩展。
* [Full text tabs forever (FTTF)](https://github.com/iansinnott/full-text-tabs-forever)：历史访问页面的全文搜索。其优势是不会产生网络开销（不执行额外的 HTTP 请求），因此没有速率限制/IP 禁止的风险。Chrome 扩展。
* [Floccus](https://github.com/floccusaddon/floccus)：浏览器之间自动同步书签（如果使用 InfiniTabs 也可同步会话），也可在移动端通过 F-Droid 上的原生 Floccus 应用或 [Mises](https://github.com/mises-id/mises-browser-core) 或 [Cromite](https://github.com/uazo/cromite/) 使用。Chrome 扩展。
* [TidyMark](https://github.com/PanHywel/TidyMark)：重组/分组书签（支持云或离线 ollama）。Chrome 扩展。
* [Wherewasi](https://github.com/Jay-Karia/wherewasi)：使用云 Gemini AI 的时间和语义标签聚类到会话。Chrome 扩展。
* LinkWarden 或 ArchiveBox：BookmarkSummarizer 的替代方案，用于索引/归档书签指向的全文内容。


[1]: https://img.shields.io/pypi/v/bookmark-summarizer.svg
[2]: https://pypi.org/project/bookmark-summarizer
[3]: https://img.shields.io/pypi/pyversions/bookmark-summarizer.svg?logo=python&logoColor=white
[5]: https://img.shields.io/pypi/dm/bookmark-summarizer.svg?label=pypi%20downloads&logo=python&logoColor=white
[7]: https://github.com/lrq3000/BookmarkSummarizer/actions/workflows/ci-build.yml/badge.svg?event=push
[8]: https://github.com/lrq3000/BookmarkSummarizer/actions/workflows/ci-build.yml
[9]: https://codecov.io/gh/lrq3000/BookmarkSummarizer/graph/badge.svg?token=NuNgXwZqAO
[10]: https://codecov.io/gh/lrq3000/BookmarkSummarizer
