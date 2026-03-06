# langchain-youdotcom

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-youdotcom?label=%20)](https://pypi.org/project/langchain-youdotcom/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-youdotcom)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-youdotcom)](https://pypistats.org/packages/langchain-youdotcom)

LangChain partner package for [You.com](https://you.com) search, content extraction, and research APIs.

[Installation](#installation) | [Credentials](#credentials) | [Tools](#tools) | [Retriever](#retriever) | [API Wrapper](#yousearchapiwrapper) | [Resources](#resources)

## Installation

```bash
pip install -U langchain-youdotcom
```

## Credentials

Get an API key at [you.com/platform/api-keys](https://you.com/platform/api-keys), then set it as an environment variable:

```bash
export YDC_API_KEY="your-api-key"
```

Or pass it directly when instantiating any component:

```python
from langchain_youdotcom import YouSearchTool, YouSearchAPIWrapper

tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(ydc_api_key="your-api-key"))
```

## Tools

### YouSearchTool

Search the web with up to date results. Supports geographic filtering, freshness controls, and optional live-crawling for full page content. Great for monitoring mentions, pulling recent news, or feeding live data into agent workflows.

**Instantiation parameters** (set on `YouSearchAPIWrapper`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count` | `int \| None` | `None` | Max results per section, 1-100 |
| `country` | `str \| None` | `None` | Two-letter country code to focus results geographically |
| `freshness` | `str \| None` | `None` | Filter by recency: `day`, `week`, `month`, or `year` |
| `language` | `str \| None` | `None` | BCP-47 language code for results |
| `livecrawl` | `str \| None` | `None` | Fetch full page content: `web`, `news`, or `all` |
| `livecrawl_formats` | `str \| None` | `None` | Format for livecrawled content: `html` or `markdown` |
| `offset` | `int \| None` | `None` | Pagination offset, 0-9 |
| `safesearch` | `str \| None` | `None` | Content filter: `off`, `moderate`, or `strict` |
| `k` | `int \| None` | `None` | Max documents to return |

**Invocation args:**

- `query` (required, `str`): The search query.

```python
from langchain_youdotcom import YouSearchTool, YouSearchAPIWrapper

tool = YouSearchTool(
    api_wrapper=YouSearchAPIWrapper(
        count=5,
        country="US",
        freshness="week",
        livecrawl="web",
        livecrawl_formats="markdown",
    ),
)

# invoke directly
result = tool.invoke("latest AI news")
print(result)
```

**Using with an agent:**

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_youdotcom import YouSearchTool

tools = [YouSearchTool()]
agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what happened in AI today?"}]}
)
```

### YouContentsTool

Extract clean, structured content from one or more web pages. Returns page text as markdown or HTML, plus metadata like JSON-LD, OpenGraph, and Twitter Cards. Useful for scraping product pages, pulling article text, or extracting structured data from any URL.

**Instantiation parameters** (set on `YouSearchAPIWrapper`):

No tool-level configuration. All parameters are passed at invocation time.

**Invocation args:**

- `urls` (required, `list[str]`): URLs to fetch content from.

Content format and timeout are configured when calling the wrapper directly (see [YouSearchAPIWrapper](#yousearchapiwrapper)).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `list[str]` | — | URLs to extract content from (required) |
| `formats` | `list[str] \| None` | `["markdown", "metadata"]` | Output formats: `markdown`, `html`, and/or `metadata` |
| `crawl_timeout` | `float \| None` | `None` | Per-URL crawl timeout in seconds |

```python
from langchain_youdotcom import YouContentsTool

tool = YouContentsTool()
result = tool.invoke({"urls": ["https://example.com"]})
print(result)
```

### YouResearchTool

Get a comprehensive, cited answer to a complex question. The Research API searches the web, reads multiple sources, and synthesizes a detailed markdown response with inline numbered citations. Perfect for competitive analysis, market research, technical due diligence, or any question that needs more than a simple search result.

**Instantiation parameters** (set on `YouSearchAPIWrapper`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_effort` | `str \| None` | `None` | Controls depth and speed (see levels below) |

**Research effort levels:**

| Level | Description |
|-------|-------------|
| `lite` | Quick answers for straightforward questions |
| `standard` | Balanced speed and depth (default) |
| `deep` | More time researching and cross-referencing sources |
| `exhaustive` | Most thorough option for complex research tasks |

**Invocation args:**

- `query` (required, `str`): The research question.

```python
from langchain_youdotcom import YouResearchTool, YouSearchAPIWrapper

# default effort
tool = YouResearchTool()
result = tool.invoke("what are the latest advances in quantum computing")
print(result)

# deep research
tool = YouResearchTool(
    api_wrapper=YouSearchAPIWrapper(research_effort="deep"),
)
result = tool.invoke("compare transformer architectures for long-context tasks")
print(result)
```

## Retriever

The simplest way to get You.com search results as LangChain documents. Accepts all search parameters from [YouSearchTool](#yousearchtool).

```python
from langchain_youdotcom import YouRetriever

retriever = YouRetriever()
docs = retriever.invoke("latest AI news")

for doc in docs:
    print(doc.metadata["title"])
    print(doc.page_content[:200])
    print()
```

With search parameters:

```python
retriever = YouRetriever(
    k=5,
    count=10,
    livecrawl="web",
    livecrawl_formats="markdown",
    country="US",
    freshness="week",
    safesearch="moderate",
)
```

## YouSearchAPIWrapper

Lower-level wrapper that powers the tools and retriever under the hood. Use it directly when you need full control over API calls and response formats.

**Search:**

```python
from langchain_youdotcom import YouSearchAPIWrapper

wrapper = YouSearchAPIWrapper()

# search -> list[Document]
docs = wrapper.results("latest AI news")

# raw SDK response
raw = wrapper.raw_results("latest AI news")
```

**Contents:**

```python
pages = wrapper.contents(
    ["https://example.com"],
    formats=["markdown", "metadata"],
    crawl_timeout=30,
)
```

**Research:**

```python
# research -> formatted markdown with sources
text = wrapper.research_text("explain quantum entanglement")

# raw SDK response
raw = wrapper.raw_research("explain quantum entanglement")
```

Async variants are available for all methods: `results_async`, `raw_results_async`, `contents_async`, `research_text_async`, `raw_research_async`.

## Resources

- [You.com API docs](https://docs.you.com)
- [Search API reference](https://docs.you.com/api-reference/search)
- [Contents API reference](https://docs.you.com/api-reference/contents)
- [Research API reference](https://docs.you.com/api-reference/research)
- [You.com API keys](https://you.com/platform/api-keys)

## Development

```bash
uv sync --all-groups
make format            # ruff format + fix
make lint              # ruff check + format diff + mypy
make test              # unit tests
make integration_tests # requires YDC_API_KEY
make check_imports     # verify all modules importable
```
