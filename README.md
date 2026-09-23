# langchain-youdotcom

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-youdotcom?label=%20)](https://pypi.org/project/langchain-youdotcom/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-youdotcom)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-youdotcom)](https://pypistats.org/packages/langchain-youdotcom)

LangChain partner package for [You.com](https://you.com) search, content extraction, research, finance research, and answer APIs.

[Installation](#installation) | [Credentials](#credentials) | [Tools](#tools) | [Retriever](#retriever) | [API Wrapper](#youapiwrapper) | [Resources](#resources)

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
from langchain_youdotcom import YouSearchTool, YouAPIWrapper

tool = YouSearchTool(api_wrapper=YouAPIWrapper(ydc_api_key="your-api-key"))
```

Every outbound request emits an `X-Client-Info` header that identifies the call as originating from `langchain-youdotcom`, so You.com can attribute usage correctly. No action required.

## Tools

### YouSearchTool

Search the web with up to date results. Supports geographic filtering, freshness controls, optional full-page content extraction, and licensed knowledge results. Great for monitoring mentions, pulling recent news, or feeding live data into agent workflows.

**Instantiation parameters** (set on `YouAPIWrapper`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boost_domains` | `list[str] \| None` | `None` | Domains to prefer in ranking (up to 500). Can combine with `exclude_domains` but not `include_domains` |
| `count` | `int \| None` | `None` | Max results per section, 1-100. Caps the web and news sections only, not `knowledge` |
| `country` | `str \| None` | `None` | Two-letter country code to focus results geographically |
| `crawl_timeout` | `int \| None` | `None` | Per-page crawl timeout in seconds for search; the API defaults to 10. Most useful with `extraction` in `full_page` mode. The client's own request timeout is set to this plus 20s of headroom, so a large value is actually honored rather than being cut off by httpx's 5s default. The server crawls pages in parallel, so this budget does not need to scale with `count`. Does not apply to `contents()`, which takes its own argument |
| `exclude_domains` | `list[str] \| None` | `None` | Domains to exclude (up to 500). Can combine with `boost_domains` but not `include_domains` |
| `extraction` | `dict \| None` | `None` | Controls how page content is attached to each result. Replaces the deprecated `livecrawl`. `extraction_mode` is required: `"highlights"` returns the relevant passages of each page, `"full_page"` returns whole-page content. `extraction_source` (`"blend"`, `"cache"`, or `"fetch"`) and `full_page` apply to `"full_page"` only. Cannot be combined with `livecrawl` |
| `freshness` | `str \| None` | `None` | Filter by recency: `day`, `week`, `month`, or `year` |
| `include_domains` | `list[str] \| None` | `None` | Domains to exclusively include (up to 500). Cannot combine with `exclude_domains` or `boost_domains`; the wrapper raises `ValueError` locally |
| `knowledge` | `str \| None` | `None` | Set to `"core"` to include knowledge results backed by licensed data providers. `"core"` is the only accepted value. Omit to skip them |
| `language` | `str \| None` | `None` | BCP-47 language code for results |
| `livecrawl` | `str \| None` | `None` | Deprecated by the SDK; will be removed in a future release. Use `extraction` instead. Fetch full page content: `web`, `news`, or `all` |
| `livecrawl_formats` | `list[str] \| None` | `None` | Deprecated by the SDK; will be removed in a future release. Use `extraction.full_page.extraction_formats` instead. Format for livecrawled content as a list, e.g. `["html", "markdown"]` |
| `n_snippets_per_hit` | `int \| None` | `None` | Max excerpts to keep per hit — applies to snippets, and to the passages returned by `extraction_mode="highlights"`. Unset keeps all |
| `offset` | `int \| None` | `None` | Pagination offset, 0-9 |
| `safesearch` | `str \| None` | `None` | Content filter: `off`, `moderate`, or `strict` |
| `k` | `int \| None` | `None` | Max documents to return |

**Invocation args:**

- `query` (required, `str`): The search query.

```python
from langchain_youdotcom import YouSearchTool, YouAPIWrapper

tool = YouSearchTool(
    api_wrapper=YouAPIWrapper(
        count=5,
        country="US",
        freshness="week",
        safesearch="moderate",
    ),
)

# invoke directly
result = tool.invoke("latest AI news")
print(result)
```

**Full-page content instead of snippets:**

```python
from langchain_youdotcom import YouAPIWrapper

wrapper = YouAPIWrapper(
    count=5,
    extraction={
        "extraction_mode": "full_page",
        "extraction_source": "blend",
        "full_page": {"extraction_formats": ["markdown"]},
    },
)
docs = wrapper.results("latest AI news")
```

`extraction` replaces the deprecated `livecrawl` / `livecrawl_formats`. Passing both raises a `ValueError` from the SDK.

**Licensed knowledge results:**

```python
wrapper = YouAPIWrapper(count=5, knowledge="core")
docs = wrapper.results("NVIDIA revenue FY2025")
```

Knowledge results are cards backed by licensed data providers such as encyclopedias, market-data firms, and reference publishers. Two things to know:

- **They have no `url`.** Attribution entries are provider credits rather than citations, so knowledge documents carry `title`, `type`, `source`, and `attribution` in `metadata` but no `url` key. Code that reads `doc.metadata["url"]` should check `doc.metadata["source"] != "knowledge"` first.
- **`count` does not cap them.** Up to 25 relevant knowledge results can arrive regardless of `count`. Only `k` bounds the returned list. They are emitted before web and news results, so a pinned `k` keeps them.

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

**Instantiation parameters** (set on `YouAPIWrapper`):

No tool-level configuration. All parameters are passed at invocation time.

**Invocation args:**

- `urls` (required, `list[str]`): URLs to fetch content from.

Content format and timeout are configured when calling the wrapper directly (see [YouAPIWrapper](#youapiwrapper)).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `list[str]` | — | URLs to extract content from (required). **Maximum 10 per request** — the API rejects more with a 422, and the wrapper raises `ValueError` locally instead. Split larger batches across calls |
| `formats` | `list[str] \| None` | `["markdown"]` | Output formats: `markdown`, `html`, and/or `metadata`. `metadata` is deprecated by the SDK and emits a `DeprecationWarning`, so it is no longer requested by default — pass it explicitly if you need the `site_name` and `favicon_url` metadata keys |
| `crawl_timeout` | `int \| None` | `None` | Per-URL crawl timeout in seconds |
| `max_age` | `int \| None` | `None` | Maximum allowed age of cached content in seconds. Cached content older than this is ignored and the page is re-fetched. Must be 0 or greater. Defaults to no age limit |

```python
from langchain_youdotcom import YouContentsTool

tool = YouContentsTool()
result = tool.invoke({"urls": ["https://example.com"]})
print(result)
```

### YouResearchTool

Get a comprehensive, cited answer to a complex question. The Research API searches the web, reads multiple sources, and synthesizes a detailed markdown response with inline numbered citations. Perfect for competitive analysis, market research, technical due diligence, or any question that needs more than a simple search result.

The output ends with a `## Sources` section listing each source by title and URL, followed by that source's supporting excerpts. Those excerpts can be long, so deep research output can grow large — use `raw_research()` when you want the structured response without the excerpts.

**Instantiation parameters** (set on `YouAPIWrapper`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_effort` | `str \| None` | `None` | Controls depth and speed (see levels below) |
| `output_schema` | `dict \| None` | `None` | JSON Schema constraining the structured output. Passed through to the SDK unmodified |
| `source_control` | `dict \| None` | `None` | Controls which sources research may draw on: `include_domains`, `exclude_domains`, `boost_domains`, `freshness`, `country`. Research only — for Search, use the top-level domain fields. Unlike `extraction`, the SDK does not reject unknown keys here, so a misspelled key is silently ignored |

**Research effort levels:**

| Level | Description |
|-------|-------------|
| `lite` | Quick answers for straightforward questions |
| `standard` | Balanced speed and depth (default) |
| `deep` | More time researching and cross-referencing sources |
| `exhaustive` | Most thorough option for complex research tasks |
| `frontier` | Highest-quality tier. Only supported by the task-based API (`background=true`); sending it to the sync API returns a 422, so `YouResearchTool` does not handle it — use the SDK directly for `frontier` runs. |

**Structured output:**

Setting `output_schema` makes the API return a JSON object in `output.content` instead of markdown, and `content_type` becomes `"object"`. Two consequences:

- `research_text()` serializes that object to indented JSON so it still returns a string. Use `raw_research()` when you want the parsed dict.
- `output_schema` is not supported with `research_effort="lite"` — the API returns a 422, so the wrapper raises `ValueError` locally instead.

The API requires every object in the schema to define `properties`, set `additionalProperties: false`, and list every property in `required`:

```python
wrapper = YouAPIWrapper(
    research_effort="deep",
    output_schema={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
)
raw = wrapper.raw_research("explain quantum entanglement")
print(raw.output.content)  # {"summary": "..."}
```

**Invocation args:**

- `query` (required, `str`): The research question.

```python
from langchain_youdotcom import YouResearchTool, YouAPIWrapper

# default effort
tool = YouResearchTool()
result = tool.invoke("what are the latest advances in quantum computing")
print(result)

# deep research
tool = YouResearchTool(
    api_wrapper=YouAPIWrapper(research_effort="deep"),
)
result = tool.invoke("compare transformer architectures for long-context tasks")
print(result)
```

### YouFinanceResearchTool

Get a comprehensive, citation-backed answer to a financial question. The Finance Research API works like the Research API but searches a finance-optimized index covering SEC filings, earnings reports, equity prices, macro indicators, and financial news. Ideal for earnings analysis, competitive benchmarking, due diligence, and market research.

**Instantiation parameters** (set on `YouAPIWrapper`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_effort` | `str \| None` | `None` | Controls depth (shared with YouResearchTool; see levels below) |

Finance research only accepts `deep` and `exhaustive`. Other values raise `ValueError`.

| Level | Description |
|-------|-------------|
| `deep` | Multi-source analysis, earnings summaries, competitive benchmarking (default) |
| `exhaustive` | Comprehensive research, deep due diligence, full 10-K analysis |

**Invocation args:**

- `query` (required, `str`): The financial research question.

```python
from langchain_youdotcom import YouFinanceResearchTool, YouAPIWrapper

tool = YouFinanceResearchTool()
result = tool.invoke("what were NVIDIA's key revenue drivers in FY2025")
print(result)

# exhaustive research for complex due diligence
tool = YouFinanceResearchTool(
    api_wrapper=YouAPIWrapper(research_effort="exhaustive"),
)
result = tool.invoke("compare gross margins of Apple, Microsoft, and Google over the past three fiscal years")
print(result)
```

### YouAnswerTool

Get a single, synthesized answer to a focused live-web question with inline citations. The Answer API is optimized for short, single-question lookups (faster than Research) and returns a synthesized response plus a `## Citations` section listing each cited source by title and URL, with its description and supporting excerpts.

The 0.4.0 release adds `YouAnswerTool` as the recommended entry point for any single-question workflow. Reach for `YouResearchTool` only when the question needs synthesis across multiple sources.

**Invocation args:**

- `query` (required, `str`): The live-web question. Max 400 characters.
- `freshness` (optional, `str`): Recency filter: `day`, `week`, `month`, `year`, or a date range.
- `country` (optional, `str`): ISO 3166-1 alpha-2 country code for geographical focus.
- `language` (optional, `str`): BCP 47 language tag.
- `safesearch` (optional, `str`): Content filter: `off`, `moderate`, or `strict`.
- `include_domains` (optional, `list[str]`): Restrict results to specific domains.
- `exclude_domains` (optional, `list[str]`): Exclude specific domains. Cannot be combined with `include_domains`.
- `boost_domains` (optional, `list[str]`): Boost specific domains in the ranking. Cannot be combined with `include_domains`.

Every one of these is also a `YouAPIWrapper` field. Wrapper-level config acts as the default and a per-call filter overrides it, so a wrapper configured with `country="US"` behaves consistently across Search and Answer. The `include_domains` combinations are validated against the merged result, so an illegal pairing formed partly by wrapper config is still rejected locally.

```python
from langchain_youdotcom import YouAnswerTool

tool = YouAnswerTool()
result = tool.invoke({"query": "what is retrieval augmented generation"})
print(result)
```

With filters:

```python
result = tool.invoke(
    {
        "query": "latest python release",
        "freshness": "week",
        "country": "US",
        "language": "EN",
        "safesearch": "moderate",
    }
)
```

With domain restriction:

```python
result = tool.invoke(
    {
        "query": "langchain release notes",
        "exclude_domains": ["pinterest.com"],
        "boost_domains": ["github.com"],
    }
)
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
    country="US",
    freshness="week",
    safesearch="moderate",
    knowledge="core",
    extraction={
        "extraction_mode": "full_page",
        "extraction_source": "blend",
        "full_page": {"extraction_formats": ["markdown"]},
    },
)
```

`YouRetriever` subclasses `YouAPIWrapper`, so every parameter above — including `extraction` and `knowledge` — is inherited. The [knowledge result caveats](#yousearchtool) apply here too: those documents have no `url` metadata key, and `count` does not cap them.

## YouAPIWrapper

Lower-level wrapper that powers the tools and retriever under the hood. Use it directly when you need full control over API calls and response formats.

**Search:**

```python
from langchain_youdotcom import YouAPIWrapper

wrapper = YouAPIWrapper()

# search -> list[Document]
docs = wrapper.results("latest AI news")

# raw SDK response
raw = wrapper.raw_results("latest AI news")
```

**Contents:**

```python
pages = wrapper.contents(
    ["https://example.com"],
    formats=["markdown"],
    crawl_timeout=30,
)

# opt back into the deprecated metadata format for site_name / favicon_url
pages = wrapper.contents(
    ["https://example.com"],
    formats=["markdown", "metadata"],
)
```

**Research:**

```python
# research -> formatted markdown with sources
text = wrapper.research_text("explain quantum entanglement")

# raw SDK response
raw = wrapper.raw_research("explain quantum entanglement")
```

**Finance Research:**

```python
# finance research -> formatted markdown with sources
text = wrapper.finance_text("what drove NVIDIA's revenue growth in FY2025")

# raw SDK response (FinanceResearchResponse)
raw = wrapper.raw_finance("compare AAPL and MSFT gross margins")
```

**Answer:**

```python
# answer -> formatted markdown with citations
text = wrapper.answer_text(
    "what is retrieval augmented generation",
    freshness="week",
    country="US",
    safesearch="moderate",
)

# raw SDK response
raw = wrapper.raw_answer(
    "what is retrieval augmented generation",
    include_domains=["arxiv.org"],
)
```

Async variants are available for all methods: `results_async`, `raw_results_async`, `contents_async`, `research_text_async`, `raw_research_async`, `finance_text_async`, `raw_finance_async`, `answer_text_async`, `raw_answer_async`.

## Resources

- [You.com API docs](https://docs.you.com)
- [Search API reference](https://docs.you.com/api-reference/search)
- [Contents API reference](https://docs.you.com/api-reference/contents)
- [Research API reference](https://docs.you.com/api-reference/research)
- [Finance Research API reference](https://docs.you.com/api-reference/finance-research)
- [Answer API reference](https://docs.you.com/api-reference/answer)
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
