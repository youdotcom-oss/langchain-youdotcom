# langchain-youdotcom

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-youdotcom?label=%20)](https://pypi.org/project/langchain-youdotcom/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-youdotcom)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-youdotcom)](https://pypistats.org/packages/langchain-youdotcom)

LangChain partner package for [You.com](https://you.com) search, content, and research APIs.

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
retriever = YouRetriever(ydc_api_key="your-api-key")
```

## Retriever

The simplest way to get You.com search results as LangChain documents.

```python
from langchain_youdotcom import YouRetriever

retriever = YouRetriever()
docs = retriever.invoke("latest AI news")

for doc in docs:
    print(doc.metadata["title"])
    print(doc.page_content[:200])
    print()
```

All search parameters are available directly on the retriever:

```python
retriever = YouRetriever(
    k=5,                    # max documents to return
    count=10,               # max results per API section
    livecrawl="web",        # fetch live page content
    livecrawl_formats="markdown",
    country="US",
    freshness="week",       # day, week, month, year
    safesearch="moderate",  # off, moderate, strict
)
```

## Tools

### YouSearchTool

Search the web with You.com. Works with any LangChain agent.

```python
from langchain_youdotcom import YouSearchTool

tool = YouSearchTool()
result = tool.invoke("what is retrieval augmented generation")
print(result)
```

### YouResearchTool

Get comprehensive, research-grade answers with cited sources.

```python
from langchain_youdotcom import YouResearchTool

tool = YouResearchTool()
result = tool.invoke("what are the latest advances in quantum computing")
print(result)
```

Control research depth with `research_effort` (lite, standard, deep, exhaustive):

```python
from langchain_youdotcom import YouResearchTool, YouSearchAPIWrapper

tool = YouResearchTool(
    api_wrapper=YouSearchAPIWrapper(research_effort="deep"),
)
```

### YouContentsTool

Fetch and extract content from web pages.

```python
from langchain_youdotcom import YouContentsTool

tool = YouContentsTool()
result = tool.invoke({"urls": ["https://example.com"]})
print(result)
```

### Using with an agent

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

## YouSearchAPIWrapper

For more control, use the wrapper directly. It powers the retriever and tools under the hood.

```python
from langchain_youdotcom import YouSearchAPIWrapper

wrapper = YouSearchAPIWrapper()

# search -> list[Document]
docs = wrapper.results("latest AI news")

# raw SDK response
raw = wrapper.raw_results("latest AI news")

# contents API -> list[Document]
pages = wrapper.contents(
    ["https://example.com"],
    formats=["markdown", "metadata"],
)
```

Research API:

```python
# research -> formatted markdown with sources
text = wrapper.research_text("explain quantum entanglement")

# raw SDK response
raw = wrapper.raw_research("explain quantum entanglement")
```

Async variants are available for all methods (`results_async`, `raw_results_async`, `contents_async`, `research_text_async`, `raw_research_async`).

## Documentation

- [You.com API docs](https://docs.you.com)
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
