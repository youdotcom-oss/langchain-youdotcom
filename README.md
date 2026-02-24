# langchain-youdotcom

LangChain partner package for [You.com](https://you.com).

## Installation

```bash
pip install langchain-youdotcom
```

## Setup

Set your API key:

```bash
export YDC_API_KEY="your-api-key"
```

Get an API key at [api.you.com](https://api.you.com).

## Usage

```python
from langchain_youdotcom import YouRetriever, YouSearchTool, YouContentsTool
```

### Retriever

```python
retriever = YouRetriever()
docs = retriever.invoke("latest AI news")
```

### Tools

```python
search = YouSearchTool()
contents = YouContentsTool()
```

## Development

```bash
uv sync --all-groups
make format
make lint
make test
```
