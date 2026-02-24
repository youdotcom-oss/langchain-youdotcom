# langchain-youdotcom

LangChain partner package for You.com search and content APIs.

## Build & test

```bash
uv sync --all-groups     # install all deps
make format              # ruff format + fix
make lint                # ruff check + format diff + mypy
make test                # unit tests (pytest)
make integration_tests   # integration tests (needs YDC_API_KEY)
make check_imports       # verify all .py files importable
```

## Architecture

- `langchain_youdotcom/_utilities.py` — `YouSearchAPIWrapper` (API client)
- `langchain_youdotcom/retrievers.py` — `YouRetriever` (BaseRetriever subclass)
- `langchain_youdotcom/tools.py` — `YouSearchTool` + `YouContentsTool` (BaseTool subclasses)

## Conventions

- Depends on `langchain-core` only, never full `langchain`
- Delegates HTTP to the `youdotcom` SDK
- Google-style docstrings, full type hints
- Ruff for linting/formatting, mypy for type checking
