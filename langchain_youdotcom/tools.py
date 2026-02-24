"""You.com tools for LangChain."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_youdotcom._utilities import YouSearchAPIWrapper


class YouSearchInput(BaseModel):
    """Input schema for :class:`YouSearchTool`."""

    query: str = Field(description="Search query to look up on You.com.")


class YouContentsInput(BaseModel):
    """Input schema for :class:`YouContentsTool`."""

    urls: list[str] = Field(description="URLs to fetch content from.")


def _format_docs(docs: list[Document]) -> str:
    """Join document contents with separators."""
    parts: list[str] = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        header = f"{title}\n{url}\n\n" if title or url else ""
        parts.append(f"{header}{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class YouSearchTool(BaseTool):
    """Tool that queries the You.com Search API.

    Requires a ``YDC_API_KEY`` environment variable or an explicit key on
    the ``api_wrapper``.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouSearchTool

            tool = YouSearchTool()
            result = tool.invoke("latest AI news")
    """

    name: str = "you_search"
    description: str = "Search the web using You.com and return relevant results."
    api_wrapper: YouSearchAPIWrapper = Field(default_factory=YouSearchAPIWrapper)
    args_schema: type[BaseModel] = YouSearchInput

    def _run(self, query: str, **kwargs: Any) -> str:
        """Run the You.com search tool.

        Args:
            query: The search query.
            **kwargs: Additional keyword arguments.

        Returns:
            Search results formatted as a string.
        """
        return _format_docs(self.api_wrapper.results(query))

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async run the You.com search tool.

        Args:
            query: The search query.
            **kwargs: Additional keyword arguments.

        Returns:
            Search results formatted as a string.
        """
        return _format_docs(await self.api_wrapper.results_async(query))


class YouContentsTool(BaseTool):
    """Tool that fetches page contents via the You.com Contents API.

    Requires a ``YDC_API_KEY`` environment variable or an explicit key on
    the ``api_wrapper``.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouContentsTool

            tool = YouContentsTool()
            result = tool.invoke({"urls": ["https://example.com"]})
    """

    name: str = "you_contents"
    description: str = "Fetch and extract content from web pages using You.com."
    api_wrapper: YouSearchAPIWrapper = Field(default_factory=YouSearchAPIWrapper)
    args_schema: type[BaseModel] = YouContentsInput

    def _run(self, urls: list[str], **kwargs: Any) -> str:
        """Run the You.com contents tool.

        Args:
            urls: URLs to fetch content from.
            **kwargs: Additional keyword arguments.

        Returns:
            Page contents formatted as a string.
        """
        return _format_docs(self.api_wrapper.contents(urls))

    async def _arun(self, urls: list[str], **kwargs: Any) -> str:
        """Async run the You.com contents tool.

        Args:
            urls: URLs to fetch content from.
            **kwargs: Additional keyword arguments.

        Returns:
            Page contents formatted as a string.
        """
        return _format_docs(await self.api_wrapper.contents_async(urls))
