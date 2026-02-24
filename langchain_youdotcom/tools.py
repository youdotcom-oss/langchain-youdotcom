"""You.com tools for LangChain."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


class YouSearchTool(BaseTool):
    """Tool that queries the You.com Search API.

    Requires a ``YDC_API_KEY`` environment variable.
    """

    name: str = "you_search"
    description: str = "Search the web using You.com and return relevant results."

    def _run(self, query: str, **kwargs: Any) -> str:
        """Run the You.com search tool.

        Args:
            query: The search query.
            **kwargs: Additional keyword arguments.

        Returns:
            Search results as a string.

        Raises:
            NotImplementedError: Stub — will be implemented in DX-335.
        """
        raise NotImplementedError


class YouContentsTool(BaseTool):
    """Tool that fetches page contents via the You.com Contents API.

    Requires a ``YDC_API_KEY`` environment variable.
    """

    name: str = "you_contents"
    description: str = "Fetch and extract content from web pages using You.com."

    def _run(self, url: str, **kwargs: Any) -> str:
        """Run the You.com contents tool.

        Args:
            url: The URL to fetch content from.
            **kwargs: Additional keyword arguments.

        Returns:
            Page content as a string.

        Raises:
            NotImplementedError: Stub — will be implemented in DX-335.
        """
        raise NotImplementedError
