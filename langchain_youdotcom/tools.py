"""You.com tools for LangChain."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_youdotcom._utilities import YouAPIWrapper


class YouSearchInput(BaseModel):
    """Input schema for :class:`YouSearchTool`."""

    query: str = Field(description="Search query to look up on You.com.")


class YouResearchInput(BaseModel):
    """Input schema for :class:`YouResearchTool`."""

    query: str = Field(description="Research query to investigate with You.com.")


class YouContentsInput(BaseModel):
    """Input schema for :class:`YouContentsTool`."""

    urls: list[str] = Field(description="URLs to fetch content from.")


class YouFinanceResearchInput(BaseModel):
    """Input schema for :class:`YouFinanceResearchTool`."""

    query: str = Field(
        description="Financial research query to investigate with You.com."
    )


class YouAnswerInput(BaseModel):
    """Input schema for :class:`YouAnswerTool`.

    Mirrors the ``POST /v1/answer`` request body. The Answer API rejects
    ``include_domains`` paired with either ``exclude_domains`` or
    ``boost_domains``; the wrapper raises a ``ValueError`` locally so callers
    fail fast instead of round-tripping a 422.
    """

    query: str = Field(
        description="Focused live-web question. Max 400 characters.",
    )
    freshness: str | None = Field(
        default=None,
        description=(
            "Freshness filter: ``day``, ``week``, ``month``, ``year``, or "
            "``YYYY-MM-DDtoYYYY-MM-DD``."
        ),
    )
    country: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 country code for geographical focus.",
    )
    language: str | None = Field(
        default=None,
        description="BCP 47 language tag.",
    )
    safesearch: str | None = Field(
        default=None,
        description="Content filter: ``off``, ``moderate``, or ``strict``.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to exclusively include (up to 500). Cannot combine with "
            "``exclude_domains`` or ``boost_domains``."
        ),
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to exclude (up to 500). Can combine with ``boost_domains`` "
            "but not ``include_domains``."
        ),
    )
    boost_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to prefer in ranking (up to 500). Can combine with "
            "``exclude_domains`` but not ``include_domains``."
        ),
    )


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
    api_wrapper: YouAPIWrapper = Field(default_factory=YouAPIWrapper)
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


class YouResearchTool(BaseTool):
    """Tool that queries the You.com Research API.

    Returns comprehensive, research-grade answers with multi-step reasoning
    and cited sources.

    Requires a ``YDC_API_KEY`` environment variable or an explicit key on
    the ``api_wrapper``.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouResearchTool

            tool = YouResearchTool()
            result = tool.invoke("what are the latest advances in quantum computing")
    """

    name: str = "you_research"
    description: str = (
        "Research a topic in depth using You.com and return a comprehensive "
        "answer with cited sources."
    )
    api_wrapper: YouAPIWrapper = Field(default_factory=YouAPIWrapper)
    args_schema: type[BaseModel] = YouResearchInput

    def _run(self, query: str, **kwargs: Any) -> str:
        """Run the You.com research tool.

        Args:
            query: The research query.
            **kwargs: Additional keyword arguments.

        Returns:
            Research answer formatted as markdown with sources.
        """
        return self.api_wrapper.research_text(query)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async run the You.com research tool.

        Args:
            query: The research query.
            **kwargs: Additional keyword arguments.

        Returns:
            Research answer formatted as markdown with sources.
        """
        return await self.api_wrapper.research_text_async(query)


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
    api_wrapper: YouAPIWrapper = Field(default_factory=YouAPIWrapper)
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


class YouFinanceResearchTool(BaseTool):
    """Tool that queries the You.com Finance Research API.

    Returns comprehensive, citation-backed answers to financial questions
    from a finance-optimized index covering SEC filings, earnings, equity
    prices, macro indicators, and financial news.

    Requires a ``YDC_API_KEY`` environment variable or an explicit key on
    the ``api_wrapper``.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouFinanceResearchTool

            tool = YouFinanceResearchTool()
            result = tool.invoke("what drove NVIDIA's revenue growth in FY2025")
    """

    name: str = "you_finance_research"
    description: str = (
        "Research a financial question in depth using You.com Finance Research "
        "and return a comprehensive answer with cited sources from a "
        "finance-optimized index (SEC filings, earnings, equity prices, "
        "macro indicators)."
    )
    api_wrapper: YouAPIWrapper = Field(default_factory=YouAPIWrapper)
    args_schema: type[BaseModel] = YouFinanceResearchInput

    def _run(self, query: str, **kwargs: Any) -> str:
        """Run the You.com finance research tool.

        Args:
            query: The financial research query.
            **kwargs: Additional keyword arguments.

        Returns:
            Finance research answer formatted as markdown with sources.
        """
        return self.api_wrapper.finance_text(query)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async run the You.com finance research tool.

        Args:
            query: The financial research query.
            **kwargs: Additional keyword arguments.

        Returns:
            Finance research answer formatted as markdown with sources.
        """
        return await self.api_wrapper.finance_text_async(query)


class YouAnswerTool(BaseTool):
    """Tool that queries the You.com Answer API.

    Returns a single synthesized natural-language answer with inline citations
    and an appended ``## Citations`` section that lists each cited source URL
    along with any supporting excerpts the server provided.

    Requires a ``YDC_API_KEY`` environment variable or an explicit key on the
    ``api_wrapper``.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouAnswerTool

            tool = YouAnswerTool()
            result = tool.invoke({"query": "what is retrieval augmented generation"})
    """

    name: str = "you_answer"
    description: str = (
        "Get a synthesized, cited answer to a single focused live-web question. "
        "Supports freshness, country, language, safesearch, and domain include / "
        "exclude / boost filters. Returns the synthesized answer plus a Citations "
        "section listing each source URL."
    )
    api_wrapper: YouAPIWrapper = Field(default_factory=YouAPIWrapper)
    args_schema: type[BaseModel] = YouAnswerInput

    def _run(self, query: str, **filters: Any) -> str:
        """Run the You.com Answer tool.

        Args:
            query: The live-web question. Max 400 characters.
            **filters: Answer-API filters forwarded to ``api_wrapper.answer_text``.
                Supported keys: ``freshness``, ``country``, ``language``,
                ``safesearch``, ``include_domains``, ``exclude_domains``,
                ``boost_domains``.

        Returns:
            Synthesized answer formatted as markdown with citations.
        """
        return self.api_wrapper.answer_text(query, **filters)

    async def _arun(self, query: str, **filters: Any) -> str:
        """Async run the You.com Answer tool.

        Args:
            query: The live-web question. Max 400 characters.
            **filters: Answer filters forwarded to ``api_wrapper.answer_text_async``.
                Supported keys: ``freshness``, ``country``, ``language``,
                ``safesearch``, ``include_domains``, ``exclude_domains``,
                ``boost_domains``.

        Returns:
            Synthesized answer formatted as markdown with citations.
        """
        return await self.api_wrapper.answer_text_async(query, **filters)
