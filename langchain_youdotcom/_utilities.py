"""You.com Search, Contents, and Research API wrapper."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.documents import Document
from pydantic import BaseModel, Field, SecretStr, model_validator
from youdotcom import You
from youdotcom.models import ContentsFormats, ResearchEffort

if TYPE_CHECKING:
    from youdotcom.models import ContentsResponse, ResearchResponse, SearchResponse


class YouSearchAPIWrapper(BaseModel):
    """Wrapper around the You.com Search, Contents, and Research APIs.

    Uses the ``youdotcom`` SDK to call the You.com Search, Contents, and
    Research APIs, returning results as LangChain :class:`Document` objects
    or formatted text.

    Requires a ``YDC_API_KEY`` environment variable or explicit
    ``ydc_api_key`` parameter.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouSearchAPIWrapper

            wrapper = YouSearchAPIWrapper(ydc_api_key="...")
            docs = wrapper.results("latest AI news")
    """

    ydc_api_key: SecretStr = Field(
        default=SecretStr(""), description="You.com API key."
    )
    count: int | None = Field(default=None, description="Max results per section.")
    safesearch: str | None = Field(
        default=None,
        description="Safe-search level: off, moderate, or strict.",
    )
    country: str | None = Field(default=None, description="Country code filter.")
    freshness: str | None = Field(
        default=None,
        description="Freshness filter: day, week, month, or year.",
    )
    offset: int | None = Field(default=None, description="Pagination offset (0-9).")
    livecrawl: str | None = Field(
        default=None, description="Live-crawl mode: web, news, or all."
    )
    livecrawl_formats: str | None = Field(
        default=None, description="Live-crawl format: html or markdown."
    )
    language: str | None = Field(default=None, description="Language code (BCP-47).")
    k: int | None = Field(default=None, description="Max documents to return.")
    n_snippets_per_hit: int | None = Field(
        default=None, description="Max snippets per search hit."
    )
    research_effort: str | None = Field(
        default=None,
        description="Research effort level: lite, standard, deep, or exhaustive.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Read YDC_API_KEY from environment when not provided explicitly."""
        values["ydc_api_key"] = values.get("ydc_api_key") or os.environ.get(
            "YDC_API_KEY", ""
        )
        return values

    def _make_client(self) -> You:
        ua = {"user-agent": "langchain-youdotcom/0.1.0"}
        return You(
            api_key_auth=self.ydc_api_key.get_secret_value(),
            client=httpx.Client(headers=ua),
            async_client=httpx.AsyncClient(headers=ua),
        )

    def _search_params(self, query: str) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query}
        for field in (
            "count",
            "safesearch",
            "country",
            "freshness",
            "offset",
            "livecrawl",
            "livecrawl_formats",
            "language",
        ):
            val = getattr(self, field)
            if val is not None:
                params[field] = val
        return params

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def raw_results(self, query: str) -> SearchResponse:
        """Call the You.com Search API and return the raw SDK response.

        Args:
            query: The search query.

        Returns:
            The raw ``SearchResponse`` from the ``youdotcom`` SDK.
        """
        with self._make_client() as client:
            return client.search.unified(**self._search_params(query))

    async def raw_results_async(self, query: str) -> SearchResponse:
        """Async variant of :meth:`raw_results`.

        Args:
            query: The search query.

        Returns:
            The raw ``SearchResponse`` from the ``youdotcom`` SDK.
        """
        async with self._make_client() as client:
            return await client.search.unified_async(**self._search_params(query))

    def results(self, query: str) -> list[Document]:
        """Search You.com and return parsed :class:`Document` objects.

        Args:
            query: The search query.

        Returns:
            A list of documents with page content and metadata.
        """
        return self._parse_search_response(self.raw_results(query))

    async def results_async(self, query: str) -> list[Document]:
        """Async variant of :meth:`results`.

        Args:
            query: The search query.

        Returns:
            A list of documents with page content and metadata.
        """
        return self._parse_search_response(await self.raw_results_async(query))

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    def contents(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: float | None = None,
    ) -> list[Document]:
        """Fetch page contents via the You.com Contents API.

        Args:
            urls: URLs to crawl.
            formats: Content formats to return (``"html"``, ``"markdown"``,
                ``"metadata"``). Defaults to ``["markdown", "metadata"]``.
            crawl_timeout: Per-URL crawl timeout in seconds.

        Returns:
            A list of documents with page content and metadata.
        """
        params = self._contents_params(
            urls, formats=formats, crawl_timeout=crawl_timeout
        )
        with self._make_client() as client:
            pages = client.contents.generate(**params)
        return self._parse_contents_response(pages)

    async def contents_async(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: float | None = None,
    ) -> list[Document]:
        """Async variant of :meth:`contents`.

        Args:
            urls: URLs to crawl.
            formats: Content formats to return (``"html"``, ``"markdown"``,
                ``"metadata"``). Defaults to ``["markdown", "metadata"]``.
            crawl_timeout: Per-URL crawl timeout in seconds.

        Returns:
            A list of documents with page content and metadata.
        """
        params = self._contents_params(
            urls, formats=formats, crawl_timeout=crawl_timeout
        )
        async with self._make_client() as client:
            pages = await client.contents.generate_async(**params)
        return self._parse_contents_response(pages)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _research_params(self, query: str) -> dict[str, Any]:
        """Build kwargs for ``client.research()``, converting effort string to enum."""
        params: dict[str, Any] = {"input": query}
        if self.research_effort is not None:
            params["research_effort"] = ResearchEffort(self.research_effort)
        return params

    def raw_research(self, query: str) -> ResearchResponse:
        """Call the You.com Research API and return the raw SDK response.

        Args:
            query: The research query.

        Returns:
            The raw ``ResearchResponse`` from the ``youdotcom`` SDK.
        """
        with self._make_client() as client:
            return client.research(**self._research_params(query))

    async def raw_research_async(self, query: str) -> ResearchResponse:
        """Async variant of :meth:`raw_research`.

        Args:
            query: The research query.

        Returns:
            The raw ``ResearchResponse`` from the ``youdotcom`` SDK.
        """
        async with self._make_client() as client:
            return await client.research_async(**self._research_params(query))

    def research_text(self, query: str) -> str:
        """Research a query and return formatted markdown with sources.

        Args:
            query: The research query.

        Returns:
            Markdown answer followed by a sources section.
        """
        return self._format_research_response(self.raw_research(query))

    async def research_text_async(self, query: str) -> str:
        """Async variant of :meth:`research_text`.

        Args:
            query: The research query.

        Returns:
            Markdown answer followed by a sources section.
        """
        return self._format_research_response(await self.raw_research_async(query))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contents_params(
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"urls": urls}
        if formats is not None:
            params["formats"] = [ContentsFormats(f) for f in formats]
        else:
            params["formats"] = [
                ContentsFormats.MARKDOWN,
                ContentsFormats.METADATA,
            ]
        if crawl_timeout is not None:
            params["crawl_timeout"] = crawl_timeout
        return params

    def _parse_search_response(self, response: SearchResponse) -> list[Document]:
        docs: list[Document] = []
        if not response.results:
            return docs

        for hit in response.results.web or []:
            docs.extend(self._parse_web_hit(hit))
            if self.k is not None and len(docs) >= self.k:
                return docs[: self.k]

        for news_hit in response.results.news or []:
            doc = self._parse_news_hit(news_hit)
            if doc:
                docs.append(doc)
                if self.k is not None and len(docs) >= self.k:
                    return docs[: self.k]

        return docs[: self.k] if self.k is not None else docs

    def _parse_web_hit(self, hit: Any) -> list[Document]:
        metadata: dict[str, Any] = {
            "url": hit.url or "",
            "title": hit.title or "",
            "description": hit.description or "",
        }
        if hit.thumbnail_url:
            metadata["thumbnail_url"] = hit.thumbnail_url
        if hit.favicon_url:
            metadata["favicon_url"] = hit.favicon_url
        if hit.page_age:
            metadata["page_age"] = str(hit.page_age)

        if hit.contents:
            md = hit.contents.markdown
            html = hit.contents.html
            if md:
                return [Document(page_content=md, metadata=metadata)]
            if html:
                return [Document(page_content=html, metadata=metadata)]

        snippets = hit.snippets or []
        if self.n_snippets_per_hit is not None:
            snippets = snippets[: self.n_snippets_per_hit]

        if snippets:
            return [Document(page_content="\n".join(snippets), metadata=metadata)]
        if hit.description:
            return [Document(page_content=hit.description, metadata=metadata)]
        return []

    @staticmethod
    def _parse_news_hit(hit: Any) -> Document | None:
        metadata: dict[str, Any] = {
            "url": hit.url or "",
            "title": hit.title or "",
            "description": hit.description or "",
        }
        if hit.thumbnail_url:
            metadata["thumbnail_url"] = hit.thumbnail_url
        if hit.page_age:
            metadata["page_age"] = str(hit.page_age)

        content = ""
        if hit.contents:
            content = hit.contents.markdown or hit.contents.html or ""
        if not content:
            content = hit.description or ""

        if content:
            return Document(page_content=content, metadata=metadata)
        return None

    @staticmethod
    def _parse_contents_response(
        pages: list[ContentsResponse],
    ) -> list[Document]:
        docs: list[Document] = []
        for page in pages:
            # OptionalNullable fields: UNSET is falsy, so truthiness works.
            content = page.markdown if page.markdown else page.html
            if not content:
                continue

            metadata: dict[str, Any] = {
                "url": page.url or "",
                "title": page.title or "",
            }
            page_meta = page.metadata
            if page_meta:
                if page_meta.site_name:
                    metadata["site_name"] = page_meta.site_name
                if page_meta.favicon_url:
                    metadata["favicon_url"] = page_meta.favicon_url

            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    @staticmethod
    def _format_research_response(response: ResearchResponse) -> str:
        """Format a research response as markdown with a numbered sources section."""
        parts: list[str] = [response.output.content]
        if response.output.sources:
            lines = ["", "## Sources", ""]
            for i, src in enumerate(response.output.sources, 1):
                title = src.title or src.url
                lines.append(f"{i}. [{title}]({src.url})")
            parts.append("\n".join(lines))
        return "\n".join(parts)
