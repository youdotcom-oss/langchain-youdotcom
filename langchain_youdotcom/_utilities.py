"""You.com Search, Contents, Research, Answer, and Finance Research API wrapper."""

from __future__ import annotations

import os
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.documents import Document
from pydantic import BaseModel, Field, SecretStr, model_validator
from youdotcom import You
from youdotcom.models import (
    AnswerResponse,
    ContentsFormats,
    FinanceResearchEffort,
    FinanceResearchResponse,
    ResearchEffort,
)

if TYPE_CHECKING:
    from youdotcom.models import ResearchResponse, SearchResponse

_CLIENT_APP_NAME = "langchain-youdotcom"
_CLIENT_APP_VERSION = pkg_version("langchain-youdotcom")

# Research and Finance Research ``exhaustive`` can take up to 300 s; the
# SDK/httpx default is 5 s, which would time out before the study completes.
_RESEARCH_TIMEOUT_MS: int = 300_000


class YouAPIWrapper(BaseModel):
    r"""Wrapper around the You.com Search, Contents, Research, Answer APIs.

    Also includes the Finance Research API. Uses the ``youdotcom`` SDK for every
    call. Search and Contents return LangChain :class:`Document` objects;
    Research, Finance Research, and Answer return formatted markdown strings
    (with a numbered ``## Sources`` section for the research-shaped endpoints).

    Every outbound request emits an ``X-Client-Info`` attribution header of the form
    ``sdk; client=langchain-youdotcom/<version>; ua=python/<v> httpx/<v>`` so backend
    analytics can identify calls that originate from this package.

    Requires a ``YDC_API_KEY`` environment variable or an explicit ``ydc_api_key``
    parameter.

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouAPIWrapper

            wrapper = YouAPIWrapper(ydc_api_key="...")
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
        default=None,
        deprecated=True,
        description="Deprecated; use the Search API's ``extraction`` object. "
        "Will be removed in a future release.",
    )
    livecrawl_formats: list[str] | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated; use ``extraction.full_page.extraction_formats``. "
        "Will be removed in a future release.",
    )
    language: str | None = Field(default=None, description="Language code (BCP-47).")
    k: int | None = Field(default=None, description="Max documents to return.")
    n_snippets_per_hit: int | None = Field(
        default=None, description="Max snippets per search hit."
    )
    research_effort: str | None = Field(
        default=None,
        description=(
            "Research effort level. For the Research API: lite, standard, deep, "
            "or exhaustive. For the Finance Research API: deep or exhaustive "
            "only."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Read ``YDC_API_KEY`` from environment when not provided explicitly."""
        values["ydc_api_key"] = values.get("ydc_api_key") or os.environ.get(
            "YDC_API_KEY", ""
        )
        return values

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _make_client(self) -> You:
        """Build a fresh SDK client with attribution headers wired up.

        Letting the SDK construct its own httpx clients keeps ``__exit__`` /
        the GC finalizer in a position to close them. ``app_name`` and
        ``app_version`` cause the SDK to emit the ``X-Client-Info``
        attribution header on every outbound request, so we no longer need
        a custom user-agent on a user-supplied client.

        ``api_key_auth=None`` so SDK 3.0.0's stricter empty-string rejection
        does not fire when ``YDC_API_KEY`` is unset; the SDK reads the
        environment variable directly.
        """
        key = self.ydc_api_key.get_secret_value()
        return You(
            api_key_auth=key if key else None,
            app_name=_CLIENT_APP_NAME,
            app_version=_CLIENT_APP_VERSION,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_params(self, query: str) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query}
        for field in (
            "count",
            "safesearch",
            "country",
            "freshness",
            "offset",
            "language",
        ):
            val = getattr(self, field)
            if val is not None:
                params[field] = val
        # Forward deprecated livecrawl fields via __dict__ to bypass the
        # pydantic deprecated-field descriptor (which would warn on every
        # call). The SDK emits its own DeprecationWarning when set.
        for field in ("livecrawl", "livecrawl_formats"):
            val = self.__dict__.get(field)
            if val is not None:
                params[field] = val
        return params

    def raw_results(self, query: str) -> SearchResponse:
        """Call the You.com Search API and return the raw SDK response."""
        with self._make_client() as client:
            return client.search(**self._search_params(query))

    async def raw_results_async(self, query: str) -> SearchResponse:
        """Async variant of :meth:`raw_results`."""
        async with self._make_client() as client:
            return await client.search_async(**self._search_params(query))

    def results(self, query: str) -> list[Document]:
        """Search You.com and return parsed :class:`Document` objects."""
        return self._parse_search_response(self.raw_results(query))

    async def results_async(self, query: str) -> list[Document]:
        """Async variant of :meth:`results`."""
        return self._parse_search_response(await self.raw_results_async(query))

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    def contents(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: int | None = None,
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
            pages = client.contents(**params)
        return self._parse_contents_response(pages)

    async def contents_async(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: int | None = None,
    ) -> list[Document]:
        """Async variant of :meth:`contents`."""
        params = self._contents_params(
            urls, formats=formats, crawl_timeout=crawl_timeout
        )
        async with self._make_client() as client:
            pages = await client.contents_async(**params)
        return self._parse_contents_response(pages)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _research_params(self, query: str) -> dict[str, Any]:
        """Build kwargs for ``client.research()``, converting effort string to enum."""
        params: dict[str, Any] = {"input": query}
        if self.research_effort is not None:
            effort = ResearchEffort(self.research_effort)
            # ``frontier`` only works on the task-based API (``background=true``);
            # the sync path returns a 422. Reject it here so callers get a
            # clear local error instead of an SDK 422 after the round-trip.
            if effort is ResearchEffort.FRONTIER:
                _msg = (
                    "research_effort='frontier' is only supported by the "
                    "task-based API (background=true). YouResearchTool "
                    "targets the sync API; use the SDK directly for frontier."
                )
                raise ValueError(_msg)
            params["research_effort"] = effort
        return params

    def raw_research(self, query: str) -> ResearchResponse:
        """Call the You.com Research API and return the raw SDK response."""
        with self._make_client() as client:
            return client.research(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._research_params(query),
            )

    async def raw_research_async(self, query: str) -> ResearchResponse:
        """Async variant of :meth:`raw_research`."""
        async with self._make_client() as client:
            return await client.research_async(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._research_params(query),
            )

    def research_text(self, query: str) -> str:
        """Research a query and return formatted markdown with sources."""
        return self._format_research_response(self.raw_research(query))

    async def research_text_async(self, query: str) -> str:
        """Async variant of :meth:`research_text`."""
        return self._format_research_response(await self.raw_research_async(query))

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    _ANSWER_MAX_QUERY_CHARS: ClassVar[int] = 400

    def _answer_params(self, query: str, **filters: Any) -> dict[str, Any]:
        """Build kwargs for ``client.answer()``.

        Accepted filter kwargs:

        - ``freshness``: ``day`` / ``week`` / ``month`` / ``year``, or a date range.
        - ``country``: ISO 3166-1 alpha-2 country code.
        - ``language``: BCP 47 language tag.
        - ``safesearch``: ``off`` / ``moderate`` / ``strict``.
        - ``include_domains``: domains to restrict to (up to 500).
        - ``exclude_domains``: domains to exclude (up to 500). Cannot be
          combined with ``include_domains``.
        - ``boost_domains``: domains to prefer in ranking (up to 500). Cannot
          be combined with ``include_domains``.

        Returns:
            A kwargs dict suitable for ``client.answer(**kwargs)``.

        Raises:
            ValueError: When the include / exclude / boost combinations
                violate the Answer API's contract, when the query is empty,
                or when the query exceeds 400 characters.
        """
        if not query or not query.strip():
            _msg = "query is required for the Answer API"
            raise ValueError(_msg)
        if len(query) > self._ANSWER_MAX_QUERY_CHARS:
            _msg = (
                f"Answer API query is limited to "
                f"{self._ANSWER_MAX_QUERY_CHARS} characters; "
                f"got {len(query)}."
            )
            raise ValueError(_msg)
        if filters.get("include_domains") and filters.get("exclude_domains"):
            _msg = (
                "Cannot combine include_domains and exclude_domains on the Answer API"
            )
            raise ValueError(_msg)
        if filters.get("include_domains") and filters.get("boost_domains"):
            _msg = "Cannot combine include_domains and boost_domains on the Answer API"
            raise ValueError(_msg)

        params: dict[str, Any] = {"query": query}
        params.update((k, v) for k, v in filters.items() if v is not None)
        return params

    def raw_answer(self, query: str, **filters: Any) -> AnswerResponse:
        """Call the You.com Answer API and return the raw SDK response.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        with self._make_client() as client:
            return client.answer(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._answer_params(query, **filters),
            )

    async def raw_answer_async(self, query: str, **filters: Any) -> AnswerResponse:
        """Async variant of :meth:`raw_answer`.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        async with self._make_client() as client:
            return await client.answer_async(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._answer_params(query, **filters),
            )

    def answer_text(self, query: str, **filters: Any) -> str:
        """Answer a query and return formatted markdown with citations.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        return self._format_answer_response(self.raw_answer(query, **filters))

    async def answer_text_async(self, query: str, **filters: Any) -> str:
        """Async variant of :meth:`answer_text`.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        return self._format_answer_response(
            await self.raw_answer_async(query, **filters)
        )

    # ------------------------------------------------------------------
    # Finance Research
    # ------------------------------------------------------------------

    def _finance_research_params(self, query: str) -> dict[str, Any]:
        """Build kwargs for ``client.finance_research()``.

        The Finance Research API only accepts ``deep`` and ``exhaustive``; other
        values raise :class:`ValueError` so the user picks a compatible level.
        """
        effort_str = self.research_effort or "deep"
        try:
            effort = FinanceResearchEffort(effort_str)
        except ValueError:
            _msg = (
                "Finance Research only supports 'deep' and 'exhaustive', "
                f"got '{effort_str}'. Set research_effort to a compatible level."
            )
            raise ValueError(_msg) from None
        return {"input": query, "research_effort": effort}

    def raw_finance(self, query: str) -> FinanceResearchResponse:
        """Call the You.com Finance Research API and return the SDK response."""
        with self._make_client() as client:
            return client.finance_research(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._finance_research_params(query),
            )

    async def raw_finance_async(self, query: str) -> FinanceResearchResponse:
        """Async variant of :meth:`raw_finance`."""
        async with self._make_client() as client:
            return await client.finance_research_async(
                timeout_ms=_RESEARCH_TIMEOUT_MS,
                **self._finance_research_params(query),
            )

    def finance_text(self, query: str) -> str:
        """Research a financial query and return formatted markdown with sources."""
        return self._format_research_response(self.raw_finance(query))

    async def finance_text_async(self, query: str) -> str:
        """Async variant of :meth:`finance_text`."""
        return self._format_research_response(await self.raw_finance_async(query))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contents_params(
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: int | None = None,
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

    def _parse_search_response(self, response: Any) -> list[Document]:
        """Turn a :class:`SearchResponse` into a list of :class:`Document`.

        Honors ``k`` as a hard cap. For each Web hit whose
        ``contents.html``/``contents.markdown`` is populated by ``extraction``, the
        Document carries that content; otherwise snippets (capped by
        ``n_snippets_per_hit``) collapse into a single Document. News hits follow the
        same pattern. Web hits are emitted before news hits so a pinned ``k`` prefetches
        the more relevant corpus.
        """
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

    @staticmethod
    def _hit_metadata(
        hit: Any, source: str, *, include_favicon: bool = False
    ) -> dict[str, Any]:
        """Build a metadata dict shared by web and news hit parsers."""
        metadata: dict[str, Any] = {
            "url": hit.url or "",
            "title": hit.title or "",
            "description": hit.description or "",
            "source": source,
        }
        if hit.thumbnail_url:
            metadata["thumbnail_url"] = hit.thumbnail_url
        if include_favicon and hit.favicon_url:
            metadata["favicon_url"] = hit.favicon_url
        if hit.page_age:
            metadata["page_age"] = str(hit.page_age)
        return metadata

    def _parse_web_hit(self, hit: Any) -> list[Document]:
        metadata = self._hit_metadata(hit, "web", include_favicon=True)

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
        metadata = YouAPIWrapper._hit_metadata(hit, "news")

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
        pages: list[Any],
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
    def _numbered_section(
        body: str,
        entries: list[tuple[str, list[str]]] | None,
        *,
        section: str,
    ) -> str:
        """Append a numbered ``## <section>`` block after ``body``.

        Args:
            body: The leading markdown body. Always emitted first.
            entries: Pairs of ``(link_line, extra_lines)``. ``extra_lines``
                are appended after the numbered link (used for citation
                excerpts).
            section: The header name, prefixed with ``##``.

        Returns:
            The combined markdown string.
        """
        parts: list[str] = [body]
        if entries:
            lines = ["", f"## {section}", ""]
            for i, (link_line, extras) in enumerate(entries, 1):
                lines.append(f"{i}. {link_line}")
                lines.extend(extras)
            parts.append("\n".join(lines))
        return "\n".join(parts)

    @staticmethod
    def _format_research_response(response: Any) -> str:
        """Format a Research or Finance Research response as markdown with sources.

        The Research API and Finance Research API share a response shape
        (``output.content`` + ``output.sources``). Uses the URL as the link text when
        a source has no title.
        """
        sources = response.output.sources or []
        entries: list[tuple[str, list[str]]] = [
            (f"[{src.title or src.url}]({src.url})", []) for src in sources
        ]
        return YouAPIWrapper._numbered_section(
            response.output.content, entries, section="Sources"
        )

    @staticmethod
    def _format_answer_response(response: Any) -> str:
        """Format an Answer response as markdown with citations.

        The answer text is server-formatted markdown with ``[1]``, ``[2]``-style
        inline references; we only need to append a numbered citation list so the
        reader can resolve each reference.
        """
        entries: list[tuple[str, list[str]]] = []
        for citation in response.citations or []:
            url = citation.source
            excerpts = citation.excerpts or []
            entries.append((f"[{url}]({url})", [f"   > {e}" for e in excerpts]))
        return YouAPIWrapper._numbered_section(
            response.answer, entries, section="Citations"
        )


# Backward-compatible alias; remove in the next major version.
YouSearchAPIWrapper = YouAPIWrapper
