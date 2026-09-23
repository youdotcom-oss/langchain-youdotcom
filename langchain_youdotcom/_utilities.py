"""You.com Search, Contents, Research, Answer, and Finance Research API wrapper."""

from __future__ import annotations

import json
import os
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.documents import Document
from pydantic import BaseModel, Field, SecretStr, model_validator
from youdotcom import You
from youdotcom.models import (
    ContentsFormats,
    FinanceResearchEffort,
    ResearchEffort,
)

if TYPE_CHECKING:
    from youdotcom.models import (
        AnswerResponse,
        FinanceResearchResponse,
        ResearchResponse,
        SearchResponse,
    )

_CLIENT_APP_NAME = "langchain-youdotcom"
_CLIENT_APP_VERSION = pkg_version("langchain-youdotcom")

# Research, Finance Research, and Answer ``exhaustive`` runs can take up to
# 300 s; the SDK/httpx default is 5 s, which would time out first. Shared by
# all three rather than per-endpoint, since none of them has a documented
# budget tighter than this and a hung call is the failure mode being bounded.
_LONG_RUNNING_TIMEOUT_MS: int = 300_000

# The SDK builds its httpx clients without an explicit timeout, so httpx's 5 s
# default applies to any call that does not pass ``timeout_ms``. Search and
# Contents can legitimately exceed that: ``crawl_timeout`` is accepted up to 60 s
# server-side, and ``extraction`` in ``full_page`` mode crawls every result.
# Scale the client budget to the server-side crawl budget plus headroom, so the
# client never gives up while the server is still inside its own limit.
_DEFAULT_CRAWL_TIMEOUT_S: int = 10
_CRAWL_HEADROOM_S: int = 20

# Filters the Search and Answer endpoints both accept, and which therefore exist
# as wrapper fields. Answer seeds these from the wrapper so shared config applies
# to both endpoints; Search reads them in ``_search_params``.
_SHARED_FILTER_FIELDS: tuple[str, ...] = (
    "country",
    "language",
    "safesearch",
    "freshness",
    "include_domains",
    "exclude_domains",
    "boost_domains",
)


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
    knowledge: str | None = Field(
        default=None,
        description=(
            'Set to ``"core"`` to request knowledge results: cards backed by '
            "licensed data providers such as encyclopedias, market-data firms, "
            "and reference publishers. They arrive in their own ``knowledge`` "
            'section of the response. ``"core"`` is the only accepted value; '
            "anything else raises a local ``ValidationError``. Omit to skip "
            "them. ``count`` does not cap this section; ``k`` does."
        ),
    )
    extraction: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Controls how page content is attached to each result. Preferred "
            "over the deprecated ``livecrawl`` / ``livecrawl_formats``. Mirrors "
            "the SDK's ``ExtractionTypedDict``: ``extraction_mode`` is required "
            '(``"highlights"`` or ``"full_page"``); ``extraction_source`` '
            '(``"blend"``, ``"cache"``, or ``"fetch"``) and ``full_page`` apply '
            'to ``"full_page"`` only. Cannot be combined with ``livecrawl``.'
        ),
    )
    include_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to exclusively include (up to 500). Cannot be combined "
            "with ``exclude_domains`` or ``boost_domains``."
        ),
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to exclude (up to 500). Can be combined with "
            "``boost_domains`` but not ``include_domains``."
        ),
    )
    boost_domains: list[str] | None = Field(
        default=None,
        description=(
            "Domains to prefer in ranking (up to 500). Can be combined with "
            "``exclude_domains`` but not ``include_domains``."
        ),
    )
    crawl_timeout: int | None = Field(
        default=None,
        description=(
            "Per-page crawl timeout in seconds for Search; the API defaults to "
            "10. Most useful with ``extraction`` in ``full_page`` mode. Note "
            "that :meth:`contents` takes its own ``crawl_timeout`` argument "
            "rather than reading this field."
        ),
    )
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
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema constraining the Research API's structured output. "
            "Applies to :meth:`research_text` / :meth:`raw_research` only and "
            "is passed through to the SDK unmodified."
        ),
    )
    source_control: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Controls which sources the Research API may draw on. Mirrors the "
            "SDK's ``SourceControl``: ``include_domains``, ``exclude_domains``, "
            "``boost_domains``, ``freshness``, and ``country``. Applies to the "
            "Research API only — for Search, set the top-level "
            "``include_domains`` / ``exclude_domains`` / ``boost_domains`` "
            "fields instead."
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

        ``api_key_auth=None`` when no key is available so SDK 3.0.0's stricter
        empty-string rejection does not fire; the SDK then falls back to its own
        ``YDC_API_KEY`` / ``YOU_API_KEY_AUTH`` environment lookup.

        Raises:
            ValueError: When no API key is available from either the
                ``ydc_api_key`` field or the environment variables the SDK
                honors. Sending the request unauthenticated instead produces a
                ``402 payment_required`` telling the caller to buy credits,
                which misdirects them away from the real cause.
        """
        key = self.ydc_api_key.get_secret_value()
        if not key and not os.getenv("YOU_API_KEY_AUTH"):
            _msg = (
                "No You.com API key found. Pass ydc_api_key=... or set the "
                "YDC_API_KEY environment variable. Get a key at "
                "https://you.com/platform/api-keys."
            )
            raise ValueError(_msg)
        return You(
            api_key_auth=key if key else None,
            app_name=_CLIENT_APP_NAME,
            app_version=_CLIENT_APP_VERSION,
        )

    @staticmethod
    def _request_timeout_ms(crawl_timeout: int | None) -> int:
        """Client timeout in ms that can outlast the server-side crawl budget.

        Args:
            crawl_timeout: The caller's per-page crawl budget in seconds, or
                ``None`` to use the API's 10 s default.

        Returns:
            The budget plus headroom for queueing and response transfer.
        """
        crawl = crawl_timeout or _DEFAULT_CRAWL_TIMEOUT_S
        return (crawl + _CRAWL_HEADROOM_S) * 1000

    @staticmethod
    def _validate_domain_filters(params: dict[str, Any], api_name: str) -> None:
        """Reject domain filter combinations the API answers with a 422.

        ``include_domains`` is mutually exclusive with both ``exclude_domains``
        and ``boost_domains``, which may be combined with each other. Shared by
        the Search and Answer paths so the rule lives in one place.

        Args:
            params: The merged kwargs about to be sent, so wrapper-level config
                and per-call filters are validated together.
            api_name: Endpoint name used in the error message.

        Raises:
            ValueError: On an illegal combination.
        """
        if params.get("include_domains") and params.get("exclude_domains"):
            _msg = (
                f"Cannot combine include_domains and exclude_domains on the "
                f"{api_name} API"
            )
            raise ValueError(_msg)
        if params.get("include_domains") and params.get("boost_domains"):
            _msg = (
                f"Cannot combine include_domains and boost_domains on the "
                f"{api_name} API"
            )
            raise ValueError(_msg)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_params(self, query: str) -> dict[str, Any]:
        """Build kwargs for ``client.search()``.

        Raises:
            ValueError: When ``include_domains`` is combined with
                ``exclude_domains`` or ``boost_domains``. The API rejects the
                combination with a 422; this mirrors ``_answer_params`` so
                callers get a clear local error instead of a round-trip.
        """
        params: dict[str, Any] = {"query": query}
        for field in (
            "count",
            "safesearch",
            "country",
            "freshness",
            "offset",
            "language",
            "knowledge",
            "extraction",
            "include_domains",
            "exclude_domains",
            "boost_domains",
            "crawl_timeout",
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
        self._validate_domain_filters(params, "Search")
        return params

    def raw_results(self, query: str) -> SearchResponse:
        """Call the You.com Search API and return the raw SDK response."""
        with self._make_client() as client:
            return client.search(
                **self._search_params(query),
                timeout_ms=self._request_timeout_ms(self.crawl_timeout),
            )

    async def raw_results_async(self, query: str) -> SearchResponse:
        """Async variant of :meth:`raw_results`."""
        async with self._make_client() as client:
            return await client.search_async(
                **self._search_params(query),
                timeout_ms=self._request_timeout_ms(self.crawl_timeout),
            )

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
        max_age: int | None = None,
    ) -> list[Document]:
        """Fetch page contents via the You.com Contents API.

        Args:
            urls: URLs to crawl. At most 10 per request.
            formats: Content formats to return (``"html"``, ``"markdown"``,
                ``"metadata"``). Defaults to ``["markdown"]``. ``"metadata"`` is
                deprecated by the SDK and emits a ``DeprecationWarning`` at
                request time; pass it explicitly only if you need the
                ``site_name`` / ``favicon_url`` metadata keys. A page is only
                returned when it produced content in a requested format, so
                passing only ``["metadata"]`` yields no documents — include
                ``markdown`` or ``html`` as well.
            crawl_timeout: Per-URL crawl timeout in seconds.
            max_age: Maximum allowed age of cached content in seconds. Cached
                content older than this is ignored and the page is re-fetched.
                Must be 0 or greater. Defaults to no age limit.

        Returns:
            A list of documents with page content and metadata.

        Raises:
            ValueError: When more than 10 URLs are passed. The API rejects the
                request with a 422 and the SDK does not validate this locally.
        """
        params = self._contents_params(
            urls, formats=formats, crawl_timeout=crawl_timeout, max_age=max_age
        )
        with self._make_client() as client:
            pages = client.contents(
                **params, timeout_ms=self._request_timeout_ms(crawl_timeout)
            )
        return self._parse_contents_response(pages)

    async def contents_async(
        self,
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: int | None = None,
        max_age: int | None = None,
    ) -> list[Document]:
        """Async variant of :meth:`contents`."""
        params = self._contents_params(
            urls, formats=formats, crawl_timeout=crawl_timeout, max_age=max_age
        )
        async with self._make_client() as client:
            pages = await client.contents_async(
                **params, timeout_ms=self._request_timeout_ms(crawl_timeout)
            )
        return self._parse_contents_response(pages)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _research_params(self, query: str) -> dict[str, Any]:
        """Build kwargs for ``client.research()``, converting effort string to enum."""
        params: dict[str, Any] = {"input": query}
        effort: ResearchEffort | None = None
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
        if self.output_schema is not None:
            # Same fail-fast rationale as ``frontier``: ``lite`` plus
            # ``output_schema`` is a 422 from the API.
            if effort is ResearchEffort.LITE:
                _msg = (
                    "output_schema is not supported with "
                    "research_effort='lite'; the API returns a 422. Use "
                    "standard, deep, or exhaustive."
                )
                raise ValueError(_msg)
            params["output_schema"] = self.output_schema
        if self.source_control is not None:
            params["source_control"] = self.source_control
        return params

    def raw_research(self, query: str) -> ResearchResponse:
        """Call the You.com Research API and return the raw SDK response."""
        with self._make_client() as client:
            return client.research(
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
                **self._research_params(query),
            )

    async def raw_research_async(self, query: str) -> ResearchResponse:
        """Async variant of :meth:`raw_research`."""
        async with self._make_client() as client:
            return await client.research_async(
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
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

        Each of these is also a ``YouAPIWrapper`` field. Wrapper-level config
        applies as the default and a per-call filter overrides it, matching how
        ``_search_params`` treats the same fields — so a wrapper configured with
        ``country="US"`` behaves consistently across Search and Answer.

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
        params: dict[str, Any] = {"query": query}
        # Wrapper-level config is the default; a per-call filter overrides it.
        for field in _SHARED_FILTER_FIELDS:
            val = getattr(self, field)
            if val is not None:
                params[field] = val
        params.update((k, v) for k, v in filters.items() if v is not None)
        # Validate the merged result so an illegal combination formed partly by
        # wrapper config is still caught before the round-trip.
        self._validate_domain_filters(params, "Answer")
        return params

    def raw_answer(self, query: str, **filters: Any) -> AnswerResponse:
        """Call the You.com Answer API and return the raw SDK response.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        with self._make_client() as client:
            return client.answer(
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
                **self._answer_params(query, **filters),
            )

    async def raw_answer_async(self, query: str, **filters: Any) -> AnswerResponse:
        """Async variant of :meth:`raw_answer`.

        See :meth:`_answer_params` for the accepted filter kwargs.
        """
        async with self._make_client() as client:
            return await client.answer_async(
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
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
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
                **self._finance_research_params(query),
            )

    async def raw_finance_async(self, query: str) -> FinanceResearchResponse:
        """Async variant of :meth:`raw_finance`."""
        async with self._make_client() as client:
            return await client.finance_research_async(
                timeout_ms=_LONG_RUNNING_TIMEOUT_MS,
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

    # The Contents API rejects a request carrying more URLs than this with a
    # 422. The SDK neither documents nor validates the limit, so surface it
    # locally in the same spirit as ``_ANSWER_MAX_QUERY_CHARS``.
    _CONTENTS_MAX_URLS: ClassVar[int] = 10

    @staticmethod
    def _contents_params(
        urls: list[str],
        *,
        formats: list[str] | None = None,
        crawl_timeout: int | None = None,
        max_age: int | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for ``client.contents()``.

        Raises:
            ValueError: When more than 10 URLs are passed. The API answers with
                a 422 and the SDK does not check this locally.
        """
        limit = YouAPIWrapper._CONTENTS_MAX_URLS
        if len(urls) > limit:
            _msg = (
                f"The Contents API accepts at most {limit} URLs per request; "
                f"got {len(urls)}. Split the call into batches."
            )
            raise ValueError(_msg)
        params: dict[str, Any] = {"urls": urls}
        if formats is not None:
            params["formats"] = [ContentsFormats(f) for f in formats]
        else:
            # ``metadata`` is deliberately excluded from the default: the SDK
            # deprecated the format in 3.4.0 and warns at request time whenever
            # it appears in ``formats``, so defaulting to it would surface a
            # DeprecationWarning on every call. Pass it explicitly to opt back in.
            params["formats"] = [ContentsFormats.MARKDOWN]
        if crawl_timeout is not None:
            params["crawl_timeout"] = crawl_timeout
        if max_age is not None:
            params["max_age"] = max_age
        return params

    def _parse_search_response(self, response: Any) -> list[Document]:
        """Turn a :class:`SearchResponse` into a list of :class:`Document`.

        Honors ``k`` as a hard cap. For each Web hit whose
        ``contents.html``/``contents.markdown`` is populated by ``extraction``, the
        Document carries that content; otherwise snippets (capped by
        ``n_snippets_per_hit``) collapse into a single Document. News hits follow the
        same pattern.

        Sections are emitted in relevance order — knowledge, then web, then news —
        so a pinned ``k`` prefetches the most authoritative corpus. Knowledge
        results only appear when ``knowledge`` is set on the wrapper, so this
        ordering cannot change results for callers that do not opt in.

        Note that the API's ``count`` parameter caps the web and news sections
        only; up to 25 knowledge results can arrive regardless. ``k`` is the sole
        bound on the returned list.
        """
        docs: list[Document] = []
        if not response.results:
            return docs

        for knowledge_hit in response.results.knowledge or []:
            doc = self._parse_knowledge_hit(knowledge_hit)
            if doc:
                docs.append(doc)
                if self.k is not None and len(docs) >= self.k:
                    return docs[: self.k]

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

    @staticmethod
    def _contents_text(contents: Any, *, cap: int | None = None) -> str:
        """Best available page content from a hit's ``contents`` block.

        Precedence is markdown, then html, then highlights. ``highlights`` is
        what the API populates for ``extraction_mode="highlights"``: in that
        mode markdown and html are both absent and ``snippets`` comes back
        empty, so without this branch the extracted content is silently
        discarded in favour of the far shorter ``description``.

        Args:
            contents: The hit's ``contents`` object, or ``None``.
            cap: Maximum number of highlights to keep, mirroring how
                ``n_snippets_per_hit`` bounds snippets.

        Returns:
            The page content, or an empty string when there is none.
        """
        if not contents:
            return ""
        if contents.markdown:
            return contents.markdown
        if contents.html:
            return contents.html
        highlights = contents.highlights or []
        if cap is not None:
            highlights = highlights[:cap]
        return "\n".join(highlights)

    def _parse_web_hit(self, hit: Any) -> list[Document]:
        metadata = self._hit_metadata(hit, "web", include_favicon=True)

        content = self._contents_text(hit.contents, cap=self.n_snippets_per_hit)
        if content:
            return [Document(page_content=content, metadata=metadata)]

        snippets = hit.snippets or []
        if self.n_snippets_per_hit is not None:
            snippets = snippets[: self.n_snippets_per_hit]

        if snippets:
            return [Document(page_content="\n".join(snippets), metadata=metadata)]
        if hit.description:
            return [Document(page_content=hit.description, metadata=metadata)]
        return []

    def _parse_news_hit(self, hit: Any) -> Document | None:
        metadata = self._hit_metadata(hit, "news")

        content = self._contents_text(hit.contents, cap=self.n_snippets_per_hit)
        if not content:
            content = hit.description or ""

        if content:
            return Document(page_content=content, metadata=metadata)
        return None

    @staticmethod
    def _parse_knowledge_hit(hit: Any) -> Document | None:
        """Build a Document from a knowledge result.

        Knowledge results are cards backed by licensed data providers. Unlike web
        and news hits they carry **no URL** — ``attribution`` entries are provider
        credits rather than citations and have no link — so the ``url`` metadata key
        is omitted rather than faked. Callers that index on ``metadata["url"]``
        must handle its absence for ``source == "knowledge"`` documents.

        ``type`` is passed through unmapped. ``"answer"`` is the only kind returned
        today; the SDK asks consumers to ignore unrecognized values rather than
        fail, since a new kind may populate a different set of fields.
        """
        content = hit.description or ""
        if not content:
            return None

        metadata: dict[str, Any] = {
            "title": hit.title or "",
            "type": hit.type or "",
            "source": "knowledge",
        }
        if hit.as_of:
            metadata["as_of"] = hit.as_of
        providers = [a.name for a in (hit.attribution or []) if a.name]
        if providers:
            metadata["attribution"] = providers
        return Document(page_content=content, metadata=metadata)

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

        ``output.content`` is ``Union[str, dict]``: passing ``output_schema`` makes
        the API return structured JSON there with ``content_type == "object"``.
        A dict is serialized so this text-returning helper stays total; callers
        who want the parsed object should use :meth:`raw_research`.
        """
        content = response.output.content
        body = (
            content
            if isinstance(content, str)
            else json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True)
        )
        sources = response.output.sources or []
        # Render each source's supporting snippets beneath its link, mirroring
        # how the Answer path renders citation excerpts. The Finance Research
        # source model gained ``snippets`` in SDK 3.5.0 but production was not
        # returning it at release, so finance renders no extras until it does;
        # the Research API's sibling ``Source`` model already populated it.
        entries: list[tuple[str, list[str]]] = [
            (
                f"[{src.title or src.url}]({src.url})",
                [
                    f"   > {line}"
                    for s in (src.snippets or [])
                    for line in s.splitlines()
                ],
            )
            for src in sources
        ]
        return YouAPIWrapper._numbered_section(body, entries, section="Sources")

    @staticmethod
    def _format_answer_response(response: Any) -> str:
        """Format an Answer response as markdown with citations.

        The answer text is server-formatted markdown with ``[1]``, ``[2]``-style
        inline references; we append a numbered citation list so the reader can
        resolve each reference.

        Each citation carries only ``source`` and ``excerpts``. Where a citation
        has a matching entry in ``results.web[]`` — joined by URL, and the cited
        sources are a subset of those results — the entry is enriched with that
        hit's ``title`` (as the link text, instead of the bare URL) and
        ``description``. ``thumbnail_url`` is intentionally left out of the text;
        it is available to callers who need it via :meth:`raw_answer`.
        """
        results = response.results
        web = (results.web if results else None) or []
        web_by_url = {hit.url: hit for hit in web}

        entries: list[tuple[str, list[str]]] = []
        for citation in response.citations or []:
            url = citation.source
            hit = web_by_url.get(url)
            link_text = (hit.title if hit else None) or url
            extras: list[str] = []
            if hit and hit.description:
                extras.extend(f"   {line}" for line in hit.description.splitlines())
            extras.extend(
                f"   > {line}"
                for excerpt in citation.excerpts or []
                for line in excerpt.splitlines()
            )
            entries.append((f"[{link_text}]({url})", extras))
        return YouAPIWrapper._numbered_section(
            response.answer, entries, section="Citations"
        )


# Backward-compatible alias; remove in the next major version.
YouSearchAPIWrapper = YouAPIWrapper
