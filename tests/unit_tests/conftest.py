"""Shared test fixtures and mock helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of the developer's ambient environment.

    ``YouAPIWrapper`` reads ``YDC_API_KEY`` at construction time, so without
    this the suite passes or fails depending on whether the key happens to be
    exported in the shell running it — it was green locally and red in CI.
    Every unit test mocks the SDK client, so the value never reaches a real
    request. Tests that care about key handling override or unset it
    themselves.
    """
    monkeypatch.setenv("YDC_API_KEY", "unit-test-key")


def make_web_hit(
    *,
    url: str = "https://example.com",
    title: str = "Example",
    description: str = "An example page",
    snippets: list[str] | None = None,
    contents: Any = None,
    thumbnail_url: str | None = None,
    favicon_url: str | None = None,
    page_age: datetime | str | None = None,
) -> MagicMock:
    """Build a mock Web hit matching the SDK's ``Web`` model."""
    hit = MagicMock()
    hit.url = url
    hit.title = title
    hit.description = description
    hit.snippets = snippets if snippets is not None else ["snippet one"]
    hit.contents = contents
    hit.thumbnail_url = thumbnail_url
    hit.favicon_url = favicon_url
    hit.page_age = page_age
    return hit


def make_news_hit(
    *,
    url: str = "https://news.example.com",
    title: str = "News Title",
    description: str = "A news story",
    contents: Any = None,
    thumbnail_url: str | None = None,
    page_age: datetime | str | None = None,
) -> MagicMock:
    """Build a mock News hit matching the SDK's ``News`` model."""
    hit = MagicMock()
    hit.url = url
    hit.title = title
    hit.description = description
    hit.contents = contents
    hit.thumbnail_url = thumbnail_url
    hit.page_age = page_age
    return hit


def make_knowledge_attribution(
    *, name: str = "Provider", source_description: str | None = None
) -> MagicMock:
    """Build a mock attribution matching the SDK's ``KnowledgeAttribution``.

    Attribution entries are provider credits rather than citations and carry no
    URL.
    """
    attr = MagicMock()
    attr.name = name
    attr.source_description = source_description
    return attr


def make_knowledge_result(
    *,
    result_type: str = "answer",
    title: str = "Knowledge Title",
    description: str | None = "Licensed knowledge answer",
    as_of: str | None = None,
    attribution: list[Any] | None = None,
) -> MagicMock:
    """Build a mock knowledge result matching the SDK's ``KnowledgeResult``.

    Unlike web and news hits, knowledge results have no ``url`` attribute.
    """
    hit = MagicMock(spec=["type", "title", "description", "as_of", "attribution"])
    hit.type = result_type
    hit.title = title
    hit.description = description
    hit.as_of = as_of
    hit.attribution = (
        attribution if attribution is not None else [make_knowledge_attribution()]
    )
    return hit


def make_search_response(
    *,
    web: list[Any] | None = None,
    news: list[Any] | None = None,
    knowledge: list[Any] | None = None,
) -> MagicMock:
    """Build a mock SearchResponse matching the SDK's ``SearchResponse``.

    ``knowledge`` must be set explicitly: the SDK models it as optional and
    omits the key when nothing is relevant, so leaving it to MagicMock's
    auto-attribute would yield a truthy mock instead of ``None``.
    """
    response = MagicMock()
    results = MagicMock()
    results.web = web
    results.news = news
    results.knowledge = knowledge
    response.results = results
    return response


def make_livecrawl_contents(
    *,
    markdown: str | None = None,
    html: str | None = None,
    highlights: list[str] | None = None,
) -> MagicMock:
    """Build a mock Contents object for extracted data inside a hit.

    ``highlights`` must be set explicitly: the API populates it for
    ``extraction_mode="highlights"``, and leaving it to MagicMock's
    auto-attribute would yield a truthy mock instead of ``None``.
    """
    contents = MagicMock()
    contents.markdown = markdown
    contents.html = html
    contents.highlights = highlights
    return contents


def make_research_source(
    *,
    url: str = "https://example.com",
    title: str | None = "Example Source",
    snippets: list[str] | None = None,
) -> MagicMock:
    """Build a mock Source matching the SDK's ``Source`` model."""
    source = MagicMock()
    source.url = url
    source.title = title
    source.snippets = snippets
    return source


def make_research_response(
    *,
    content: str | dict[str, Any] = "Research answer with [1] citations.",
    sources: list[Any] | None = None,
) -> MagicMock:
    """Build a mock ResearchResponse matching the SDK's model.

    ``content`` accepts a dict because the SDK types it as
    ``Union[str, Dict[str, Any]]`` — structured output arrives as an object.
    """
    response = MagicMock()
    output = MagicMock()
    output.content = content
    output.content_type = "text" if isinstance(content, str) else "object"
    output.sources = sources if sources is not None else [make_research_source()]
    response.output = output
    return response


def make_contents_page(
    *,
    url: str = "https://example.com",
    title: str = "Page Title",
    markdown: str | None = "# Hello",
    html: str | None = None,
    site_name: str | None = None,
    favicon_url: str | None = None,
) -> MagicMock:
    """Build a mock ContentsResponse matching the SDK's model."""
    page = MagicMock()
    page.url = url
    page.title = title
    page.markdown = markdown
    page.html = html
    meta = MagicMock()
    meta.site_name = site_name
    meta.favicon_url = favicon_url
    page.metadata = meta if (site_name or favicon_url) else None
    return page


def make_finance_research_response(
    *,
    content: str = "Finance answer with [1] citations.",
    sources: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock matching the SDK's :class:`FinanceResearchResponse`."""
    if sources is None:
        sources = [{"url": "https://finance.example.com", "title": "Finance Source"}]
    # SDK's FinanceResearchSource declares only `url` and optional `title`.
    src_objects = [MagicMock(url=s["url"], title=s.get("title")) for s in sources]
    output = MagicMock(content=content, content_type="text", sources=src_objects)
    return MagicMock(output=output)


def make_answer_response(
    *,
    answer: str = "Cited answer with [1] citation.",
    citations: list[dict[str, Any]] | None = None,
    web_results: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock matching the SDK's :class:`AnswerResponse`."""
    if citations is None:
        citations = [
            {
                "source": "https://example.com",
                "excerpts": ["supporting excerpt"],
            },
        ]
    if web_results is None:
        web_results = [
            {
                "url": "https://example.com",
                "title": "Example",
                "snippets": ["snippet"],
            },
        ]
    citation_objects = [
        MagicMock(
            source=c["source"],
            excerpts=c.get("excerpts"),
        )
        for c in citations
    ]
    web_objects = [
        MagicMock(
            url=w["url"],
            title=w["title"],
            snippets=w.get("snippets"),
            description=w.get("description"),
            thumbnail_url=w.get("thumbnail_url"),
            page_age=w.get("page_age"),
        )
        for w in web_results
    ]
    return MagicMock(
        answer=answer,
        citations=citation_objects,
        results=MagicMock(web=web_objects),
    )
