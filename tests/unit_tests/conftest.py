"""Shared test fixtures and mock helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def make_web_hit(
    *,
    url: str = "https://example.com",
    title: str = "Example",
    description: str = "An example page",
    snippets: list[str] | None = None,
    contents: Any = None,
    thumbnail_url: str | None = None,
    favicon_url: str | None = None,
    page_age: str | None = None,
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
    page_age: str | None = None,
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


def make_search_response(
    *,
    web: list[Any] | None = None,
    news: list[Any] | None = None,
) -> MagicMock:
    """Build a mock SearchResponse matching the SDK's ``SearchResponse``."""
    response = MagicMock()
    results = MagicMock()
    results.web = web
    results.news = news
    response.results = results
    return response


def make_livecrawl_contents(
    *, markdown: str | None = None, html: str | None = None
) -> MagicMock:
    """Build a mock Contents object for livecrawl data inside a hit."""
    contents = MagicMock()
    contents.markdown = markdown
    contents.html = html
    return contents


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
