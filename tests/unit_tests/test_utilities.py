"""Unit tests for YouSearchAPIWrapper."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_youdotcom import YouSearchAPIWrapper
from tests.unit_tests.conftest import (
    make_contents_page,
    make_livecrawl_contents,
    make_news_hit,
    make_search_response,
    make_web_hit,
)


class TestInit:
    """Initialization and API key handling."""

    def test_init_default_empty_key(self) -> None:
        """Wrapper initializes with empty key when env var is unset."""
        env = os.environ.copy()
        os.environ.pop("YDC_API_KEY", None)
        try:
            wrapper = YouSearchAPIWrapper()
            assert wrapper.ydc_api_key.get_secret_value() == ""
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_init_with_explicit_key(self) -> None:
        """Wrapper accepts an explicit API key."""
        wrapper = YouSearchAPIWrapper(ydc_api_key="test-key")
        assert wrapper.ydc_api_key.get_secret_value() == "test-key"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrapper reads YDC_API_KEY from environment."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouSearchAPIWrapper()
        assert wrapper.ydc_api_key.get_secret_value() == "env-key"

    def test_explicit_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit key overrides environment variable."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouSearchAPIWrapper(ydc_api_key="explicit-key")
        assert wrapper.ydc_api_key.get_secret_value() == "explicit-key"


class TestSearchParsing:
    """Parsing of search responses into Documents."""

    def test_web_hit_with_snippets(self) -> None:
        """Web hit with snippets produces a Document."""
        hit = make_web_hit(
            snippets=["first snippet", "second snippet"],
        )
        response = make_search_response(web=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "first snippet\nsecond snippet"
        assert docs[0].metadata["url"] == "https://example.com"
        assert docs[0].metadata["title"] == "Example"

    def test_web_hit_prefers_livecrawl_markdown(self) -> None:
        """Livecrawl markdown takes priority over snippets."""
        contents = make_livecrawl_contents(markdown="# Live Content")
        hit = make_web_hit(
            snippets=["ignored snippet"],
            contents=contents,
        )
        response = make_search_response(web=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "# Live Content"

    def test_web_hit_falls_back_to_html(self) -> None:
        """Livecrawl HTML used when markdown is absent."""
        contents = make_livecrawl_contents(markdown=None, html="<h1>HTML</h1>")
        hit = make_web_hit(contents=contents)
        response = make_search_response(web=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "<h1>HTML</h1>"

    def test_respects_k_limit(self) -> None:
        """Only k documents are returned."""
        hits = [make_web_hit(url=f"https://example.com/{i}") for i in range(5)]
        response = make_search_response(web=hits)
        wrapper = YouSearchAPIWrapper(ydc_api_key="k", k=2)
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 2

    def test_respects_n_snippets_per_hit(self) -> None:
        """Snippet count is capped by n_snippets_per_hit."""
        hit = make_web_hit(snippets=["a", "b", "c"])
        response = make_search_response(web=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k", n_snippets_per_hit=1)
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "a"

    def test_news_hit_parsed(self) -> None:
        """News hits produce Documents."""
        hit = make_news_hit()
        response = make_search_response(news=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].metadata["url"] == "https://news.example.com"

    def test_empty_response(self) -> None:
        """No results returns an empty list."""
        response = MagicMock()
        response.results = None
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs == []

    def test_metadata_includes_optional_fields(self) -> None:
        """Thumbnail, favicon, and page_age appear in metadata."""
        hit = make_web_hit(
            thumbnail_url="https://img.example.com/thumb.jpg",
            favicon_url="https://example.com/favicon.ico",
            page_age="2025-01-01",
        )
        response = make_search_response(web=[hit])
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].metadata["thumbnail_url"] == "https://img.example.com/thumb.jpg"
        assert docs[0].metadata["favicon_url"] == "https://example.com/favicon.ico"
        assert docs[0].metadata["page_age"] == "2025-01-01"


class TestContentsParsing:
    """Parsing of contents responses into Documents."""

    def test_markdown_page(self) -> None:
        """Contents page with markdown produces a Document."""
        page = make_contents_page(markdown="# Hello World")
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert len(docs) == 1
        assert docs[0].page_content == "# Hello World"

    def test_html_fallback(self) -> None:
        """Contents page falls back to HTML when markdown is absent."""
        page = make_contents_page(markdown=None, html="<p>Hello</p>")
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert len(docs) == 1
        assert docs[0].page_content == "<p>Hello</p>"

    def test_metadata_includes_site_name(self) -> None:
        """Site name and favicon appear in metadata when present."""
        page = make_contents_page(
            site_name="Example Site",
            favicon_url="https://example.com/fav.ico",
        )
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert docs[0].metadata["site_name"] == "Example Site"
        assert docs[0].metadata["favicon_url"] == "https://example.com/fav.ico"

    def test_skips_empty_pages(self) -> None:
        """Pages with no content are skipped."""
        page = make_contents_page(markdown=None, html=None)
        wrapper = YouSearchAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert docs == []


class TestSDKIntegration:
    """Verify the wrapper calls the SDK correctly."""

    @patch("langchain_youdotcom._utilities.You")
    def test_results_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """results() creates a client and calls search.unified."""
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.unified.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouSearchAPIWrapper(ydc_api_key="test-key", count=5)
        docs = wrapper.results("test query")

        assert mock_you_cls.call_args.kwargs["api_key_auth"] == "test-key"
        mock_client.search.unified.assert_called_once_with(query="test query", count=5)
        assert len(docs) == 1

    @patch("langchain_youdotcom._utilities.You")
    def test_contents_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """contents() creates a client and calls contents.generate."""
        page = make_contents_page()
        mock_client = MagicMock()
        mock_client.contents.generate.return_value = [page]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouSearchAPIWrapper(ydc_api_key="test-key")
        docs = wrapper.contents(["https://example.com"])

        mock_client.contents.generate.assert_called_once()
        assert len(docs) == 1
