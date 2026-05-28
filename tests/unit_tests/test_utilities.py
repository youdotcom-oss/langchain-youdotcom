"""Unit tests for YouAPIWrapper."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_youdotcom import YouAPIWrapper
from tests.unit_tests.conftest import (
    make_contents_page,
    make_finance_research_json,
    make_livecrawl_contents,
    make_news_hit,
    make_research_response,
    make_research_source,
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
            wrapper = YouAPIWrapper()
            assert wrapper.ydc_api_key.get_secret_value() == ""
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_init_with_explicit_key(self) -> None:
        """Wrapper accepts an explicit API key."""
        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        assert wrapper.ydc_api_key.get_secret_value() == "test-key"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrapper reads YDC_API_KEY from environment."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouAPIWrapper()
        assert wrapper.ydc_api_key.get_secret_value() == "env-key"

    def test_explicit_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit key overrides environment variable."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouAPIWrapper(ydc_api_key="explicit-key")
        assert wrapper.ydc_api_key.get_secret_value() == "explicit-key"


class TestSearchParsing:
    """Parsing of search responses into Documents."""

    def test_web_hit_with_snippets(self) -> None:
        """Web hit with snippets produces a Document."""
        hit = make_web_hit(
            snippets=["first snippet", "second snippet"],
        )
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
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
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "# Live Content"

    def test_web_hit_falls_back_to_html(self) -> None:
        """Livecrawl HTML used when markdown is absent."""
        contents = make_livecrawl_contents(markdown=None, html="<h1>HTML</h1>")
        hit = make_web_hit(contents=contents)
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "<h1>HTML</h1>"

    def test_respects_k_limit(self) -> None:
        """Only k documents are returned."""
        hits = [make_web_hit(url=f"https://example.com/{i}") for i in range(5)]
        response = make_search_response(web=hits)
        wrapper = YouAPIWrapper(ydc_api_key="k", k=2)
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 2

    def test_respects_n_snippets_per_hit(self) -> None:
        """Snippet count is capped by n_snippets_per_hit."""
        hit = make_web_hit(snippets=["a", "b", "c"])
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k", n_snippets_per_hit=1)
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "a"

    def test_news_hit_parsed(self) -> None:
        """News hits produce Documents."""
        hit = make_news_hit()
        response = make_search_response(news=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].metadata["url"] == "https://news.example.com"

    def test_empty_response(self) -> None:
        """No results returns an empty list."""
        response = MagicMock()
        response.results = None
        wrapper = YouAPIWrapper(ydc_api_key="k")
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
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].metadata["thumbnail_url"] == "https://img.example.com/thumb.jpg"
        assert docs[0].metadata["favicon_url"] == "https://example.com/favicon.ico"
        assert docs[0].metadata["page_age"] == "2025-01-01"


class TestContentsParsing:
    """Parsing of contents responses into Documents."""

    def test_markdown_page(self) -> None:
        """Contents page with markdown produces a Document."""
        page = make_contents_page(markdown="# Hello World")
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert len(docs) == 1
        assert docs[0].page_content == "# Hello World"

    def test_html_fallback(self) -> None:
        """Contents page falls back to HTML when markdown is absent."""
        page = make_contents_page(markdown=None, html="<p>Hello</p>")
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert len(docs) == 1
        assert docs[0].page_content == "<p>Hello</p>"

    def test_metadata_includes_site_name(self) -> None:
        """Site name and favicon appear in metadata when present."""
        page = make_contents_page(
            site_name="Example Site",
            favicon_url="https://example.com/fav.ico",
        )
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_contents_response([page])

        assert docs[0].metadata["site_name"] == "Example Site"
        assert docs[0].metadata["favicon_url"] == "https://example.com/fav.ico"

    def test_skips_empty_pages(self) -> None:
        """Pages with no content are skipped."""
        page = make_contents_page(markdown=None, html=None)
        wrapper = YouAPIWrapper(ydc_api_key="k")
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

        wrapper = YouAPIWrapper(ydc_api_key="test-key", count=5)
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

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        docs = wrapper.contents(["https://example.com"])

        mock_client.contents.generate.assert_called_once()
        assert len(docs) == 1

    @patch("langchain_youdotcom._utilities.You")
    def test_raw_research_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """raw_research() creates a client and calls client.research."""
        from youdotcom.models import ResearchEffort

        response = make_research_response()
        mock_client = MagicMock()
        mock_client.research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key", research_effort="lite")
        result = wrapper.raw_research("test query")

        mock_client.research.assert_called_once()
        call_kwargs = mock_client.research.call_args.kwargs
        assert call_kwargs["input"] == "test query"
        assert call_kwargs["research_effort"] == ResearchEffort.LITE
        assert result.output.content == "Research answer with [1] citations."

    @patch("langchain_youdotcom._utilities.You")
    def test_research_text_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """research_text() returns formatted markdown with sources."""
        response = make_research_response()
        mock_client = MagicMock()
        mock_client.research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = wrapper.research_text("test query")

        assert isinstance(result, str)
        assert "Research answer" in result
        assert "## Sources" in result

    @patch("langchain_youdotcom._utilities.You")
    async def test_research_text_async_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """research_text_async() returns formatted markdown with sources."""
        response = make_research_response()
        mock_client = MagicMock()
        mock_client.research_async = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = await wrapper.research_text_async("test query")

        mock_client.research_async.assert_called_once()
        assert isinstance(result, str)
        assert "Research answer" in result
        assert "## Sources" in result


class TestResearchFormatting:
    """Formatting of research responses."""

    def test_format_with_sources(self) -> None:
        """Response includes markdown answer and numbered sources."""
        sources = [
            make_research_source(url="https://a.com", title="Source A"),
            make_research_source(url="https://b.com", title="Source B"),
        ]
        response = make_research_response(content="The answer is 42.", sources=sources)
        wrapper = YouAPIWrapper(ydc_api_key="k")
        result = wrapper._format_research_response(response)

        assert result.startswith("The answer is 42.")
        assert "## Sources" in result
        assert "1. [Source A](https://a.com)" in result
        assert "2. [Source B](https://b.com)" in result

    def test_format_without_sources(self) -> None:
        """Response without sources omits the sources section."""
        response = make_research_response(content="Just an answer.", sources=[])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        result = wrapper._format_research_response(response)

        assert result == "Just an answer."
        assert "## Sources" not in result

    def test_format_source_falls_back_to_url(self) -> None:
        """Source with no title uses URL as link text."""
        source = make_research_source(url="https://c.com", title=None)
        response = make_research_response(sources=[source])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        result = wrapper._format_research_response(response)

        assert "[https://c.com](https://c.com)" in result

    def test_research_params_without_effort(self) -> None:
        """Params omit research_effort when not set."""
        wrapper = YouAPIWrapper(ydc_api_key="k")
        params = wrapper._research_params("my query")

        assert params == {"input": "my query"}

    def test_research_params_with_effort(self) -> None:
        """Params include ResearchEffort enum when set."""
        from youdotcom.models import ResearchEffort

        wrapper = YouAPIWrapper(ydc_api_key="k", research_effort="deep")
        params = wrapper._research_params("my query")

        assert params["input"] == "my query"
        assert params["research_effort"] == ResearchEffort.DEEP


class TestFinanceResearchParams:
    """Finance Research parameter building."""

    def test_default_effort_is_deep(self) -> None:
        """Default finance research effort is deep."""
        wrapper = YouAPIWrapper(ydc_api_key="k")
        params = wrapper._finance_research_params("NVDA earnings")
        assert params == {"input": "NVDA earnings", "research_effort": "deep"}

    def test_explicit_effort(self) -> None:
        """Explicit research_effort is passed through."""
        wrapper = YouAPIWrapper(ydc_api_key="k", research_effort="exhaustive")
        params = wrapper._finance_research_params("AAPL revenue")
        assert params["research_effort"] == "exhaustive"

    def test_incompatible_effort_raises(self) -> None:
        """Incompatible effort level raises ValueError."""
        wrapper = YouAPIWrapper(ydc_api_key="k", research_effort="lite")
        with pytest.raises(ValueError, match="Finance Research"):
            wrapper._finance_research_params("query")


class TestFinanceResearchParsing:
    """Parsing of Finance Research JSON responses."""

    def test_parse_finance_research_json(self) -> None:
        """Parsed object works with _format_research_response."""
        data = make_finance_research_json(
            content="NVIDIA revenue grew due to data center.",
            sources=[
                {"url": "https://sec.gov/nvda", "title": "NVIDIA 10-K"},
            ],
        )
        wrapper = YouAPIWrapper(ydc_api_key="k")
        response = wrapper._parse_finance_research_json(data)
        result = wrapper._format_research_response(response)

        assert "NVIDIA revenue grew" in result
        assert "## Sources" in result
        assert "[NVIDIA 10-K](https://sec.gov/nvda)" in result

    def test_parse_finance_research_json_no_sources(self) -> None:
        """Parsed object with no sources omits sources section."""
        data = make_finance_research_json(content="Just an answer.", sources=[])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        response = wrapper._parse_finance_research_json(data)
        result = wrapper._format_research_response(response)

        assert result == "Just an answer."
        assert "## Sources" not in result

    def test_parse_finance_research_json_source_no_title(self) -> None:
        """Source with no title falls back to URL."""
        data = make_finance_research_json(
            sources=[{"url": "https://sec.gov/filing", "title": None}],
        )
        wrapper = YouAPIWrapper(ydc_api_key="k")
        response = wrapper._parse_finance_research_json(data)
        result = wrapper._format_research_response(response)

        assert "[https://sec.gov/filing](https://sec.gov/filing)" in result


class TestFinanceResearchHTTPCalls:
    """Verify the wrapper makes correct HTTP calls for Finance Research."""

    @patch("langchain_youdotcom._utilities.httpx.Client")
    def test_raw_finance_calls_httpx(self, mock_client_cls: MagicMock) -> None:
        """raw_finance() posts to the Finance Research API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = make_finance_research_json()
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = wrapper.raw_finance("NVDA earnings")

        mock_client.post.assert_called_once_with(
            "https://api.you.com/v1/finance_research",
            json={"input": "NVDA earnings", "research_effort": "deep"},
        )
        assert result.output.content == "Finance answer with [1] citations."

    @patch("langchain_youdotcom._utilities.httpx.AsyncClient")
    async def test_raw_finance_async_calls_httpx(
        self, mock_client_cls: MagicMock
    ) -> None:
        """raw_finance_async() posts to the Finance Research API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = make_finance_research_json()
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = await wrapper.raw_finance_async("AAPL cash flow")

        mock_client.post.assert_called_once()
        assert result.output.content == "Finance answer with [1] citations."

    @patch("langchain_youdotcom._utilities.httpx.Client")
    def test_finance_text_returns_markdown(self, mock_client_cls: MagicMock) -> None:
        """finance_text() returns formatted markdown."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = make_finance_research_json(
            content="Revenue grew 40% YoY."
        )
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = wrapper.finance_text("NVDA revenue")

        assert "Revenue grew 40% YoY." in result
        assert "## Sources" in result
