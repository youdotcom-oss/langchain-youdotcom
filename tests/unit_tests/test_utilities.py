"""Unit tests for YouAPIWrapper."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_youdotcom import YouAPIWrapper
from langchain_youdotcom._utilities import (
    _CLIENT_APP_NAME,
    _CLIENT_APP_VERSION,
)
from tests.unit_tests.conftest import (
    make_answer_response,
    make_contents_page,
    make_finance_research_response,
    make_livecrawl_contents,
    make_news_hit,
    make_research_response,
    make_research_source,
    make_search_response,
    make_web_hit,
)


class TestInit:
    """Initialization, API key handling, and X-Client-Info attribution."""

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

    @patch("langchain_youdotcom._utilities.You")
    def test_zero_key_passes_api_key_auth_none(self, mock_you_cls: MagicMock) -> None:
        """Empty key is normalized to ``api_key_auth=None`` for SDK 3.x."""
        mock_client = MagicMock()
        mock_you_cls.return_value = mock_client

        env = os.environ.copy()
        os.environ.pop("YDC_API_KEY", None)
        try:
            wrapper = YouAPIWrapper()
            wrapper._make_client()
        finally:
            os.environ.clear()
            os.environ.update(env)

        assert mock_you_cls.call_args.kwargs["api_key_auth"] is None

    @patch("langchain_youdotcom._utilities.You")
    def test_make_client_sets_app_attribution(self, mock_you_cls: MagicMock) -> None:
        """Every outbound call goes out with ``X-Client-Info`` attribution."""
        mock_client = MagicMock()
        mock_you_cls.return_value = mock_client

        env = os.environ.copy()
        os.environ.pop("YDC_API_KEY", None)
        try:
            wrapper = YouAPIWrapper(ydc_api_key="k")
            wrapper._make_client()
        finally:
            os.environ.clear()
            os.environ.update(env)

        assert mock_you_cls.call_args.kwargs["app_name"] == _CLIENT_APP_NAME
        assert mock_you_cls.call_args.kwargs["app_version"] == _CLIENT_APP_VERSION


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
        assert docs[0].metadata["source"] == "web"

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
        assert docs[0].metadata["source"] == "news"

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
    """Verify the wrapper calls the SDK correctly with the supported surface."""

    @patch("langchain_youdotcom._utilities.You")
    def test_results_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """results() creates a client and calls client.search."""
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key", count=5)
        docs = wrapper.results("test query")

        assert mock_you_cls.call_args.kwargs["api_key_auth"] == "test-key"
        mock_client.search.assert_called_once_with(query="test query", count=5)
        assert len(docs) == 1

    @patch("langchain_youdotcom._utilities.You")
    def test_search_forwards_deprecated_livecrawl(
        self, mock_you_cls: MagicMock
    ) -> None:
        """Deprecated livecrawl/livecrawl_formats are forwarded to the SDK."""
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(
            ydc_api_key="test-key",
            livecrawl="web",
            livecrawl_formats=["html", "markdown"],
        )
        wrapper.results("test query")

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["livecrawl"] == "web"
        assert call_kwargs["livecrawl_formats"] == ["html", "markdown"]

    @patch("langchain_youdotcom._utilities.You")
    def test_contents_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """contents() creates a client and calls client.contents."""
        page = make_contents_page()
        mock_client = MagicMock()
        mock_client.contents.return_value = [page]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        docs = wrapper.contents(["https://example.com"])

        mock_client.contents.assert_called_once()
        call_kwargs = mock_client.contents.call_args.kwargs
        assert call_kwargs["urls"] == ["https://example.com"]
        assert mock_client.contents.call_args is not None
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
        assert call_kwargs["timeout_ms"] == 300_000
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

    @patch("langchain_youdotcom._utilities.You")
    def test_raw_finance_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """raw_finance() creates a client and calls client.finance_research."""
        response = make_finance_research_response()
        mock_client = MagicMock()
        mock_client.finance_research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = wrapper.raw_finance("NVDA earnings")

        mock_client.finance_research.assert_called_once()
        call_kwargs = mock_client.finance_research.call_args.kwargs
        assert call_kwargs["input"] == "NVDA earnings"
        assert call_kwargs["research_effort"] == "deep"
        assert call_kwargs["timeout_ms"] == 300_000
        assert result.output.content == "Finance answer with [1] citations."

    @patch("langchain_youdotcom._utilities.You")
    async def test_raw_finance_async_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """raw_finance_async() calls ``client.finance_research_async``."""
        response = make_finance_research_response()
        mock_client = MagicMock()
        mock_client.finance_research_async = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = await wrapper.raw_finance_async("AAPL cash flow")

        mock_client.finance_research_async.assert_called_once()
        call_kwargs = mock_client.finance_research_async.call_args.kwargs
        assert call_kwargs["input"] == "AAPL cash flow"
        assert call_kwargs["research_effort"] == "deep"
        assert call_kwargs["timeout_ms"] == 300_000
        assert result.output.content == "Finance answer with [1] citations."

    @patch("langchain_youdotcom._utilities.You")
    def test_finance_text_returns_markdown(self, mock_you_cls: MagicMock) -> None:
        """finance_text() returns formatted markdown via SDK."""
        response = make_finance_research_response(
            content="Revenue grew 40% YoY.",
        )
        mock_client = MagicMock()
        mock_client.finance_research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        result = wrapper.finance_text("NVDA revenue")

        assert "Revenue grew 40% YoY." in result
        assert "## Sources" in result

    @patch("langchain_youdotcom._utilities.You")
    def test_raw_answer_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """raw_answer() calls ``client.answer`` with all filter kwargs."""
        response = make_answer_response(answer="cited")
        mock_client = MagicMock()
        mock_client.answer.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="k")
        wrapper.raw_answer(
            "what is RAG",
            freshness="week",
            country="US",
            language="EN",
            safesearch="moderate",
            include_domains=["example.com"],
        )

        call_kwargs = mock_client.answer.call_args.kwargs
        assert call_kwargs["query"] == "what is RAG"
        assert call_kwargs["freshness"] == "week"
        assert call_kwargs["country"] == "US"
        assert call_kwargs["language"] == "EN"
        assert call_kwargs["safesearch"] == "moderate"
        assert call_kwargs["include_domains"] == ["example.com"]
        assert call_kwargs["timeout_ms"] == 300_000

    @patch("langchain_youdotcom._utilities.You")
    async def test_raw_answer_async_calls_sdk(self, mock_you_cls: MagicMock) -> None:
        """raw_answer_async() calls ``client.answer_async``."""
        response = make_answer_response(answer="async cited")
        mock_client = MagicMock()
        mock_client.answer_async = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="k")
        await wrapper.raw_answer_async("what is RAG")

        mock_client.answer_async.assert_called_once()

    def test_answer_text_includes_citations(self) -> None:
        """answer_text() returns markdown with a Citations section."""
        response = make_answer_response(
            answer="RAG combines retrieval and generation [1].",
            citations=[
                {
                    "source": "https://arxiv.org/abs/2005.11401",
                    "excerpts": ["RAG paper abstract."],
                },
            ],
        )
        with patch(
            "langchain_youdotcom.YouAPIWrapper.raw_answer",
            return_value=response,
        ):
            result = YouAPIWrapper(ydc_api_key="k").answer_text("what is RAG")

        assert "RAG combines retrieval and generation [1]." in result
        assert "## Citations" in result
        assert "https://arxiv.org/abs/2005.11401" in result
        assert "RAG paper abstract." in result


class TestAnswerValidation:
    """Pre-flight validation of Answer Args."""

    def test_query_over_400_chars_rejected(self) -> None:
        """Server rejects queries over 400 chars; surface locally."""
        with pytest.raises(ValueError, match="400"):
            YouAPIWrapper()._answer_params("x" * 401)

    def test_empty_query_rejected(self) -> None:
        """Empty query is rejected without an HTTP call."""
        with pytest.raises(ValueError, match="query"):
            YouAPIWrapper()._answer_params("")

    def test_whitespace_only_query_rejected(self) -> None:
        """Whitespace-only query is rejected."""
        with pytest.raises(ValueError, match="query"):
            YouAPIWrapper()._answer_params("   ")

    def test_include_with_exclude_rejected(self) -> None:
        """include_domains conflicts with exclude_domains."""
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper()._answer_params(
                "q",
                include_domains=["a.com"],
                exclude_domains=["b.com"],
            )

    def test_include_with_boost_rejected(self) -> None:
        """include_domains conflicts with boost_domains."""
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper()._answer_params(
                "q",
                include_domains=["a.com"],
                boost_domains=["b.com"],
            )

    def test_exclude_and_boost_pass(self) -> None:
        """exclude_domains and boost_domains can co-exist."""
        params = YouAPIWrapper()._answer_params(
            "q",
            exclude_domains=["b.com"],
            boost_domains=["c.com"],
        )
        assert params["exclude_domains"] == ["b.com"]
        assert params["boost_domains"] == ["c.com"]

    def test_params_include_filters_when_set(self) -> None:
        """All filter kwargs reach the param dict."""
        params = YouAPIWrapper()._answer_params(
            "q",
            freshness="week",
            country="US",
            language="EN",
            safesearch="moderate",
        )
        assert params == {
            "query": "q",
            "freshness": "week",
            "country": "US",
            "language": "EN",
            "safesearch": "moderate",
        }


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

    def test_research_params_rejects_frontier(self) -> None:
        """Frontier is task-only; sync API returns 422, so reject locally."""
        wrapper = YouAPIWrapper(ydc_api_key="k", research_effort="frontier")
        with pytest.raises(ValueError, match="frontier"):
            wrapper._research_params("my query")


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
