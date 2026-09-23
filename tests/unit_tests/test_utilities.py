"""Unit tests for YouAPIWrapper."""

from __future__ import annotations

import os
from datetime import datetime, timezone
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
    make_knowledge_attribution,
    make_knowledge_result,
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

    def test_zero_key_raises_clear_error(self) -> None:
        """A missing key fails locally rather than sending an unauthenticated call.

        Without this the request goes out with no auth and the API answers
        ``402 payment_required`` telling the caller to buy credits, which
        points at billing instead of the real cause: no key was configured.
        """
        env = os.environ.copy()
        os.environ.pop("YDC_API_KEY", None)
        os.environ.pop("YOU_API_KEY_AUTH", None)
        try:
            wrapper = YouAPIWrapper()
            with pytest.raises(ValueError, match=r"No You\.com API key found"):
                wrapper._make_client()
        finally:
            os.environ.clear()
            os.environ.update(env)

    @patch("langchain_youdotcom._utilities.You")
    def test_sdk_env_fallback_key_is_accepted(
        self, mock_you_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``YOU_API_KEY_AUTH`` still satisfies the SDK's own env fallback.

        The wrapper reads ``YDC_API_KEY`` into its field at construction, but
        the SDK also honors ``YOU_API_KEY_AUTH``. Passing ``api_key_auth=None``
        lets that lookup happen instead of tripping SDK 3.0.0's empty-string
        rejection.
        """
        mock_you_cls.return_value = MagicMock()
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        monkeypatch.setenv("YOU_API_KEY_AUTH", "sdk-env-key")

        YouAPIWrapper()._make_client()

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

    def test_highlights_used_when_markdown_and_html_absent(self) -> None:
        """``extraction_mode="highlights"`` populates ``contents.highlights`` only.

        In that mode the API returns ``markdown=None``, ``html=None`` and empty
        ``snippets``, so without reading ``highlights`` the extracted content is
        silently discarded in favour of the far shorter ``description``.
        """
        contents = make_livecrawl_contents(highlights=["first", "second"])
        hit = make_web_hit(snippets=[], description="short desc", contents=contents)
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "first\nsecond"

    def test_markdown_preferred_over_highlights(self) -> None:
        """Full-page content wins when both are present."""
        contents = make_livecrawl_contents(markdown="# Full", highlights=["h"])
        hit = make_web_hit(contents=contents)
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "# Full"

    def test_n_snippets_per_hit_caps_highlights(self) -> None:
        """The per-hit excerpt cap bounds highlights as it does snippets."""
        contents = make_livecrawl_contents(highlights=["a", "b", "c"])
        hit = make_web_hit(snippets=[], contents=contents)
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k", n_snippets_per_hit=2)
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "a\nb"

    def test_n_snippets_per_hit_caps_news_highlights(self) -> None:
        """The cap applies to news hits as well, not only web hits.

        ``_parse_news_hit`` has to thread the cap through explicitly; leaving
        it at the ``_contents_text`` default of ``None`` silently returns
        unbounded content for news while the README promises the cap applies to
        highlights generally.
        """
        contents = make_livecrawl_contents(highlights=["a", "b", "c"])
        hit = make_news_hit(description="desc", contents=contents)
        response = make_search_response(news=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k", n_snippets_per_hit=2)
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "a\nb"

    def test_news_hit_uses_highlights(self) -> None:
        """News hits read highlights too, not just markdown/html."""
        contents = make_livecrawl_contents(highlights=["news highlight"])
        hit = make_news_hit(description="desc", contents=contents)
        response = make_search_response(news=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].page_content == "news highlight"

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

    def test_page_age_datetime_is_stringified(self) -> None:
        """SDK 3.2.0 widened ``page_age`` to ``datetime | str | None``.

        ``_hit_metadata`` wraps it in ``str()``, so a datetime still lands in
        ``Document.metadata`` as a string instead of leaking a datetime object
        into a dict that consumers expect to be JSON-friendly.
        """
        hit = make_web_hit(page_age=datetime(2026, 3, 4, tzinfo=timezone.utc))
        response = make_search_response(web=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs[0].metadata["page_age"] == "2026-03-04 00:00:00+00:00"


class TestKnowledgeParsing:
    """Parsing of the Search API's ``knowledge`` section (SDK 3.5.0)."""

    def test_knowledge_hit_parsed(self) -> None:
        """Knowledge hits produce Documents from the licensed description."""
        hit = make_knowledge_result(
            title="GDP",
            description="Licensed answer text",
            as_of="2026-01-01",
            attribution=[make_knowledge_attribution(name="Example Data Co")],
        )
        response = make_search_response(knowledge=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].page_content == "Licensed answer text"
        assert docs[0].metadata["title"] == "GDP"
        assert docs[0].metadata["type"] == "answer"
        assert docs[0].metadata["source"] == "knowledge"
        assert docs[0].metadata["as_of"] == "2026-01-01"
        assert docs[0].metadata["attribution"] == ["Example Data Co"]

    def test_knowledge_hit_has_no_url_key(self) -> None:
        """Knowledge Documents omit ``url`` because the SDK provides none.

        Attribution entries are provider credits rather than citations and
        carry no link, so the key is omitted rather than faked. Callers that
        index on ``metadata["url"]`` must handle its absence for documents
        whose ``source`` is ``"knowledge"``.
        """
        response = make_search_response(knowledge=[make_knowledge_result()])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert "url" not in docs[0].metadata

    def test_knowledge_omits_unset_optional_metadata(self) -> None:
        """``as_of`` and ``attribution`` are absent when the API omits them."""
        hit = make_knowledge_result(as_of=None, attribution=[])
        response = make_search_response(knowledge=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert "as_of" not in docs[0].metadata
        assert "attribution" not in docs[0].metadata

    def test_knowledge_hit_without_description_skipped(self) -> None:
        """A knowledge result carrying no description yields no Document."""
        response = make_search_response(
            knowledge=[make_knowledge_result(description=None)]
        )
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert docs == []

    def test_unknown_knowledge_type_passes_through(self) -> None:
        """An unrecognized ``type`` is passed through, not rejected.

        The SDK documents ``answer`` as the only kind returned today and asks
        consumers to ignore unrecognized values rather than fail, since a new
        kind may populate a different set of fields.
        """
        hit = make_knowledge_result(result_type="future_kind")
        response = make_search_response(knowledge=[hit])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].metadata["type"] == "future_kind"

    def test_absent_knowledge_section_is_safe(self) -> None:
        """The API omits ``knowledge`` entirely when nothing is relevant.

        The field is ``None`` rather than an empty list, so the parser needs
        its ``or []`` guard.
        """
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        wrapper = YouAPIWrapper(ydc_api_key="k")
        docs = wrapper._parse_search_response(response)

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "web"

    def test_knowledge_precedes_web_under_k_limit(self) -> None:
        """A pinned ``k`` keeps knowledge, the most authoritative tier."""
        response = make_search_response(
            web=[make_web_hit(url=f"https://example.com/{i}") for i in range(3)],
            knowledge=[make_knowledge_result()],
        )
        wrapper = YouAPIWrapper(ydc_api_key="k", k=2)
        docs = wrapper._parse_search_response(response)

        assert [d.metadata["source"] for d in docs] == ["knowledge", "web"]


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
        # 30_000 = the API's 10 s default crawl budget plus 20 s of headroom,
        # so the client cannot give up while the server is still inside its own
        # limit (httpx defaults to 5 s when timeout_ms is not passed).
        mock_client.search.assert_called_once_with(
            query="test query", count=5, timeout_ms=30_000
        )
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
    def test_search_forwards_extraction(self, mock_you_cls: MagicMock) -> None:
        """The ``extraction`` dict reaches the SDK verbatim.

        The wrapper does not reshape or validate it — the SDK's ``Extraction``
        model rejects unknown keys and wrong-mode couplings itself.
        """
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        extraction = {
            "extraction_mode": "full_page",
            "extraction_source": "blend",
            "full_page": {"extraction_formats": ["markdown"]},
        }
        wrapper = YouAPIWrapper(ydc_api_key="test-key", extraction=extraction)
        wrapper.results("test query")

        assert mock_client.search.call_args.kwargs["extraction"] == extraction

    def test_extraction_conflicts_with_livecrawl(self) -> None:
        """Combining ``extraction`` with deprecated ``livecrawl`` is rejected.

        Deliberately unmocked: the SDK mirrors the server's ``422`` locally so
        the caller fails before a round-trip, which means the real client raises
        without network access or an API key. The wrapper forwards both fields
        and leaves the conflict rule to the SDK rather than duplicating it.
        """
        wrapper = YouAPIWrapper(
            ydc_api_key="test-key",
            extraction={"extraction_mode": "highlights"},
            livecrawl="web",
        )

        with pytest.raises(ValueError, match="extraction cannot be combined"):
            wrapper.results("test query")

    @patch("langchain_youdotcom._utilities.You")
    def test_search_forwards_knowledge(self, mock_you_cls: MagicMock) -> None:
        """``knowledge`` reaches the SDK when set (SDK 3.5.0)."""
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key", knowledge="core")
        wrapper.results("test query")

        assert mock_client.search.call_args.kwargs["knowledge"] == "core"

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
    def test_contents_default_formats_exclude_metadata(
        self, mock_you_cls: MagicMock
    ) -> None:
        """The default requests markdown only, not the deprecated metadata.

        The SDK deprecated the ``metadata`` format in 3.4.0 and emits a
        ``DeprecationWarning`` at request time whenever it appears in
        ``formats``. Keeping it out of the default is what stops every
        ``YouContentsTool()`` call from warning. Callers who need
        ``site_name`` / ``favicon_url`` can still pass it explicitly.
        """
        from youdotcom.models import ContentsFormats

        mock_client = MagicMock()
        mock_client.contents.return_value = [make_contents_page()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        wrapper.contents(["https://example.com"])

        formats = mock_client.contents.call_args.kwargs["formats"]
        assert formats == [ContentsFormats.MARKDOWN]
        assert ContentsFormats.METADATA not in formats

    @patch("langchain_youdotcom._utilities.You")
    def test_contents_explicit_metadata_still_forwarded(
        self, mock_you_cls: MagicMock
    ) -> None:
        """Opting back into ``metadata`` still works after the default changed."""
        from youdotcom.models import ContentsFormats

        mock_client = MagicMock()
        mock_client.contents.return_value = [make_contents_page()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        wrapper.contents(["https://example.com"], formats=["markdown", "metadata"])

        formats = mock_client.contents.call_args.kwargs["formats"]
        assert formats == [ContentsFormats.MARKDOWN, ContentsFormats.METADATA]

    @patch("langchain_youdotcom._utilities.You")
    def test_contents_forwards_max_age(self, mock_you_cls: MagicMock) -> None:
        """``max_age`` bounds how stale cached content may be."""
        mock_client = MagicMock()
        mock_client.contents.return_value = [make_contents_page()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        wrapper.contents(["https://example.com"], max_age=3600)

        assert mock_client.contents.call_args.kwargs["max_age"] == 3600

    def test_contents_max_age_omitted_by_default(self) -> None:
        """An unset ``max_age`` is not sent, preserving the API's no-limit."""
        params = YouAPIWrapper._contents_params(["https://example.com"])

        assert "max_age" not in params

    def test_contents_rejects_more_than_ten_urls(self) -> None:
        """The API answers an 11-URL request with a 422; reject it locally.

        The SDK neither documents nor validates this limit, so without the
        local check the caller gets a raw 422 after a round-trip.
        """
        urls = [f"https://example.com/{i}" for i in range(11)]

        with pytest.raises(ValueError, match="at most 10 URLs"):
            YouAPIWrapper._contents_params(urls)

    def test_contents_accepts_exactly_ten_urls(self) -> None:
        """The 10-URL limit is inclusive."""
        urls = [f"https://example.com/{i}" for i in range(10)]

        assert len(YouAPIWrapper._contents_params(urls)["urls"]) == 10

    @patch("langchain_youdotcom._utilities.You")
    def test_crawl_timeout_scales_client_timeout(self, mock_you_cls: MagicMock) -> None:
        """The client budget must outlast the server-side crawl budget.

        httpx defaults to 5 s and the SDK only overrides it when ``timeout_ms``
        is passed, so without this a caller-set ``crawl_timeout`` of 60 could
        never be honored — the client would raise ``ReadTimeout`` first.
        """
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key", crawl_timeout=60)
        wrapper.results("test query")

        assert mock_client.search.call_args.kwargs["timeout_ms"] == 80_000

    @patch("langchain_youdotcom._utilities.You")
    def test_contents_crawl_timeout_scales_client_timeout(
        self, mock_you_cls: MagicMock
    ) -> None:
        """Contents scales its client timeout from its own argument."""
        mock_client = MagicMock()
        mock_client.contents.return_value = [make_contents_page()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(ydc_api_key="test-key")
        wrapper.contents(["https://example.com"], crawl_timeout=45)

        assert mock_client.contents.call_args.kwargs["timeout_ms"] == 65_000

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

    def test_answer_text_enriches_citations_from_web_results(self) -> None:
        """Citations gain the matching web result's title and description.

        ``response.citations`` carries only ``source`` and ``excerpts``; the
        richer ``title`` and ``description`` live on ``results.web[]``. The two
        are joined by URL — cited sources are a subset of the web results.
        """
        response = make_answer_response(
            answer="RAG combines retrieval and generation [1].",
            citations=[
                {
                    "source": "https://arxiv.org/abs/2005.11401",
                    "excerpts": ["RAG abstract."],
                }
            ],
            web_results=[
                {
                    "url": "https://arxiv.org/abs/2005.11401",
                    "title": "Retrieval-Augmented Generation",
                    "description": "The original RAG paper.",
                }
            ],
        )
        with patch(
            "langchain_youdotcom.YouAPIWrapper.raw_answer",
            return_value=response,
        ):
            result = YouAPIWrapper(ydc_api_key="k").answer_text("what is RAG")

        assert (
            "[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)"
            in result
        )
        assert "The original RAG paper." in result
        assert "   > RAG abstract." in result

    def test_answer_text_falls_back_to_url_when_no_web_match(self) -> None:
        """A citation with no results.web entry keeps the bare-URL format."""
        response = make_answer_response(
            citations=[{"source": "https://other.org/x", "excerpts": ["e"]}],
            web_results=[{"url": "https://unrelated.example", "title": "No match"}],
        )
        with patch(
            "langchain_youdotcom.YouAPIWrapper.raw_answer",
            return_value=response,
        ):
            result = YouAPIWrapper(ydc_api_key="k").answer_text("q")

        assert "[https://other.org/x](https://other.org/x)" in result

    def test_answer_text_handles_missing_results_block(self) -> None:
        """``response.results`` is Optional; a citation-only response still works."""
        response = make_answer_response(
            citations=[{"source": "https://other.org/x", "excerpts": ["e"]}],
        )
        response.results = None
        with patch(
            "langchain_youdotcom.YouAPIWrapper.raw_answer",
            return_value=response,
        ):
            result = YouAPIWrapper(ydc_api_key="k").answer_text("q")

        assert "## Citations" in result
        assert "[https://other.org/x](https://other.org/x)" in result


class TestSearchValidation:
    """Pre-flight validation and forwarding of Search parameters."""

    def test_include_with_exclude_rejected(self) -> None:
        """include_domains conflicts with exclude_domains on Search too.

        The API answers this combination with a 422; surfacing it locally
        matches what ``_answer_params`` already does for the Answer API.
        """
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper(
                ydc_api_key="k",
                include_domains=["a.com"],
                exclude_domains=["b.com"],
            )._search_params("q")

    def test_include_with_boost_rejected(self) -> None:
        """include_domains conflicts with boost_domains."""
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper(
                ydc_api_key="k",
                include_domains=["a.com"],
                boost_domains=["b.com"],
            )._search_params("q")

    def test_exclude_with_boost_allowed(self) -> None:
        """exclude_domains and boost_domains combine legally."""
        params = YouAPIWrapper(
            ydc_api_key="k",
            exclude_domains=["spam.example"],
            boost_domains=["preferred.example"],
        )._search_params("q")

        assert params["exclude_domains"] == ["spam.example"]
        assert params["boost_domains"] == ["preferred.example"]

    @patch("langchain_youdotcom._utilities.You")
    def test_new_search_params_forwarded(self, mock_you_cls: MagicMock) -> None:
        """Domain filters and crawl_timeout all reach the SDK."""
        response = make_search_response(web=[make_web_hit(snippets=["result"])])
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        wrapper = YouAPIWrapper(
            ydc_api_key="test-key",
            include_domains=["a.com", "b.com"],
            crawl_timeout=30,
        )
        wrapper.results("test query")

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["include_domains"] == ["a.com", "b.com"]
        assert call_kwargs["crawl_timeout"] == 30


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

    def test_answer_params_seed_from_wrapper_config(self) -> None:
        """Wrapper-level config applies to Answer, not only to Search.

        A caller who configures ``country`` / ``safesearch`` / domain filters
        once on the wrapper expects them to hold across every endpoint.
        """
        wrapper = YouAPIWrapper(
            ydc_api_key="k",
            country="US",
            safesearch="strict",
            exclude_domains=["spam.example"],
        )
        params = wrapper._answer_params("q")

        assert params["country"] == "US"
        assert params["safesearch"] == "strict"
        assert params["exclude_domains"] == ["spam.example"]

    def test_answer_call_filters_override_wrapper_config(self) -> None:
        """A per-call filter wins over the wrapper-level default."""
        wrapper = YouAPIWrapper(ydc_api_key="k", country="US")
        params = wrapper._answer_params("q", country="DE")

        assert params["country"] == "DE"

    def test_answer_rejects_combo_formed_by_config_and_call(self) -> None:
        """Validation runs on the merged result, not just the call filters.

        ``include_domains`` from wrapper config plus ``exclude_domains`` from
        the call is still illegal, and checking only ``filters`` would miss it
        and cost a round-trip to a 422.
        """
        wrapper = YouAPIWrapper(ydc_api_key="k", include_domains=["a.com"])

        with pytest.raises(ValueError, match="include_domains"):
            wrapper._answer_params("q", exclude_domains=["b.com"])


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

    def test_format_structured_output_content(self) -> None:
        r"""``output_schema`` makes ``output.content`` a dict, not a string.

        ``_numbered_section`` joins its parts with ``"\\n"``, so a dict body
        would raise ``TypeError``. Structured output is serialized to JSON so
        the text-returning helpers stay total; callers who want the parsed
        object use ``raw_research()``.
        """
        response = make_research_response(
            content={"summary": "The answer is 42.", "confidence": 0.9},
            sources=[make_research_source(url="https://a.com", title="Source A")],
        )
        wrapper = YouAPIWrapper(ydc_api_key="k")
        result = wrapper._format_research_response(response)

        assert '"summary": "The answer is 42."' in result
        assert "## Sources" in result
        assert "1. [Source A](https://a.com)" in result

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

    def test_format_includes_source_snippets(self) -> None:
        """Source snippets render as excerpts beneath each source link.

        Shared by Research and Finance Research — both go through
        ``_format_research_response``. ``Source.snippets`` was already present;
        ``FinanceResearchSource.snippets`` was added in SDK 3.5.0 and production
        was not returning it yet, so finance renders nothing until it does.
        """
        sources = [
            make_research_source(
                url="https://a.com", title="A", snippets=["from source A"]
            ),
            make_research_source(url="https://b.com", title="B"),  # no snippets
        ]
        response = make_research_response(content="The answer.", sources=sources)
        wrapper = YouAPIWrapper(ydc_api_key="k")
        result = wrapper._format_research_response(response)

        assert "[A](https://a.com)" in result
        assert "   > from source A" in result
        assert "[B](https://b.com)" in result

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

    def test_research_params_with_output_schema_and_source_control(self) -> None:
        """Research-only params pass through to the SDK unmodified.

        Omission when unset is already pinned by
        ``test_research_params_without_effort``, which asserts the exact dict.
        """
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        control = {"include_domains": ["sec.gov"], "freshness": "year"}
        wrapper = YouAPIWrapper(
            ydc_api_key="k",
            output_schema=schema,
            source_control=control,
        )
        params = wrapper._research_params("my query")

        assert params["output_schema"] == schema
        assert params["source_control"] == control

    def test_research_params_rejects_frontier(self) -> None:
        """Frontier is task-only; sync API returns 422, so reject locally."""
        wrapper = YouAPIWrapper(ydc_api_key="k", research_effort="frontier")
        with pytest.raises(ValueError, match="frontier"):
            wrapper._research_params("my query")

    def test_research_params_rejects_output_schema_with_lite(self) -> None:
        """``output_schema`` with ``lite`` is a 422, so reject locally."""
        wrapper = YouAPIWrapper(
            ydc_api_key="k",
            research_effort="lite",
            output_schema={"type": "object"},
        )
        with pytest.raises(ValueError, match="output_schema"):
            wrapper._research_params("my query")

    def test_research_params_allows_output_schema_with_deep(self) -> None:
        """``output_schema`` is valid with standard/deep/exhaustive."""
        wrapper = YouAPIWrapper(
            ydc_api_key="k",
            research_effort="deep",
            output_schema={"type": "object"},
        )
        params = wrapper._research_params("my query")

        assert params["output_schema"] == {"type": "object"}


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
