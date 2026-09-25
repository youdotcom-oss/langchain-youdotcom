"""Unit tests for You.com tool classes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from langchain_youdotcom import (
    YouAnswerTool,
    YouAPIWrapper,
    YouContentsTool,
    YouFinanceResearchTool,
    YouResearchTool,
    YouSearchTool,
)
from langchain_youdotcom.tools import _format_docs
from tests.unit_tests.conftest import (
    make_answer_response,
    make_contents_page,
    make_finance_research_response,
    make_research_response,
    make_search_response,
    make_web_hit,
)


class TestYouSearchTool:
    """Tests for YouSearchTool."""

    def test_is_base_tool_subclass(self) -> None:
        """YouSearchTool must extend BaseTool."""
        assert issubclass(YouSearchTool, BaseTool)

    def test_default_name(self) -> None:
        """Tool should have a sensible default name."""
        tool = YouSearchTool()
        assert tool.name == "you_search"

    def test_default_description(self) -> None:
        """Tool should have a non-empty description."""
        tool = YouSearchTool()
        assert len(tool.description) > 0

    @patch("langchain_youdotcom._utilities.You")
    def test_run_returns_formatted_results(self, mock_you_cls: MagicMock) -> None:
        """_run delegates to api_wrapper.results and formats output."""
        response = make_search_response(
            web=[
                make_web_hit(
                    url="https://a.com",
                    title="A",
                    snippets=["content a"],
                ),
            ]
        )
        mock_client = MagicMock()
        mock_client.search.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouSearchTool()
        result = tool._run("test query")

        assert isinstance(result, str)
        assert "content a" in result

    def test_run_with_patched_wrapper(self) -> None:
        """_run delegates to the api_wrapper."""
        docs = [
            Document(
                page_content="hello",
                metadata={"title": "T", "url": "https://x.com"},
            )
        ]
        with patch.object(YouAPIWrapper, "results", return_value=docs) as mock_results:
            tool = YouSearchTool()
            result = tool._run("test")

        mock_results.assert_called_once_with("test")
        assert "hello" in result


class TestYouContentsTool:
    """Tests for YouContentsTool."""

    def test_is_base_tool_subclass(self) -> None:
        """YouContentsTool must extend BaseTool."""
        assert issubclass(YouContentsTool, BaseTool)

    def test_default_name(self) -> None:
        """Tool should have a sensible default name."""
        tool = YouContentsTool()
        assert tool.name == "you_contents"

    def test_default_description(self) -> None:
        """Tool should have a non-empty description."""
        tool = YouContentsTool()
        assert len(tool.description) > 0

    @patch("langchain_youdotcom._utilities.You")
    def test_run_returns_formatted_contents(self, mock_you_cls: MagicMock) -> None:
        """_run delegates to api_wrapper.contents and formats output."""
        page = make_contents_page(
            url="https://example.com",
            title="Example",
            markdown="# Page Content",
        )
        mock_client = MagicMock()
        mock_client.contents.return_value = [page]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouContentsTool()
        result = tool._run(["https://example.com"])

        assert isinstance(result, str)
        assert "# Page Content" in result

    def test_run_with_patched_wrapper(self) -> None:
        """_run delegates to the api_wrapper."""
        docs = [
            Document(
                page_content="page text",
                metadata={"title": "P", "url": "https://x.com"},
            )
        ]
        with patch.object(
            YouAPIWrapper, "contents", return_value=docs
        ) as mock_contents:
            tool = YouContentsTool()
            result = tool._run(["https://x.com"])

        mock_contents.assert_called_once_with(["https://x.com"])
        assert "page text" in result


class TestYouResearchTool:
    """Tests for YouResearchTool."""

    def test_is_base_tool_subclass(self) -> None:
        """YouResearchTool must extend BaseTool."""
        assert issubclass(YouResearchTool, BaseTool)

    def test_default_name(self) -> None:
        """Tool should have a sensible default name."""
        tool = YouResearchTool()
        assert tool.name == "you_research"

    def test_default_description(self) -> None:
        """Tool should have a non-empty description."""
        tool = YouResearchTool()
        assert len(tool.description) > 0

    @patch("langchain_youdotcom._utilities.You")
    def test_run_returns_formatted_research(self, mock_you_cls: MagicMock) -> None:
        """_run delegates to api_wrapper.research_text and returns markdown."""
        response = make_research_response(
            content="Deep research answer.",
        )
        mock_client = MagicMock()
        mock_client.research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouResearchTool()
        result = tool._run("test query")

        assert isinstance(result, str)
        assert "Deep research answer." in result
        assert "## Sources" in result

    def test_run_with_patched_wrapper(self) -> None:
        """_run delegates to the api_wrapper."""
        with patch.object(
            YouAPIWrapper,
            "research_text",
            return_value="mocked answer",
        ) as mock_research:
            tool = YouResearchTool()
            result = tool._run("test")

        mock_research.assert_called_once_with("test")
        assert result == "mocked answer"

    async def test_arun_with_patched_wrapper(self) -> None:
        """_arun delegates to the api_wrapper async method."""
        with patch.object(
            YouAPIWrapper,
            "research_text_async",
            return_value="async mocked answer",
        ) as mock_research:
            tool = YouResearchTool()
            result = await tool._arun("test")

        mock_research.assert_called_once_with("test")
        assert result == "async mocked answer"


class TestYouFinanceResearchTool:
    """Tests for YouFinanceResearchTool."""

    def test_is_base_tool_subclass(self) -> None:
        """YouFinanceResearchTool must extend BaseTool."""
        assert issubclass(YouFinanceResearchTool, BaseTool)

    def test_default_name(self) -> None:
        """Tool should have a sensible default name."""
        tool = YouFinanceResearchTool()
        assert tool.name == "you_finance_research"

    def test_default_description(self) -> None:
        """Tool should have a non-empty description."""
        tool = YouFinanceResearchTool()
        assert len(tool.description) > 0

    @patch("langchain_youdotcom._utilities.You")
    def test_run_delegates_to_sdk_finance_research(
        self, mock_you_cls: MagicMock
    ) -> None:
        """_run paths through YouAPIWrapper to ``client.finance_research``."""
        response = make_finance_research_response(
            content="NVIDIA revenue grew 40%.",
        )
        mock_client = MagicMock()
        mock_client.finance_research.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouFinanceResearchTool()
        result = tool._run("NVDA earnings")

        assert isinstance(result, str)
        assert "NVIDIA revenue grew 40%." in result
        assert "## Sources" in result

    def test_run_with_patched_wrapper(self) -> None:
        """_run delegates to the api_wrapper."""
        with patch.object(
            YouAPIWrapper,
            "finance_text",
            return_value="mocked finance answer",
        ) as mock_finance:
            tool = YouFinanceResearchTool()
            result = tool._run("test")

        mock_finance.assert_called_once_with("test")
        assert result == "mocked finance answer"

    async def test_arun_with_patched_wrapper(self) -> None:
        """_arun delegates to the api_wrapper async method."""
        with patch.object(
            YouAPIWrapper,
            "finance_text_async",
            return_value="async mocked finance answer",
        ) as mock_finance:
            tool = YouFinanceResearchTool()
            result = await tool._arun("test")

        mock_finance.assert_called_once_with("test")
        assert result == "async mocked finance answer"


class TestYouAnswerTool:
    """Tests for YouAnswerTool."""

    def test_is_base_tool_subclass(self) -> None:
        """YouAnswerTool must extend BaseTool."""
        assert issubclass(YouAnswerTool, BaseTool)

    def test_default_name(self) -> None:
        """Tool should default to ``you_answer``."""
        tool = YouAnswerTool()
        assert tool.name == "you_answer"

    def test_default_description(self) -> None:
        """Tool should have a non-empty description."""
        tool = YouAnswerTool()
        assert len(tool.description) > 0

    @patch("langchain_youdotcom._utilities.You")
    def test_run_return_formatted_answer(self, mock_you_cls: MagicMock) -> None:
        """_run delegates to api_wrapper.answer_text and returns markdown."""
        response = make_answer_response(
            answer="RAG stands for retrieval augmented generation.",
            citations=[
                {"source": "https://example.com/rag", "excerpts": ["..."]},
            ],
        )
        mock_client = MagicMock()
        mock_client.answer.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouAnswerTool()
        result = tool.invoke({"query": "what is RAG"})

        assert isinstance(result, str)
        assert "RAG stands for retrieval augmented generation." in result
        assert "## Citations" in result
        assert "https://example.com/rag" in result

    @patch("langchain_youdotcom._utilities.You")
    def test_run_forwards_filter_kwargs(self, mock_you_cls: MagicMock) -> None:
        """All filter kwargs reach ``client.answer``."""
        response = make_answer_response(answer="result")
        mock_client = MagicMock()
        mock_client.answer.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouAnswerTool()
        tool.invoke(
            {
                "query": "quantum",
                "freshness": "week",
                "country": "US",
                "language": "EN",
                "safesearch": "moderate",
                "include_domains": ["example.com"],
                "exclude_domains": None,
                "boost_domains": None,
            }
        )

        call_kwargs = mock_client.answer.call_args.kwargs
        assert call_kwargs["query"] == "quantum"
        assert call_kwargs["freshness"] == "week"
        assert call_kwargs["country"] == "US"
        assert call_kwargs["language"] == "EN"
        assert call_kwargs["safesearch"] == "moderate"
        assert call_kwargs["include_domains"] == ["example.com"]

    def test_run_with_patched_wrapper(self) -> None:
        """_run delegates to api_wrapper.answer_text."""
        with patch.object(
            YouAPIWrapper,
            "answer_text",
            return_value="mocked answer",
        ) as mock_answer:
            tool = YouAnswerTool()
            result = tool.invoke({"query": "test"})

        mock_answer.assert_called_once()
        assert result == "mocked answer"

    async def test_arun_with_patched_wrapper(self) -> None:
        """_arun delegates to api_wrapper.answer_text_async."""
        with patch.object(
            YouAPIWrapper,
            "answer_text_async",
            return_value="async mocked answer",
        ) as mock_answer:
            tool = YouAnswerTool()
            result = await tool._arun("test")

        mock_answer.assert_called_once()
        assert result == "async mocked answer"

    def test_run_rejects_include_with_exclude(self) -> None:
        """Combine-validation happens before any HTTP call."""
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper().answer_text(
                "what is RAG",
                include_domains=["a.com"],
                exclude_domains=["b.com"],
            )

    def test_run_rejects_include_with_boost(self) -> None:
        """Combine-validation happens before any HTTP call."""
        with pytest.raises(ValueError, match="include_domains"):
            YouAPIWrapper().answer_text(
                "what is RAG",
                include_domains=["a.com"],
                boost_domains=["b.com"],
            )

    def test_run_rejects_query_over_400_chars(self) -> None:
        """Server rejects queries longer than 400 chars; surface error locally."""
        too_long = "x" * 401
        with pytest.raises(ValueError, match="400"):
            YouAPIWrapper().answer_text(too_long)

    def test_run_rejects_empty_query(self) -> None:
        """Empty queries are rejected without an HTTP call."""
        with pytest.raises(ValueError, match="query"):
            YouAPIWrapper().answer_text("")

    @patch("langchain_youdotcom._utilities.You")
    async def test_arun_delegates_to_sdk_answer(self, mock_you_cls: MagicMock) -> None:
        """Async _arun calls ``client.answer_async``."""
        response = make_answer_response(answer="async result")
        mock_client = MagicMock()
        mock_client.answer_async = AsyncMock(return_value=response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_you_cls.return_value = mock_client

        tool = YouAnswerTool()
        result = await tool._arun("test")

        assert "async result" in result
        assert "## Citations" in result


class TestFormatDocs:
    """Formatting of Documents into the string the search tools return."""

    def test_web_doc_renders_title_and_url(self) -> None:
        """A document with a URL renders both in the header."""
        docs = [
            Document(
                page_content="body",
                metadata={"title": "Title", "url": "https://example.com"},
            )
        ]

        assert _format_docs(docs) == "Title\nhttps://example.com\n\nbody"

    def test_doc_without_url_does_not_raise(self) -> None:
        """Knowledge documents carry no ``url``; formatting must not KeyError.

        ``_format_docs`` reads ``metadata.get("url", "")`` rather than
        ``metadata["url"]``, so a knowledge document renders its title and
        content with an empty URL line instead of blowing up. This is the seam
        where the no-URL knowledge design meets user-visible tool output.
        """
        docs = [
            Document(
                page_content="licensed answer",
                metadata={"title": "GDP", "source": "knowledge", "type": "answer"},
            )
        ]

        assert _format_docs(docs) == "GDP\n\n\nlicensed answer"

    def test_doc_without_title_or_url_omits_header(self) -> None:
        """A bare document renders its content with no header."""
        assert _format_docs([Document(page_content="body", metadata={})]) == "body"

    def test_multiple_docs_are_separated(self) -> None:
        """Documents are joined with a horizontal rule."""
        docs = [
            Document(page_content="one", metadata={}),
            Document(page_content="two", metadata={}),
        ]

        assert _format_docs(docs) == "one\n\n---\n\ntwo"
