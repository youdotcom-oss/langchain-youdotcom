"""Unit tests for You.com tool classes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from langchain_youdotcom import (
    YouContentsTool,
    YouResearchTool,
    YouSearchAPIWrapper,
    YouSearchTool,
)
from tests.unit_tests.conftest import (
    make_contents_page,
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
        mock_client.search.unified.return_value = response
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
        with patch.object(
            YouSearchAPIWrapper, "results", return_value=docs
        ) as mock_results:
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
        mock_client.contents.generate.return_value = [page]
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
            YouSearchAPIWrapper, "contents", return_value=docs
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
            YouSearchAPIWrapper,
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
            YouSearchAPIWrapper,
            "research_text_async",
            return_value="async mocked answer",
        ) as mock_research:
            tool = YouResearchTool()
            result = await tool._arun("test")

        mock_research.assert_called_once_with("test")
        assert result == "async mocked answer"
