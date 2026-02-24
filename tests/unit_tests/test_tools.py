"""Unit tests for You.com tool classes."""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from langchain_youdotcom import YouContentsTool, YouSearchTool


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

    def test_run_raises_not_implemented(self) -> None:
        """Stub implementation raises NotImplementedError."""
        tool = YouSearchTool()
        with pytest.raises(NotImplementedError):
            tool._run("test query")


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

    def test_run_raises_not_implemented(self) -> None:
        """Stub implementation raises NotImplementedError."""
        tool = YouContentsTool()
        with pytest.raises(NotImplementedError):
            tool._run("https://example.com")
