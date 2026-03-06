"""Integration tests for YouSearchTool and YouContentsTool."""

from __future__ import annotations

from langchain_youdotcom import (
    YouContentsTool,
    YouResearchTool,
    YouSearchAPIWrapper,
    YouSearchTool,
)


def test_search_tool_basic() -> None:
    """YouSearchTool returns a non-empty string."""
    tool = YouSearchTool()
    result = tool.invoke("what is retrieval augmented generation")

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_tool_contains_content() -> None:
    """Search result string contains actual content."""
    tool = YouSearchTool()
    result = tool.invoke("python langchain")

    assert "http" in result  # URLs appear in formatted output


def test_research_tool_basic() -> None:
    """YouResearchTool returns a non-empty string with sources."""
    tool = YouResearchTool(
        api_wrapper=YouSearchAPIWrapper(research_effort="lite"),
    )
    result = tool.invoke("what is retrieval augmented generation")

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
    assert "## Sources" in result


def test_contents_tool_basic() -> None:
    """YouContentsTool returns a non-empty string."""
    tool = YouContentsTool()
    result = tool.invoke({"urls": ["https://example.com"]})

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
