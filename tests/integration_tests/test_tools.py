"""Integration tests for You.com tools."""

from __future__ import annotations

from langchain_youdotcom import (
    YouAnswerTool,
    YouAPIWrapper,
    YouContentsTool,
    YouFinanceResearchTool,
    YouResearchTool,
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


def test_search_full_page_extraction() -> None:
    """``extraction`` full_page returns page content rather than snippets."""
    wrapper = YouAPIWrapper(
        count=3,
        extraction={
            "extraction_mode": "full_page",
            "extraction_source": "blend",
            "full_page": {"extraction_formats": ["markdown"]},
        },
    )
    docs = wrapper.results("python langchain tutorial")

    assert len(docs) > 0
    # Full-page markdown dwarfs a snippet; the longest doc proves extraction ran.
    longest = max(len(doc.page_content) for doc in docs)
    print(longest)  # noqa: T201
    assert longest > 500


def test_search_knowledge_results() -> None:
    """``knowledge="core"`` yields knowledge documents when relevant.

    The API omits the section entirely when nothing licensed matches, so this
    asserts the invariants of any knowledge documents returned rather than a
    fixed count.
    """
    wrapper = YouAPIWrapper(count=5, knowledge="core")
    docs = wrapper.results("NVIDIA revenue fiscal year 2025")

    assert len(docs) > 0
    knowledge_docs = [d for d in docs if d.metadata.get("source") == "knowledge"]
    print(len(knowledge_docs))  # noqa: T201
    for doc in knowledge_docs:
        assert doc.page_content
        assert doc.metadata["title"]
        assert doc.metadata["type"]
        # Knowledge results are provider credits, not citations: no URL exists.
        assert "url" not in doc.metadata


def test_research_tool_basic() -> None:
    """YouResearchTool returns a non-empty string with sources."""
    tool = YouResearchTool(
        api_wrapper=YouAPIWrapper(research_effort="lite"),
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


def test_finance_research_tool_basic() -> None:
    """YouFinanceResearchTool returns a non-empty string with sources."""
    tool = YouFinanceResearchTool(
        api_wrapper=YouAPIWrapper(research_effort="deep"),
    )
    result = tool.invoke("what were NVIDIA's key revenue drivers in FY2025")

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
    assert "## Sources" in result


def test_answer_tool_basic() -> None:
    """YouAnswerTool returns a non-empty string with citations."""
    tool = YouAnswerTool()
    result = tool.invoke({"query": "what is retrieval augmented generation"})

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
    assert "http" in result  # citation URLs appear in formatted output


def test_answer_tool_with_freshness() -> None:
    """YouAnswerTool accepts the freshness filter."""
    tool = YouAnswerTool()
    result = tool.invoke(
        {"query": "python release", "freshness": "week"},
    )

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
