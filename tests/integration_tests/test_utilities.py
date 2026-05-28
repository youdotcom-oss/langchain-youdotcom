"""Integration tests for YouAPIWrapper."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_youdotcom import YouAPIWrapper


def test_results_basic() -> None:
    """results() returns documents for a simple query."""
    wrapper = YouAPIWrapper()
    docs = wrapper.results("what is langchain")

    print(docs)  # noqa: T201
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert docs[0].page_content


def test_raw_results() -> None:
    """raw_results() returns the SDK response object."""
    wrapper = YouAPIWrapper()
    response = wrapper.raw_results("python tutorials")

    print(response)  # noqa: T201
    assert response.results is not None
    assert response.results.web is not None
    assert len(response.results.web) > 0


def test_results_k_limit() -> None:
    """K limits the total documents returned."""
    wrapper = YouAPIWrapper(k=2)
    docs = wrapper.results("machine learning")

    assert len(docs) <= 2


def test_results_with_livecrawl() -> None:
    """Livecrawl fetches full page content."""
    wrapper = YouAPIWrapper(k=1, livecrawl="web")
    docs = wrapper.results("langchain documentation")

    assert len(docs) > 0
    assert docs[0].page_content


def test_contents_basic() -> None:
    """contents() fetches and parses page content."""
    wrapper = YouAPIWrapper()
    docs = wrapper.contents(["https://example.com"])

    print(docs)  # noqa: T201
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert docs[0].page_content
    assert docs[0].metadata.get("url") == "https://example.com"


def test_contents_multiple_urls() -> None:
    """contents() handles multiple URLs."""
    wrapper = YouAPIWrapper()
    urls = ["https://example.com", "https://httpbin.org/html"]
    docs = wrapper.contents(urls)

    assert len(docs) > 0


def test_raw_research() -> None:
    """raw_research() returns the SDK response object."""
    wrapper = YouAPIWrapper(research_effort="lite")
    response = wrapper.raw_research("what is langchain")

    print(response)  # noqa: T201
    assert response.output is not None
    assert response.output.content
    assert isinstance(response.output.sources, list)


def test_research_text() -> None:
    """research_text() returns formatted markdown with sources."""
    wrapper = YouAPIWrapper(research_effort="lite")
    result = wrapper.research_text("what is retrieval augmented generation")

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
    assert "## Sources" in result


def test_finance_text() -> None:
    """finance_text() returns formatted markdown with sources."""
    wrapper = YouAPIWrapper(research_effort="deep")
    result = wrapper.finance_text("what were NVIDIA's key revenue drivers in FY2025")

    print(result)  # noqa: T201
    assert isinstance(result, str)
    assert len(result) > 0
    assert "## Sources" in result
