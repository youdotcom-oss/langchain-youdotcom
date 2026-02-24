"""Integration tests for YouRetriever."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_youdotcom import YouRetriever


def test_retriever_basic() -> None:
    """Retriever returns documents for a simple query."""
    retriever = YouRetriever()
    docs = retriever.invoke("latest AI news")

    print(docs)  # noqa: T201
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert docs[0].page_content
    assert docs[0].metadata.get("url")


def test_retriever_k_limit() -> None:
    """K parameter limits the number of documents returned."""
    retriever = YouRetriever(k=3)
    docs = retriever.invoke("python programming")

    assert len(docs) <= 3


def test_retriever_metadata_fields() -> None:
    """Documents include expected metadata fields."""
    retriever = YouRetriever(k=1)
    docs = retriever.invoke("openai gpt-4")

    doc = docs[0]
    assert "url" in doc.metadata
    assert "title" in doc.metadata
