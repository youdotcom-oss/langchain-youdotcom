"""Unit tests for YouRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_youdotcom import YouRetriever
from tests.unit_tests.conftest import make_search_response, make_web_hit


class TestYouRetriever:
    """Tests for YouRetriever."""

    def test_is_base_retriever_subclass(self) -> None:
        """YouRetriever must extend BaseRetriever."""
        assert issubclass(YouRetriever, BaseRetriever)

    def test_init_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retriever initializes without arguments."""
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        retriever = YouRetriever()
        assert retriever.ydc_api_key == ""

    def test_init_with_api_key(self) -> None:
        """Retriever accepts an explicit API key."""
        retriever = YouRetriever(ydc_api_key="test-key")
        assert retriever.ydc_api_key == "test-key"

    def test_inherits_search_config(self) -> None:
        """Retriever exposes search wrapper config fields."""
        retriever = YouRetriever(ydc_api_key="k", k=3, count=10)
        assert retriever.k == 3
        assert retriever.count == 10

    @patch("langchain_youdotcom._utilities.You")
    def test_invoke_returns_documents(self, mock_you_cls: MagicMock) -> None:
        """invoke() returns documents from search results."""
        response = make_search_response(web=[make_web_hit(snippets=["hello world"])])
        mock_client = MagicMock()
        mock_client.search.unified.return_value = response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_you_cls.return_value = mock_client

        retriever = YouRetriever(ydc_api_key="test-key")
        docs = retriever.invoke("test query")

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "hello world"
