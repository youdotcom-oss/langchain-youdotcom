"""Unit tests for YouRetriever."""

from __future__ import annotations

import pytest
from langchain_core.retrievers import BaseRetriever

from langchain_youdotcom import YouRetriever


class TestYouRetriever:
    """Tests for YouRetriever."""

    def test_is_base_retriever_subclass(self) -> None:
        """YouRetriever must extend BaseRetriever."""
        assert issubclass(YouRetriever, BaseRetriever)

    def test_init_default(self) -> None:
        """Retriever initializes without arguments."""
        retriever = YouRetriever()
        assert retriever.ydc_api_key == ""

    def test_init_with_api_key(self) -> None:
        """Retriever accepts an explicit API key."""
        retriever = YouRetriever(ydc_api_key="test-key")
        assert retriever.ydc_api_key == "test-key"

    def test_invoke_raises_not_implemented(self) -> None:
        """Stub implementation raises NotImplementedError."""
        retriever = YouRetriever()
        with pytest.raises(NotImplementedError):
            retriever.invoke("test query")
