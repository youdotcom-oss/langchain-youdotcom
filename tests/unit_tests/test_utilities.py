"""Unit tests for YouSearchAPIWrapper."""

from __future__ import annotations

import os

import pytest

from langchain_youdotcom import YouSearchAPIWrapper


class TestYouSearchAPIWrapper:
    """Tests for YouSearchAPIWrapper."""

    def test_init_default_empty_key(self) -> None:
        """Wrapper initializes with empty key when env var is unset."""
        env = os.environ.copy()
        os.environ.pop("YDC_API_KEY", None)
        try:
            wrapper = YouSearchAPIWrapper()
            assert wrapper.ydc_api_key == ""
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_init_with_explicit_key(self) -> None:
        """Wrapper accepts an explicit API key."""
        wrapper = YouSearchAPIWrapper(ydc_api_key="test-key")
        assert wrapper.ydc_api_key == "test-key"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrapper reads YDC_API_KEY from environment."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouSearchAPIWrapper()
        assert wrapper.ydc_api_key == "env-key"

    def test_explicit_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit key overrides environment variable."""
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        wrapper = YouSearchAPIWrapper(ydc_api_key="explicit-key")
        assert wrapper.ydc_api_key == "explicit-key"

    def test_raw_results_raises_not_implemented(self) -> None:
        """Stub implementation raises NotImplementedError."""
        wrapper = YouSearchAPIWrapper()
        with pytest.raises(NotImplementedError):
            wrapper.raw_results("test")

    def test_results_raises_not_implemented(self) -> None:
        """Stub implementation raises NotImplementedError."""
        wrapper = YouSearchAPIWrapper()
        with pytest.raises(NotImplementedError):
            wrapper.results("test")
