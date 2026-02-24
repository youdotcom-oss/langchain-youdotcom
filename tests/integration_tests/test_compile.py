"""Compile check for integration tests — no API calls made."""

import pytest


@pytest.mark.compile
def test_imports_compile() -> None:
    """Verify integration test modules import without errors."""
    import tests.integration_tests.test_retriever
    import tests.integration_tests.test_tools
    import tests.integration_tests.test_utilities  # noqa: F401
