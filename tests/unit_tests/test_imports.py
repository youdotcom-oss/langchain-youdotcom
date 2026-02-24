"""Test that all expected symbols are importable from the package."""

from langchain_youdotcom import __all__

EXPECTED_EXPORTS = [
    "YouSearchAPIWrapper",
    "YouRetriever",
    "YouSearchTool",
    "YouContentsTool",
]


def test_all_exports_present() -> None:
    """Verify __all__ matches expected exports."""
    assert sorted(__all__) == sorted(EXPECTED_EXPORTS)
