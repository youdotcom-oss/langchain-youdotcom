"""Test that all expected symbols are importable from the package."""

from langchain_youdotcom import __all__

EXPECTED_EXPORTS = [
    "YouContentsTool",
    "YouRetriever",
    "YouSearchAPIWrapper",
    "YouSearchTool",
]


def test_all_exports_present() -> None:
    """Verify __all__ matches expected exports."""
    assert sorted(__all__) == sorted(EXPECTED_EXPORTS)


def test_all_exports_importable() -> None:
    """Verify each symbol in __all__ is actually importable."""
    import langchain_youdotcom

    for name in EXPECTED_EXPORTS:
        assert hasattr(langchain_youdotcom, name), f"{name} not importable"
