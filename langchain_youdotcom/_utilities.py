"""You.com Search API wrapper."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator


class YouSearchAPIWrapper(BaseModel):
    """Wrapper around You.com Search API.

    Requires a YDC_API_KEY environment variable or explicit ``ydc_api_key``.
    """

    ydc_api_key: str = Field(default="", description="You.com API key.")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that the API key is set."""
        values["ydc_api_key"] = values.get("ydc_api_key") or os.environ.get(
            "YDC_API_KEY", ""
        )
        return values

    def raw_results(self, query: str) -> list[dict[str, Any]]:
        """Fetch raw results from the You.com Search API.

        Args:
            query: The search query.

        Returns:
            A list of result dictionaries.

        Raises:
            NotImplementedError: Stub -- will be implemented in DX-335.
        """
        raise NotImplementedError

    def results(self, query: str) -> list[dict[str, Any]]:
        """Fetch parsed results from the You.com Search API.

        Args:
            query: The search query.

        Returns:
            A list of result dictionaries.

        Raises:
            NotImplementedError: Stub -- will be implemented in DX-335.
        """
        raise NotImplementedError
