"""You.com retriever."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class YouRetriever(BaseRetriever):
    """Retriever that uses the You.com Search API.

    Requires a ``YDC_API_KEY`` environment variable.
    """

    ydc_api_key: str = ""
    """You.com API key."""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from You.com Search.

        Args:
            query: The search query.
            run_manager: Callback manager for the retriever run.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of relevant documents.

        Raises:
            NotImplementedError: Stub — will be implemented in DX-335.
        """
        raise NotImplementedError
