"""You.com retriever."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_youdotcom._utilities import YouSearchAPIWrapper


class YouRetriever(BaseRetriever, YouSearchAPIWrapper):
    """Retriever that uses the You.com Search API.

    Inherits all configuration fields from
    :class:`~langchain_youdotcom.YouSearchAPIWrapper` (``ydc_api_key``,
    ``k``, ``count``, ``livecrawl``, etc.).

    Example:
        .. code-block:: python

            from langchain_youdotcom import YouRetriever

            retriever = YouRetriever(ydc_api_key="...")
            docs = retriever.invoke("latest AI news")
    """

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
        """
        return self.results(query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        """Async retrieve documents from You.com Search.

        Args:
            query: The search query.
            run_manager: Async callback manager for the retriever run.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of relevant documents.
        """
        return await self.results_async(query)
