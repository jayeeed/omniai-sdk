"""Models catalog resource for OmniAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from omniai.types.models import ModelInfo, ModelList

if TYPE_CHECKING:
    from omniai._base import BaseClient, AsyncBaseClient


class ModelsResource:
    """Synchronous models resource."""

    def __init__(self, client: BaseClient):
        self._client = client

    def list(self, timeout: Optional[float] = None) -> ModelList:
        """Fetches list of all AI models supported by the OmniAI platform."""
        response = self._client.client.get(
            "/v1/models",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return ModelList.model_validate(response.json())

    def retrieve(self, model_id: str, timeout: Optional[float] = None) -> ModelInfo:
        """Retrieves metadata for a specific model."""
        response = self._client.client.get(
            f"/v1/models/{model_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return ModelInfo.model_validate(response.json())


class AsyncModelsResource:
    """Asynchronous models resource."""

    def __init__(self, client: AsyncBaseClient):
        self._client = client

    async def list(self, timeout: Optional[float] = None) -> ModelList:
        """Fetches list of all AI models asynchronously."""
        response = await self._client.client.get(
            "/v1/models",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return ModelList.model_validate(response.json())

    async def retrieve(self, model_id: str, timeout: Optional[float] = None) -> ModelInfo:
        """Retrieves metadata for a specific model asynchronously."""
        response = await self._client.client.get(
            f"/v1/models/{model_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return ModelInfo.model_validate(response.json())
