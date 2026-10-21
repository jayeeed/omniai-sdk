"""OmniAI Python Client SDK.

Provides synchronous and asynchronous clients with nested resource namespaces
and top-level convenience methods for chat, streaming, background tasks, and LiveKit sessions.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)
import httpx

from omniai._base import AsyncBaseClient, BaseClient
from omniai._constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    SUPPORTED_PROVIDERS,
)
from omniai.resources.chat import AsyncChatResource, ChatResource
from omniai.resources.livekit import AsyncLiveKitResource, LiveKitResource
from omniai.resources.models import AsyncModelsResource, ModelsResource
from omniai.resources.tasks import AsyncTasksResource, TasksResource
from omniai.types.livekit import LiveSessionInfo
from omniai.types.models import ModelInfo, ModelList
from omniai.types.tasks import AsyncTaskJob, TaskJob

logger = logging.getLogger("omniai.client")

T = TypeVar("T")


class OmniaiClient(BaseClient):
    """
    Synchronous client for OmniAI.

    Usage:
        >>> from omniai import OmniaiClient
        >>> client = OmniaiClient()
        >>> response = client.chat(model="gemini:gemini-2.5-flash", prompt="Hello!")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            http_client=http_client,
        )
        self.chat = ChatResource(self)
        self.models = ModelsResource(self)
        self.tasks = TasksResource(self)
        self.livekit = LiveKitResource(self)

    def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Union[str, T]:
        """Convenience method to execute a chat completion."""
        return self.chat.completions.create(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Convenience method to stream a chat completion chunk-by-chunk."""
        return self.chat.completions.stream(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def submit_task(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> TaskJob[T]:
        """Submits a background task to the OmniAI Celery queue."""
        return self.tasks.submit(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def create_live_session(
        self,
        model: str = "gemini:gemini-2.5-flash",
        voice: str = "Aoede",
        video_input: bool = False,
        timeout: Optional[float] = None,
    ) -> LiveSessionInfo:
        """Requests room and participant credentials for a LiveKit WebRTC session."""
        return self.livekit.create_session(
            model=model,
            voice=voice,
            video_input=video_input,
            timeout=timeout,
        )

    def list_models(self, timeout: Optional[float] = None) -> ModelList:
        """Lists available AI models."""
        return self.models.list(timeout=timeout)


class AsyncOmniaiClient(AsyncBaseClient):
    """
    Asynchronous client for OmniAI.

    Usage:
        >>> import asyncio
        >>> from omniai import AsyncOmniaiClient
        >>> async def main():
        ...     async with AsyncOmniaiClient() as client:
        ...         res = await client.chat(model="gemini:gemini-2.5-flash", prompt="Hello!")
        ...         print(res)
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            http_client=http_client,
        )
        self.chat = AsyncChatResource(self)
        self.models = AsyncModelsResource(self)
        self.tasks = AsyncTasksResource(self)
        self.livekit = AsyncLiveKitResource(self)

    async def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Union[str, T]:
        """Convenience method to execute an async chat completion."""
        return await self.chat.completions.create(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Convenience method to stream an async chat completion chunk-by-chunk."""
        async for chunk in self.chat.completions.stream(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        ):
            yield chunk

    async def submit_task(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncTaskJob[T]:
        """Submits an async background task to the OmniAI Celery queue."""
        return await self.tasks.submit(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def create_live_session(
        self,
        model: str = "gemini:gemini-2.5-flash",
        voice: str = "Aoede",
        video_input: bool = False,
        timeout: Optional[float] = None,
    ) -> LiveSessionInfo:
        """Requests room and participant credentials for a LiveKit WebRTC session asynchronously."""
        return await self.livekit.create_session(
            model=model,
            voice=voice,
            video_input=video_input,
            timeout=timeout,
        )

    async def list_models(self, timeout: Optional[float] = None) -> ModelList:
        """Lists available AI models asynchronously."""
        return await self.models.list(timeout=timeout)


# Aliases for brand and convenience consistency
OmniAIClient = OmniaiClient
AsyncOmniAIClient = AsyncOmniaiClient
OmniClient = OmniaiClient
AsyncOmniClient = AsyncOmniaiClient

__all__ = [
    "OmniaiClient",
    "AsyncOmniaiClient",
    "OmniAIClient",
    "AsyncOmniAIClient",
    "OmniClient",
    "AsyncOmniClient",
]
