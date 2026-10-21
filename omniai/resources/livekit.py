"""LiveKit multimodal audio/video sessions resource for OmniAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from omniai.types.livekit import LiveSessionInfo

if TYPE_CHECKING:
    from omniai._base import BaseClient, AsyncBaseClient


class LiveKitResource:
    """Synchronous LiveKit multimodal resource."""

    def __init__(self, client: BaseClient):
        self._client = client

    def create_session(
        self,
        *,
        model: str = "gemini:gemini-2.5-flash",
        voice: str = "Aoede",
        video_input: bool = False,
        timeout: Optional[float] = None,
    ) -> LiveSessionInfo:
        """Requests room and participant credentials for a LiveKit WebRTC session."""
        response = self._client.client.post(
            "/v1/live/token",
            json={
                "model": model,
                "voice": voice,
                "video_input": video_input,
            },
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)

        data = response.json()
        return LiveSessionInfo.model_validate(data)


class AsyncLiveKitResource:
    """Asynchronous LiveKit multimodal resource."""

    def __init__(self, client: AsyncBaseClient):
        self._client = client

    async def create_session(
        self,
        *,
        model: str = "gemini:gemini-2.5-flash",
        voice: str = "Aoede",
        video_input: bool = False,
        timeout: Optional[float] = None,
    ) -> LiveSessionInfo:
        """Requests room and participant credentials for a LiveKit WebRTC session asynchronously."""
        response = await self._client.client.post(
            "/v1/live/token",
            json={
                "model": model,
                "voice": voice,
                "video_input": video_input,
            },
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)

        data = response.json()
        return LiveSessionInfo.model_validate(data)
