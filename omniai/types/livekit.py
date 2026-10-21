"""Type definitions for LiveKit multimodal audio/video sessions."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class LiveSessionInfo(BaseModel):
    """LiveKit WebRTC multimodal audio/video session credentials and connection info."""
    server_url: str = Field(..., description="WebSocket/WebRTC URL of the LiveKit server")
    room_name: str = Field(..., description="Unique room name for the multimodal session")
    participant_token: str = Field(..., description="JWT token for authenticating to the room")
    participant_identity: str = Field(..., description="Unique identity assigned to the participant")
    model_used: str = Field(default="gemini:gemini-2.5-flash", description="Underlying model configured for voice/video agent")
    voice: str = Field(default="Aoede", description="Voice personality for realtime synthesis")
    video_input: bool = Field(default=False, description="Whether realtime video streaming input is active")
