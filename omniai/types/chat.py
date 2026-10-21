"""Type definitions for Chat Completions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message in a conversation thread."""
    role: str = Field(default="user", description="Role of the message author (e.g. user, system, assistant)")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion response."""
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Usage statistics for a completion request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Full chat completion response from the OmniAI gateway."""
    id: Optional[str] = None
    object: str = "chat.completion"
    created: Optional[int] = None
    model: str
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: Optional[ChatCompletionUsage] = None
    result: Optional[Union[str, Dict[str, Any]]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk of a chat completion."""
    id: Optional[str] = None
    object: str = "chat.completion.chunk"
    created: Optional[int] = None
    model: Optional[str] = None
    delta: Optional[str] = None
    finish_reason: Optional[str] = None
