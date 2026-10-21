"""Type definitions for OmniAI Model catalog and metadata."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata describing an AI model supported by OmniAI."""
    id: str
    object: str = "model"
    created: Optional[int] = None
    created_at: Optional[str] = None
    release_date: Optional[str] = None
    owned_by: Optional[str] = None
    provider: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = True


class ModelList(BaseModel):
    """List of supported AI models."""
    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)
