from __future__ import annotations

from typing import Any, Dict, Optional


class OmniaiError(Exception):
    """Base exception for all errors raised by the OmniAI Python SDK."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} message={self.message!r}>"


class OmniaiAuthError(OmniaiError):
    """Raised when authentication or authorization fails (HTTP 401 / 403)."""
    pass


class OmniaiPaymentRequiredError(OmniaiError):
    """Raised when API credit balance is exhausted or payment is required (HTTP 402)."""
    pass


class OmniaiNotFoundError(OmniaiError):
    """Raised when a requested resource, model, or task is not found (HTTP 404)."""
    pass


class OmniaiRateLimitError(OmniaiError):
    """Raised when request rate limits or concurrency quotas are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


class OmniaiTimeoutError(OmniaiError):
    """Raised when a network request, streaming chunk, or task polling times out."""
    pass


class OmniaiConnectionError(OmniaiError):
    """Raised when network connectivity fails (DNS resolution, connection refused, broken pipe)."""
    pass


class OmniaiExecutionError(OmniaiError):
    """Raised when the OmniAI gateway or upstream LLM provider encounters an execution error (HTTP 5xx / provider failure)."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.status_code = status_code


class OmniaiValidationError(OmniaiError):
    """Raised when structured outputs or response_model validation fails against expected schema."""

    def __init__(
        self,
        message: str,
        raw_output: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.raw_output = raw_output


class OmniaiUnsupportedProviderError(OmniaiError):
    """
    Raised when an unsupported provider is requested.
    The OmniAI SDK strictly supports: 'openai', 'gemini', 'anthropic', and 'vllm'.
    """
    pass


# Backward-compatible aliases for legacy imports
OmniError = OmniaiError
OmniAuthError = OmniaiAuthError
OmniExecutionError = OmniaiExecutionError
OmniTimeoutError = OmniaiTimeoutError
OmniRateLimitError = OmniaiRateLimitError
