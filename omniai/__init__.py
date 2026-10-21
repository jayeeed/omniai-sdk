"""
OmniAI Python Client SDK (`omniai`)

Official, production-grade Python SDK for the OmniAI AI Gateway and Dynamic Multi-Provider Router.

Supported Providers:
  - OpenAI (`openai:...`)
  - Google Gemini (`gemini:...`)
  - Anthropic Claude (`anthropic:...`)
  - vLLM Self-Hosted Inference (`vllm:...`)

Quickstart:
    >>> from omniai import OmniaiClient
    >>> client = OmniaiClient(api_key="your-api-key")
    >>> response = client.chat(model="openai:gpt-4o", prompt="Hello world!")
    >>> print(response)
"""

from omniai.client import (
    AsyncOmniaiClient,
    OmniaiClient,
    AsyncOmniAIClient,
    OmniAIClient,
    AsyncOmniClient,
    OmniClient,
)
from omniai.exceptions import (
    OmniAuthError,
    OmniError,
    OmniExecutionError,
    OmniRateLimitError,
    OmniTimeoutError,
    OmniaiAuthError,
    OmniaiConnectionError,
    OmniaiError,
    OmniaiExecutionError,
    OmniaiNotFoundError,
    OmniaiPaymentRequiredError,
    OmniaiRateLimitError,
    OmniaiTimeoutError,
    OmniaiUnsupportedProviderError,
    OmniaiValidationError,
)
from omniai.live import create_live_session, generate_livekit_token
from omniai.models import (
    AsyncTaskJob,
    LiveSessionInfo,
    TaskJob,
    TaskStatus,
)
from omniai.retry import RetryPolicy
from omniai.router import RouterEngine, SUPPORTED_PROVIDERS

__version__ = "1.0.0"
__author__ = "OmniAI Team"

__all__ = [
    # Primary Client Interfaces
    "OmniaiClient",
    "AsyncOmniaiClient",
    # Convenient Casing & Backward-Compatible Aliases
    "OmniAIClient",
    "AsyncOmniAIClient",
    "OmniClient",
    "AsyncOmniClient",
    # Core Engine & Retries
    "RouterEngine",
    "RetryPolicy",
    "SUPPORTED_PROVIDERS",
    # LiveKit Multimodal Realtime
    "create_live_session",
    "generate_livekit_token",
    "LiveSessionInfo",
    # Background Tasks
    "TaskJob",
    "AsyncTaskJob",
    "TaskStatus",
    # Typed Exceptions
    "OmniaiError",
    "OmniaiAuthError",
    "OmniaiPaymentRequiredError",
    "OmniaiNotFoundError",
    "OmniaiRateLimitError",
    "OmniaiExecutionError",
    "OmniaiTimeoutError",
    "OmniaiConnectionError",
    "OmniaiValidationError",
    "OmniaiUnsupportedProviderError",
    # Backward-Compatible Exception Aliases
    "OmniError",
    "OmniAuthError",
    "OmniExecutionError",
    "OmniTimeoutError",
    "OmniRateLimitError",
]
