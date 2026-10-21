"""Constants used throughout the OmniAI Python SDK."""

from typing import Set, Tuple

DEFAULT_BASE_URL: str = "https://omniai.go-agents.app"
DEFAULT_TIMEOUT: float = 60.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: float = 0.5
DEFAULT_MAX_BACKOFF: float = 30.0

DEFAULT_RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}

SUPPORTED_PROVIDERS: Tuple[str, ...] = (
    "openai",
    "gemini",
    "anthropic",
    "vllm",
    "openrouter",
    "ollama",
    "custom",
    "local",
)

ENV_API_KEY_NAMES: Tuple[str, ...] = (
    "OMNIAI_API_KEY",
    "OMNI_API_KEY",
    "X_API_KEY",
)

ENV_BASE_URL_NAMES: Tuple[str, ...] = (
    "OMNIAI_BASE_URL",
    "OMNI_BASE_URL",
    "GATEWAY_URL",
)

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}
