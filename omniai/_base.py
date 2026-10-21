"""Base HTTP clients for the OmniAI Python SDK."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Mapping, Optional, Union
import httpx

from omniai._constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ENV_API_KEY_NAMES,
    ENV_BASE_URL_NAMES,
)
from omniai._version import __version__
from omniai.exceptions import (
    OmniaiAuthError,
    OmniaiConnectionError,
    OmniaiExecutionError,
    OmniaiNotFoundError,
    OmniaiPaymentRequiredError,
    OmniaiRateLimitError,
    OmniaiTimeoutError,
)
import email.utils
import logging
import random
import time
from typing import Callable, Coroutine, Set, Tuple, Type, TypeVar

logger = logging.getLogger("omniai._base")

T = TypeVar("T")

DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

DEFAULT_RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}


class RetryPolicy:
    """
    Enterprise-grade exponential backoff retry policy with full jitter
    and Retry-After header parsing for resilient API interactions.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_backoff: float = 30.0,
        retryable_statuses: Optional[Set[int]] = None,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max(0, max_retries)
        self.backoff_factor = max(0.01, backoff_factor)
        self.max_backoff = max_backoff
        self.retryable_statuses = retryable_statuses or DEFAULT_RETRYABLE_STATUSES
        self.retryable_exceptions = retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS

    def parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Parses Retry-After header as either seconds or HTTP-date."""
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            try:
                date_tuple = email.utils.parsedate_tz(raw)
                if date_tuple:
                    timestamp = email.utils.mktime_tz(date_tuple)
                    delta = timestamp - time.time()
                    return max(0.0, delta)
            except Exception:
                pass
        return None

    def calculate_delay(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None,
    ) -> float:
        """Calculates retry delay in seconds using exponential backoff with full jitter."""
        if response is not None:
            retry_after = self.parse_retry_after(response)
            if retry_after is not None:
                return min(self.max_backoff, retry_after + random.uniform(0.05, 0.5))

        base_delay = min(self.max_backoff, self.backoff_factor * (2 ** (attempt - 1)))
        jittered = base_delay * random.uniform(0.5, 1.5)
        return min(self.max_backoff, max(0.05, jittered))

    def should_retry(
        self,
        attempt: int,
        response: Optional[httpx.Response] = None,
        exception: Optional[Exception] = None,
    ) -> bool:
        """Determines if a failed request attempt should be retried."""
        if attempt >= self.max_retries:
            return False

        if exception is not None:
            return isinstance(exception, self.retryable_exceptions)

        if response is not None:
            return response.status_code in self.retryable_statuses

        return False

    def execute_sync(
        self,
        func: Callable[[], httpx.Response],
        operation_name: str = "request",
    ) -> httpx.Response:
        """Executes a synchronous HTTP call with retry logic."""
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = func()
                if self.should_retry(attempt, response=resp):
                    delay = self.calculate_delay(attempt, response=resp)
                    logger.warning(
                        f"OmniAI {operation_name} returned status {resp.status_code}. "
                        f"Retrying (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                return resp
            except Exception as exc:
                if self.should_retry(attempt, exception=exc):
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"OmniAI {operation_name} failed with {exc.__class__.__name__}: {exc}. "
                        f"Retrying (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                raise exc

    async def execute_async(
        self,
        func: Callable[[], Coroutine[Any, Any, httpx.Response]],
        operation_name: str = "request",
    ) -> httpx.Response:
        """Executes an asynchronous HTTP call with retry logic."""
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await func()
                if self.should_retry(attempt, response=resp):
                    delay = self.calculate_delay(attempt, response=resp)
                    logger.warning(
                        f"OmniAI {operation_name} returned status {resp.status_code}. "
                        f"Retrying async (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return resp
            except Exception as exc:
                if self.should_retry(attempt, exception=exc):
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"OmniAI {operation_name} failed with {exc.__class__.__name__}: {exc}. "
                        f"Retrying async (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise exc


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Discovers API key from parameters or environment variables."""
    if api_key:
        return api_key.strip()
    for env_var in ENV_API_KEY_NAMES:
        val = os.getenv(env_var)
        if val:
            return val.strip()
    return ""


def _resolve_base_url(base_url: Optional[str]) -> str:
    """Discovers base URL from parameters or environment variables, falling back to default."""
    if base_url:
        return base_url.strip().rstrip("/")
    for env_var in ENV_BASE_URL_NAMES:
        val = os.getenv(env_var)
        if val:
            return val.strip().rstrip("/")
    return DEFAULT_BASE_URL.rstrip("/")


def _mask_key(key: str) -> str:
    """Masks secret API key for safe representation."""
    if not key:
        return "None"
    if len(key) <= 12:
        return "***"
    return f"{key[:7]}...{key[-4:]}"


class BaseClient:
    """Synchronous base HTTP client handling authentication, sessions, and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.api_key: str = _resolve_api_key(api_key)
        self.base_url: str = _resolve_base_url(base_url)
        self.timeout: float = timeout
        self.retry_policy = RetryPolicy(max_retries=max_retries)

        default_headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": f"OmniAI-Python-SDK/{__version__}",
        }
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
            default_headers["X-API-KEY"] = self.api_key

        if headers:
            default_headers.update(headers)

        self._headers = default_headers
        self._custom_client = http_client is not None
        self._http: httpx.Client = http_client or httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self.timeout,
            follow_redirects=False,
        )
        self._client = self._http

    @property
    def client(self) -> httpx.Client:
        return self._http

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Parses error responses and maps them to appropriate typed exceptions."""
        status = response.status_code
        try:
            data = response.json()
            message = (
                data.get("error", {}).get("message")
                or data.get("detail")
                or data.get("message")
                or response.text
            )
        except Exception:
            message = response.text or f"HTTP {status}"

        if status in (401, 403):
            raise OmniaiAuthError(f"Authentication failed ({status}): {message}")
        elif status == 402:
            raise OmniaiPaymentRequiredError(f"Payment required ({status}): {message}")
        elif status == 404:
            raise OmniaiNotFoundError(f"Resource not found ({status}): {message}")
        elif status == 429:
            retry_after_raw = response.headers.get("Retry-After")
            retry_after = float(retry_after_raw) if retry_after_raw and retry_after_raw.isdigit() else None
            raise OmniaiRateLimitError(f"Rate limit exceeded: {message}", retry_after=retry_after)
        elif status >= 500:
            raise OmniaiExecutionError(f"Server error ({status}): {message}", status_code=status)
        else:
            raise OmniaiExecutionError(f"Request failed ({status}): {message}", status_code=status)

    def close(self) -> None:
        if not self._custom_client and hasattr(self, "_http") and not self._http.is_closed:
            self._http.close()

    def __enter__(self) -> BaseClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        masked = _mask_key(self.api_key)
        return f"<{self.__class__.__name__} base_url={self.base_url!r} api_key={masked!r}>"


class AsyncBaseClient:
    """Asynchronous base HTTP client handling authentication, sessions, and retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key: str = _resolve_api_key(api_key)
        self.base_url: str = _resolve_base_url(base_url)
        self.timeout: float = timeout
        self.retry_policy = RetryPolicy(max_retries=max_retries)

        default_headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": f"OmniAI-Python-SDK/{__version__}",
        }
        if self.api_key:
            default_headers["Authorization"] = f"Bearer {self.api_key}"
            default_headers["X-API-KEY"] = self.api_key

        if headers:
            default_headers.update(headers)

        self._headers = default_headers
        self._custom_client = http_client is not None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self.timeout,
            follow_redirects=False,
        )
        self._client = self._http

    @property
    def client(self) -> httpx.AsyncClient:
        return self._http

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Parses error responses and maps them to appropriate typed exceptions."""
        status = response.status_code
        try:
            data = response.json()
            message = (
                data.get("error", {}).get("message")
                or data.get("detail")
                or data.get("message")
                or response.text
            )
        except Exception:
            message = response.text or f"HTTP {status}"

        if status in (401, 403):
            raise OmniaiAuthError(f"Authentication failed ({status}): {message}")
        elif status == 402:
            raise OmniaiPaymentRequiredError(f"Payment required ({status}): {message}")
        elif status == 404:
            raise OmniaiNotFoundError(f"Resource not found ({status}): {message}")
        elif status == 429:
            retry_after_raw = response.headers.get("Retry-After")
            retry_after = float(retry_after_raw) if retry_after_raw and retry_after_raw.isdigit() else None
            raise OmniaiRateLimitError(f"Rate limit exceeded: {message}", retry_after=retry_after)
        elif status >= 500:
            raise OmniaiExecutionError(f"Server error ({status}): {message}", status_code=status)
        else:
            raise OmniaiExecutionError(f"Request failed ({status}): {message}", status_code=status)

    async def aclose(self) -> None:
        if not self._custom_client and hasattr(self, "_http") and not self._http.is_closed:
            await self._http.aclose()

    async def close(self) -> None:
        """Alias for aclose() for interface consistency."""
        await self.aclose()

    async def __aenter__(self) -> AsyncBaseClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        masked = _mask_key(self.api_key)
        return f"<{self.__class__.__name__} base_url={self.base_url!r} api_key={masked!r}>"
