from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from typing import Any, Callable, Coroutine, Optional, Set, Tuple, Type, TypeVar
import httpx

logger = logging.getLogger("omniai.retry")

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
        """
        Calculates retry delay in seconds using exponential backoff with full jitter,
        honoring Retry-After header if returned by the upstream server.
        """
        if response is not None:
            retry_after = self.parse_retry_after(response)
            if retry_after is not None:
                # Add small jitter to avoid thundering herd on retry-after window
                return min(self.max_backoff, retry_after + random.uniform(0.05, 0.5))

        base_delay = min(self.max_backoff, self.backoff_factor * (2 ** (attempt - 1)))
        # Full jitter (0.5x to 1.5x)
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
                        f"OmniAI async {operation_name} returned status {resp.status_code}. "
                        f"Retrying (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return resp
            except Exception as exc:
                if self.should_retry(attempt, exception=exc):
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"OmniAI async {operation_name} failed with {exc.__class__.__name__}: {exc}. "
                        f"Retrying (attempt {attempt}/{self.max_retries}) in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise exc
