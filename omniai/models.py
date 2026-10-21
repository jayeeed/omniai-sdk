from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union
from pydantic import BaseModel

from omniai.exceptions import (
    OmniaiError,
    OmniaiExecutionError,
    OmniaiTimeoutError,
    OmniaiValidationError,
)

logger = logging.getLogger("omniai.models")

T = TypeVar("T")


def _clean_json_text(text: str) -> str:
    """Strips markdown code blocks (e.g. ```json ... ```) from model output."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


class LiveSessionInfo(BaseModel):
    """LiveKit WebRTC multimodal audio/video session credentials and connection info."""
    server_url: str
    room_name: str
    participant_token: str
    participant_identity: str
    model_used: str
    voice: str
    video_input: bool = False


class TaskStatus(BaseModel):
    """Current status and progress of a background Celery task."""
    task_id: str
    status: str
    ready: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskJob(Generic[T]):
    """
    Synchronous handle to a background task running on the OmniAI Celery queue.
    Features fault-tolerant polling that survives transient network dropouts.
    """

    def __init__(
        self,
        task_id: str,
        status_checker: Callable[[str], TaskStatus],
        canceller: Callable[[str], dict],
        response_model: Optional[Type[T]] = None,
    ):
        self.task_id = task_id
        self._status_checker = status_checker
        self._canceller = canceller
        self._response_model = response_model
        self._last_status: Optional[TaskStatus] = None

    def get_status(self) -> TaskStatus:
        """Fetches the latest status of the task."""
        self._last_status = self._status_checker(self.task_id)
        return self._last_status

    def wait(
        self,
        timeout: float = 120.0,
        interval: float = 1.5,
        max_consecutive_poll_errors: int = 3,
    ) -> T:
        """
        Polls until the task completes and returns the result, parsed into response_model if specified.
        Tolerates transient polling connection failures up to max_consecutive_poll_errors.
        """
        start_time = time.time()
        consecutive_errors = 0

        while time.time() - start_time < timeout:
            try:
                status = self.get_status()
                consecutive_errors = 0  # Reset error count on successful check
            except Exception as poll_err:
                consecutive_errors += 1
                logger.warning(
                    f"Transient error polling task {self.task_id} ({consecutive_errors}/{max_consecutive_poll_errors}): {poll_err}"
                )
                if consecutive_errors >= max_consecutive_poll_errors:
                    raise OmniaiExecutionError(
                        f"Failed polling task {self.task_id} after {consecutive_errors} consecutive errors: {poll_err}"
                    ) from poll_err
                time.sleep(interval)
                continue

            if status.ready:
                if status.status == "SUCCESS":
                    raw_result = status.result
                    if self._response_model and issubclass(self._response_model, BaseModel):
                        if isinstance(raw_result, dict):
                            try:
                                return self._response_model.model_validate(raw_result)
                            except Exception as val_err:
                                raise OmniaiValidationError(
                                    f"Task result validation failed: {val_err}",
                                    raw_output=str(raw_result),
                                    details={"error": str(val_err)},
                                ) from val_err
                        elif isinstance(raw_result, str):
                            cleaned = _clean_json_text(raw_result)
                            try:
                                return self._response_model.model_validate_json(cleaned)
                            except Exception as val_err:
                                raise OmniaiValidationError(
                                    f"Task JSON string validation failed: {val_err}",
                                    raw_output=raw_result,
                                    details={"error": str(val_err)},
                                ) from val_err
                    return raw_result
                else:
                    raise OmniaiExecutionError(
                        f"Task {self.task_id} failed with status {status.status}: {status.error or 'Unknown error'}",
                        details={"status": status.status, "error": status.error},
                    )

            time.sleep(interval)

        raise OmniaiTimeoutError(
            f"Task {self.task_id} timed out after {timeout} seconds (last status: {self._last_status.status if self._last_status else 'UNKNOWN'})"
        )

    def cancel(self) -> dict:
        """Cancels/revokes the background task."""
        return self._canceller(self.task_id)


class AsyncTaskJob(Generic[T]):
    """
    Asynchronous handle to a background task running on the OmniAI Celery queue.
    Features fault-tolerant polling that survives transient network dropouts.
    """

    def __init__(
        self,
        task_id: str,
        status_checker: Callable[[str], Any],
        canceller: Callable[[str], Any],
        response_model: Optional[Type[T]] = None,
    ):
        self.task_id = task_id
        self._status_checker = status_checker
        self._canceller = canceller
        self._response_model = response_model
        self._last_status: Optional[TaskStatus] = None

    async def get_status(self) -> TaskStatus:
        """Fetches the latest status of the task asynchronously."""
        self._last_status = await self._status_checker(self.task_id)
        return self._last_status

    async def wait(
        self,
        timeout: float = 120.0,
        interval: float = 1.5,
        max_consecutive_poll_errors: int = 3,
    ) -> T:
        """
        Polls asynchronously until the task completes and returns the result.
        Tolerates transient polling connection failures up to max_consecutive_poll_errors.
        """
        start_time = time.time()
        consecutive_errors = 0

        while time.time() - start_time < timeout:
            try:
                status = await self.get_status()
                consecutive_errors = 0
            except Exception as poll_err:
                consecutive_errors += 1
                logger.warning(
                    f"Transient error polling async task {self.task_id} ({consecutive_errors}/{max_consecutive_poll_errors}): {poll_err}"
                )
                if consecutive_errors >= max_consecutive_poll_errors:
                    raise OmniaiExecutionError(
                        f"Failed polling async task {self.task_id} after {consecutive_errors} consecutive errors: {poll_err}"
                    ) from poll_err
                await asyncio.sleep(interval)
                continue

            if status.ready:
                if status.status == "SUCCESS":
                    raw_result = status.result
                    if self._response_model and issubclass(self._response_model, BaseModel):
                        if isinstance(raw_result, dict):
                            try:
                                return self._response_model.model_validate(raw_result)
                            except Exception as val_err:
                                raise OmniaiValidationError(
                                    f"Async task result validation failed: {val_err}",
                                    raw_output=str(raw_result),
                                    details={"error": str(val_err)},
                                ) from val_err
                        elif isinstance(raw_result, str):
                            cleaned = _clean_json_text(raw_result)
                            try:
                                return self._response_model.model_validate_json(cleaned)
                            except Exception as val_err:
                                raise OmniaiValidationError(
                                    f"Async task JSON string validation failed: {val_err}",
                                    raw_output=raw_result,
                                    details={"error": str(val_err)},
                                ) from val_err
                    return raw_result
                else:
                    raise OmniaiExecutionError(
                        f"Task {self.task_id} failed with status {status.status}: {status.error or 'Unknown error'}",
                        details={"status": status.status, "error": status.error},
                    )

            await asyncio.sleep(interval)

        raise OmniaiTimeoutError(
            f"Async task {self.task_id} timed out after {timeout} seconds (last status: {self._last_status.status if self._last_status else 'UNKNOWN'})"
        )

    async def cancel(self) -> dict:
        """Cancels/revokes the task asynchronously."""
        return await self._canceller(self.task_id)
