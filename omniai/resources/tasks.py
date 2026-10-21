"""Background tasks resource for OmniAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel

from omniai.types.tasks import TaskJob, AsyncTaskJob, TaskStatus

if TYPE_CHECKING:
    from omniai._base import BaseClient, AsyncBaseClient

T = TypeVar("T")


class TasksResource:
    """Synchronous background tasks resource."""

    def __init__(self, client: BaseClient):
        self._client = client

    def submit(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> TaskJob[T]:
        """Submits an asynchronous task to Celery queue and returns a TaskJob handle."""
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if output_format:
            payload["output_format"] = output_format
        elif response_model and issubclass(response_model, BaseModel):
            payload["output_format"] = response_model.model_json_schema()

        response = self._client.client.post(
            "/v1/chat/completions/async",
            json=payload,
            timeout=timeout or self._client.timeout,
        )
        if response.status_code not in (200, 202):
            self._client._handle_error_response(response)

        data = response.json()
        task_id = data.get("task_id")

        return TaskJob(
            task_id=task_id,
            status_checker=self.get_status,
            canceller=self.cancel,
            response_model=response_model,
        )

    def get_status(self, task_id: str, timeout: Optional[float] = None) -> TaskStatus:
        """Checks status of background task."""
        response = self._client.client.get(
            f"/v1/chat/completions/tasks/{task_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return TaskStatus.model_validate(response.json())

    def cancel(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Cancels a background task."""
        response = self._client.client.delete(
            f"/v1/chat/completions/tasks/{task_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return response.json()


class AsyncTasksResource:
    """Asynchronous background tasks resource."""

    def __init__(self, client: AsyncBaseClient):
        self._client = client

    async def submit(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Any]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncTaskJob[T]:
        """Submits an asynchronous task to Celery queue and returns an AsyncTaskJob handle."""
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if output_format:
            payload["output_format"] = output_format
        elif response_model and issubclass(response_model, BaseModel):
            payload["output_format"] = response_model.model_json_schema()

        response = await self._client.client.post(
            "/v1/chat/completions/async",
            json=payload,
            timeout=timeout or self._client.timeout,
        )
        if response.status_code not in (200, 202):
            self._client._handle_error_response(response)

        data = response.json()
        task_id = data.get("task_id")

        return AsyncTaskJob(
            task_id=task_id,
            status_checker=self.get_status,
            canceller=self.cancel,
            response_model=response_model,
        )

    async def get_status(self, task_id: str, timeout: Optional[float] = None) -> TaskStatus:
        """Checks status of background task asynchronously."""
        response = await self._client.client.get(
            f"/v1/chat/completions/tasks/{task_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return TaskStatus.model_validate(response.json())

    async def cancel(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Cancels a background task asynchronously."""
        response = await self._client.client.delete(
            f"/v1/chat/completions/tasks/{task_id}",
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)
        return response.json()
