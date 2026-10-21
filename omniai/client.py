from __future__ import annotations

import json
import logging
import os
import re
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
import httpx
from pydantic import BaseModel

from omniai.exceptions import (
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
from omniai.models import AsyncTaskJob, LiveSessionInfo, TaskJob, TaskStatus
from omniai.retry import RetryPolicy
from omniai.router import SUPPORTED_PROVIDERS, RouterEngine

logger = logging.getLogger("omniai.client")

T = TypeVar("T")

DEFAULT_BASE_URL = "https://omniai.go-agents.app"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3


def _clean_json_str(raw: str) -> str:
    """Strips markdown code fences (e.g. ```json ... ```) from LLM output."""
    stripped = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def _auto_load_env():
    """Attempts to auto-load .env from current directory or parents without third-party dependencies."""
    try:
        cur_dir = os.path.abspath(os.getcwd())
        for _ in range(5):
            env_path = os.path.join(cur_dir, ".env")
            if os.path.isfile(env_path):
                with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            parent = os.path.dirname(cur_dir)
            if parent == cur_dir:
                break
            cur_dir = parent
    except Exception:
        pass


def _extract_tools_and_format(
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
    output_format: Optional[Union[str, Dict[str, Any]]] = None,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[Union[str, Dict[str, Any]]]]:
    """Converts Pydantic classes and tool schemas into payload-compatible JSON schemas."""
    resolved_tools: Optional[List[Dict[str, Any]]] = None
    resolved_format = output_format

    if response_model is not None:
        schema = response_model.model_json_schema()
        resolved_tools = [schema]
    elif tools is not None:
        resolved_tools = []
        for t in tools:
            if isinstance(t, type) and issubclass(t, BaseModel):
                resolved_tools.append(t.model_json_schema())
            elif isinstance(t, dict):
                resolved_tools.append(t)

    return resolved_tools, resolved_format


def _build_payload(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    system_instruction: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
    temperature: Optional[float] = None,
    thinking: Optional[int] = None,
    base_url_override: Optional[str] = None,
    output_format: Optional[Union[str, Dict[str, Any]]] = None,
    stream: bool = False,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_tools, resolved_format = _extract_tools_and_format(response_model, tools, output_format)
    payload: Dict[str, Any] = {
        "model": model,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if messages is not None:
        payload["messages"] = messages
    if system_instruction is not None:
        payload["system_instruction"] = system_instruction
    if temperature is not None:
        payload["temperature"] = temperature
    if thinking is not None:
        payload["thinking"] = thinking
    if base_url_override is not None:
        payload["base_url"] = base_url_override
    if resolved_format is not None:
        payload["output_format"] = resolved_format
    if resolved_tools is not None:
        payload["tools"] = resolved_tools
    if stream:
        payload["stream"] = True
    if webhook_url:
        payload["webhook_url"] = webhook_url
    return payload


def _handle_response_error(response: httpx.Response) -> None:
    """Translates HTTP error status codes into strongly-typed SDK exceptions."""
    if response.status_code < 400:
        return

    status = response.status_code
    try:
        data = response.json()
        message = data.get("detail", response.text)
        details = data if isinstance(data, dict) else {}
    except Exception:
        message = response.text or f"HTTP error {status}"
        details = {}

    if status in (401, 403):
        raise OmniaiAuthError(
            f"Authentication failed ({status}): {message}",
            details=details,
        )
    elif status == 402:
        raise OmniaiPaymentRequiredError(
            f"Payment required or credit balance exhausted ({status}): {message}",
            details=details,
        )
    elif status == 404:
        raise OmniaiNotFoundError(
            f"Resource not found ({status}): {message}",
            details=details,
        )
    elif status == 429:
        retry_after = RetryPolicy().parse_retry_after(response)
        raise OmniaiRateLimitError(
            f"Rate limit exceeded ({status}): {message}",
            retry_after=retry_after,
            details=details,
        )
    elif status >= 500:
        raise OmniaiExecutionError(
            f"OmniAI server/provider execution error ({status}): {message}",
            status_code=status,
            details=details,
        )
    else:
        raise OmniaiError(
            f"OmniAI request error ({status}): {message}",
            details=details,
        )


def _validate_model_provider(model: str) -> None:
    """
    Validates model identifier against permitted providers:
    openai, gemini, anthropic, vllm, openrouter, ollama, custom, local.
    """
    if ":" in model:
        prefix = model.split(":", 1)[0].strip().lower()
        if prefix in SUPPORTED_PROVIDERS:
            return
    if "/" in model:
        prefix = model.split("/", 1)[0].strip().lower()
        if prefix in SUPPORTED_PROVIDERS:
            return

    lower = model.lower()
    if lower.startswith(("gemini", "claude", "gpt-", "o1", "o3", "text-", "dall-e")):
        return

    if ":" in model:
        provider = model.split(":", 1)[0].strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise OmniaiUnsupportedProviderError(
                f"Unsupported model provider '{provider}'. "
                f"The OmniAI SDK strictly supports: {', '.join(SUPPORTED_PROVIDERS)}."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Synchronous Client: OmniaiClient
# ──────────────────────────────────────────────────────────────────────────────

class OmniaiClient:
    """
    Official Synchronous Python Client for the OmniAI Gateway.
    Compatible with OpenAI, Google Gemini, Anthropic Claude, and vLLM models.
    Features automatic retries with exponential backoff & jitter for high fault tolerance.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = 0.5,
    ):
        _auto_load_env()
        self.base_url = (
            base_url
            or os.getenv("OMNI_BASE_URL")
            or os.getenv("OMNIAI_BASE_URL")
            or os.getenv("GATEWAY_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("OMNI_API_KEY")
            or os.getenv("OMNIAI_API_KEY")
            or os.getenv("X_API_KEY")
            or ""
        )
        self.timeout = timeout
        self.retry_policy = RetryPolicy(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

        self._headers: Dict[str, str] = {
            "User-Agent": "Omniai-Python-SDK/1.0.0",
        }
        if self.api_key:
            self._headers["X-API-KEY"] = self.api_key
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        self._http = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0),
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        masked_key = (
            f"{self.api_key[:7]}...{self.api_key[-4:]}"
            if len(self.api_key) > 11
            else ("***" if self.api_key else "None")
        )
        return f"<OmniaiClient base_url={self.base_url!r} api_key={masked_key!r}>"

    def close(self) -> None:
        """Closes the underlying HTTP client and releases connection pools."""
        self._http.close()

    def __enter__(self) -> OmniaiClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> Union[str, Dict[str, Any], T]:
        """
        Executes a synchronous completion at /v1/chat/completions with automatic retries.
        Supports models from: openai, gemini, anthropic, and vllm.
        """
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            output_format=output_format,
            stream=False,
        )

        effective_timeout = timeout or self.timeout

        def _do_request() -> httpx.Response:
            return self._http.post(
                "/v1/chat/completions",
                json=payload,
                timeout=effective_timeout,
            )

        try:
            resp = self.retry_policy.execute_sync(_do_request, operation_name=f"chat[{model}]")
            _handle_response_error(resp)
            data = resp.json()
            raw_result = data.get("result")

            # Structured Output Validation
            if response_model is not None and isinstance(response_model, type) and issubclass(response_model, BaseModel):
                if isinstance(raw_result, dict):
                    try:
                        return response_model.model_validate(raw_result)
                    except Exception as err:
                        raise OmniaiValidationError(
                            f"Failed to validate response into {response_model.__name__}: {err}",
                            raw_output=str(raw_result),
                            details={"error": str(err)},
                        ) from err
                elif isinstance(raw_result, str):
                    cleaned = _clean_json_str(raw_result)
                    try:
                        return response_model.model_validate_json(cleaned)
                    except Exception as err:
                        raise OmniaiValidationError(
                            f"Failed to validate JSON string response into {response_model.__name__}: {err}",
                            raw_output=raw_result,
                            details={"error": str(err)},
                        ) from err

            return raw_result

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OmniaiConnectionError(f"Connection to OmniAI gateway failed: {exc}") from exc
        except httpx.ReadTimeout as exc:
            raise OmniaiTimeoutError(f"OmniAI chat request timed out after {effective_timeout}s: {exc}") from exc

    def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Streams real-time token chunks from /v1/chat/completions via Server-Sent Events (SSE).
        Tolerant to keep-alive comments, carriage returns, and heartbeat frames.
        """
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            stream=True,
        )

        effective_timeout = timeout or self.timeout

        try:
            with self._http.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=effective_timeout,
            ) as resp:
                _handle_response_error(resp)
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        # Keep-alive heartbeat or comment frame
                        continue
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(raw_data)
                            content = parsed.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            if raw_data:
                                yield raw_data
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OmniaiConnectionError(f"Connection failed during streaming: {exc}") from exc
        except httpx.ReadTimeout as exc:
            raise OmniaiTimeoutError(f"Streaming token stream timed out: {exc}") from exc

    def submit_task(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        webhook_url: Optional[str] = None,
    ) -> TaskJob[T]:
        """Queues an LLM completion job to the OmniAI Celery background workers."""
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            output_format=output_format,
            webhook_url=webhook_url,
        )

        def _do_submit() -> httpx.Response:
            return self._http.post("/v1/chat/completions/async", json=payload)

        resp = self.retry_policy.execute_sync(_do_submit, operation_name="submit_task")
        _handle_response_error(resp)
        task_id = resp.json()["task_id"]

        return TaskJob(
            task_id=task_id,
            status_checker=self.get_task_status,
            canceller=self.cancel_task,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
        )

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Fetches status and result of a background Celery task."""
        def _do_status() -> httpx.Response:
            return self._http.get(f"/v1/chat/completions/tasks/{task_id}")

        resp = self.retry_policy.execute_sync(_do_status, operation_name=f"get_task_status[{task_id}]")
        _handle_response_error(resp)
        return TaskStatus.model_validate(resp.json())

    def cancel_task(self, task_id: str) -> dict:
        """Cancels/revokes an active background task."""
        def _do_cancel() -> httpx.Response:
            return self._http.post(f"/v1/chat/completions/tasks/{task_id}/cancel")

        resp = self.retry_policy.execute_sync(_do_cancel, operation_name=f"cancel_task[{task_id}]")
        _handle_response_error(resp)
        return resp.json()

    def create_live_session(
        self,
        model: str = "gemini:gemini-3.1-flash-live-preview",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        video_input: bool = False,
        room_name: Optional[str] = None,
        participant_identity: Optional[str] = None,
        participant_name: Optional[str] = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> LiveSessionInfo:
        """Generates signed LiveKit WebRTC session credentials directly on the client."""
        from omniai.live import create_live_session as _create_live_session

        return _create_live_session(
            server_url=server_url,
            api_key=api_key,
            api_secret=api_secret,
            model=model,
            voice=voice,
            system_instruction=system_instruction,
            temperature=temperature,
            video_input=video_input,
            room_name=room_name,
            participant_identity=participant_identity,
            participant_name=participant_name,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Asynchronous Client: AsyncOmniaiClient
# ──────────────────────────────────────────────────────────────────────────────

class AsyncOmniaiClient:
    """
    Official Asynchronous (async/await) Python Client for the OmniAI Gateway.
    Compatible with OpenAI, Google Gemini, Anthropic Claude, and vLLM models.
    Features automatic retries with exponential backoff & jitter for high fault tolerance.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = 0.5,
    ):
        _auto_load_env()
        self.base_url = (
            base_url
            or os.getenv("OMNI_BASE_URL")
            or os.getenv("OMNIAI_BASE_URL")
            or os.getenv("GATEWAY_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("OMNI_API_KEY")
            or os.getenv("OMNIAI_API_KEY")
            or os.getenv("X_API_KEY")
            or ""
        )
        self.timeout = timeout
        self.retry_policy = RetryPolicy(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

        self._headers: Dict[str, str] = {
            "User-Agent": "Omniai-Async-Python-SDK/1.0.0",
        }
        if self.api_key:
            self._headers["X-API-KEY"] = self.api_key
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0),
            follow_redirects=False,
        )

    def __repr__(self) -> str:
        masked_key = (
            f"{self.api_key[:7]}...{self.api_key[-4:]}"
            if len(self.api_key) > 11
            else ("***" if self.api_key else "None")
        )
        return f"<AsyncOmniaiClient base_url={self.base_url!r} api_key={masked_key!r}>"

    async def close(self) -> None:
        """Closes the underlying AsyncHTTP client and releases connection pools."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncOmniaiClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def chat(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> Union[str, Dict[str, Any], T]:
        """
        Executes an asynchronous completion at /v1/chat/completions with automatic retries.
        Supports models from: openai, gemini, anthropic, and vllm.
        """
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            output_format=output_format,
            stream=False,
        )

        effective_timeout = timeout or self.timeout

        async def _do_request() -> httpx.Response:
            return await self._http.post(
                "/v1/chat/completions",
                json=payload,
                timeout=effective_timeout,
            )

        try:
            resp = await self.retry_policy.execute_async(_do_request, operation_name=f"chat[{model}]")
            _handle_response_error(resp)
            data = resp.json()
            raw_result = data.get("result")

            if response_model is not None and isinstance(response_model, type) and issubclass(response_model, BaseModel):
                if isinstance(raw_result, dict):
                    try:
                        return response_model.model_validate(raw_result)
                    except Exception as err:
                        raise OmniaiValidationError(
                            f"Failed to validate response into {response_model.__name__}: {err}",
                            raw_output=str(raw_result),
                            details={"error": str(err)},
                        ) from err
                elif isinstance(raw_result, str):
                    cleaned = _clean_json_str(raw_result)
                    try:
                        return response_model.model_validate_json(cleaned)
                    except Exception as err:
                        raise OmniaiValidationError(
                            f"Failed to validate JSON string response into {response_model.__name__}: {err}",
                            raw_output=raw_result,
                            details={"error": str(err)},
                        ) from err

            return raw_result

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OmniaiConnectionError(f"Connection to OmniAI gateway failed: {exc}") from exc
        except httpx.ReadTimeout as exc:
            raise OmniaiTimeoutError(f"OmniAI async chat request timed out after {effective_timeout}s: {exc}") from exc

    async def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams real-time token chunks from /v1/chat/completions asynchronously via SSE.
        Tolerant to keep-alive comments, carriage returns, and heartbeat frames.
        """
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            stream=True,
        )

        effective_timeout = timeout or self.timeout

        try:
            async with self._http.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=effective_timeout,
            ) as resp:
                _handle_response_error(resp)
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(raw_data)
                            content = parsed.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            if raw_data:
                                yield raw_data
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OmniaiConnectionError(f"Connection failed during async streaming: {exc}") from exc
        except httpx.ReadTimeout as exc:
            raise OmniaiTimeoutError(f"Async streaming token stream timed out: {exc}") from exc

    async def submit_task(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Union[Type[BaseModel], Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        webhook_url: Optional[str] = None,
    ) -> AsyncTaskJob[T]:
        """Queues an asynchronous LLM completion job to the OmniAI Celery background workers."""
        _validate_model_provider(model)
        payload = _build_payload(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
            tools=tools,
            temperature=temperature,
            thinking=thinking,
            base_url_override=base_url,
            output_format=output_format,
            webhook_url=webhook_url,
        )

        async def _do_submit() -> httpx.Response:
            return await self._http.post("/v1/chat/completions/async", json=payload)

        resp = await self.retry_policy.execute_async(_do_submit, operation_name="submit_task_async")
        _handle_response_error(resp)
        task_id = resp.json()["task_id"]

        return AsyncTaskJob(
            task_id=task_id,
            status_checker=self.get_task_status,
            canceller=self.cancel_task,
            response_model=response_model if isinstance(response_model, type) and issubclass(response_model, BaseModel) else None,
        )

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Fetches status and result of a background Celery task."""
        async def _do_status() -> httpx.Response:
            return await self._http.get(f"/v1/chat/completions/tasks/{task_id}")

        resp = await self.retry_policy.execute_async(_do_status, operation_name=f"get_task_status_async[{task_id}]")
        _handle_response_error(resp)
        return TaskStatus.model_validate(resp.json())

    async def cancel_task(self, task_id: str) -> dict:
        """Cancels/revokes an active background task."""
        async def _do_cancel() -> httpx.Response:
            return await self._http.post(f"/v1/chat/completions/tasks/{task_id}/cancel")

        resp = await self.retry_policy.execute_async(_do_cancel, operation_name=f"cancel_task_async[{task_id}]")
        _handle_response_error(resp)
        return resp.json()

    async def create_live_session(
        self,
        model: str = "gemini:gemini-3.1-flash-live-preview",
        voice: str = "Aoede",
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        video_input: bool = False,
        room_name: Optional[str] = None,
        participant_identity: Optional[str] = None,
        participant_name: Optional[str] = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> LiveSessionInfo:
        """Generates signed LiveKit WebRTC session credentials directly on the client."""
        from omniai.live import create_live_session as _create_live_session

        return _create_live_session(
            server_url=server_url,
            api_key=api_key,
            api_secret=api_secret,
            model=model,
            voice=voice,
            system_instruction=system_instruction,
            temperature=temperature,
            video_input=video_input,
            room_name=room_name,
            participant_identity=participant_identity,
            participant_name=participant_name,
        )


# Convenient and backward-compatible client aliases
OmniAIClient = OmniaiClient
AsyncOmniAIClient = AsyncOmniaiClient
OmniClient = OmniaiClient
AsyncOmniClient = AsyncOmniaiClient
