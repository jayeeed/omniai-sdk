"""Chat completions resource for OmniAI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel

from omniai.types.tasks import clean_json_text
from omniai.exceptions import OmniaiValidationError

if TYPE_CHECKING:
    from omniai._base import BaseClient, AsyncBaseClient

T = TypeVar("T")


class CompletionsResource:
    """Synchronous completions endpoint resource."""

    def __init__(self, client: BaseClient):
        self._client = client

    def create(
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
    ) -> Union[str, T]:
        # Early provider validation
        if ":" in model:
            provider = model.split(":", 1)[0].strip().lower()
            if provider not in ("openai", "gemini", "anthropic", "vllm", "openrouter", "ollama", "custom", "local"):
                from omniai.exceptions import OmniaiUnsupportedProviderError
                raise OmniaiUnsupportedProviderError(
                    f"Unsupported model provider '{provider}'. Strictly supported providers are: openai, gemini, anthropic, vllm, openrouter, ollama, custom, local."
                )

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
            "/v1/chat/completions",
            json=payload,
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)

        data = response.json()
        raw_result = data.get("result", "")

        if response_model and issubclass(response_model, BaseModel):
            if isinstance(raw_result, dict):
                try:
                    return response_model.model_validate(raw_result)
                except Exception as e:
                    raise OmniaiValidationError(
                        f"Failed validating response dictionary against {response_model.__name__}: {e}",
                        raw_output=str(raw_result),
                    ) from e
            elif isinstance(raw_result, str):
                cleaned = clean_json_text(raw_result)
                try:
                    return response_model.model_validate_json(cleaned)
                except Exception as e:
                    raise OmniaiValidationError(
                        f"Failed validating JSON string against {response_model.__name__}: {e}",
                        raw_output=raw_result,
                    ) from e

        return raw_result

    def stream(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Streams a chat completion chunk-by-chunk using Server-Sent Events."""
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "stream": True,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        with self._client.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=timeout or self._client.timeout,
        ) as response:
            if response.status_code != 200:
                response.read()
                self._client._handle_error_response(response)

            for line in response.iter_lines():
                if not line or not line.strip():
                    continue
                line_str = line.strip()
                if line_str.startswith("data: "):
                    content = line_str[6:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        delta = (
                            chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            or chunk.get("content")
                            or chunk.get("delta")
                            or chunk.get("text", "")
                        )
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        yield content


class AsyncCompletionsResource:
    """Asynchronous completions endpoint resource."""

    def __init__(self, client: AsyncBaseClient):
        self._client = client

    async def create(
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
    ) -> Union[str, T]:
        # Early provider validation
        if ":" in model:
            provider = model.split(":", 1)[0].strip().lower()
            if provider not in ("openai", "gemini", "anthropic", "vllm", "openrouter", "ollama", "custom", "local"):
                from omniai.exceptions import OmniaiUnsupportedProviderError
                raise OmniaiUnsupportedProviderError(
                    f"Unsupported model provider '{provider}'. Strictly supported providers are: openai, gemini, anthropic, vllm, openrouter, ollama, custom, local."
                )

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
            "/v1/chat/completions",
            json=payload,
            timeout=timeout or self._client.timeout,
        )
        if response.status_code != 200:
            self._client._handle_error_response(response)

        data = response.json()
        raw_result = data.get("result", "")

        if response_model and issubclass(response_model, BaseModel):
            if isinstance(raw_result, dict):
                try:
                    return response_model.model_validate(raw_result)
                except Exception as e:
                    raise OmniaiValidationError(
                        f"Failed validating response dictionary against {response_model.__name__}: {e}",
                        raw_output=str(raw_result),
                    ) from e
            elif isinstance(raw_result, str):
                cleaned = clean_json_text(raw_result)
                try:
                    return response_model.model_validate_json(cleaned)
                except Exception as e:
                    raise OmniaiValidationError(
                        f"Failed validating JSON string against {response_model.__name__}: {e}",
                        raw_output=raw_result,
                    ) from e

        return raw_result

    async def stream(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams an async chat completion chunk-by-chunk using Server-Sent Events."""
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "stream": True,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if messages is not None:
            payload["messages"] = messages
        if system_instruction is not None:
            payload["system_instruction"] = system_instruction
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with self._client.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=timeout or self._client.timeout,
        ) as response:
            if response.status_code != 200:
                await response.aread()
                self._client._handle_error_response(response)

            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                line_str = line.strip()
                if line_str.startswith("data: "):
                    content = line_str[6:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        delta = (
                            chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            or chunk.get("content")
                            or chunk.get("delta")
                            or chunk.get("text", "")
                        )
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        yield content


class ChatResource:
    """Synchronous chat resource with nested completions."""

    def __init__(self, client: BaseClient):
        self._client = client
        self.completions = CompletionsResource(client)

    def __call__(
        self,
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
    ) -> Union[str, T]:
        """Allows direct invocation: client.chat(...)"""
        return self.completions.create(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Allows direct streaming invocation: client.chat.stream(...)"""
        return self.completions.stream(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


class AsyncChatResource:
    """Asynchronous chat resource with nested completions."""

    def __init__(self, client: AsyncBaseClient):
        self._client = client
        self.completions = AsyncCompletionsResource(client)

    async def __call__(
        self,
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
    ) -> Union[str, T]:
        """Allows direct async invocation: await client.chat(...)"""
        return await self.completions.create(
            model=model,
            prompt=prompt,
            messages=messages,
            response_model=response_model,
            tools=tools,
            output_format=output_format,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def stream(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Allows direct async streaming invocation: async for chunk in client.chat.stream(...)"""
        return self.completions.stream(
            model=model,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
