from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlparse
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Set, Tuple, Union
import httpx
from pydantic import BaseModel

from omniai.exceptions import OmniaiUnsupportedProviderError

logger = logging.getLogger("omniai.router")

SUPPORTED_PROVIDERS: Tuple[str, ...] = ("openai", "gemini", "anthropic", "vllm", "openrouter", "ollama", "custom", "local")


def validate_sdk_base_url(url: Optional[str]) -> None:
    """
    SEC-08: Validates base_url in SDK RouterEngine.
    Enforces https:// scheme for external hosts; allows http:// only for localhost/127.0.0.1/private testnets.
    """
    if not url:
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid base_url scheme '{parsed.scheme}'. Only http and https are permitted.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Invalid base_url: missing hostname.")
    if parsed.scheme == "http" and hostname not in ("localhost", "127.0.0.1", "::1", "host.docker.internal"):
        # For local container networking or internal vLLM/Ollama, allow internal hosts
        if not (
            hostname.endswith(".local")
            or hostname.startswith("vllm")
            or hostname.startswith("ollama")
            or hostname.startswith("10.")
            or hostname.startswith("192.168.")
            or hostname.startswith("172.")
        ):
            raise ValueError(
                f"Insecure plaintext HTTP endpoint '{url}' is prohibited for remote hosts. Use HTTPS."
            )


def _get_langsmith_tracer():
    """Returns LangSmith traceable decorator if enabled, otherwise a pass-through decorator."""
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith import traceable
            return traceable
        except ImportError:
            pass

    def dummy_decorator(*args, **kwargs):
        def inner(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return inner

    return dummy_decorator


traceable = _get_langsmith_tracer()


class RouterEngine:
    """
    Production-grade multi-provider LLM inference engine.
    Strictly compatible with 4 providers:
      1. OpenAI ('openai:model-name')
      2. Google Gemini ('gemini:model-name')
      3. Anthropic Claude ('anthropic:model-name')
      4. vLLM Self-Hosted Inference ('vllm:model-name')
    """

    SUPPORTED_PROVIDERS: Tuple[str, ...] = SUPPORTED_PROVIDERS

    @classmethod
    def parse_provider(cls, model_str: str) -> tuple[str, str]:
        """
        Splits 'provider:model-name' into (provider, model_name).
        Enforces strict whitelisting for: openai, gemini, anthropic, vllm.
        """
        if ":" in model_str:
            provider, _, model_name = model_str.partition(":")
            provider_clean = provider.strip().lower()
            if provider_clean not in cls.SUPPORTED_PROVIDERS:
                raise OmniaiUnsupportedProviderError(
                    f"Unsupported provider '{provider_clean}'. The OmniAI SDK strictly supports: "
                    f"{', '.join(cls.SUPPORTED_PROVIDERS)}."
                )
            return (provider_clean, model_name.strip())

        # Implicit provider detection
        lower = model_str.lower()
        if lower.startswith("gemini"):
            return ("gemini", model_str)
        elif lower.startswith("claude"):
            return ("anthropic", model_str)
        elif lower.startswith("openrouter/"):
            return ("openrouter", model_str[11:])
        elif lower.startswith("vllm/"):
            return ("vllm", model_str[5:])
        elif lower.startswith("ollama/"):
            return ("ollama", model_str[7:])
        elif lower.startswith("custom/"):
            return ("custom", model_str[7:])
        elif lower.startswith("local/"):
            return ("local", model_str[6:])
        else:
            return ("openai", model_str)

    @staticmethod
    def resolve_output_schema(
        tools: Optional[Union[List[Any], Dict[str, Any]]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Resolves `tools` and `output_format` into a single schema dictionary or 'json' mode."""
        if tools:
            if isinstance(tools, list):
                schemas: List[Dict[str, Any]] = []
                for item in tools:
                    if isinstance(item, type) and issubclass(item, BaseModel):
                        schemas.append(item.model_json_schema())
                    elif isinstance(item, dict):
                        if "function" in item and isinstance(item["function"], dict) and "parameters" in item["function"]:
                            schemas.append(item["function"]["parameters"])
                        else:
                            schemas.append(item)
                    elif isinstance(item, str):
                        schemas.append({"title": item, "type": "object"})

                if len(schemas) == 1:
                    return schemas[0]
                elif len(schemas) > 1:
                    return {"anyOf": schemas}

            elif isinstance(payload_tools := tools, dict):
                if "function" in payload_tools and isinstance(payload_tools["function"], dict) and "parameters" in payload_tools["function"]:
                    return payload_tools["function"]["parameters"]
                return payload_tools

        return output_format

    @staticmethod
    def _normalize_message(msg: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalizes dict, Pydantic, or object message format to standard dict."""
        if isinstance(msg, dict):
            return msg
        return {
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", ""),
        }

    @classmethod
    def build_openai_messages(
        cls,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Constructs standardized OpenAI-compatible messages list."""
        built: List[Dict[str, Any]] = []
        if system_instruction:
            built.append({"role": "system", "content": system_instruction})

        if messages:
            norm_messages = [cls._normalize_message(m) for m in messages]
            for msg in norm_messages:
                role = "assistant" if msg.get("role") in ("assistant", "ai") else msg.get("role", "user")
                if role == "system" and system_instruction:
                    continue
                built.append({"role": role, "content": msg.get("content", "")})
        elif prompt:
            built.append({"role": "user", "content": prompt})

        return built

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Google Gemini Provider
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _execute_gemini(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        api_key: str,
    ) -> Any:
        """Executes Google Gemini inference via google-genai SDK or OpenAI-compatible endpoint."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config_args: Dict[str, Any] = {"temperature": temperature}
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            if thinking is not None and thinking >= 0:
                config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking)
            if schema:
                config_args["response_mime_type"] = "application/json"
                if isinstance(schema, dict):
                    config_args["response_schema"] = schema

            config = types.GenerateContentConfig(**config_args)

            if messages:
                contents = []
                for msg in messages:
                    role = "model" if msg.get("role") in ("assistant", "ai") else "user"
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.get("content", ""))]))
                response = client.models.generate_content(model=model_name, contents=contents, config=config)
            else:
                response = client.models.generate_content(model=model_name, contents=prompt or "", config=config)

            text = response.text or ""
            if schema and isinstance(schema, dict):
                try:
                    return json.loads(text)
                except Exception:
                    pass
            elif schema and str(schema).lower() == "json":
                try:
                    return json.loads(text)
                except Exception:
                    pass
            return text

        except Exception:
            # Fallback to Google Generative Language REST
            return cls._execute_openai(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
            )

    @classmethod
    async def _execute_gemini_stream(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        api_key: str,
    ) -> AsyncGenerator[str, None]:
        """Streams Google Gemini inference chunks via google-genai SDK."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config_args: Dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if thinking is not None and thinking >= 0:
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking)
        if schema:
            config_args["response_mime_type"] = "application/json"
            if isinstance(schema, dict):
                config_args["response_schema"] = schema

        config = types.GenerateContentConfig(**config_args)

        if messages:
            contents = []
            for msg in messages:
                role = "model" if msg.get("role") in ("assistant", "ai") else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.get("content", ""))]))
            stream = client.models.generate_content_stream(model=model_name, contents=contents, config=config)
        else:
            stream = client.models.generate_content_stream(model=model_name, contents=prompt or "", config=config)

        for chunk in stream:
            if chunk.text:
                yield chunk.text


    # ──────────────────────────────────────────────────────────────────────────
    # 2. Anthropic Claude Provider
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strips `thinking` content blocks from assistant messages.
        Anthropic rejects requests where an assistant turn contains only a thinking
        block and no text block — this causes a 400 "model output must contain
        either output text or tool calls" error in multi-turn thinking sessions.
        """
        sanitized = []
        for msg in messages:
            content = msg.get("content")
            if msg.get("role") in ("assistant", "ai") and isinstance(content, list):
                # Keep only text/tool_use blocks — discard thinking blocks
                filtered = [b for b in content if isinstance(b, dict) and b.get("type") != "thinking"]
                if not filtered:
                    # All content was thinking — convert to a minimal text placeholder
                    # so the conversation turn is still valid
                    text_parts = [
                        b.get("thinking", "") or b.get("text", "")
                        for b in content
                        if isinstance(b, dict)
                    ]
                    combined = " ".join(t for t in text_parts if t).strip()
                    sanitized.append({"role": "assistant", "content": combined or "..."})
                else:
                    # Collapse single-text list to a plain string for cleanliness
                    if len(filtered) == 1 and filtered[0].get("type") == "text":
                        sanitized.append({"role": msg["role"], "content": filtered[0].get("text", "")})
                    else:
                        sanitized.append({**msg, "content": filtered})
            else:
                sanitized.append(msg)
        return sanitized

    @classmethod
    def _execute_anthropic(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        api_key: str,
    ) -> Any:
        """Executes Anthropic Claude inference via direct HTTP REST API."""
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        anth_messages: List[Dict[str, Any]] = []
        if messages:
            for msg in messages:
                role = "assistant" if msg.get("role") in ("assistant", "ai") else "user"
                if msg.get("role") == "system" and not system_instruction:
                    system_instruction = msg.get("content")
                    continue
                anth_messages.append({"role": role, "content": msg.get("content", "")})
        else:
            anth_messages.append({"role": "user", "content": prompt or ""})

        # Strip thinking blocks from assistant history to avoid Anthropic 400
        anth_messages = cls._sanitize_anthropic_messages(anth_messages)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": anth_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if system_instruction:
            payload["system"] = system_instruction

        if thinking and thinking > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking}
            payload["max_tokens"] = max(4096, thinking + 2048)
            payload["temperature"] = 1.0
            # Required beta header for extended thinking on older Anthropic API versions
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        with httpx.Client(timeout=90.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            content_blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

            # Fallback: if model returned only thinking blocks (no text), extract thinking content
            if not text and content_blocks:
                text = "".join(
                    b.get("thinking", "") or b.get("text", "")
                    for b in content_blocks
                    if b.get("type") in ("thinking", "text")
                )

            if schema and (isinstance(schema, dict) or str(schema).lower() == "json"):
                try:
                    return json.loads(text)
                except Exception:
                    pass
            return text

    @classmethod
    async def _execute_anthropic_stream(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        api_key: str,
    ) -> AsyncGenerator[str, None]:
        """Streams Anthropic Claude inference chunks via SSE."""
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        anth_messages: List[Dict[str, Any]] = []
        if messages:
            for msg in messages:
                role = "assistant" if msg.get("role") in ("assistant", "ai") else "user"
                if msg.get("role") == "system" and not system_instruction:
                    system_instruction = msg.get("content")
                    continue
                anth_messages.append({"role": role, "content": msg.get("content", "")})
        else:
            anth_messages.append({"role": "user", "content": prompt or ""})

        # Strip thinking blocks from assistant history to avoid Anthropic 400
        anth_messages = cls._sanitize_anthropic_messages(anth_messages)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": anth_messages,
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": True,
        }
        if system_instruction:
            payload["system"] = system_instruction

        if thinking and thinking > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking}
            payload["max_tokens"] = max(4096, thinking + 2048)
            payload["temperature"] = 1.0
            # Required beta header for extended thinking on older Anthropic API versions
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    raise RuntimeError(f"Anthropic streaming error ({resp.status_code}): {err.decode('utf-8')}")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            evt = json.loads(raw_data)
                            if evt.get("type") == "content_block_delta":
                                delta = evt.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except Exception:
                            pass

    # ──────────────────────────────────────────────────────────────────────────
    # 3. OpenAI-Compatible Core Execution (OpenAI, OpenRouter, vLLM)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _execute_openai_compatible(
        cls,
        base_url: Optional[str],
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
    ) -> Any:
        """Executes OpenAI / OpenRouter / vLLM inference via OpenAI client with robust parameter handling."""
        validate_sdk_base_url(base_url)
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )
        built_messages = cls.build_openai_messages(prompt, messages, system_instruction)

        # OpenAI reasoning models (o1, o3, etc.) and newer official API endpoints require max_completion_tokens
        is_official_openai = base_url is None or "api.openai.com" in (base_url or "")
        is_reasoning_model = any(model_name.lower().startswith(p) for p in ("o1", "o3", "gpt-5"))

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": built_messages,
        }

        # Reasoning models reject custom temperature
        if not is_reasoning_model:
            kwargs["temperature"] = temperature

        if is_official_openai or is_reasoning_model:
            kwargs["max_completion_tokens"] = 4096
        else:
            kwargs["max_tokens"] = 4096

        if thinking and thinking > 0:
            effort = "low" if thinking <= 2000 else "medium" if thinking <= 8000 else "high"
            kwargs["reasoning_effort"] = effort
            kwargs.pop("temperature", None)

        if schema:
            if isinstance(schema, dict):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output_schema", "schema": schema, "strict": True},
                }
            elif str(schema).lower() == "json":
                kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                if "max_tokens" in kwargs:
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = 4096
                else:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = 4096
                if "temperature" in err_msg:
                    kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
            elif "temperature" in err_msg and ("not supported" in err_msg or "unsupported_parameter" in err_msg):
                kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        content = response.choices[0].message.content or ""

        if schema and (isinstance(schema, dict) or str(schema).lower() == "json"):
            try:
                return json.loads(content)
            except Exception:
                pass
        return content

    @classmethod
    async def _execute_openai_compatible_stream(
        cls,
        base_url: Optional[str],
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
    ) -> AsyncGenerator[str, None]:
        """Streams OpenAI / OpenRouter / vLLM inference chunks via OpenAI async client."""
        validate_sdk_base_url(base_url)
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )
        built_messages = cls.build_openai_messages(prompt, messages, system_instruction)

        is_official_openai = base_url is None or "api.openai.com" in (base_url or "")
        is_reasoning_model = any(model_name.lower().startswith(p) for p in ("o1", "o3", "gpt-5"))

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": built_messages,
            "stream": True,
        }

        if not is_reasoning_model:
            kwargs["temperature"] = temperature

        if is_official_openai or is_reasoning_model:
            kwargs["max_completion_tokens"] = 4096
        else:
            kwargs["max_tokens"] = 4096

        if thinking and thinking > 0:
            effort = "low" if thinking <= 2000 else "medium" if thinking <= 8000 else "high"
            kwargs["reasoning_effort"] = effort
            kwargs.pop("temperature", None)

        if schema:
            if isinstance(schema, dict):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output_schema", "schema": schema, "strict": True},
                }
            elif str(schema).lower() == "json":
                kwargs["response_format"] = {"type": "json_object"}

        try:
            stream = await client.chat.completions.create(**kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                if "max_tokens" in kwargs:
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = 4096
                else:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = 4096
                if "temperature" in err_msg:
                    kwargs.pop("temperature", None)
                stream = await client.chat.completions.create(**kwargs)
            elif "temperature" in err_msg and ("not supported" in err_msg or "unsupported_parameter" in err_msg):
                kwargs.pop("temperature", None)
                stream = await client.chat.completions.create(**kwargs)
            else:
                raise

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @classmethod
    def _execute_openai(
        cls,
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
    ) -> Any:
        return cls._execute_openai_compatible(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        )

    @classmethod
    async def _execute_openai_stream(
        cls,
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in cls._execute_openai_compatible_stream(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        ):
            yield chunk

    @classmethod
    def _execute_openrouter(
        cls,
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
    ) -> Any:
        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return cls._execute_openai_compatible(
            base_url=resolved_base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        )

    @classmethod
    async def _execute_openrouter_stream(
        cls,
        api_key: str,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        async for chunk in cls._execute_openai_compatible_stream(
            base_url=resolved_base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        ):
            yield chunk

    @classmethod
    def _execute_vllm(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        resolved_base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        return cls._execute_openai_compatible(
            base_url=resolved_base_url,
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        )

    @classmethod
    async def _execute_vllm_stream(
        cls,
        model_name: str,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        system_instruction: Optional[str],
        temperature: float,
        thinking: Optional[int],
        schema: Optional[Union[str, Dict[str, Any]]],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        resolved_base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        async for chunk in cls._execute_openai_compatible_stream(
            base_url=resolved_base_url,
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking=thinking,
            schema=schema,
        ):
            yield chunk

    # ──────────────────────────────────────────────────────────────────────────
    # Unified Execution Entry Points
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def execute(
        cls,
        model: Union[str, Any],
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        tools: Optional[Union[List[Any], Dict[str, Any]]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Main execution entry point for synchronous completions across supported providers:
        openai, gemini, anthropic, vllm, openrouter.
        Supports passing an LLMPayload object or individual arguments.
        """
        if not isinstance(model, str):
            payload = model
            model = getattr(payload, "model", "")
            prompt = getattr(payload, "prompt", prompt)
            messages = getattr(payload, "messages", messages)
            system_instruction = getattr(payload, "system_instruction", system_instruction)
            temperature = getattr(payload, "temperature", temperature)
            thinking = getattr(payload, "thinking", thinking)
            base_url = getattr(payload, "base_url", base_url)
            tools = getattr(payload, "tools", tools)
            output_format = getattr(payload, "output_format", output_format)

        provider, model_name = cls.parse_provider(model)
        schema = cls.resolve_output_schema(tools, output_format)
        native_model = model_name.replace(":batch", "") if provider in ("gemini", "openai", "anthropic") else model_name

        if provider == "gemini":
            key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
            return cls._execute_gemini(
                model_name=native_model,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                api_key=key,
            )

        elif provider == "anthropic":
            key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            return cls._execute_anthropic(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                api_key=key,
            )

        elif provider == "openrouter":
            key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
            return cls._execute_openrouter(
                api_key=key,
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=base_url,
            )

        elif provider in ("vllm", "ollama", "custom", "local"):
            default_url = "http://localhost:11434/v1" if provider == "ollama" else "http://localhost:8000/v1"
            custom_url = base_url or os.getenv(f"{provider.upper()}_BASE_URL") or os.getenv("VLLM_BASE_URL") or default_url
            custom_key = vllm_api_key or os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")
            return cls._execute_vllm(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=custom_url,
                api_key=custom_key,
            )

        elif provider == "openai":
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            return cls._execute_openai(
                api_key=key,
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=base_url,
            )

        raise OmniaiUnsupportedProviderError(
            f"Unsupported provider '{provider}'. The OmniAI SDK strictly supports: "
            f"{', '.join(cls.SUPPORTED_PROVIDERS)}."
        )

    @classmethod
    async def execute_stream(
        cls,
        model: Union[str, Any],
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        thinking: Optional[int] = None,
        base_url: Optional[str] = None,
        tools: Optional[Union[List[Any], Dict[str, Any]]] = None,
        output_format: Optional[Union[str, Dict[str, Any]]] = None,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Main execution entry point for streaming token chunks across supported providers:
        openai, gemini, anthropic, vllm, openrouter.
        Supports passing an LLMPayload object or individual arguments.
        """
        if not isinstance(model, str):
            payload = model
            model = getattr(payload, "model", "")
            prompt = getattr(payload, "prompt", prompt)
            messages = getattr(payload, "messages", messages)
            system_instruction = getattr(payload, "system_instruction", system_instruction)
            temperature = getattr(payload, "temperature", temperature)
            thinking = getattr(payload, "thinking", thinking)
            base_url = getattr(payload, "base_url", base_url)
            tools = getattr(payload, "tools", tools)
            output_format = getattr(payload, "output_format", output_format)

        provider, model_name = cls.parse_provider(model)
        schema = cls.resolve_output_schema(tools, output_format)
        native_model = model_name.replace(":batch", "") if provider in ("gemini", "openai", "anthropic") else model_name

        if provider == "gemini":
            key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
            async for chunk in cls._execute_gemini_stream(
                model_name=native_model,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                api_key=key,
            ):
                yield chunk

        elif provider == "anthropic":
            key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            async for chunk in cls._execute_anthropic_stream(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                api_key=key,
            ):
                yield chunk

        elif provider == "openrouter":
            key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
            async for chunk in cls._execute_openrouter_stream(
                api_key=key,
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=base_url,
            ):
                yield chunk

        elif provider in ("vllm", "ollama", "custom", "local"):
            default_url = "http://localhost:11434/v1" if provider == "ollama" else "http://localhost:8000/v1"
            custom_url = base_url or os.getenv(f"{provider.upper()}_BASE_URL") or os.getenv("VLLM_BASE_URL") or default_url
            custom_key = vllm_api_key or os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("VLLM_API_KEY", "EMPTY")
            async for chunk in cls._execute_vllm_stream(
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=custom_url,
                api_key=custom_key,
            ):
                yield chunk

        elif provider == "openai":
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            async for chunk in cls._execute_openai_stream(
                api_key=key,
                model_name=model_name,
                prompt=prompt,
                messages=messages,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking=thinking,
                schema=schema,
                base_url=base_url,
            ):
                yield chunk

        else:
            raise OmniaiUnsupportedProviderError(
                f"Unsupported provider '{provider}'. The OmniAI SDK strictly supports: "
                f"{', '.join(cls.SUPPORTED_PROVIDERS)}."
            )
