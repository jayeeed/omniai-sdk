from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field
from omniai.resources.router import RouterEngine, SUPPORTED_PROVIDERS
from omniai.exceptions import OmniaiUnsupportedProviderError


class UserInfo(BaseModel):
    name: str = Field(..., description="User's full name")
    age: int = Field(..., description="Age in years")


def test_router_supported_providers():
    assert "openai" in SUPPORTED_PROVIDERS
    assert "gemini" in SUPPORTED_PROVIDERS
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "vllm" in SUPPORTED_PROVIDERS
    assert "openrouter" in SUPPORTED_PROVIDERS
    assert "ollama" in SUPPORTED_PROVIDERS
    assert len(SUPPORTED_PROVIDERS) == 8


def test_resolve_output_schema_pydantic():
    schema = RouterEngine.resolve_output_schema(tools=[UserInfo])
    assert schema is not None
    assert isinstance(schema, dict)
    assert schema["title"] == "UserInfo"
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_resolve_output_schema_json_mode():
    schema = RouterEngine.resolve_output_schema(output_format="json")
    assert schema == "json"


def test_build_openai_messages():
    # Prompt only
    msgs = RouterEngine.build_openai_messages(prompt="Hello")
    assert msgs == [{"role": "user", "content": "Hello"}]

    # System instruction + prompt
    msgs = RouterEngine.build_openai_messages(prompt="Hello", system_instruction="Be concise")
    assert msgs == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]


def test_vllm_execution():
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Hello from vLLM!"))]
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.dict("sys.modules", {"openai": mock_openai}):
        result = RouterEngine.execute(
            model="vllm:meta-llama/Llama-3.1-8B-Instruct",
            prompt="Hi vLLM",
            base_url="http://localhost:8000/v1",
            vllm_api_key="EMPTY",
        )

        assert result == "Hello from vLLM!"
        mock_openai.OpenAI.assert_called_once_with(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "meta-llama/Llama-3.1-8B-Instruct"
