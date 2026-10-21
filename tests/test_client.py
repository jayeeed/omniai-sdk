import httpx
import pytest
from pydantic import BaseModel
from omniai import OmniaiClient, OmniClient
from omniai.exceptions import (
    OmniaiAuthError,
    OmniaiPaymentRequiredError,
    OmniaiValidationError,
)


class MockReport(BaseModel):
    summary: str
    score: int


def test_client_init_and_repr_masking():
    client = OmniaiClient(
        base_url="https://test.omniai.ai",
        api_key="sk-omni-secret-key-1234567890",
    )
    assert client.base_url == "https://test.omniai.ai"
    assert client.api_key == "sk-omni-secret-key-1234567890"

    # Verify secret is masked in repr
    repr_str = repr(client)
    assert "sk-omni...7890" in repr_str
    assert "secret-key" not in repr_str


def test_backward_compatibility_alias():
    # Verify OmniClient is identical to OmniaiClient
    assert OmniClient is OmniaiClient
    client = OmniClient(api_key="test-key")
    assert isinstance(client, OmniaiClient)


def test_context_manager_lifecycle():
    with OmniaiClient(api_key="test-key") as client:
        assert not client._http.is_closed
    assert client._http.is_closed


def test_chat_success():
    def mock_handler(request: httpx.Request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(200, json={"result": "Hello from mock OmniAI!"})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport, headers=client._headers)

    res = client.chat(model="openai:gpt-4o", prompt="Hello")
    assert res == "Hello from mock OmniAI!"


def test_chat_structured_output_validation():
    def mock_handler(request: httpx.Request):
        return httpx.Response(200, json={"result": {"summary": "Passed test", "score": 98}})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport)

    report = client.chat(
        model="openai:gpt-4o",
        prompt="Score this test",
        response_model=MockReport,
    )
    assert isinstance(report, MockReport)
    assert report.summary == "Passed test"
    assert report.score == 98


def test_chat_structured_output_markdown_stripping():
    def mock_handler(request: httpx.Request):
        # LLM wrapped output in markdown json block
        raw_json_in_markdown = '```json\n{"summary": "Stripped markdown", "score": 100}\n```'
        return httpx.Response(200, json={"result": raw_json_in_markdown})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport)

    report = client.chat(
        model="openai:gpt-4o",
        prompt="Score this",
        response_model=MockReport,
    )
    assert isinstance(report, MockReport)
    assert report.summary == "Stripped markdown"
    assert report.score == 100


def test_chat_auth_failure_raises_omniai_auth_error():
    def mock_handler(request: httpx.Request):
        return httpx.Response(401, json={"detail": "Invalid API Key"})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="invalid-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport)

    with pytest.raises(OmniaiAuthError) as exc_info:
        client.chat(model="openai:gpt-4o", prompt="Hello")
    assert "Authentication failed" in str(exc_info.value)


def test_chat_balance_failure_raises_payment_required():
    def mock_handler(request: httpx.Request):
        return httpx.Response(402, json={"detail": "Prepaid credit balance exhausted"})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="empty-balance-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport)

    with pytest.raises(OmniaiPaymentRequiredError) as exc_info:
        client.chat(model="openai:gpt-4o", prompt="Hello")
    assert "Payment required" in str(exc_info.value)


def test_stream_generator():
    def mock_handler(request: httpx.Request):
        sse_body = (
            ": keep-alive\r\n\r\n"
            'data: {"content": "Hello "}\r\n\r\n'
            ": ping\r\n\r\n"
            'data: {"content": "world!"}\r\n\r\n'
            "data: [DONE]\r\n\r\n"
        )
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(mock_handler)
    client = OmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.Client(base_url="https://mock.omniai.ai", transport=transport)

    tokens = list(client.stream(model="openai:gpt-4o", prompt="Stream hello"))
    assert tokens == ["Hello ", "world!"]
