import httpx
import pytest
from pydantic import BaseModel
from omniai import AsyncOmniaiClient, AsyncOmniClient


class MockAnalysis(BaseModel):
    category: str
    confidence: float


@pytest.mark.asyncio
async def test_async_client_lifecycle():
    async with AsyncOmniaiClient(api_key="sk-omni-async-1234567890") as client:
        assert "sk-omni...7890" in repr(client)
        assert not client._http.is_closed
    assert client._http.is_closed


def test_async_backward_compatibility_alias():
    assert AsyncOmniClient is AsyncOmniaiClient


@pytest.mark.asyncio
async def test_async_chat_success():
    def mock_handler(request: httpx.Request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"result": "Async completion success"})

    transport = httpx.MockTransport(mock_handler)
    client = AsyncOmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.AsyncClient(base_url="https://mock.omniai.ai", transport=transport)

    res = await client.chat(model="openai:gpt-4o", prompt="Async test")
    assert res == "Async completion success"
    await client.close()


@pytest.mark.asyncio
async def test_async_chat_response_model():
    def mock_handler(request: httpx.Request):
        return httpx.Response(200, json={"result": {"category": "database", "confidence": 0.99}})

    transport = httpx.MockTransport(mock_handler)
    client = AsyncOmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.AsyncClient(base_url="https://mock.omniai.ai", transport=transport)

    res = await client.chat(model="openai:gpt-4o", prompt="Analyze", response_model=MockAnalysis)
    assert isinstance(res, MockAnalysis)
    assert res.category == "database"
    assert res.confidence == 0.99
    await client.close()


@pytest.mark.asyncio
async def test_async_streaming():
    def mock_handler(request: httpx.Request):
        sse_body = (
            'data: {"content": "Async "}\r\n\r\n'
            'data: {"content": "Token "}\r\n\r\n'
            'data: {"content": "Stream"}\r\n\r\n'
            "data: [DONE]\r\n\r\n"
        )
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(mock_handler)
    client = AsyncOmniaiClient(base_url="https://mock.omniai.ai", api_key="test-key")
    client._http = httpx.AsyncClient(base_url="https://mock.omniai.ai", transport=transport)

    chunks = []
    async for chunk in client.stream(model="openai:gpt-4o", prompt="Stream"):
        chunks.append(chunk)

    assert chunks == ["Async ", "Token ", "Stream"]
    await client.close()
