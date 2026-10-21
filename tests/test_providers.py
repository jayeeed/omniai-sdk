import pytest
from omniai import OmniaiClient, AsyncOmniaiClient, RouterEngine
from omniai.exceptions import OmniaiUnsupportedProviderError


def test_supported_providers_whitelisted():
    assert RouterEngine.SUPPORTED_PROVIDERS == (
        "openai", "gemini", "anthropic", "vllm", "openrouter", "ollama", "custom", "local"
    )


def test_parse_valid_providers():
    # OpenAI
    prov, model = RouterEngine.parse_provider("openai:gpt-4o")
    assert prov == "openai"
    assert model == "gpt-4o"

    # Gemini
    prov, model = RouterEngine.parse_provider("gemini:gemini-2.5-flash")
    assert prov == "gemini"
    assert model == "gemini-2.5-flash"

    # Anthropic
    prov, model = RouterEngine.parse_provider("anthropic:claude-3-5-sonnet")
    assert prov == "anthropic"
    assert model == "claude-3-5-sonnet"

    # vLLM
    prov, model = RouterEngine.parse_provider("vllm:meta-llama/Llama-3.1-8B-Instruct")
    assert prov == "vllm"
    assert model == "meta-llama/Llama-3.1-8B-Instruct"

    # OpenRouter
    prov, model = RouterEngine.parse_provider("openrouter:meta-llama/llama-3")
    assert prov == "openrouter"
    assert model == "meta-llama/llama-3"


def test_implicit_provider_parsing():
    assert RouterEngine.parse_provider("gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert RouterEngine.parse_provider("claude-3-5-sonnet") == ("anthropic", "claude-3-5-sonnet")
    assert RouterEngine.parse_provider("gpt-4o") == ("openai", "gpt-4o")


def test_unsupported_providers_rejected():
    unsupported = [
        "replicate:meta-llama",
        "mistral:mistral-large",
        "cohere:command-r-plus",
        "groq:llama3-70b",
        "deepseek:deepseek-chat",
        "bedrock:anthropic.claude",
    ]

    for model_str in unsupported:
        with pytest.raises(OmniaiUnsupportedProviderError) as exc_info:
            RouterEngine.parse_provider(model_str)
        assert "Unsupported provider" in str(exc_info.value)
        assert "The OmniAI SDK strictly supports: openai, gemini, anthropic, vllm, openrouter" in str(exc_info.value)


def test_client_rejects_unsupported_provider_early():
    client = OmniaiClient(api_key="test-key")

    with pytest.raises(OmniaiUnsupportedProviderError) as exc_info:
        client.chat(model="replicate:meta-llama", prompt="test")
    assert "Unsupported model provider 'replicate'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_client_rejects_unsupported_provider_early():
    client = AsyncOmniaiClient(api_key="test-key")

    with pytest.raises(OmniaiUnsupportedProviderError) as exc_info:
        await client.chat(model="cohere:command-r-plus", prompt="test")
    assert "Unsupported model provider 'cohere'" in str(exc_info.value)
