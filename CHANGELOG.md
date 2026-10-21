# Changelog

All notable changes to the `omniai` Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-21

### Added
- **Unified Multi-Provider Routing:** Synchronous (`OmniaiClient`, `OmniAIClient`, `OmniClient`) and asynchronous (`AsyncOmniaiClient`, `AsyncOmniAIClient`, `AsyncOmniClient`) client interfaces.
- **Provider Protocol Handlers:** Native support for:
  - OpenAI (`openai:...`)
  - Google Gemini (`gemini:...`) with direct native execution and batch suffix normalization
  - Anthropic Claude (`anthropic:...`)
  - vLLM self-hosted private inference (`vllm:...`)
- **Real-Time Token Streaming:** Server-Sent Events (SSE) streaming engine with clean frame parsing and generator yield.
- **Strongly-Typed Structured Outputs:** Pydantic model validation with automatic JSON schema extraction and enforcement (`response_model=...`).
- **LiveKit Multimodal Real-Time:** Generation of signed WebRTC voice and video session tokens (`create_live_session`, `generate_livekit_token`).
- **Background Async Tasks:** Non-blocking job dispatch and polling resilience (`submit_task`, `TaskJob`, `AsyncTaskJob`).
- **Production Resilience:** Automatic exponential backoff with full jitter on HTTP 429, 500, 502, 503, 504 and connection drops. Respects `Retry-After` headers.
- **Type Checking:** PEP 561 compliance with `omniai/py.typed` marker.
- **Documentation & Tests:** Full test suite and comprehensive developer quickstart guide.
