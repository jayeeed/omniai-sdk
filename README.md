# OmniAI Kit (`omniai-kit`) 🚀

[![PyPI version](https://img.shields.io/pypi/v/omniai-kit.svg?color=blue)](https://pypi.org/project/omniai-kit/)
[![GitHub repository](https://img.shields.io/badge/GitHub-omniai--sdk-181717?logo=github)](https://github.com/jayeeed/omniai-sdk)
[![Python versions](https://img.shields.io/pypi/pyversions/omniai-kit.svg)](https://pypi.org/project/omniai-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: PEP 561](https://img.shields.io/badge/typing-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)
[![Status: Production/Stable](https://img.shields.io/badge/status-production-success.svg)](https://omniai.go-agents.app)

Official Python Client SDK for the **OmniAI Platform & AI Gateway** — providing zero-config multi-provider LLM routing, real-time token streaming, native Pydantic structured outputs, background task queues, and WebRTC multimodal live voice/video sessions.

---

## ⚡ Features

- **Zero-Configuration Setup:** Automatically loads credentials (`OMNIAI_API_KEY`, `OMNI_API_KEY`, `X_API_KEY`) and defaults to `https://omniai.go-agents.app`.
- **Modern Resource Architecture:** Clean OpenAI-style resource hierarchy (`client.chat.completions.create`, `client.models.list`, `client.tasks.submit`, `client.livekit.create_session`) plus ergonomic top-level shortcuts (`client.chat`, `client.stream`).
- **Synchronous & Asynchronous:** Identical API surfaces for both synchronous (`OmniaiClient`) and async (`AsyncOmniaiClient`) applications.
- **Strongly Typed Pydantic Schemas:** Pass Pydantic models directly to `response_model` for guaranteed schema validation and markdown JSON stripping.
- **Resilient Real-Time Streaming:** Server-Sent Events (SSE) streaming engine with auto-reconnect and keep-alive handling.
- **Enterprise Fault Tolerance:** Built-in exponential backoff with full jitter, honoring upstream `Retry-After` headers on 429 and 5xx errors.
- **Background Tasks:** Submit compute-heavy completions to the Celery background queue and poll reliably with `job.wait()`.
- **LiveKit Multimodal Sessions:** Generate signed WebRTC access tokens for ultra-low latency Gemini Live audio and video agents.

---

## 📦 Installation

```bash
pip install omniai-kit
```

### Optional Provider Dependencies (for direct standalone routing)
```bash
pip install "omniai-kit[all]"        # Installs all upstream provider SDKs
pip install "omniai-kit[openai]"     # Installs OpenAI SDK
pip install "omniai-kit[gemini]"     # Installs Google GenAI SDK
pip install "omniai-kit[anthropic]"  # Installs Anthropic SDK
```

---

## 🏛️ Supported Model Providers

OmniAI natively routes and normalizes parameters across the following providers:

| Provider | Schema | Example Models |
| :--- | :--- | :--- |
| **OpenAI** | `openai:<model>` | `openai:gpt-4o`, `openai:o3-mini`, `openai:gpt-4o-mini` |
| **Google Gemini** | `gemini:<model>` | `gemini:gemini-2.5-flash`, `gemini:gemini-3.7-flash` |
| **Anthropic** | `anthropic:<model>` | `anthropic:claude-3-5-sonnet-20241022`, `anthropic:claude-3-opus` |
| **vLLM (Self-Hosted)** | `vllm:<model>` | `vllm:meta-llama/Llama-3.1-8B-Instruct` |
| **OpenRouter** | `openrouter:<model>` | `openrouter:anthropic/claude-3.5-sonnet` |
| **Ollama (Local)** | `ollama:<model>` | `ollama:qwen2.5-coder:7b` |

---

## 🚀 Quickstart

### 1. Basic Chat Completion

```python
from omniai import OmniaiClient

# Automatically discovers OMNIAI_API_KEY from environment or .env
client = OmniaiClient()

response = client.chat(
    model="gemini:gemini-2.5-flash",
    prompt="Explain distributed event streaming in two concise sentences.",
)
print(response)
```

Both direct shortcut and resource styles are supported:
```python
# Shortcut style
response = client.chat(model="gemini:gemini-2.5-flash", prompt="Hello!")

# Resource hierarchy style (OpenAI-compatible)
response = client.chat.completions.create(model="gemini:gemini-2.5-flash", prompt="Hello!")
```

---

### 2. Real-Time Token Streaming

```python
from omniai import OmniaiClient

client = OmniaiClient()

print("Assistant: ", end="", flush=True)
for token in client.stream(
    model="gemini:gemini-2.5-flash",
    prompt="List 3 key architectural benefits of event-driven microservices.",
):
    print(token, end="", flush=True)
print()
```

---

### 3. Strongly Typed Structured Outputs (Pydantic V2)

Pass any Pydantic class to `response_model`. The SDK extracts JSON schema, enforces structured output from the model, and automatically parses and validates the response into your Python model:

```python
from typing import List
from pydantic import BaseModel, Field
from omniai import OmniaiClient

class ServiceArchitecture(BaseModel):
    service_name: str = Field(description="Name of the service")
    patterns: List[str] = Field(description="Design patterns applied")
    availability_sla: float = Field(description="Target SLA percentage")

client = OmniaiClient()

analysis: ServiceArchitecture = client.chat(
    model="gemini:gemini-2.5-flash",
    prompt="Design a high-throughput payment processing service.",
    response_model=ServiceArchitecture,
)

print(f"Service: {analysis.service_name}")
print(f"Patterns: {analysis.patterns}")
print(f"Target SLA: {analysis.availability_sla}%")
```

---

### 4. Asynchronous Client (`AsyncOmniaiClient`)

Full async/await parity for high-concurrency microservices, FastAPI, and async workers:

```python
import asyncio
from omniai import AsyncOmniaiClient

async def main():
    async with AsyncOmniaiClient() as client:
        # Async chat
        res = await client.chat(
            model="gemini:gemini-2.5-flash",
            prompt="Summarize CQRS in 20 words.",
        )
        print("Response:", res)

        # Async streaming
        print("Streaming: ", end="", flush=True)
        async for chunk in client.stream(
            model="gemini:gemini-2.5-flash",
            prompt="Count from 1 to 5.",
        ):
            print(chunk, end="", flush=True)
        print()

asyncio.run(main())
```

---

### 5. Background Task Queue (Celery Integration)

Queue long-running jobs to the background Celery cluster with fault-tolerant polling that survives transient network dropouts:

```python
from omniai import OmniaiClient
from pydantic import BaseModel

class ComprehensiveAudit(BaseModel):
    vulnerabilities: list[str]
    risk_score: int

client = OmniaiClient()

# 1. Submit task asynchronously
job = client.submit_task(
    model="gemini:gemini-2.5-flash",
    prompt="Perform in-depth threat modeling for an open S3 bucket with public access.",
    response_model=ComprehensiveAudit,
)
print(f"Dispatched Celery Task ID: {job.task_id}")

# 2. Poll until complete with automatic error-tolerance
result: ComprehensiveAudit = job.wait(timeout=120.0, interval=2.0)
print("Task finished successfully!")
print(f"Risk Score: {result.risk_score}")
```

---

### 6. LiveKit Multimodal Real-Time Voice & Video

Create WebRTC sessions for sub-second bidirectional voice and vision agents:

```python
from omniai import OmniaiClient

client = OmniaiClient()

session = client.create_live_session(
    model="gemini:gemini-2.5-flash",
    voice="Aoede",
    video_input=True,
)

print(f"LiveKit Server: {session.server_url}")
print(f"Room:           {session.room_name}")
print(f"Token:          {session.participant_token}")
```

---

### 7. Model Discovery & Catalog

Query available models dynamically from the gateway:

```python
from omniai import OmniaiClient

client = OmniaiClient()

models = client.models.list()
print(f"Total available models: {len(models.data)}")
for m in models.data[:5]:
    print(f"  • {m.id} (provider: {m.provider}, release: {m.release_date})")
```

---

## 🔑 Authentication & Configuration

`OmniaiClient` and `AsyncOmniaiClient` resolve configuration in the following order:

| Setting | 1. Explicit Argument | 2. Environment Variables (`.env`) | 3. Default Fallback |
| :--- | :--- | :--- | :--- |
| **`api_key`** | `OmniaiClient(api_key="...")` | `OMNIAI_API_KEY` &rarr; `OMNI_API_KEY` &rarr; `X_API_KEY` | `""` |
| **`base_url`** | `OmniaiClient(base_url="...")` | `OMNIAI_BASE_URL` &rarr; `OMNI_BASE_URL` &rarr; `GATEWAY_URL` | `"https://omniai.go-agents.app"` |
| **`timeout`** | `OmniaiClient(timeout=30.0)` | — | `60.0` seconds |
| **`max_retries`**| `OmniaiClient(max_retries=5)`| — | `3` attempts |

Create a `.env` file in your project directory:
```env
OMNIAI_API_KEY=sk-or-v1-your-api-key-here
```

---

## 🛡️ Exception Hierarchy

All SDK exceptions inherit from `OmniaiError`:

```text
OmniaiError
├── OmniaiAuthError             # HTTP 401 / 403 (Invalid or revoked API key)
├── OmniaiPaymentRequiredError  # HTTP 402 (Prepaid credit balance exhausted)
├── OmniaiNotFoundError         # HTTP 404 (Resource or model not found)
├── OmniaiRateLimitError        # HTTP 429 (Rate limit exceeded; includes .retry_after)
├── OmniaiTimeoutError          # Request or polling timeout
├── OmniaiConnectionError       # DNS, connection drop, network failure
├── OmniaiExecutionError        # HTTP 5xx or upstream provider execution failure
├── OmniaiValidationError      # Output failed Pydantic schema validation
└── OmniaiUnsupportedProviderError # Request for an unsupported provider prefix
```

---

## 📂 Repository Directory Pattern

```text
omniai-sdk/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Python 3.9 - 3.13 test matrix
│       └── publish.yml         # PyPI Trusted Publishing workflow
├── omniai/
│   ├── resources/              # Modular resource namespaces
│   │   ├── chat.py
│   │   ├── livekit.py
│   │   ├── models.py
│   │   ├── router.py
│   │   └── tasks.py
│   ├── types/                  # Typed Pydantic schemas
│   │   ├── chat.py
│   │   ├── livekit.py
│   │   ├── models.py
│   │   └── tasks.py
│   ├── __init__.py             # Top-level exports
│   ├── _base.py                # Base HTTP transport & retry policy
│   ├── _constants.py           # Default constants & endpoints
│   ├── _version.py             # Package version (1.0.0)
│   ├── client.py               # OmniaiClient & AsyncOmniaiClient
│   ├── exceptions.py           # Typed exception hierarchy
│   └── py.typed                # PEP 561 typing compliance
├── examples/
│   ├── quickstart.py
│   ├── streaming.py
│   ├── structured_output.py
│   └── async_chat.py
├── tests/                      # Unit and integration test suite
├── .env.example
├── .gitignore
├── LICENSE                     # MIT License
├── README.md                   # Documentation & Guides
├── pyproject.toml              # PEP 517/621 packaging metadata
├── requirements.txt            # Core dependencies (httpx, pydantic)
└── test.py                     # Root test script for quick validation
```

---

## 🧪 Testing

Run the test suite with `pytest`:

```bash
pytest tests
```

Or run the quick root validation script:

```bash
python test.py
```

---

## 📄 License

MIT © [OmniAI Team](https://omniai.go-agents.app) — Jayed Bin Jahangir
