# OmniAI Python Client SDK (`omniai`)

[![PyPI version](https://img.shields.io/pypi/v/omniai.svg)](https://pypi.org/project/omniai/)
[![Python versions](https://img.shields.io/pypi/pyversions/omniai.svg)](https://pypi.org/project/omniai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: PEP 561](https://img.shields.io/badge/typing-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

Official, production-ready Python client SDK for the **OmniAI AI Gateway** and **Dynamic Multi-Provider LLM Router**.

---

## 🏛️ Supported Providers

The OmniAI SDK is strictly engineered and validated for the following **4 model providers**:

| Provider | Identifier Schema | Description & Examples |
| :--- | :--- | :--- |
| **OpenAI** | `openai:<model>` | Standard and reasoning models (`openai:gpt-4o`, `openai:o3-mini`) |
| **Google Gemini** | `gemini:<model>` | Google AI Multimodal models (`gemini:gemini-2.5-flash`, `gemini:gemini-3.7-flash`) |
| **Anthropic** | `anthropic:<model>` | Claude reasoning & coding models (`anthropic:claude-3-5-sonnet`, `anthropic:claude-3-opus`) |
| **vLLM (Self-Hosted)** | `vllm:<model>` | High-throughput self-hosted inference servers (`vllm:meta-llama/Llama-3.1-8B-Instruct`) |

---

## 📦 Installation

Install the lightweight client from PyPI:

```bash
pip install omniai
```

### Optional Provider Dependencies (for direct local routing)
If you wish to use the standalone `RouterEngine` directly within your Python process without an external gateway:

```bash
pip install "omniai[all]"      # Installs all provider SDKs (openai, google-genai)
pip install "omniai[openai]"   # Installs openai
pip install "omniai[gemini]"   # Installs google-genai
```

---

## 🚀 Quickstart

### 1. Synchronous Client (`OmniaiClient`)

```python
from omniai import OmniaiClient

# Automatically defaults base_url="https://omniai.go-agents.app"
# and auto-loads OMNI_API_KEY from your .env file
client = OmniaiClient()

# Or pass parameters explicitly:
# client = OmniaiClient(api_key="your-api-key")

# Call flagship model openai:gpt-5.6-luna
response = client.chat(
    model="openai:gpt-5.6-luna",
    prompt="Explain quantum entanglement in 2 sentences.",
    temperature=0.7,
)
print(response)
```

### 2. Real-Time Token Streaming

The SDK features a resilient Server-Sent Events (SSE) streaming engine with automatic frame filtering and clean termination:

```python
from omniai import OmniaiClient

client = OmniaiClient()

for token in client.stream(
    model="openai:gpt-5.6-luna",
    prompt="Write a haiku about high-availability microservices.",
):
    print(token, end="", flush=True)
```

### 3. Strongly-Typed Structured Outputs with Pydantic

Pass your Pydantic model directly into `response_model`. The SDK extracts JSON schemas automatically, instructs the upstream provider, and parses the returned JSON into a validated Python instance:

```python
from pydantic import BaseModel, Field
from omniai import OmniaiClient

class SecurityAudit(BaseModel):
    summary: str = Field(..., description="High-level assessment")
    vulnerabilities: list[str] = Field(..., description="Discovered vulnerability IDs")
    passed: bool = Field(..., description="Overall security compliance status")

client = OmniaiClient()

audit: SecurityAudit = client.chat(
    model="openai:gpt-5.6-luna",
    prompt="Audit this configuration: CORS allowed * with credentials enabled.",
    response_model=SecurityAudit,
)

print(audit.summary)
print(f"Passed: {audit.passed}")
```

### 4. Self-Hosted vLLM Inference

Connect seamlessly to any self-hosted or cloud vLLM server:

```python
from omniai import OmniaiClient

# Point to your internal vLLM cluster (or route through OmniAI Gateway)
client = OmniaiClient(
    base_url="http://vllm-node.internal:8000/v1",
    api_key="EMPTY",
)

response = client.chat(
    model="vllm:meta-llama/Llama-3.1-8B-Instruct",
    prompt="Hello from OmniAI!",
)
print(response)
```

### 5. Asynchronous Client (`AsyncOmniaiClient`)

Full `async`/`await` parity for high-concurrency applications:

```python
import asyncio
from omniai import AsyncOmniaiClient

async def main():
    async with AsyncOmniaiClient() as client:
        # Async chat
        result = await client.chat(
            model="openai:gpt-5.6-luna",
            prompt="Hello async world!",
        )
        print(result)

        # Async streaming
        async for chunk in client.stream(model="openai:gpt-5.6-luna", prompt="Count to 3."):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### 6. Fault-Tolerant Background Celery Tasks

Queue long-running jobs to the background Celery cluster with automatic polling resilience:

```python
from omniai import OmniaiClient

client = OmniaiClient()

# Submit task asynchronously to the queue
job = client.submit_task(
    model="openai:gpt-5.6-luna",
    prompt="Analyze 50 pages of financial records.",
)
print(f"Dispatched task ID: {job.task_id}")

# Fault-tolerant polling that survives temporary network dropouts
result = job.wait(timeout=180.0, interval=2.0)
print(result)
```

### 7. LiveKit Multimodal Real-Time Voice & Video

Generate signed WebRTC credentials on the fly:

```python
from omniai import OmniaiClient

client = OmniaiClient()

session = client.create_live_session(
    model="gemini:gemini-3.1-flash-live-preview",
    voice="Aoede",
    participant_identity="agent-caller-01",
)
print(f"Connect to room '{session.room_name}' at {session.server_url}")
print(f"JWT Token: {session.participant_token}")
```

---

## 🛡️ Production Reliability & Fault Tolerance

The SDK includes enterprise resilience patterns built-in:

1. **Exponential Backoff with Full Jitter**:
   - Automatically retries transient failures: HTTP 429 (Rate Limit), 500, 502, 503, 504, connection timeouts, and connection reset errors.
   - Respects upstream `Retry-After` headers.
   - Configurable via `max_retries: int = 3` and `backoff_factor: float = 0.5`.

2. **Connection Pooling**:
   - Pre-configured with `httpx.Limits(max_keepalive_connections=20, max_connections=100)`.
   - Supports explicit lifecycle management via `client.close()` and context managers (`with OmniaiClient() as client:`).

3. **Granular Exception Hierarchy**:
   - `OmniaiAuthError`: HTTP 401 / 403 invalid or expired credentials.
   - `OmniaiPaymentRequiredError`: HTTP 402 insufficient prepaid balance.
   - `OmniaiRateLimitError`: HTTP 429 rate limit exceeded (includes `.retry_after` attribute).
   - `OmniaiExecutionError`: HTTP 5xx or upstream LLM provider error.
   - `OmniaiValidationError`: Model output failed schema validation.
   - `OmniaiTimeoutError`: Polling or network timeout.
   - `OmniaiConnectionError`: Network drops, DNS resolution failures.
   - `OmniaiUnsupportedProviderError`: Raised if attempting to route to an unsupported provider.


---

## 📄 License

MIT © [OmniAI Team](https://omniai.go-agents.app)
