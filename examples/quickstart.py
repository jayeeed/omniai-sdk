"""OmniAI SDK Quickstart Example."""

import os
from omniai import OmniaiClient

# Initialize client (automatically reads OMNIAI_API_KEY from environment)
client = OmniaiClient()

response = client.chat(
    model="gemini:gemini-2.5-flash",
    prompt="Explain the difference between synchronous and asynchronous programming in one sentence.",
)

print(response)
