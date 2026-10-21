"""OmniAI SDK Streaming Example."""

from omniai import OmniaiClient

client = OmniaiClient()

print("Streaming real-time tokens:")
for chunk in client.stream(
    model="gemini:gemini-2.5-flash",
    prompt="List 3 best practices for building production microservices.",
):
    print(chunk, end="", flush=True)

print()
