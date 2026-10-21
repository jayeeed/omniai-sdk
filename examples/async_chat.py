"""OmniAI SDK Asynchronous Chat Example."""

import asyncio
from omniai import AsyncOmniaiClient


async def main():
    async with AsyncOmniaiClient() as client:
        response = await client.chat(
            model="gemini:gemini-2.5-flash",
            prompt="What are the three pillars of observability in software engineering?",
        )
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
