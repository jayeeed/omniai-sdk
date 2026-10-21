#!/usr/bin/env python3
"""
OmniAI SDK Verification Script.

Usage:
    python test.py
"""

import os
import sys

# Ensure local omniai package is on path if running directly from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import omniai
    from omniai import (
        OmniaiClient,
        AsyncOmniaiClient,
        RouterEngine,
        RetryPolicy,
        __version__,
    )
    from omniai.types import LiveSessionInfo, TaskStatus, ModelList
    from omniai.exceptions import OmniaiError, OmniaiAuthError
except ImportError as e:
    print(f"❌ Failed to import omniai SDK: {e}")
    sys.exit(1)


def main():
    print("=" * 60)
    print(f"🧪 Testing OmniAI Python SDK v{__version__}")
    print("=" * 60)

    # 1. Verify Router Engine
    router = RouterEngine()
    print(f"✅ RouterEngine initialized: {type(router).__name__}")
    print(f"✅ Supported Providers: {router.SUPPORTED_PROVIDERS}")

    provider, model = RouterEngine.parse_provider("gemini:gemini-2.5-flash")
    print(f"✅ Provider parsing: gemini:gemini-2.5-flash -> {provider}:{model}")

    # 2. Verify Retry Policy
    policy = RetryPolicy(max_retries=3, backoff_factor=0.5)
    print(f"✅ RetryPolicy initialized: max_retries={policy.max_retries}")

    # 3. Discover API Key
    api_key = (
        os.getenv("OMNIAI_API_KEY")
        or os.getenv("OMNI_API_KEY")
        or os.getenv("X_API_KEY")
    )

    # Check for monorepo .env fallback if testing locally
    if not api_key:
        local_env = os.path.join(os.path.dirname(__file__), "../omniai/.env")
        if os.path.exists(local_env):
            with open(local_env) as f:
                for line in f:
                    if line.startswith("X_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    client = OmniaiClient(api_key=api_key)
    print(f"✅ OmniaiClient initialized: {client}")

    async_client = AsyncOmniaiClient(api_key=api_key)
    print(f"✅ AsyncOmniaiClient initialized: {async_client}")

    # 4. Optional Live Chat Completion if API Key is Present
    if api_key:
        print("\n📡 Testing live chat completion against OmniAI Gateway...")
        try:
            res = client.chat(
                model="gemini:gemini-2.5-flash",
                prompt="Say 'OmniAI SDK verified!' in exactly 3 words.",
            )
            print(f"🎉 Live Response: {res}")
        except Exception as err:
            print(f"⚠️ Live chat completion returned error: {type(err).__name__}: {err}")
    else:
        print("\nℹ️ No API key detected in environment. Live completion skipped.")
        print("   Set OMNIAI_API_KEY to test live calls against the platform.")

    print("\n" + "=" * 60)
    print("✅ All OmniAI SDK structure and import tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
