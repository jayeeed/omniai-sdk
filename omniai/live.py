from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, Optional
from omniai.models import LiveSessionInfo


def generate_livekit_token(
    api_key: str,
    api_secret: str,
    identity: str,
    name: Optional[str] = None,
    room_name: Optional[str] = None,
    metadata: Optional[str] = None,
    ttl_seconds: int = 86400,
) -> str:
    """Generates a signed LiveKit JWT access token with fallback to HMAC-SHA256."""
    try:
        from livekit import api

        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(name or identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
        )
        if metadata:
            token.with_metadata(metadata)
        return token.to_jwt()
    except ImportError:
        # High-reliability fallback without livekit dependency using standard library
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload: Dict[str, Any] = {
            "iss": api_key,
            "sub": identity,
            "name": name or identity,
            "nbf": now - 10,
            "exp": now + ttl_seconds,
            "video": {
                "roomJoin": True,
                "room": room_name,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            },
        }
        if metadata:
            payload["metadata"] = metadata

        def b64_encode(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

        json_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        json_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        segments = [b64_encode(json_header), b64_encode(json_payload)]
        signing_input = ".".join(segments).encode("utf-8")
        signature = hmac.new(api_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        segments.append(b64_encode(signature))

        return ".".join(segments)


def create_live_session(
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    model: str = "gemini:gemini-3.1-flash-live-preview",
    voice: str = "Aoede",
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
    video_input: bool = False,
    room_name: Optional[str] = None,
    participant_identity: Optional[str] = None,
    participant_name: Optional[str] = None,
) -> LiveSessionInfo:
    """Creates a local or cloud LiveKit session with signed JWT credentials."""
    url = (
        server_url
        or os.getenv("LIVEKIT_URL")
        or os.getenv("LIVEKIT_WS_URL")
        or "wss://omniai.go-agents.app/livekit"
    )
    key = api_key or os.getenv("LIVEKIT_API_KEY") or "devkey"
    secret = api_secret or os.getenv("LIVEKIT_API_SECRET") or "secret"

    resolved_room = room_name or f"omni-live-{uuid.uuid4().hex[:8]}"
    resolved_identity = participant_identity or f"user-{uuid.uuid4().hex[:6]}"
    resolved_name = participant_name or resolved_identity

    metadata_dict = {
        "model": model,
        "voice": voice,
        "temperature": temperature,
        "video_input": video_input,
    }
    if system_instruction:
        metadata_dict["system_instruction"] = system_instruction

    token = generate_livekit_token(
        api_key=key,
        api_secret=secret,
        identity=resolved_identity,
        name=resolved_name,
        room_name=resolved_room,
        metadata=json.dumps(metadata_dict),
    )

    return LiveSessionInfo(
        server_url=url,
        room_name=resolved_room,
        participant_token=token,
        participant_identity=resolved_identity,
        model_used=model,
        voice=voice,
        video_input=video_input,
    )
