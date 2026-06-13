"""Thin async client for xAI's realtime voice WebSocket API.

NOTE ON THE PROTOCOL
--------------------
xAI's public REST API is OpenAI-compatible, so this client is modeled on the
OpenAI Realtime API event shape (session.update / input_audio_buffer.append /
response.audio.delta / etc.). Confirm the exact endpoint, model id, voice names,
and event names against xAI's realtime docs and adjust the constants below +
the event names in `iter_events` / `append_audio` if they differ. The Telnyx
side of the bridge does not depend on any of these details.
"""
import asyncio
import base64
import json
import logging

import websockets

import config

log = logging.getLogger("xai")


class XAIRealtimeClient:
    def __init__(self):
        self._ws = None

    async def connect(self):
        url = f"{config.XAI_REALTIME_URL}?model={config.XAI_REALTIME_MODEL}"
        headers = {
            "Authorization": f"Bearer {config.XAI_API_KEY}",
            # Some realtime APIs require this beta header; harmless if ignored.
            "OpenAI-Beta": "realtime=v1",
        }
        log.info("Connecting to xAI realtime: %s", url)
        # websockets >=14 renamed extra_headers -> additional_headers.
        try:
            self._ws = await websockets.connect(
                url, additional_headers=headers, max_size=None
            )
        except TypeError:
            self._ws = await websockets.connect(
                url, extra_headers=headers, max_size=None
            )
        await self._configure_session()
        return self

    async def _configure_session(self):
        """Set audio formats + persona + server-side voice activity detection."""
        await self._send({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": config.XAI_INSTRUCTIONS,
                "voice": config.XAI_VOICE,
                "input_audio_format": config.TELNYX_AUDIO_FORMAT,   # g711_ulaw
                "output_audio_format": config.TELNYX_AUDIO_FORMAT,  # g711_ulaw
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500,
                },
            },
        })

    async def _send(self, payload: dict):
        await self._ws.send(json.dumps(payload))

    async def append_audio(self, ulaw_b64: str):
        """Push a chunk of caller audio (base64 μ-law) to xAI."""
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": ulaw_b64,
        })

    async def iter_events(self):
        """Yield ('audio', b64_str) or ('speech_started', None) or ('other', evt)."""
        async for raw in self._ws:
            try:
                evt = json.loads(raw)
            except (ValueError, TypeError):
                continue
            etype = evt.get("type", "")
            if etype in ("response.audio.delta", "response.output_audio.delta"):
                delta = evt.get("delta") or evt.get("audio")
                if delta:
                    yield "audio", delta
            elif etype == "input_audio_buffer.speech_started":
                # Caller started talking -> barge-in (interrupt the bot).
                yield "speech_started", None
            elif etype == "error":
                log.error("xAI error event: %s", evt)
                yield "error", evt
            else:
                yield "other", evt

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
