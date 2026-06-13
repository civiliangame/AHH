"""Async client for xAI's Voice Agent API (realtime WebSocket).

Protocol verified against xAI's docs + cookbook telephony example:
https://docs.x.ai/developers/model-capabilities/audio/voice-agent
https://github.com/xai-org/xai-cookbook/tree/main/voice-examples/agent/telephony/xai

Connection handshake (order matters):
  1. open WS to wss://api.x.ai/v1/realtime?model=...  (Bearer auth header)
  2. server sends `conversation.created`  -> we send `session.update` (config)
  3. server sends `session.updated`       -> NOW safe to stream audio;
                                              optionally send a greeting.
Audio in/out is base64 G.711 μ-law (audio/pcmu @ 8 kHz) — same as Telnyx,
so no transcoding.
"""
import json
import logging

import websockets

import config

log = logging.getLogger("xai")


class XAIRealtimeClient:
    def __init__(self):
        self._ws = None
        self.ready = False  # True once session.updated arrives

    async def connect(self):
        url = f"{config.XAI_REALTIME_URL}?model={config.XAI_REALTIME_MODEL}"
        headers = {"Authorization": f"Bearer {config.XAI_API_KEY}"}
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
        return self

    async def _send(self, payload: dict):
        await self._ws.send(json.dumps(payload))

    async def _configure_session(self):
        """Set persona, voice, μ-law audio format, and server-side VAD."""
        await self._send({
            "type": "session.update",
            "session": {
                "instructions": config.XAI_INSTRUCTIONS,
                "voice": config.XAI_VOICE,
                "audio": {
                    "input": {"format": {"type": config.XAI_AUDIO_FORMAT}},
                    "output": {"format": {"type": config.XAI_AUDIO_FORMAT}},
                },
                "turn_detection": {"type": "server_vad"},
            },
        })

    async def _greet(self):
        """Make the bot speak first (inbound call greeting)."""
        if not config.XAI_GREETING:
            return
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": config.XAI_GREETING}],
            },
        })
        await self._send({"type": "response.create"})

    async def append_audio(self, ulaw_b64: str):
        """Push a chunk of caller audio (base64 μ-law) to xAI. No-op until ready."""
        if not self.ready or self._ws is None:
            return
        await self._send({"type": "input_audio_buffer.append", "audio": ulaw_b64})

    async def iter_events(self):
        """Drive the xAI protocol and yield high-level (kind, data) tuples:
        ('audio', b64) ('speech_started', None) ('ready', None)
        ('user_transcript', str) ('bot_transcript', str) ('error', evt)
        """
        async for raw in self._ws:
            try:
                evt = json.loads(raw)
            except (ValueError, TypeError):
                continue
            etype = evt.get("type", "")

            if etype == "conversation.created":
                await self._configure_session()
            elif etype == "session.updated":
                self.ready = True
                await self._greet()
                yield "ready", None
            elif etype == "response.output_audio.delta":
                delta = evt.get("delta")
                if delta:
                    yield "audio", delta
            elif etype == "input_audio_buffer.speech_started":
                # Caller started talking -> barge-in (interrupt the bot).
                yield "speech_started", None
            elif etype == "conversation.item.input_audio_transcription.completed":
                if evt.get("transcript"):
                    yield "user_transcript", evt["transcript"]
            elif etype == "response.output_audio_transcript.delta":
                if evt.get("delta"):
                    yield "bot_transcript", evt["delta"]
            elif etype == "error":
                log.error("xAI error event: %s", evt.get("error") or evt)
                yield "error", evt
            else:
                log.debug("xAI event: %s", etype)

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            self.ready = False
