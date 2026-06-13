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

import agents
import config

log = logging.getLogger("xai")


class XAIRealtimeClient:
    def __init__(self, persona=None):
        self._ws = None
        self.ready = False  # True once session.updated arrives
        self._needs_followup = False  # tool output submitted; respond on done
        self.persona = persona or agents.TRIAGE  # which agent this call runs

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
                "instructions": self.persona.instructions,
                "voice": self.persona.voice,
                "audio": {
                    "input": {"format": {"type": config.XAI_AUDIO_FORMAT}},
                    "output": {"format": {"type": config.XAI_AUDIO_FORMAT}},
                },
                "turn_detection": {
                    "type": "server_vad",
                    # Wait this long after the caller stops before ending their
                    # turn. Higher = fewer mid-sentence splits (they can pause to
                    # think), at the cost of a bit more response latency.
                    "silence_duration_ms": 800,
                    "prefix_padding_ms": 300,
                },
                "tools": self.persona.tools,
            },
        })

    async def send_function_result(self, call_id: str, result: dict):
        """Submit one tool result. Does NOT ask for a response yet.

        A single model turn can contain multiple function calls. We submit each
        output as it arrives, then trigger exactly one `response.create` when the
        turn finishes (`response.done`). Triggering a response per tool call
        makes the model speak once per call — the double-reply bug.
        """
        if self._ws is None:
            return
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        })
        self._needs_followup = True

    async def send_function_output(self, call_id: str, result: dict):
        """Submit a tool result WITHOUT asking the model to respond.

        Unlike send_function_result, this does NOT set _needs_followup, so the
        model stays silent on response.done. Use it when you drive the next turn
        yourself via force_message (the recordSymptom flow) instead of letting
        the model speak.
        """
        if self._ws is None:
            return
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        })

    async def force_message(self, text: str, interruptible: bool = False, voice=None):
        """Speak a hard-coded line verbatim via TTS, with no model involvement.

        The force_message IS the turn — do NOT follow it with response.create.
        It emits its own response.created -> output_audio.delta -> response.done
        lifecycle, so the existing audio relay forwards it to the caller as-is.
        """
        if self._ws is None or not text:
            return
        item = {
            "type": "force_message",
            "role": "assistant",
            "interruptible": interruptible,
            "content": [{"type": "output_text", "text": text}],
        }
        if voice:
            item["voice"] = voice
        await self._send({"type": "conversation.item.create", "item": item})

    async def greet(self):
        """Have Grok speak the opening line in its own voice.

        Triggered as a one-off response instruction (not a fake user turn) so
        the model says the line once and stops, instead of echoing it and then
        adding its own continuation.
        """
        greeting = self.persona.greeting
        if not greeting or self._ws is None:
            return
        await self._send({
            "type": "response.create",
            "response": {
                "instructions": (
                    f'Open the call by greeting the caller with this line, then '
                    f'stop and wait for them to respond: "{greeting}"'
                ),
            },
        })

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
                # Ready to stream audio. The greeting is triggered by the
                # server (via greet()) once Telnyx's stream is also up.
                self.ready = True
                yield "ready", None
            elif etype == "response.output_audio.delta":
                # Relay all agent audio to the caller — both the model's spoken
                # empathy and force_message lines (e.g. the next question).
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
            elif etype == "response.function_call_arguments.done":
                # evt carries: name, call_id, arguments (JSON string).
                yield "function_call", evt
            elif etype == "response.created":
                yield "response_created", evt.get("response", {}).get("id")
            elif etype == "response.done":
                # The turn finished. If we submitted any tool outputs during it,
                # ask for exactly one follow-up response (regardless of how many
                # tools were called) so the model speaks once, not per-tool.
                if self._needs_followup:
                    self._needs_followup = False
                    await self._send({"type": "response.create"})
                # Surface the finished response's id (the recordSymptom flow
                # gates on this to send the next question after empathy plays).
                yield "response_done", evt.get("response", {}).get("id")
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
