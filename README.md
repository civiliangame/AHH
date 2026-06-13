# Telnyx ↔ xAI Voice Agent Bridge

Scaffolding for a voice AI you can **call on the phone**. Telnyx streams the
call audio to this server over a WebSocket; the server relays it to xAI's
**Voice Agent API** and pipes Grok's spoken responses back to the caller.

```
Phone ──call──> Telnyx ──Media Streaming WS──> server.py (:5000 via ngrok) ──WS──> xAI Voice Agent API
                         G.711 μ-law 8kHz                                    audio/pcmu (no transcoding)
```

The xAI protocol here is verified against xAI's docs and their official
cookbook telephony example (which uses Twilio — Telnyx is the same protocol
with `streamSid` renamed to `stream_id`):
- https://docs.x.ai/developers/model-capabilities/audio/voice-agent
- https://github.com/xai-org/xai-cookbook/tree/main/voice-examples/agent/telephony/xai

## Files

| File | Purpose |
|------|---------|
| `server.py` | WebSocket **server** Telnyx connects to (:5000). Bridges each call to an xAI session. |
| `xai_client.py` | WebSocket **client** for the xAI Voice Agent API. |
| `config.py` | Loads settings from `.env`. |
| `audio.py` | Optional μ-law ↔ PCM16 transcoding. Unused in the telephony path (μ-law passes through). |
| `.env` | Your secrets (gitignored). |

## Run locally

```bash
pip install -r requirements.txt
python server.py            # listens on ws://0.0.0.0:5000
ngrok http 5000             # in another terminal -> https://<sub>.ngrok-free.app
```

## Point Telnyx at it

In Telnyx Mission Control, on inbound call start a Media Stream to your ngrok
URL using the **`wss://`** scheme:

- **TeXML**: `<Start><Stream url="wss://<sub>.ngrok-free.app" /></Start>`
- **Call Control**: `streaming_start` with `stream_url = wss://<sub>.ngrok-free.app`,
  `stream_track = "inbound_track"`.

Assign a number to the app and call it. You'll see `Telnyx connected` →
`Stream start` → `xAI session ready` in the logs, then Grok greets the caller.

## How the xAI handshake works (in `xai_client.py`)

Order matters — audio is rejected until the session is configured:

1. Open WS to `wss://api.x.ai/v1/realtime?model=grok-voice-latest` with
   `Authorization: Bearer <key>`.
2. Server sends `conversation.created` → we send `session.update` (voice,
   instructions, `audio.{input,output}.format.type = "audio/pcmu"`, server VAD).
3. Server sends `session.updated` → now safe to stream audio. We send a greeting
   (`conversation.item.create` + `response.create`) so the bot speaks first.

Server events handled:
- `response.output_audio.delta` → base64 μ-law audio → forwarded to Telnyx as `media`.
- `input_audio_buffer.speech_started` → **barge-in**: send Telnyx a `clear` to drop queued bot audio.
- `conversation.item.input_audio_transcription.completed` → caller transcript (logged).
- `response.output_audio_transcript.delta` → bot transcript (logged at DEBUG).
- `error` → logged.

Caller audio → xAI: `input_audio_buffer.append` with the base64 μ-law payload
straight from Telnyx (gated until the session is ready).

## Config (.env)

| Var | Default | Notes |
|-----|---------|-------|
| `XAI_REALTIME_MODEL` | `grok-voice-latest` | or pin `grok-voice-think-fast-1.0` |
| `XAI_VOICE` | `ara` | eve, ara, rex, sal, leo, or custom voice ID |
| `XAI_INSTRUCTIONS` | (persona) | system prompt |
| `XAI_GREETING` | greeting line | blank = wait for caller instead of speaking first |

## Adding tools (function calling)

xAI supports `tools` in `session.update` and emits
`response.output_item.done` with `item.type == "function_call"`. To wire tools,
add a `tools` array to the session config in `xai_client._configure_session`,
handle the `function_call` event in `iter_events`, run the function, and send
back `conversation.item.create` with `type: "function_call_output"` then
`response.create`. The cookbook `index.ts` has a full `generate_random_number`
example.

## Security

The xAI key was shared in plaintext during setup — **rotate it** at
console.x.ai. `.env` is gitignored; never commit real keys. For browser/mobile
clients use xAI **ephemeral tokens** instead of the raw key (not needed here —
this is server-side).
