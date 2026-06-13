# Telnyx ↔ xAI Voice Bridge

Scaffolding for a voice AI you can **call on the phone**. Telnyx streams the
call audio to this server over a WebSocket; the server relays it to xAI's
realtime voice API and pipes the spoken responses back to the caller.

```
Phone ──call──> Telnyx ──Media Streaming WS──> server.py (:5000 via ngrok) ──WS──> xAI Realtime Voice
                         G.711 μ-law 8kHz                                    g711_ulaw (no transcoding)
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | WebSocket **server** Telnyx connects to (:5000). Bridges each call to an xAI session. |
| `xai_client.py` | WebSocket **client** for xAI's realtime voice API. |
| `config.py` | Loads settings from `.env`. |
| `audio.py` | Optional μ-law ↔ PCM16 transcoding (only if xAI needs PCM16 instead of μ-law). |
| `.env` | Your secrets (gitignored). |

## Run locally

```bash
pip install -r requirements.txt
python server.py            # listens on ws://0.0.0.0:5000
```

In another terminal, expose it (you said ngrok is already pointed at 5000):

```bash
ngrok http 5000             # -> https://<sub>.ngrok-free.app
```

## Point Telnyx at it

In your Telnyx Mission Control portal:

1. **Voice API / Call Control** app, or a **TeXML** application.
2. On inbound call, start media streaming to your ngrok URL using the **`wss://`**
   scheme (not `https://`), e.g. `wss://<sub>.ngrok-free.app`.
   - TeXML: use a `<Stream>` verb inside `<Start>`/`<Connect>`.
   - Call Control: issue the `streaming_start` command with `stream_url` = your `wss://` URL,
     `stream_track = "inbound_track"` (or `both_tracks`).
3. Assign a phone number to the application and call it.

You'll see `Telnyx connected` → `Stream start` in the server logs, then audio flows.

## ⚠️ Verify the xAI side against xAI's docs

The Telnyx half is implemented to spec. The xAI half is modeled on the
**OpenAI Realtime API** event shape (xAI's REST API is OpenAI-compatible, and a
test handshake to `wss://api.x.ai/v1/realtime` succeeded). Confirm and adjust
these in `.env` / `xai_client.py` if xAI's realtime spec differs:

- **Endpoint** `XAI_REALTIME_URL` — `wss://api.x.ai/v1/realtime`
- **Model** `XAI_REALTIME_MODEL` — `grok-realtime` *(placeholder — set the real id)*
- **Voice** `XAI_VOICE` — `ara` *(placeholder)*
- **Event names** in `xai_client.py`: `session.update`, `input_audio_buffer.append`,
  `response.audio.delta`, `input_audio_buffer.speech_started`.
- **Audio format** — we request `g711_ulaw` both ways to match Telnyx and skip
  transcoding. If xAI only supports PCM16/24kHz, switch the formats in
  `_configure_session()` and transcode with `audio.py` in `server.py`.

## Features already wired

- One xAI session per call, cleaned up on hangup.
- **Barge-in**: when xAI detects the caller speaking (`speech_started`), the
  server sends Telnyx a `clear` to stop the bot's queued audio.
- Server-side VAD turn detection configured in the xAI session.

## Security

The xAI key was shared in plaintext during setup — **rotate it** at
console.x.ai. `.env` is gitignored; never commit real keys.
