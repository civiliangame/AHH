# Telnyx ↔ xAI Voice Agent Bridge

A voice AI you can **call on the phone**. Telnyx (Call Control) answers the
call, streams the audio to this server, and this server bridges it to xAI's
**Voice Agent API** so Grok talks to your caller in real time.

```
Phone ──call──> Telnyx ──webhook POST /webhook──> server.py answers + starts
                                                   bidirectional streaming (rtp/PCMU)
            └── Telnyx opens WS ──> /media-stream ──> bridge ──WS──> xAI Voice Agent
                         G.711 μ-law 8kHz                       audio/pcmu (no transcoding)
```

Verified against:
- xAI: https://docs.x.ai/developers/model-capabilities/audio/voice-agent + cookbook telephony example
- Telnyx: https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming

## Files

| File | Purpose |
|------|---------|
| `server.py` | aiohttp server (:5000). `POST /webhook` (Telnyx Call Control) + `GET /media-stream` (audio bridge). |
| `xai_client.py` | WebSocket client for the xAI Voice Agent API. |
| `config.py` | Loads settings from `.env`. |
| `audio.py` | Optional μ-law ↔ PCM16 transcoding. Unused — μ-law passes through. |
| `.env` | Secrets, gitignored. |

## Run

```bash
pip install -r requirements.txt
python server.py            # http+ws on :5000
ngrok http 5000             # -> https://<sub>.ngrok-free.app
```

Optionally set `PUBLIC_HOSTNAME=<sub>.ngrok-free.app` in `.env` so the media
stream URL is pinned (otherwise it's derived from the webhook's Host header).

## Telnyx setup (Call Control / Voice API)

1. portal.telnyx.com → **Voice → Call Control / Voice API Applications** → your app.
2. Set the **webhook URL** to `https://<sub>.ngrok-free.app/webhook` (HTTP POST).
3. Assign a phone number to the application.
4. Call the number. Logs show:
   `Webhook: call.initiated` → `Telnyx answer ok` → `Telnyx media stream connected`
   → `xAI session ready`, then Grok greets the caller.

What the server does on an incoming call: issues the Call Control **`answer`**
command with streaming params, which answers the call *and* starts a
bidirectional `rtp` / `PCMU` media stream pointed at `wss://<host>/media-stream`.

## Audio path

- **Caller → xAI:** Telnyx `media` events (base64 μ-law) → xAI `input_audio_buffer.append`.
- **xAI → caller:** xAI `response.output_audio.delta` (base64 μ-law) → Telnyx `{"event":"media","media":{"payload":...}}`.
- **Barge-in:** xAI `input_audio_buffer.speech_started` → Telnyx `{"event":"clear"}` to flush queued bot audio.

xAI handshake (in `xai_client.py`): open WS → `conversation.created` → send
`session.update` (voice, instructions, `audio.{input,output}.format.type="audio/pcmu"`,
server VAD) → `session.updated` → stream audio + send greeting.

## Config (.env)

| Var | Notes |
|-----|-------|
| `XAI_API_KEY` | xAI key (console.x.ai) |
| `XAI_REALTIME_MODEL` | `grok-voice-latest` or pin `grok-voice-think-fast-1.0` |
| `XAI_VOICE` | eve, ara, rex, sal, leo, or custom voice ID |
| `XAI_INSTRUCTIONS` / `XAI_GREETING` | persona / first line (blank = wait for caller) |
| `TELNYX_API_KEY` | Telnyx Call Control key (`KEY...`) |
| `PUBLIC_HOSTNAME` | ngrok domain, no scheme; blank = derive from webhook |
| `STREAM_PATH` | media WS path (default `/media-stream`) |

## Verified locally

- `GET /health` → 200.
- `POST /webhook` with `call.initiated` → issues Telnyx `answer`; the key
  authenticates (real calls get a valid call_control_id).
- `/media-stream` WS → connects to xAI, configures the session, and relays
  Grok's spoken greeting back as μ-law frames.

Not yet tested with a real phone call — the field names match Telnyx's docs,
but the first live call is the real confirmation.
