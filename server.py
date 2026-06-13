"""Telnyx (Call Control) <-> xAI Voice Agent bridge — FastAPI app.

One server on :5000 does two jobs:

  POST  /webhook        Telnyx Call Control webhooks (JSON). On an incoming
                        call we answer it AND start bidirectional media
                        streaming (rtp / PCMU) pointed back at /media-stream.
  WS    /media-stream   Telnyx opens this WebSocket and streams call audio.
                        We bridge it to an xAI Voice Agent session and relay
                        Grok's audio back to the caller.

Audio is G.711 μ-law (PCMU) @ 8 kHz end to end — no transcoding.

Run:  python server.py      # auto-reloads on file save
      (or: uvicorn server:app --reload --host 0.0.0.0 --port 5000)
Point your Telnyx Call Control app's webhook at https://<ngrok>/webhook
"""
import asyncio
import json
import logging

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

import config
import triage
from xai_client import XAIRealtimeClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")

app = FastAPI()


# ----------------------------------------------------------------------------
# Telnyx Call Control REST helper
# ----------------------------------------------------------------------------
async def telnyx_command(call_control_id: str, action: str, body: dict):
    """POST a Call Control action, e.g. answer / streaming_start / hangup."""
    if not config.TELNYX_API_KEY:
        log.error("TELNYX_API_KEY not set — cannot issue '%s'", action)
        return
    url = f"{config.TELNYX_API_BASE}/calls/{call_control_id}/actions/{action}"
    headers = {"Authorization": f"Bearer {config.TELNYX_API_KEY}"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 300:
            log.error("Telnyx %s failed (%s): %s", action, resp.status_code, resp.text)
        else:
            log.info("Telnyx %s ok (%s)", action, resp.status_code)


def stream_url_for(request: Request) -> str:
    """wss:// URL Telnyx should connect the media stream to."""
    host = config.PUBLIC_HOSTNAME or request.headers.get("host", "")
    return f"wss://{host}{config.STREAM_PATH}"


# ----------------------------------------------------------------------------
# HTTP: Telnyx Call Control webhook
# ----------------------------------------------------------------------------
@app.post("/webhook")
@app.post("/")
async def handle_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        log.warning("Webhook with non-JSON body")
        return Response(status_code=200)

    data = body.get("data", {})
    event_type = data.get("event_type", "")
    payload = data.get("payload", {})
    ccid = payload.get("call_control_id")
    log.info("Webhook: %s", event_type)

    if event_type == "call.initiated" and payload.get("direction") == "incoming":
        # Answer AND start bidirectional streaming in one command.
        await telnyx_command(ccid, "answer", {
            "stream_url": stream_url_for(request),
            "stream_track": "inbound_track",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
        })
    # Other events (call.answered / streaming.* / call.hangup): just ack 2xx.
    return Response(status_code=200)


@app.get("/health")
async def handle_health():
    return JSONResponse({"status": "ok"})


# ----------------------------------------------------------------------------
# WS: Telnyx media stream  <->  xAI Voice Agent
# ----------------------------------------------------------------------------
@app.websocket(config.STREAM_PATH)
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("Telnyx media stream connected")

    xai = XAIRealtimeClient()
    try:
        await xai.connect()
    except Exception:
        log.exception("Could not connect to xAI; closing call")
        await ws.close()
        return

    # Have Grok speak the greeting once BOTH the Telnyx stream and the xAI
    # session are ready — triggering it before Telnyx's `start` would clip the
    # opening audio. Either side may become ready first, so both paths call this.
    greet = {"done": False, "telnyx_started": False}

    async def maybe_greet():
        if greet["done"] or not greet["telnyx_started"] or not xai.ready:
            return
        greet["done"] = True
        await xai.greet()
        log.info("Triggered greeting")

    async def pump_xai_to_telnyx():
        """Drive the xAI session and relay its output to the caller."""
        try:
            async for kind, data in xai.iter_events():
                if kind == "audio":
                    # Bidirectional Telnyx: media frame, no stream_id needed.
                    await ws.send_text(json.dumps(
                        {"event": "media", "media": {"payload": data}}))
                elif kind == "speech_started":
                    # Barge-in: caller interrupted -> flush queued bot audio.
                    await ws.send_text(json.dumps({"event": "clear"}))
                elif kind == "function_call":
                    # Model invoked a tool. Run it and return the result so the
                    # model can continue. arguments arrives as a JSON string.
                    args = json.loads(data.get("arguments") or "{}")
                    result = triage.handle(data["name"], args)
                    log.info("Tool %s -> %s", data["name"], result)
                    await xai.send_function_result(data["call_id"], result)
                elif kind == "ready":
                    log.info("xAI session ready")
                    await maybe_greet()
                elif kind == "user_transcript":
                    log.info("Caller: %s", data)
                elif kind == "bot_transcript":
                    log.debug("Bot: %s", data)
        except Exception:
            log.exception("xAI->Telnyx pump failed")

    pump_task = asyncio.create_task(pump_xai_to_telnyx())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                m = json.loads(raw)
            except (ValueError, TypeError):
                continue

            event = m.get("event")
            if event == "start":
                fmt = m.get("start", {}).get("media_format", {})
                log.info("Stream start: id=%s format=%s", m.get("stream_id"), fmt)
                greet["telnyx_started"] = True
                await maybe_greet()
            elif event == "media":
                media = m.get("media", {})
                # On bidirectional streams Telnyx sends only inbound by default.
                if media.get("track", "inbound") == "inbound" and media.get("payload"):
                    await xai.append_audio(media["payload"])
            elif event == "stop":
                log.info("Telnyx stream stopped")
                break
            elif event == "dtmf":
                log.info("DTMF: %s", m.get("dtmf"))
    except WebSocketDisconnect:
        log.info("Telnyx disconnected")
    finally:
        pump_task.cancel()
        await xai.close()
        log.info("Call cleaned up")


if __name__ == "__main__":
    import uvicorn

    if not config.TELNYX_API_KEY:
        log.warning("TELNYX_API_KEY is empty — calls won't be answered. Set it in .env")
    # reload=True watches .py files and restarts on save (dev mode).
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )
