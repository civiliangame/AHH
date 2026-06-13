"""Telnyx (Call Control) <-> xAI Voice Agent bridge.

One aiohttp server on :5000 does two jobs:

  POST  /webhook        Telnyx Call Control webhooks (JSON). On an incoming
                        call we answer it AND start bidirectional media
                        streaming (rtp / PCMU) pointed back at /media-stream.
  WS    /media-stream   Telnyx opens this WebSocket and streams call audio.
                        We bridge it to an xAI Voice Agent session and relay
                        Grok's audio back to the caller.

Audio is G.711 μ-law (PCMU) @ 8 kHz end to end — no transcoding.

Run:  python server.py   (then `ngrok http 5000`)
Point your Telnyx Call Control app's webhook at https://<ngrok>/webhook
"""
import asyncio
import json
import logging

import aiohttp
from aiohttp import web, WSMsgType

import config
from xai_client import XAIRealtimeClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")


# ----------------------------------------------------------------------------
# Telnyx Call Control REST helper
# ----------------------------------------------------------------------------
async def telnyx_command(call_control_id: str, action: str, body: dict):
    """POST a Call Control action, e.g. answer / streaming_start / hangup."""
    if not config.TELNYX_API_KEY:
        log.error("TELNYX_API_KEY not set — cannot issue '%s'", action)
        return
    url = f"{config.TELNYX_API_BASE}/calls/{call_control_id}/actions/{action}"
    headers = {
        "Authorization": f"Bearer {config.TELNYX_API_KEY}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as resp:
            text = await resp.text()
            if resp.status >= 300:
                log.error("Telnyx %s failed (%s): %s", action, resp.status, text)
            else:
                log.info("Telnyx %s ok (%s)", action, resp.status)


def stream_url_for(request: web.Request) -> str:
    """wss:// URL Telnyx should connect the media stream to."""
    host = config.PUBLIC_HOSTNAME or request.host
    return f"wss://{host}{config.STREAM_PATH}"


# ----------------------------------------------------------------------------
# HTTP: Telnyx Call Control webhook
# ----------------------------------------------------------------------------
async def handle_webhook(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        log.warning("Webhook with non-JSON body")
        return web.Response(status=200)

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
    # call.answered / streaming.started / streaming.stopped / call.hangup:
    # nothing to do — just ack. (Telnyx requires a 2xx.)
    return web.Response(status=200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ----------------------------------------------------------------------------
# WS: Telnyx media stream  <->  xAI Voice Agent
# ----------------------------------------------------------------------------
async def handle_media_stream(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    log.info("Telnyx media stream connected")

    xai = XAIRealtimeClient()
    try:
        await xai.connect()
    except Exception:
        log.exception("Could not connect to xAI; closing call")
        await ws.close()
        return ws

    async def pump_xai_to_telnyx():
        """Drive the xAI session and relay its output to the caller."""
        try:
            async for kind, data in xai.iter_events():
                if kind == "audio":
                    # Bidirectional Telnyx: media frame, no stream_id needed.
                    await ws.send_str(json.dumps(
                        {"event": "media", "media": {"payload": data}}))
                elif kind == "speech_started":
                    # Barge-in: caller interrupted -> flush queued bot audio.
                    await ws.send_str(json.dumps({"event": "clear"}))
                elif kind == "ready":
                    log.info("xAI session ready")
                elif kind == "user_transcript":
                    log.info("Caller: %s", data)
                elif kind == "bot_transcript":
                    log.debug("Bot: %s", data)
        except Exception:
            log.exception("xAI->Telnyx pump failed")

    pump_task = asyncio.create_task(pump_xai_to_telnyx())

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except (ValueError, TypeError):
                continue

            event = m.get("event")
            if event == "start":
                fmt = m.get("start", {}).get("media_format", {})
                log.info("Stream start: id=%s format=%s", m.get("stream_id"), fmt)
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
    finally:
        pump_task.cancel()
        await xai.close()
        log.info("Call cleaned up")
    return ws


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/webhook", handle_webhook)
    app.router.add_post("/", handle_webhook)  # fallback if webhook set to root
    app.router.add_get("/health", handle_health)
    app.router.add_get(config.STREAM_PATH, handle_media_stream)
    return app


if __name__ == "__main__":
    if not config.TELNYX_API_KEY:
        log.warning("TELNYX_API_KEY is empty — calls won't be answered. Set it in .env")
    log.info("Listening on http://%s:%d  (webhook /webhook, media ws %s)",
             config.HOST, config.PORT, config.STREAM_PATH)
    web.run_app(build_app(), host=config.HOST, port=config.PORT, print=None)
