"""Telnyx <-> xAI voice bridge.

Telnyx connects here over a Media Streaming WebSocket (point your ngrok URL at
this server's :5000 in the Telnyx TeXML/Call Control config). We relay caller
audio to xAI's realtime voice API and stream xAI's audio responses back to the
caller. Audio stays as G.711 μ-law @ 8 kHz end to end (no transcoding).

Run:  python server.py
"""
import asyncio
import json
import logging

import websockets

import config
from xai_client import XAIRealtimeClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")


async def handle_telnyx(telnyx_ws):
    """One phone call = one Telnyx WS connection = one xAI session."""
    log.info("Telnyx connected")
    stream_id = None
    xai = XAIRealtimeClient()

    try:
        await xai.connect()
    except Exception:
        log.exception("Could not connect to xAI; closing call")
        await telnyx_ws.close()
        return

    async def pump_xai_to_telnyx():
        """Drive the xAI session and relay its output to the caller."""
        try:
            async for kind, data in xai.iter_events():
                if kind == "audio" and stream_id:
                    await telnyx_ws.send(json.dumps({
                        "event": "media",
                        "stream_id": stream_id,
                        "media": {"payload": data},  # base64 μ-law
                    }))
                elif kind == "speech_started" and stream_id:
                    # Barge-in: caller interrupted -> drop buffered bot audio.
                    await telnyx_ws.send(json.dumps({
                        "event": "clear",
                        "stream_id": stream_id,
                    }))
                elif kind == "ready":
                    log.info("xAI session ready")
                elif kind == "user_transcript":
                    log.info("Caller: %s", data)
                elif kind == "bot_transcript":
                    log.debug("Bot: %s", data)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("xAI->Telnyx pump failed")

    pump_task = asyncio.create_task(pump_xai_to_telnyx())

    try:
        async for raw in telnyx_ws:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue

            event = msg.get("event")
            if event == "connected":
                log.info("Telnyx stream connected")
            elif event == "start":
                stream_id = msg.get("stream_id") or msg.get("start", {}).get("stream_id")
                fmt = msg.get("start", {}).get("media_format", {})
                log.info("Stream start: id=%s format=%s", stream_id, fmt)
            elif event == "media":
                payload = msg.get("media", {}).get("payload")
                # Only forward the caller's inbound audio.
                track = msg.get("media", {}).get("track", "inbound")
                if payload and track == "inbound":
                    await xai.append_audio(payload)
            elif event == "stop":
                log.info("Telnyx stream stopped")
                break
            elif event == "dtmf":
                log.info("DTMF: %s", msg.get("dtmf"))
    except websockets.ConnectionClosed:
        log.info("Telnyx connection closed")
    finally:
        pump_task.cancel()
        await xai.close()
        log.info("Call cleaned up")


async def main():
    log.info("Bridge listening on ws://%s:%d", config.HOST, config.PORT)
    async with websockets.serve(handle_telnyx, config.HOST, config.PORT, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
