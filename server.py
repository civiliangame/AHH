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
import dataclasses
import json
import logging
import os
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

import agents
import config
import events
import interactions
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
        # Answer AND start bidirectional streaming in one command. Pass the
        # caller's number through the stream URL so the media handler can key
        # the patient's interaction record by phone number.
        stream_url = stream_url_for(request)
        caller = payload.get("from", "")
        if caller:
            stream_url += f"?phone={quote(caller)}"
        await telnyx_command(ccid, "answer", {
            "stream_url": stream_url,
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
# Live transcript: dashboard page + Server-Sent Events feed
# ----------------------------------------------------------------------------
@app.get("/")
async def dashboard():
    """The live transcript page. (POST / is the Telnyx webhook fallback.)"""
    return HTMLResponse(Path("dashboard.html").read_text(encoding="utf-8"))


@app.get("/events")
async def sse():
    """Stream transcript turns to the browser as they happen."""
    async def gen():
        async for ev in events.hub.subscribe():
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # don't let a proxy buffer the stream
    })


# ----------------------------------------------------------------------------
# WS: Telnyx media stream  <->  xAI Voice Agent
# ----------------------------------------------------------------------------
@app.websocket(config.STREAM_PATH)
async def media_stream(ws: WebSocket):
    await ws.accept()
    # Pick the agent persona from the stream URL (?agent=checkin for outbound).
    # No param -> triage (inbound default).
    persona = agents.get(ws.query_params.get("agent"))
    phone = ws.query_params.get("phone")  # patient's number, keys their record
    call_id = uuid.uuid4().hex[:12]  # session key for the transcript + dashboard
    log.info("Telnyx media stream connected (agent=%s, call=%s, phone=%s)",
             persona.name, call_id, phone)
    await events.emit(call_id, "system", f"Call started · agent: {persona.name}")

    # Load/append the patient's interaction record and open a session for this call.
    record = interactions.load(phone)
    session = interactions.start_session(record, call_id, persona.name)

    # For the outbound check-in agent, fold the patient's saved follow-up
    # questions into its prompt so it actually asks them, then clear them so we
    # don't re-ask on the next check-in.
    if persona.name == "checkin" and record.get("pending_checkins"):
        # pending_checkins are {days, message} dicts (older records may be strings).
        qs = "\n".join(
            f"- {c['message'] if isinstance(c, dict) else c}"
            for c in record["pending_checkins"])
        persona = dataclasses.replace(
            persona,
            instructions=persona.instructions
            + f"\n\nAsk the patient these follow-up questions, one at a time:\n{qs}",
        )
        log.info("Check-in: injected %d follow-up question(s)",
                 len(record["pending_checkins"]))
        interactions.clear_pending(record)
    interactions.save(record)

    xai = XAIRealtimeClient(persona=persona)
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
        await xai.force_message(config.XAI_GREETING, interruptible=False)
        log.info("Triggered greeting")

    # State for the recordSymptom flow: model speaks empathy + calls recordSymptom
    # in one turn; we kick off get_next_question in parallel, then force_message
    # the question when that turn's response.done arrives. None when idle.
    record_flow = {"pending": None}

    # Per-call symptom state (NOT module-global — keyed to this caller).
    descriptions: list[str] = []      # every symptom recordSymptom has logged
    last_candidates: list[str] = []   # running differential, fed back each turn
    asked: list[str] = []             # questions already asked (avoid repeats)
    complete = {"v": False}           # True once triage closed (no more questions)

    # Agent transcript arrives as deltas; buffer them into one line per turn and
    # flush (persist + finalize) when the caller speaks or the call ends.
    agent_buf: list[str] = []

    async def flush_agent():
        if agent_buf:
            line = "".join(agent_buf)
            await events.emit(call_id, "agent", line)
            interactions.add_transcript(session, "agent", line)
            interactions.save(record)
            agent_buf.clear()

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
                    # Model invoked a tool. arguments arrives as a JSON string.
                    name = data["name"]
                    args = json.loads(data.get("arguments") or "{}")
                    if name == "recordSymptom":
                        desc = args.get("description", "")
                        empathy = args.get("empathy", "")
                        descriptions.append(desc)
                        # The model is SPEAKING its empathy line in this same
                        # response. Kick off the next-question lookup in parallel
                        # (it may block, so run it in a thread) so it's ready the
                        # moment the empathy turn finishes. Pass the per-call
                        # symptom list + running differential for refinement.
                        if complete["v"]:
                            # Triage already closed — keep recording, but don't
                            # ask anything further.
                            await xai.send_function_output(
                                data["call_id"], {"status": "recorded"})
                        elif record_flow["pending"] is None:
                            record_flow["pending"] = {
                                "call_ids": [data["call_id"]],
                                "new_descriptions": [desc],
                                "empathy": empathy,
                                "q_task": asyncio.create_task(asyncio.to_thread(
                                    triage.get_next_question,
                                    list(descriptions), list(last_candidates),
                                    list(asked))),
                            }
                        else:
                            # Multiple symptoms in one turn -> record each, ask once.
                            record_flow["pending"]["call_ids"].append(data["call_id"])
                            record_flow["pending"]["new_descriptions"].append(desc)
                        log.info("recordSymptom: %r", desc)
                    else:
                        result = triage.handle(name, args)
                        log.info("Tool %s -> %s", name, result)
                        await xai.send_function_result(data["call_id"], result)
                elif kind == "response_done":
                    # The empathy turn (which contained recordSymptom) just ended.
                    # Resolve the tool call(s) and speak the next question verbatim.
                    # No response.create -> the model stays silent until the caller
                    # answers, then empathizes + records again.
                    p = record_flow["pending"]
                    if p:
                        record_flow["pending"] = None
                        result = await p["q_task"] or {}
                        next_q = result.get("next_question", "")
                        candidates = result.get("candidates", [])
                        dsm5_criteria = result.get("dsm5_criteria", [])
                        future_checkin = result.get("future_checkin", [])
                        # Running memory: feed this turn's differential into the next.
                        last_candidates[:] = candidates
                        # Persist the turn + update the patient profile (candidates,
                        # DSM-5 criteria met/not-met, next question, future check-ins).
                        interactions.add_turn(
                            record, session,
                            descriptions=p["new_descriptions"],
                            empathy=p.get("empathy", ""),
                            candidates=candidates,
                            next_question=next_q,
                            future_checkin=future_checkin,
                            dsm5_criteria=dsm5_criteria,
                        )
                        interactions.save(record)
                        # Close the call when the model is done, when no question
                        # came back, or after the max question count is reached.
                        done = (result.get("done")
                                or not next_q
                                or len(asked) >= triage.MAX_QUESTIONS)
                        for cid in p["call_ids"]:
                            await xai.send_function_output(
                                cid, {"status": "recorded", "next_question": next_q})
                        if done:
                            complete["v"] = True
                            await xai.force_message(triage.CLOSING_LINE, interruptible=False)
                            log.info("Triage complete (asked=%d, done=%s) -> closing line",
                                     len(asked), result.get("done"))
                        else:
                            asked.append(next_q)
                            await xai.force_message(next_q, interruptible=False)
                            log.info("Question %d: %r | candidates=%s | future_checkin=%s",
                                     len(asked), next_q, candidates, future_checkin)
                elif kind == "ready":
                    log.info("xAI session ready")
                    await maybe_greet()
                elif kind == "user_transcript":
                    log.info("Caller: %s", data)
                    await flush_agent()  # close any open agent turn first
                    await events.emit(call_id, "caller", data)
                    interactions.add_transcript(session, "caller", data)
                    interactions.save(record)
                elif kind == "bot_transcript":
                    log.debug("Bot: %s", data)
                    agent_buf.append(data)
                    # Stream the delta live (typewriter); the full line is
                    # persisted on flush, so partials are display-only.
                    await events.emit(call_id, "agent", data, partial=True)
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
        await flush_agent()  # persist the last agent turn
        await events.emit(call_id, "system", "Call ended")
        interactions.save(record)  # final flush of this session's record
        pump_task.cancel()
        await xai.close()
        log.info("Call cleaned up")


if __name__ == "__main__":
    import uvicorn

    if not config.TELNYX_API_KEY:
        log.warning("TELNYX_API_KEY is empty — calls won't be answered. Set it in .env")
    # Reload is OPT-IN (RELOAD=1). It is OFF by default because a file change
    # mid-call tears down the worker and kills the active call — and this repo
    # lives under OneDrive, whose background sync writes fire spurious reloads.
    # Turn it on only while iterating on prompts/code WITHOUT a call in progress.
    reload_enabled = os.getenv("RELOAD", "0").lower() in ("1", "true", "yes")
    if not reload_enabled:
        log.info("Reload disabled (set RELOAD=1 to enable hot-reload for dev).")
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=reload_enabled,
        reload_includes=["*.py", "*.txt"] if reload_enabled else None,
        log_level=config.LOG_LEVEL.lower(),
    )
