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
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

import agents
import analysis
import config
import events
import triage
from recorder import CallRecorder
from xai_client import XAIRealtimeClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the ~702 MB DAM checkpoint once at boot so per-call scoring is fast.
    # Runs in a worker thread so torch's blocking load doesn't stall startup.
    await asyncio.to_thread(analysis.init_pipeline)
    yield


app = FastAPI(lifespan=lifespan)


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
# Live transcript: dashboard page + Server-Sent Events feed
# ----------------------------------------------------------------------------
@app.get("/")
async def dashboard():
    """The live transcript page. (POST / is the Telnyx webhook fallback.)"""
    return HTMLResponse(Path("dashboard.html").read_text(encoding="utf-8"))


@app.get("/api/state")
async def api_state():
    """Current in-memory triage state — accumulated symptoms + latest candidates.
    Lets the dashboard render the metadata panel on first load."""
    return JSONResponse({
        "symptoms": list(triage.global_descriptions),
        "candidates": list(triage.latest_candidates),
    })


@app.get("/api/calls")
async def api_calls():
    """List all persisted call transcripts, newest first.
    Each entry has call_id, agent, started_at (mtime), turn count."""
    out = []
    for p in events.TRANSCRIPT_DIR.glob("*.jsonl"):
        agent = "unknown"
        turns = 0
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    ev = json.loads(line)
                    role = ev.get("role")
                    if role == "system" and "agent:" in ev.get("text", ""):
                        agent = ev["text"].split("agent:", 1)[1].strip()
                    if role in ("caller", "agent"):
                        turns += 1
        except Exception:
            pass
        out.append({
            "call_id": p.stem,
            "agent": agent,
            "started_at": p.stat().st_mtime,
            "turns": turns,
        })
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return JSONResponse({"calls": out})


@app.get("/api/calls/{call_id}")
async def api_call_transcript(call_id: str):
    """Return one call's full transcript (turns + metadata) for replay."""
    p = events.TRANSCRIPT_DIR / f"{call_id}.jsonl"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    events_list = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                events_list.append(json.loads(line))
            except Exception:
                continue
    return JSONResponse({"call_id": call_id, "events": events_list})


@app.get("/api/scores/{call_id}")
async def get_scores(call_id: str):
    """Return persisted DAM scores for a call, or 404 if not yet computed."""
    # Reject anything that isn't our uuid4().hex[:12] shape, so a crafted id
    # can't traverse out of scores/.
    if not call_id.isalnum() or len(call_id) > 32:
        return JSONResponse({"error": "invalid call_id"}, status_code=400)
    path = analysis.SCORES_DIR / f"{call_id}.json"
    if not path.exists():
        return JSONResponse({"error": "not found", "call_id": call_id}, status_code=404)
    return JSONResponse(json.loads(path.read_text()))


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
    call_id = uuid.uuid4().hex[:12]  # session key for the transcript + dashboard
    recorder = CallRecorder(call_id)  # buffers caller μ-law for post-call DAM scoring
    log.info("Telnyx media stream connected (agent=%s, call=%s)", persona.name, call_id)
    await events.emit(call_id, "system", f"Call started · agent: {persona.name}")

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

    # State for the recordSymptom flow: empathy force_message -> wait for its
    # response.done -> next-question force_message. `pending` is None when idle.
    record_flow = {"pending": None}

    # Agent transcript arrives as deltas; buffer them into one line per turn and
    # flush (persist + finalize) when the caller speaks or the call ends.
    agent_buf: list[str] = []

    async def flush_agent():
        if agent_buf:
            await events.emit(call_id, "agent", "".join(agent_buf))
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
                        description = args.get("description", "")
                        empathy = args.get("empathy", "")
                        triage.recordSymptom(description, empathy)
                        await events.emit_metadata(call_id, {
                            "kind": "symptom",
                            "description": description,
                            "empathy": empathy,
                        })
                        # Fetch the next question CONCURRENTLY. get_next_question
                        # may block/sleep, so run it in a thread — otherwise it
                        # freezes the audio relay. It resolves while the empathy
                        # line plays (the whole point of the empathy filler).
                        q_task = asyncio.create_task(
                            asyncio.to_thread(triage.get_next_question))
                        # Signal the UI that a new differential round is in
                        # flight so the dashboard can show a "computing…"
                        # state during the empathy-line latency window.
                        await events.emit_metadata(call_id, {
                            "kind": "differential_started",
                            "round": len(triage.global_descriptions),
                            "symptoms_so_far": list(triage.global_descriptions),
                        })
                        # Speak the empathy line verbatim and in full (no barge-in).
                        await xai.force_message(empathy, interruptible=False)
                        record_flow["pending"] = {
                            "phase": "await_created",
                            "call_id": data["call_id"],
                            "empathy_id": None,
                            "q_task": q_task,
                        }
                        log.info("recordSymptom: desc=%r empathy=%r", description, empathy)
                    else:
                        result = triage.handle(name, args)
                        log.info("Tool %s -> %s", name, result)
                        await xai.send_function_result(data["call_id"], result)
                elif kind == "response_created":
                    p = record_flow["pending"]
                    # The first response.created after we send the empathy line is
                    # that force_message — capture its id so we gate on the right
                    # response.done (not the model's earlier tool-call turn).
                    if p and p["phase"] == "await_created":
                        p["empathy_id"] = data
                        p["phase"] = "await_done"
                elif kind == "response_done":
                    p = record_flow["pending"]
                    if p and p["phase"] == "await_done" and data == p["empathy_id"]:
                        # Empathy finished. Resolve the tool call (handing the model
                        # the question text for context) and speak it verbatim. No
                        # response.create -> model stays silent until the caller
                        # answers, then it calls recordSymptom again.
                        result = await p["q_task"]
                        next_q = (result or {}).get("next_question", "")
                        await events.emit_metadata(call_id, {
                            "kind": "differential",
                            "round": len(triage.global_descriptions),
                            "candidates": (result or {}).get("candidates", []),
                            "next_question": next_q,
                        })
                        await xai.send_function_output(
                            p["call_id"],
                            {"status": "recorded", "next_question": next_q})
                        await xai.force_message(next_q, interruptible=False)
                        log.info("Next question: %r", next_q)
                        record_flow["pending"] = None
                elif kind == "ready":
                    log.info("xAI session ready")
                    await maybe_greet()
                elif kind == "user_transcript":
                    log.info("Caller: %s", data)
                    await flush_agent()  # close any open agent turn first
                    await events.emit(call_id, "caller", data)
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
                    payload = media["payload"]
                    recorder.feed(payload)
                    await xai.append_audio(payload)
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
        # Save the caller's audio and fire-and-forget the DAM scoring task.
        # Runs after the WS closes; results land in scores/{call_id}.json and on
        # the SSE stream as a 'scores' metadata event.
        wav_path = recorder.save()
        if wav_path is not None:
            asyncio.create_task(
                analysis.analyze_call(call_id, wav_path, recorder.duration_seconds))
            log.info("Queued DAM analysis for %s (%.1fs of audio)",
                     call_id, recorder.duration_seconds)
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
