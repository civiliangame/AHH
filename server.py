"""Telnyx (Call Control) <-> xAI Voice Agent bridge — FastAPI app.

One server on :5000 does two jobs:

  POST  /webhook        Telnyx Call Control webhooks (JSON). On an incoming
                        call we answer it AND start bidirectional media
                        streaming (rtp / PCMU) pointed back at /media-stream.
  WS    /media-stream   Telnyx opens this WebSocket and streams call audio.
                        We bridge it to an xAI Voice Agent session and relay
                        Grok's audio back to the caller.

Audio is G.711 μ-law (PCMU) @ 8 kHz end to end — no transcoding.

Two layers run on top of the raw bridge:
  - Semantic / longitudinal: a per-call DSM-5 triage state machine (triage.py)
    backed by a per-patient record keyed by phone number (interactions.py).
  - Acoustic / observability: every call's caller audio is recorded and, after
    hangup, scored by the DAM voice-biomarker model (analysis.py) — folded back
    into the patient's record AND surfaced live on the dashboard.

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
import analysis
import config
import events
import interactions
import triage
from recorder import CallRecorder
from xai_client import XAIRealtimeClient

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bridge")

# The DAM voice-biomarker model is loaded LAZILY on the first call that needs
# scoring (see analysis.analyze_call), not at boot — so the conversational
# stack starts instantly and runs fine even when the ~702 MB checkpoint or
# torch isn't installed. Scoring then degrades to a graceful "unavailable".
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


# ----------------------------------------------------------------------------
# Patient-centric API — the dashboard's primary navigation axis.
# A patient is one phone number with a longitudinal record (interactions.py):
# rolling differential, DSM-5 criteria, a depression/anxiety score trend, and
# scheduled future check-ins.
# ----------------------------------------------------------------------------
@app.get("/api/patients")
async def api_patients():
    """List every patient (one per phone), newest activity first, with a summary
    the dashboard renders in the nav column."""
    return JSONResponse({"patients": interactions.list_summaries()})


@app.get("/api/patients/{phone}")
async def api_patient(phone: str):
    """Full longitudinal record for one patient: profile (rolling differential +
    DSM-5 + score trend), every session (with its DAM scores), pending check-ins,
    and the deep background DSM-5 criteria profile."""
    # interactions.load() sanitizes the key to digits, so a crafted phone can't
    # traverse out of ./interactions.
    record = interactions.load(phone)
    if not record.get("sessions") and not record.get("profile"):
        return JSONResponse({"error": "not found", "phone": phone}, status_code=404)
    record["dsm5_profile"] = interactions.load_dsm5_profile(phone)
    return JSONResponse(record)


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
    phone = ws.query_params.get("phone")  # patient's number, keys their record
    call_id = uuid.uuid4().hex[:12]  # session key for the transcript + dashboard
    recorder = CallRecorder(call_id)  # buffers caller μ-law for post-call DAM scoring
    log.info("Telnyx media stream connected (agent=%s, call=%s, phone=%s)",
             persona.name, call_id, phone)
    await events.emit(call_id, "system", f"Call started · agent: {persona.name}")

    # Load/append the patient's interaction record and open a session for this call.
    record = interactions.load(phone)
    session = interactions.start_session(record, call_id, persona.name)
    # Let the dashboard associate this live call with its patient.
    await events.emit_metadata(call_id, {
        "kind": "call_meta",
        "phone": interactions.key_for(phone),
        "agent": persona.name,
    })

    # For the outbound check-in agent, fold the patient's DUE follow-up questions
    # into its prompt so it actually asks them. "Due" = the `days` schedule has
    # elapsed; pass ?due=all on the stream URL to force all of them (for demos).
    # We DON'T clear them here — only after the call ends (see finally), so a
    # dropped call doesn't silently lose unasked questions.
    asked_checkins: list = []
    if persona.name == "checkin":
        pending = record.get("pending_checkins", [])
        force_all = ws.query_params.get("due") == "all"
        asked_checkins = pending if force_all else interactions.filter_due(pending)
        if asked_checkins:
            qs = "\n".join(
                f"- {c['message'] if isinstance(c, dict) else c}"
                for c in asked_checkins)
            persona = dataclasses.replace(
                persona,
                instructions=persona.instructions
                + f"\n\nAsk the patient these follow-up questions, one at a time:\n{qs}",
            )
            log.info("Check-in: injected %d follow-up question(s)%s",
                     len(asked_checkins), " (forced all)" if force_all else " (due)")
        else:
            log.info("Check-in: no follow-up questions are due yet")
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
        # Use THIS persona's greeting — the triage opener vs the check-in opener
        # ("…calling to check in. Is now a good time?"), not a hard-coded line.
        await xai.force_message(xai.persona.greeting, interruptible=False)
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
    rounds = {"n": 0}                 # differential rounds completed (for the UI)

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
                        # Tell the dashboard a symptom landed (live metadata panel).
                        await events.emit_metadata(call_id, {
                            "kind": "symptom",
                            "description": desc,
                            "empathy": empathy,
                        })
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
                                    list(asked), phone)),
                            }
                            # Signal the UI that a new differential round is in
                            # flight so the dashboard can show a "computing…"
                            # state during the empathy-line latency window.
                            await events.emit_metadata(call_id, {
                                "kind": "differential_started",
                                "round": rounds["n"] + 1,
                                "symptoms_so_far": list(descriptions),
                            })
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
                        rounds["n"] += 1
                        # Broadcast the refreshed differential to the dashboard
                        # (live metadata panel + a feed card).
                        await events.emit_metadata(call_id, {
                            "kind": "differential",
                            "round": rounds["n"],
                            "candidates": candidates,
                            "next_question": next_q,
                        })
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
        # Now that the check-in call is over, drop the questions it asked (keeping
        # any not-yet-due ones) — so a mid-call drop never loses unasked questions.
        if asked_checkins:
            interactions.remove_checkins(record, asked_checkins)
        interactions.save(record)  # final flush of this session's record
        # Save the caller's audio and fire-and-forget the DAM scoring task.
        # Runs after the WS closes; results land in scores/{call_id}.json, get
        # folded into the patient's record, and stream out as a 'scores' event.
        wav_path = recorder.save()
        if wav_path is not None:
            asyncio.create_task(analysis.analyze_call(
                call_id, wav_path, recorder.duration_seconds, phone=phone))
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
    log_level = config.LOG_LEVEL.lower()

    if reload_enabled:
        # Hot-reload needs the import-string + a single bind; serve telephony only.
        # (The dashboard is reachable on this port too while iterating.)
        log.info("Reload ON — single port %s (dashboard at http://localhost:%s/).",
                 config.PORT, config.PORT)
        uvicorn.run(
            "server:app",
            host=config.HOST,
            port=config.PORT,
            reload=True,
            reload_includes=["*.py", "*.txt"],
            log_level=log_level,
        )
    else:
        # Serve the SAME app on two ports in one process so the in-memory event
        # hub is shared: Telnyx hits :PORT (telephony), the browser hits
        # :DASHBOARD_PORT (UI + SSE). Both see the same live call state.
        async def _serve_dual():
            servers = [
                uvicorn.Server(uvicorn.Config(
                    app, host=config.HOST, port=config.PORT, log_level=log_level)),
                uvicorn.Server(uvicorn.Config(
                    app, host=config.HOST, port=config.DASHBOARD_PORT, log_level=log_level)),
            ]
            log.info("Telephony on :%s · dashboard on http://localhost:%s/",
                     config.PORT, config.DASHBOARD_PORT)
            await asyncio.gather(*(s.serve() for s in servers))

        asyncio.run(_serve_dual())
