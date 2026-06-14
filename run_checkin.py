"""Trigger an outbound AI check-in call via Telnyx.

This only kicks off the dial. The running server (`python server.py`) + ngrok
must be up, and PUBLIC_HOSTNAME must point at your ngrok domain, so Telnyx can
open the media stream back to the bridge. The dialed call is connected to the
`checkin` persona (see agents.py / checkin_prompt.txt), separate from the
inbound triage agent.

Usage:
    python run_checkin.py +15551234567
    python run_checkin.py +15551234567 --agent checkin
"""
import argparse
import asyncio
import urllib.parse

import httpx

import config


async def place_call(to_number: str, agent: str) -> None:
    missing = [k for k, v in {
        "TELNYX_API_KEY": config.TELNYX_API_KEY,
        "TELNYX_CONNECTION_ID": config.TELNYX_CONNECTION_ID,
        "TELNYX_FROM_NUMBER": config.TELNYX_FROM_NUMBER,
        "PUBLIC_HOSTNAME": config.PUBLIC_HOSTNAME,
    }.items() if not v]
    if missing:
        raise SystemExit("Set these in .env first: " + ", ".join(missing))

    # Carry the dialed number through so the bridge can tag the transcript
    # with the patient's phone (no inbound webhook fires for outbound calls).
    phone_q = urllib.parse.quote(to_number)
    stream_url = f"wss://{config.PUBLIC_HOSTNAME}{config.STREAM_PATH}?agent={agent}&phone={phone_q}"
    body = {
        "connection_id": config.TELNYX_CONNECTION_ID,
        "to": to_number,
        "from": config.TELNYX_FROM_NUMBER,
        # Stream the call to our bridge, with two-way audio (rtp/PCMU). Telnyx
        # starts the stream when the callee answers.
        "stream_url": stream_url,
        "stream_track": "inbound_track",
        "stream_bidirectional_mode": "rtp",
        "stream_bidirectional_codec": "PCMU",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{config.TELNYX_API_BASE}/calls",
            headers={"Authorization": f"Bearer {config.TELNYX_API_KEY}"},
            json=body,
        )
    if resp.status_code >= 300:
        raise SystemExit(f"Telnyx dial failed ({resp.status_code}): {resp.text}")

    data = resp.json().get("data", {})
    print(f"Dialing {to_number} with agent '{agent}'.")
    print(f"  call_control_id: {data.get('call_control_id')}")
    print(f"  stream_url:      {stream_url}")
    print("Answer the phone — the bridge connects and the agent speaks first.")


def main():
    ap = argparse.ArgumentParser(description="Place an outbound AI call via Telnyx.")
    ap.add_argument("to", help="Number to call, E.164 (e.g. +15551234567)")
    ap.add_argument("--agent", default="checkin",
                    help="Persona name from agents.py (default: checkin)")
    args = ap.parse_args()
    asyncio.run(place_call(args.to, args.agent))


if __name__ == "__main__":
    main()
