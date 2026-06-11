"""
WebSocket endpoint: /ws/{session_id}

The frontend connects on page load and receives real-time pipeline updates
pushed by:
  - update_job_progress_sync() in Celery workers (via job_manager)
  - broadcast_completion() in ProcessorAgent

Message types the client can expect
-------------------------------------
{
    "type": "progress",
    "job_id": str,
    "status": "running|success|failure",
    "progress": int,              # 0-100 aggregate
    "agents": {
        "ScannerAgent":   { "status": ..., "progress": ... },
        ...
    }
}

{
    "type": "complete",
    "job_id": str,
    "message": "Done! Processed 4 docs, stored 47 chunks",
    "progress": 100
}

{
    "type": "error",
    "message": str
}

{
    "type": "pong"                 # response to client "ping"
}
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.utils.websocket_manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    Persistent WebSocket connection for a browser session.

    - Registers the connection in ws_manager so Celery workers can push to it.
    - Keeps alive until the client disconnects or the server shuts down.
    - Echoes 'pong' in response to client 'ping' messages (keepalive).
    - Any JSON the client sends is logged but otherwise ignored;
      all meaningful communication is server → client.
    """
    await ws_manager.connect(websocket, session_id)
    logger.info(f"WebSocket opened | session={session_id}")

    # Send a welcome message so the frontend knows the connection is live
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "message": "VoiceDoc Intelligence connected. Ready for commands.",
    })

    try:
        while True:
            # Wait for a client message (keepalive ping or future commands)
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            # Handle ping / pong keepalive
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    logger.debug(
                        f"WS message from session={session_id}: {raw[:120]}"
                    )
            except (json.JSONDecodeError, Exception):
                pass  # ignore malformed messages

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"WebSocket error | session={session_id}: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(websocket, session_id)
        logger.info(f"WebSocket closed | session={session_id}")
