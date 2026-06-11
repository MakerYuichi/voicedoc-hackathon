"""
WebSocket connection manager.

Maintains a registry of active WebSocket connections keyed by session_id.
The job_manager calls broadcast_to_session() to push real-time updates
to the correct browser tab.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Thread-safe registry of active WebSocket connections."""

    def __init__(self) -> None:
        # session_id → set of WebSocket connections
        # (a session can have multiple tabs open)
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    # ── lifecycle ──────────────────────────────────────────────────

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[session_id].add(websocket)
        logger.info(f"🔌 WS connected    | session={session_id} | total={self._total()}")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        async with self._lock:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]
        logger.info(f"🔌 WS disconnected | session={session_id} | total={self._total()}")

    # ── messaging ──────────────────────────────────────────────────

    async def broadcast_to_session(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Send a JSON payload to all connections for a given session."""
        message = json.dumps(payload)
        dead: List[WebSocket] = []

        async with self._lock:
            sockets = set(self._connections.get(session_id, set()))

        for ws in sockets:
            try:
                await ws.send_text(message)
            except Exception as exc:
                logger.warning(f"WS send failed for session={session_id}: {exc}")
                dead.append(ws)

        # Prune dead connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[session_id].discard(ws)

    async def broadcast_all(self, payload: Dict[str, Any]) -> None:
        """Broadcast to every connected session (e.g. system-wide alerts)."""
        message = json.dumps(payload)
        async with self._lock:
            all_sockets = [
                ws
                for sockets in self._connections.values()
                for ws in sockets
            ]
        for ws in all_sockets:
            try:
                await ws.send_text(message)
            except Exception:
                pass

    # ── introspection ──────────────────────────────────────────────

    def active_sessions(self) -> List[str]:
        return list(self._connections.keys())

    def _total(self) -> int:
        return sum(len(v) for v in self._connections.values())


# Singleton used across the app
ws_manager = WebSocketManager()
