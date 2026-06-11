"""
API routers for VoiceDoc Intelligence.
"""
from app.api.routes_process import router as process_router
from app.api.routes_query import router as query_router
from app.api.routes_websocket import router as websocket_router

__all__ = ["process_router", "query_router", "websocket_router"]
