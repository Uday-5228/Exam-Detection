"""FastAPI application serving streams, alerts, events, and dashboard assets."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.config import AppConfig, load
from src.db import count_events_since, list_events, mark_reviewed
from src.pipeline import PipelineManager

STATIC_DIR = Path(__file__).with_name("static")


class AlertBroadcaster:
    """Move worker-thread alerts through an asyncio queue to WebSocket clients."""

    def __init__(self) -> None:
        """Create a broadcaster that will bind to the application event loop."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._clients: set[WebSocket] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the queue and background broadcaster task."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Cancel the broadcaster task and close connected clients."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for client in tuple(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

    def publish_from_worker(self, payload: dict[str, Any]) -> None:
        """Schedule an alert payload from a worker thread without blocking it."""
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket alert subscriber."""
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket subscriber."""
        self._clients.discard(websocket)

    async def _broadcast_loop(self) -> None:
        """Send every queued alert to all currently connected subscribers."""
        assert self._queue is not None
        while True:
            payload = await self._queue.get()
            clients = tuple(self._clients)
            if not clients:
                continue
            outcomes = await asyncio.gather(*(client.send_json(payload) for client in clients),
                                            return_exceptions=True)
            for client, outcome in zip(clients, outcomes):
                if isinstance(outcome, Exception):
                    self.disconnect(client)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the API application with its stream workers and static dashboard."""
    app_config = config or load()
    manager = PipelineManager(app_config)
    broadcaster = AlertBroadcaster()
    database_path = Path(app_config.output.event_db)
    if not database_path.is_absolute():
        database_path = app_config.config_path.parent / database_path
    snapshot_dir = Path(app_config.output.snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = app_config.config_path.parent / snapshot_dir

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Start worker threads and the WebSocket broadcaster on startup."""
        await broadcaster.start()
        manager.start(broadcaster.publish_from_worker)
        try:
            yield
        finally:
            await asyncio.to_thread(manager.stop)
            await broadcaster.stop()

    application = FastAPI(title="Smart CCTV Exam Activity Monitor", lifespan=lifespan)
    application.state.config = app_config
    application.state.pipeline = manager
    application.state.broadcaster = broadcaster
    application.state.database_path = database_path
    application.state.snapshot_dir = snapshot_dir
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the single-page monitoring dashboard."""
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/streams")
    async def get_streams() -> list[dict[str, Any]]:
        """List configured streams with status and live FPS."""
        return manager.streams()

    @application.get("/stream/{stream_id}")
    async def stream_video(stream_id: str) -> StreamingResponse:
        """Serve one annotated stream as an MJPEG multipart response."""
        worker = manager.get(stream_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="Unknown stream")

        async def frames() -> AsyncIterator[bytes]:
            """Yield each new JPEG frame without blocking the event loop."""
            last_sequence = -1
            while True:
                sequence, jpeg = worker.latest_frame()
                if jpeg is not None and sequence != last_sequence:
                    last_sequence = sequence
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.05)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @application.websocket("/ws/alerts")
    async def websocket_alerts(websocket: WebSocket) -> None:
        """Subscribe to live alert messages over WebSocket."""
        await broadcaster.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.disconnect(websocket)

    @application.get("/api/events")
    async def get_events(
        stream_id: str | None = None,
        event_type: str | None = Query(default=None, alias="type"),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """List event log rows with optional stream and type filters."""
        return await run_in_threadpool(list_events, database_path, stream_id, event_type, limit)

    @application.post("/api/events/{event_id}/review")
    async def review_event(event_id: str) -> dict[str, bool | str]:
        """Mark an event as reviewed or return 404 if no event matches."""
        updated = await run_in_threadpool(mark_reviewed, database_path, event_id)
        if not updated:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"id": event_id, "reviewed": True}

    @application.get("/api/metrics")
    async def get_metrics() -> dict[str, Any]:
        """Return per-stream FPS/latency, alert counts, and today's total."""
        metrics = manager.metrics()
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        metrics["total_alerts_today"] = await run_in_threadpool(count_events_since, database_path, midnight)
        metrics["streams_online"] = sum(
            item["status"] == "online" for item in metrics["streams"].values()
        )
        return metrics

    @application.get("/snapshots/{filename}")
    async def get_snapshot(filename: str) -> FileResponse:
        """Serve a saved alert JPEG by its filename."""
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        snapshot = snapshot_dir / filename
        if not snapshot.is_file():
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return FileResponse(snapshot, media_type="image/jpeg")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000)
