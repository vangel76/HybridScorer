"""
HybridScorer - local image triage app.

FastAPI + Tabler UI entrypoint. Core scoring/model/cache logic lives in lib/.
"""

import asyncio
import html as _html
import logging
import os
import sys
import webbrowser

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lib.config import INPUT_FOLDER_NAME, SETUP_SCRIPT_HINT
from lib.utils import resolve_server_port
from lib.web_context import HybridScorerContext, dependency_issues


APP_NAME = "HybridScorer"
APP_DISPLAY_NAME = "HybridSelector"


def load_app_version():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    version_path = os.path.join(script_dir, "VERSION")
    try:
        with open(version_path, "r", encoding="utf-8") as handle:
            version = handle.read().strip()
            if version:
                return version
    except OSError:
        pass
    return "0.0.0"


def load_changelog():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "CHANGELOG.md"), "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return "Changelog not available."


APP_VERSION = load_app_version()
APP_GITHUB_TAG = f"v{APP_VERSION}"
APP_WINDOW_TITLE = f"{APP_DISPLAY_NAME} {APP_GITHUB_TAG}"
APP_CHANGELOG_HTML = _html.escape(load_changelog())


def create_setup_required_app(requirement_issues):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app = FastAPI(title=APP_WINDOW_TITLE)
    app.mount("/static", StaticFiles(directory=os.path.join(script_dir, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(script_dir, "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def setup_required(request: Request):
        return templates.TemplateResponse(
            request,
            "setup_required.html",
            {
                "request": request,
                "title": APP_WINDOW_TITLE,
                "setup_hint": SETUP_SCRIPT_HINT,
                "issues": requirement_issues or ["Unknown dependency mismatch"],
            },
        )

    return app


def create_fastapi_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    context = HybridScorerContext(script_dir, APP_NAME, APP_GITHUB_TAG, APP_CHANGELOG_HTML)
    app = FastAPI(title=APP_WINDOW_TITLE)
    app.state.hybrid_context = context
    app.mount("/static", StaticFiles(directory=os.path.join(script_dir, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(script_dir, "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "title": APP_WINDOW_TITLE,
                "app_name": APP_NAME,
                "version": APP_GITHUB_TAG,
                "initial_state": context.to_payload(),
            },
        )

    @app.get("/api/state")
    async def api_state():
        with context.lock:
            return JSONResponse(context.to_payload())

    @app.post("/api/controls")
    async def api_controls(payload: dict):
        with context.lock:
            context.update_inputs(payload)
            return JSONResponse(context.to_payload())

    @app.post("/api/folder/load")
    async def api_folder_load(payload: dict):
        job = context.load_folder_job(payload)
        return {"job_id": job.id, "job": context.job_payload(job)}

    @app.post("/api/score")
    async def api_score(payload: dict):
        job = context.score_job(payload)
        return {"job_id": job.id, "job": context.job_payload(job)}

    @app.post("/api/search/{kind}")
    async def api_search(kind: str, payload: dict):
        if kind not in {"similar", "same-person", "object"}:
            raise HTTPException(status_code=404, detail="Unknown search kind")
        job = context.search_job(kind, payload)
        return {"job_id": job.id, "job": context.job_payload(job)}

    @app.post("/api/thresholds")
    async def api_thresholds(payload: dict):
        return JSONResponse(context.threshold_action(payload))

    @app.post("/api/selection")
    async def api_selection(payload: dict):
        return JSONResponse(context.selection_action(payload))

    @app.post("/api/query-image")
    async def api_query_image(request: Request, image: UploadFile | None = File(default=None)):
        content_type = request.headers.get("content-type", "")
        if image is not None:
            return JSONResponse(await context.set_query_image(upload=image))
        if "application/json" in content_type:
            payload = await request.json()
            return JSONResponse(await context.set_query_image(payload=payload))
        raise HTTPException(status_code=400, detail="Expected image file or JSON data_url payload")

    @app.delete("/api/query-image")
    async def api_clear_query_image():
        return JSONResponse(context.clear_query_image())

    @app.post("/api/prompt/generate")
    async def api_prompt_generate(payload: dict):
        job = context.prompt_generate_job(payload)
        return {"job_id": job.id, "job": context.job_payload(job)}

    @app.post("/api/prompt/insert")
    async def api_prompt_insert(payload: dict):
        return JSONResponse(context.prompt_insert(payload))

    @app.post("/api/prompt/detail")
    async def api_prompt_detail(payload: dict):
        return JSONResponse(context.prompt_detail(payload))

    @app.post("/api/export")
    async def api_export(payload: dict):
        job = context.export_job(payload)
        return {"job_id": job.id, "job": context.job_payload(job)}

    @app.get("/media/{media_id}")
    async def media(media_id: str):
        response = context.media.response(media_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Media not found")
        return response

    @app.get("/api/jobs/{job_id}")
    async def api_job_status(job_id: str):
        payload = context.job_status_payload(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return JSONResponse(payload)

    @app.websocket("/ws/jobs/{job_id}")
    async def websocket_job(websocket: WebSocket, job_id: str):
        await websocket.accept()
        job = context.jobs.get(job_id)
        if job is None:
            await websocket.send_json({"type": "error", "error": "Unknown job id"})
            await websocket.close()
            return
        await websocket.send_json({"type": "progress", "job": context.job_payload(job)})
        try:
            while True:
                if job.status in {"done", "error"} and job.events.empty():
                    final = {"type": job.status, "job": context.job_payload(job)}
                    if job.result is not None:
                        final["state"] = job.result
                    if job.error:
                        final["error"] = job.error
                    await websocket.send_json(final)
                    break
                try:
                    event = await asyncio.to_thread(job.events.get, True, 30)
                except Exception:
                    event = {"type": "progress", "job": context.job_payload(job)}
                await websocket.send_json(event)
                if event.get("type") in {"done", "error"}:
                    break
        except WebSocketDisconnect:
            return
        finally:
            await websocket.close()

    return app


if __name__ == "__main__":
    issues = dependency_issues()
    if issues:
        print("[Startup check] Dependency mismatch detected. Please rerun setup.")
        for issue in issues:
            print(f"  - {issue}")
        fastapi_app = create_setup_required_app(issues)
    else:
        fastapi_app = create_fastapi_app()

    port = resolve_server_port(7862, "HYBRIDSELECTOR_PORT")

    class _MediaFilter(logging.Filter):
        def filter(self, record):
            return '/media/' not in record.getMessage()

    @fastapi_app.on_event("startup")
    async def _open_browser():
        logging.getLogger("uvicorn.access").addFilter(_MediaFilter())
        url = f"http://localhost:{port}"
        await asyncio.sleep(0.1)
        webbrowser.open(url)

    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
