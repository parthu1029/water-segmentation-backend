import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse
import logging
import uuid
import contextvars
import time

# Ensure the project root (parent of this file's directory) is on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import settings
from app.api.api_v1.api import api_router

request_id_var = contextvars.ContextVar("request_id", default=None)

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get()
        setattr(record, "request_id", rid)
        return True

class DBLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from app.db.connection import insert_log
            message = self.format(record)
            rid = getattr(record, "request_id", None)
            extra = {
                "processName": record.processName if hasattr(record, "processName") else None,
                "threadName": record.threadName if hasattr(record, "threadName") else None,
                "module": record.module,
            }
            insert_log(
                level=record.levelname,
                name=record.name,
                message=message,
                pathname=record.pathname,
                lineno=record.lineno,
                funcname=record.funcName,
                request_id=rid,
                extra=extra,
            )
        except Exception:
            pass

def configure_logging():
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.addFilter(RequestIdFilter())
    stream.setFormatter(fmt)
    dbh = DBLogHandler()
    dbh.setLevel(level)
    dbh.addFilter(RequestIdFilter())
    dbh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    logger.addHandler(dbh)

configure_logging()
logger = logging.getLogger(__name__)

# Create necessary directories
#os.makedirs(settings.INPUT_DIR, exist_ok=True)
#os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

os.makedirs("/tmp/input", exist_ok=True)
os.makedirs("/tmp/output", exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

@app.middleware("http")
async def add_request_id(request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(rid)
    try:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        try:
            logger.info(f"{request.method} {request.url.path} -> {response.status_code} in {duration_ms:.1f}ms")
        except Exception:
            pass
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response

allowed_origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^https://([a-z0-9-]+\.)?vercel\.app$|^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve static artifacts
if settings.STORE_IN_DB:
    @app.get("/static/{request_id}/{filename:path}")
    async def static_from_db(request_id: str, filename: str):
        try:
            from app.db.connection import get_artifact
            row = get_artifact(request_id, filename)
            if row:
                content, content_type = row
                return Response(content, media_type=content_type or "application/octet-stream")
        except Exception:
            pass
        disk_path = os.path.join(settings.OUTPUT_DIR, request_id, filename)
        try:
            if os.path.exists(disk_path):
                return FileResponse(disk_path)
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="Artifact not found")
else:
    # Mount static files from configured output directory
    try:
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"OUTPUT_DIR create warning: {e}")
    app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR, check_dir=False), name="static")

@app.get("/")
async def root():
    return {"message": "Waterbody Segmentation API is running"}

@app.on_event("startup")
def on_startup():
    try:
      # Lazy import to avoid psycopg2 import errors during cold start if DB not configured
      from app.db.connection import init_db
      init_db()
    except Exception as e:
      # In debug, it's useful to see this in logs, but avoid crashing startup
      logger.warning(f"DB init error: {e}")
        
'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
