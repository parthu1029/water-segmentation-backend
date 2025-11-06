import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure the project root (parent of this file's directory) is on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import settings
from app.api.api_v1.api import api_router

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

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files from configured output directory
try:
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"OUTPUT_DIR create warning: {e}")
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
      print(f"DB init error: {e}")
        
'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
