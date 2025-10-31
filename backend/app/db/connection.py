import os
import psycopg2
import psycopg2.extras as extras
from typing import Optional, Dict, Any

from ..core.config import settings


def _build_dsn_from_env() -> Optional[str]:
    # Support both DATABASE_URL and individual env vars (lowercase keys per user's snippet)
    if settings.DATABASE_URL:
        return settings.DATABASE_URL
    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")
    dbname = os.getenv("dbname")
    if all([user, password, host, port, dbname]):
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return None


def get_connection():
    dsn = _build_dsn_from_env()
    if not dsn:
        raise RuntimeError("DATABASE_URL or individual DB env variables are not set")
    # psycopg2.connect understands DSN URL
    return psycopg2.connect(dsn)


def init_db():
    """Create required tables if they do not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS rois (
        roi_id UUID PRIMARY KEY,
        geometry JSONB NOT NULL,
        properties JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
    finally:
        if conn is not None:
            conn.close()


def insert_roi(roi_id: str, geometry: Dict[str, Any], properties: Optional[Dict[str, Any]] = None):
    sql = """
    INSERT INTO rois (roi_id, geometry, properties)
    VALUES (%s, %s, %s)
    ON CONFLICT (roi_id) DO NOTHING
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (roi_id, extras.Json(geometry), extras.Json(properties or {})))
    finally:
        if conn is not None:
            conn.close()
