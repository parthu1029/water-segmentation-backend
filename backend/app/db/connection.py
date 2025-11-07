import os
import psycopg2
import psycopg2.extras as extras
from psycopg2 import errorcodes
from typing import Optional, Dict, Any, Tuple

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
    statements = [
        (
            """
            CREATE TABLE IF NOT EXISTS rois (
                roi_id UUID PRIMARY KEY,
                geometry JSONB NOT NULL,
                properties JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        ),
        (
            """
            CREATE TABLE IF NOT EXISTS jobs (
                request_id TEXT PRIMARY KEY,
                payload JSONB,
                results JSONB,
                error TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        ),
        (
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                request_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content BYTEA NOT NULL,
                content_type TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (request_id, filename)
            )
            """
        ),
        (
            """
            CREATE TABLE IF NOT EXISTS logs (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ DEFAULT NOW(),
                level TEXT,
                name TEXT,
                message TEXT,
                pathname TEXT,
                lineno INT,
                funcname TEXT,
                request_id TEXT,
                extra JSONB
            )
            """
        ),
    ]
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
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


def insert_job(request_id: str, payload: Dict[str, Any], results: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    sql = """
    INSERT INTO jobs (request_id, payload, results, error)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (request_id) DO UPDATE SET
        payload = EXCLUDED.payload,
        results = EXCLUDED.results,
        error = EXCLUDED.error
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (request_id, extras.Json(payload or {}), extras.Json(results or {}), error))
    finally:
        if conn is not None:
            conn.close()


def update_job_results(request_id: str, results: Dict[str, Any]):
    sql = """
    UPDATE jobs SET results = %s, error = NULL WHERE request_id = %s
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (extras.Json(results or {}), request_id))
    finally:
        if conn is not None:
            conn.close()


def update_job_error(request_id: str, error: str):
    sql = """
    UPDATE jobs SET error = %s WHERE request_id = %s
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (error, request_id))
    finally:
        if conn is not None:
            conn.close()


def insert_artifact(request_id: str, filename: str, content: bytes, content_type: Optional[str] = None):
    sql = """
    INSERT INTO artifacts (request_id, filename, content, content_type)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (request_id, filename) DO UPDATE SET
        content = EXCLUDED.content,
        content_type = EXCLUDED.content_type
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (request_id, filename, psycopg2.Binary(content), content_type))
    finally:
        if conn is not None:
            conn.close()


def get_artifact(request_id: str, filename: str) -> Optional[Tuple[bytes, Optional[str]]]:
    sql = """
    SELECT content, content_type FROM artifacts WHERE request_id = %s AND filename = %s
    """
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (request_id, filename))
                row = cur.fetchone()
                if not row:
                    return None
                return row[0].tobytes() if hasattr(row[0], 'tobytes') else row[0], row[1]
    finally:
        if conn is not None:
            conn.close()


def insert_log(level: str, name: str, message: str, pathname: str, lineno: int, funcname: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
    sql = """
    INSERT INTO logs (level, name, message, pathname, lineno, funcname, request_id, extra)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    for _ in range(2):
        conn = None
        try:
            conn = get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        level,
                        name,
                        message,
                        pathname,
                        int(lineno),
                        funcname,
                        request_id,
                        extras.Json(extra or {})
                    ))
            return
        except Exception as e:
            try:
                code = getattr(e, 'pgcode', None)
            except Exception:
                code = None
            if code == errorcodes.UNDEFINED_TABLE:
                try:
                    init_db()
                except Exception:
                    pass
                continue
            else:
                break
        finally:
            if conn is not None:
                conn.close()
