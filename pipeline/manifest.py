"""SQLite manifest for tracking pipeline state across all stages."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("data/manifest.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT,
    duration    REAL,
    path        TEXT,
    status      TEXT DEFAULT 'pending',
    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clips (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id),
    frames_dir  TEXT NOT NULL,
    start_sec   REAL,
    end_sec     REAL,
    fps         INTEGER,
    n_frames    INTEGER,
    status      TEXT DEFAULT 'pending',
    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(_SCHEMA)


# --- videos ---

def add_video(video_id: str, url: str, title: str, duration: float):
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO videos (id, url, title, duration) VALUES (?,?,?,?)",
            (video_id, url, title, duration),
        )


def mark_video(video_id: str, status: str, path: str = None, error: str = None):
    with get_db() as conn:
        conn.execute(
            "UPDATE videos SET status=?, path=?, error=? WHERE id=?",
            (status, path, error, video_id),
        )


def pending_videos():
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE status='pending'"
        ).fetchall()


def downloaded_videos():
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE status='downloaded'"
        ).fetchall()


# --- clips ---

def add_clip(clip_id: str, video_id: str, frames_dir: str,
             start_sec: float, end_sec: float, fps: int, n_frames: int):
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO clips
               (id, video_id, frames_dir, start_sec, end_sec, fps, n_frames)
               VALUES (?,?,?,?,?,?,?)""",
            (clip_id, video_id, frames_dir, start_sec, end_sec, fps, n_frames),
        )


def mark_clip(clip_id: str, status: str, error: str = None):
    with get_db() as conn:
        conn.execute(
            "UPDATE clips SET status=?, error=? WHERE id=?",
            (status, error, clip_id),
        )


def pending_clips():
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM clips WHERE status='pending'"
        ).fetchall()


def summary():
    with get_db() as conn:
        v = dict(conn.execute(
            "SELECT status, COUNT(*) as n FROM videos GROUP BY status"
        ).fetchall() or [])
        c = dict(conn.execute(
            "SELECT status, COUNT(*) as n FROM clips GROUP BY status"
        ).fetchall() or [])
    return {"videos": v, "clips": c}
