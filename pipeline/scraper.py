"""Download YouTube videos matching a search query via yt-dlp."""

import logging
from pathlib import Path

import yt_dlp

from .manifest import add_video, mark_video, pending_videos

VIDEO_DIR = Path("data/videos")
log = logging.getLogger(__name__)


def _ydl_opts(download: bool, output_dir: Path) -> dict:
    return {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": not download,
        "merge_output_format": "mp4",
    }


def search_and_queue(query: str, max_videos: int = 20):
    """Search YouTube and add results to the manifest without downloading."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    search_url = f"ytsearch{max_videos}:{query}"
    with yt_dlp.YoutubeDL(_ydl_opts(download=False, output_dir=VIDEO_DIR)) as ydl:
        info = ydl.extract_info(search_url, download=False)

    entries = info.get("entries", []) or []
    queued = 0
    for entry in entries:
        if not entry:
            continue
        vid_id = entry.get("id")
        url    = entry.get("webpage_url") or f"https://www.youtube.com/watch?v={vid_id}"
        title  = entry.get("title", "")
        dur    = entry.get("duration", 0.0)
        add_video(vid_id, url, title, dur)
        queued += 1

    log.info("Queued %d videos for '%s'", queued, query)
    return queued


def download_pending(max_duration: float = 600.0):
    """Download all pending videos, skipping those over max_duration seconds."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    rows = pending_videos()

    if not rows:
        log.info("No pending videos.")
        return

    from tqdm import tqdm
    for row in tqdm(rows, desc="Downloading videos", unit="video"):
        vid_id   = row["id"]
        url      = row["url"]
        duration = row["duration"] or 0.0

        if duration > max_duration:
            mark_video(vid_id, "skipped", error=f"duration {duration:.0f}s > limit {max_duration:.0f}s")
            log.info("Skipped %s (%.0fs)", vid_id, duration)
            continue

        try:
            with yt_dlp.YoutubeDL(_ydl_opts(download=True, output_dir=VIDEO_DIR)) as ydl:
                ydl.download([url])
            # find the downloaded file
            matches = list(VIDEO_DIR.glob(f"{vid_id}.*"))
            path = str(matches[0]) if matches else None
            mark_video(vid_id, "downloaded", path=path)
            log.info("Downloaded %s", vid_id)
        except Exception as exc:
            mark_video(vid_id, "failed", error=str(exc))
            log.warning("Failed %s: %s", vid_id, exc)
