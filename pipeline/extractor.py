"""Extract fixed-length clips (as frame directories) from downloaded videos."""

import logging
import subprocess
from pathlib import Path

from .manifest import add_clip, downloaded_videos, mark_video

CLIPS_DIR = Path("data/clips")
log = logging.getLogger(__name__)


def _run_ffmpeg(cmd: list):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def _get_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip() or 0)


def extract_clips(
    clip_duration: float = 5.0,
    fps: int = 8,
    size: int = 224,
    min_duration: float = 10.0,
):
    """
    Slice each downloaded video into non-overlapping clips, extract frames as JPGs.
    Skips videos shorter than min_duration.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    rows = downloaded_videos()

    if not rows:
        log.info("No downloaded videos to extract.")
        return

    from tqdm import tqdm
    for row in tqdm(rows, desc="Extracting clips", unit="video"):
        vid_id     = row["id"]
        video_path = Path(row["path"])

        if not video_path.exists():
            mark_video(vid_id, "failed", error="file missing")
            continue

        try:
            total_dur = _get_duration(video_path)
            if total_dur < min_duration:
                mark_video(vid_id, "skipped", error=f"too short ({total_dur:.1f}s)")
                continue

            n_clips = int(total_dur // clip_duration)
            for i in range(n_clips):
                start   = i * clip_duration
                clip_id = f"{vid_id}_clip{i:04d}"
                out_dir = CLIPS_DIR / clip_id
                out_dir.mkdir(exist_ok=True)

                # extract frames
                _run_ffmpeg([
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-vf", f"fps={fps},scale={size}:{size}:flags=lanczos",
                    "-q:v", "2",
                    str(out_dir / "%05d.jpg"),
                    "-loglevel", "error",
                ])

                frames = sorted(out_dir.glob("*.jpg"))
                add_clip(clip_id, vid_id, str(out_dir),
                         start, start + clip_duration, fps, len(frames))

            mark_video(vid_id, "extracted")
            log.info("Extracted %d clips from %s", n_clips, vid_id)

        except Exception as exc:
            mark_video(vid_id, "failed", error=str(exc))
            log.warning("Failed to extract %s: %s", vid_id, exc)
