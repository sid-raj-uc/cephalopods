"""
Remote CLIP scanner — streams frames from HTTP-hosted MP4s via ffmpeg,
scores with CLIP, then downloads only detected octopus segments.

No full video download needed. ffmpeg uses HTTP range requests to fetch
only the frames required for scanning, then only the detected byte ranges
for segment download.

Usage:
    from phase2.remote_scanner import RemoteScanner

    scanner = RemoteScanner(username="octopus", password="communication42")
    segments = scanner.scan_and_download(
        url="https://repo.octopus-intelligence.org/public/.../video.mp4",
        out_dir="data/aquarium",
    )
"""

import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

BASE_URL = "https://repo.octopus-intelligence.org/public"
DEFAULT_USER = "octopus"
DEFAULT_PASS = "communication42"


def _auth_url(url: str, username: str, password: str) -> str:
    """Inject Basic Auth credentials into the URL."""
    return url.replace("https://", f"https://{username}:{password}@")


def _stream_frames(
    url: str,
    username: str,
    password: str,
    scan_fps: float = 0.2,
    size: int = 224,
):
    """
    Stream frames from a remote MP4 via ffmpeg pipe.
    - scan_fps=0.2  → 1 frame every 5s (360 frames for a 30-min video vs 1800 at 1fps)
    - size=224      → resize in ffmpeg, much less data transferred over HTTP
    Yields (timestamp_sec, np.ndarray RGB frame HxWx3).
    """
    auth_url = _auth_url(url, username, password)

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", auth_url,
        "-vf", f"fps={scan_fps},scale={size}:{size}",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = size * size * 3

    timestamp = 0.0
    interval  = 1.0 / scan_fps
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((size, size, 3))
        yield timestamp, frame
        timestamp += interval

    proc.stdout.close()
    proc.wait()


def scan_url(
    url: str,
    model,
    processor,
    text_features,
    device: str,
    username: str = DEFAULT_USER,
    password: str = DEFAULT_PASS,
    scan_fps: float = 0.2,
    size: int = 224,
    batch_size: int = 64,
    clip_lock=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan a remote MP4 using CLIP. No full download.

    scan_fps=0.2 means 1 frame every 5s — sufficient for detecting octopus
    presence and 5x fewer CLIP calls than 1fps.

    Returns (timestamps, scores) arrays.
    """
    import torch

    t_total = time.perf_counter()
    log.info("Scanning: %s  (%.1f fps, %dpx)", Path(url).name, scan_fps, size)

    timestamps, scores = [], []
    batch_ts, batch_frames = [], []

    def _score_batch():
        pil_images = [Image.fromarray(f) for f in batch_frames]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        logits = (img_features @ text_features.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)
        return probs[:, 0].cpu().tolist()

    for ts, frame in _stream_frames(url, username, password, scan_fps, size):
        batch_ts.append(ts)
        batch_frames.append(frame)

        if len(batch_frames) == batch_size:
            if clip_lock:
                with clip_lock:
                    batch_scores = _score_batch()
            else:
                batch_scores = _score_batch()
            scores.extend(batch_scores)
            timestamps.extend(batch_ts)
            log.info("  [%s] scored up to %.0fs  (%d frames)", Path(url).stem[:16], batch_ts[-1], len(scores))
            batch_ts, batch_frames = [], []

    if batch_frames:
        if clip_lock:
            with clip_lock:
                batch_scores = _score_batch()
        else:
            batch_scores = _score_batch()
        scores.extend(batch_scores)
        timestamps.extend(batch_ts)

    timestamps = np.array(timestamps, dtype=np.float32)
    scores     = np.array(scores,     dtype=np.float32)

    elapsed = time.perf_counter() - t_total
    log.info(
        "Scan done: %d frames in %.1fs  (%.1f frames/sec)  |  score %.3f–%.3f  mean %.3f",
        len(scores), elapsed, len(scores) / max(elapsed, 1e-6),
        scores.min(), scores.max(), scores.mean(),
    )
    return timestamps, scores


def download_segment(
    url: str,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    username: str = DEFAULT_USER,
    password: str = DEFAULT_PASS,
):
    """
    Download a single time segment from a remote MP4 using ffmpeg range copy.
    Only fetches the bytes for [start_sec, end_sec] — not the full file.
    """
    auth_url = _auth_url(url, username, password)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", auth_url,
            "-c", "copy",
            str(out_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    size_mb = out_path.stat().st_size / 1e6
    log.info(
        "Downloaded %.1f–%.1fs → %s  (%.1fMB, %.1fs)",
        start_sec, end_sec, out_path, size_mb,
        time.perf_counter() - t0,
    )
    return out_path


class RemoteScanner:
    """
    High-level interface: scan a remote video, detect octopus segments,
    download only those segments locally.
    """

    def __init__(
        self,
        username: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        clip_threshold: float = 0.6,
        min_duration: float = 5.0,
    ):
        self.username      = username
        self.password      = password
        self.clip_threshold = clip_threshold
        self.min_duration  = min_duration
        self._model        = None

    def _load_clip(self):
        if self._model is None:
            from phase2.scanner import load_clip
            self._model, self._processor, self._text_features, self._device = load_clip()

    def scan_and_download(
        self,
        url: str,
        out_dir: str | Path,
        video_name: str = None,
    ) -> list[Path]:
        """
        Scan a remote video, detect octopus segments, download them.

        Returns list of downloaded segment paths.
        """
        from phase2.scanner import detect_segments, save_scores

        self._load_clip()
        out_dir = Path(out_dir)
        video_name = video_name or Path(url).stem

        # ── Scan ──────────────────────────────────────────────────
        timestamps, scores = scan_url(
            url, self._model, self._processor, self._text_features, self._device,
            username=self.username, password=self.password,
        )
        save_scores(f"remote_{video_name}", timestamps, scores)

        # ── Detect segments ───────────────────────────────────────
        segments = detect_segments(timestamps, scores, self.clip_threshold, self.min_duration)
        log.info(
            "%d segments detected (threshold=%.2f, min_dur=%.0fs)",
            len(segments), self.clip_threshold, self.min_duration,
        )

        if not segments:
            log.warning("No octopus detected in %s", url)
            return []

        # ── Download segments ─────────────────────────────────────
        t_dl = time.perf_counter()
        downloaded = []
        for i, (start, end) in enumerate(segments):
            out_path = out_dir / f"{video_name}_{start:.0f}_{end:.0f}.mp4"
            if out_path.exists():
                log.info("Segment already exists: %s", out_path)
                downloaded.append(out_path)
                continue
            try:
                p = download_segment(url, start, end, out_path, self.username, self.password)
                downloaded.append(p)
            except Exception as e:
                log.warning("Failed to download segment %d: %s", i, e)

        total_mb = sum(p.stat().st_size for p in downloaded) / 1e6
        log.info(
            "Downloaded %d segments  (%.1fMB total, %.1fs)",
            len(downloaded), total_mb,
            time.perf_counter() - t_dl,
        )
        return downloaded


def list_camera_urls(
    date: str,
    base_url: str = BASE_URL,
    username: str = DEFAULT_USER,
    password: str = DEFAULT_PASS,
) -> dict[str, list[str]]:
    """
    List all video URLs for a given date across all cameras.

    date format: "2026-02-20"
    Returns: {"Left Top": ["https://.../085420--vv-1.mp4", ...], ...}
    """
    import urllib.parse

    session_url = f"{base_url}/O-vulgaris-Nity-2026-2-20--/"  # session folder (fixed for this subject)
    cameras = ["Left Top", "Right Back", "Right Front", "Right Left", "Right Right", "Right Top"]
    result = {}

    for cam in cameras:
        cam_encoded = urllib.parse.quote(cam)
        listing_url = f"{session_url}{cam_encoded}/Local/{date}/"

        try:
            proc = subprocess.run(
                ["curl", "-s", "--user", f"{username}:{password}", listing_url],
                capture_output=True, text=True,
            )
            files = re.findall(r'href="([^"]+\.mp4)"', proc.stdout)
            result[cam] = [f"{listing_url}{f}" for f in files]
            log.info("%-15s  %d videos on %s", cam, len(files), date)
        except Exception as e:
            log.warning("Could not list %s / %s: %s", cam, date, e)
            result[cam] = []

    return result
