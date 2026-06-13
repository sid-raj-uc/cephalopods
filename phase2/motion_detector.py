"""
Motion detector — frame differencing to measure activity over time.

Streams frames via ffmpeg and computes mean absolute pixel difference
between consecutive frames (grayscale). Returns a per-second motion
score in [0, 1], where 1 = maximum change between frames.

Works on local files and remote HTTP streams.
"""

import logging
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_GRAY_SIZE = 224


def _stream_gray_frames(source: str, fps: float = 1.0):
    """
    Yield (timestamp_sec, H×W uint8 grayscale frame) via ffmpeg pipe.
    source: local path or http(s) URL (auth already embedded).
    """
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", source,
        "-vf", f"fps={fps},scale={_GRAY_SIZE}:{_GRAY_SIZE},format=gray",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = _GRAY_SIZE * _GRAY_SIZE
    interval = 1.0 / fps
    ts = 0.0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        yield ts, np.frombuffer(raw, dtype=np.uint8).reshape((_GRAY_SIZE, _GRAY_SIZE))
        ts += interval
    proc.stdout.close()
    proc.wait()


def scan_motion(
    source: str,
    fps: float = 1.0,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame motion score via absolute frame differencing.

    Parameters
    ----------
    source : local file path or authenticated remote URL
    fps    : frames per second to sample (1.0 is sufficient for 30-min videos)
    smooth_window : rolling average window in frames

    Returns
    -------
    timestamps   : float32 array, seconds from start
    motion_scores: float32 array in [0, 1], normalised mean abs diff
    """
    t0 = time.perf_counter()
    log.info("Motion scan: %s  (%.1f fps)", Path(source).name if not source.startswith("http") else source[-40:], fps)

    timestamps, raw_scores = [], []
    prev_frame = None

    for ts, frame in _stream_gray_frames(source, fps):
        if prev_frame is not None:
            diff = np.mean(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)))
            raw_scores.append(float(diff))
            timestamps.append(ts)
        prev_frame = frame

    if not raw_scores:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    scores = np.array(raw_scores, dtype=np.float32)
    # normalise to [0, 1] relative to the max in this video
    max_val = scores.max()
    if max_val > 0:
        scores /= max_val

    # smooth
    if smooth_window > 1 and len(scores) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        scores = np.convolve(scores, kernel, mode="same")

    log.info(
        "Motion scan done: %d frames in %.1fs  |  mean=%.3f  max=%.3f",
        len(scores), time.perf_counter() - t0, scores.mean(), scores.max(),
    )
    return np.array(timestamps, dtype=np.float32), scores
