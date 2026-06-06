"""
CLIP-based octopus presence scanner.

For each video, samples 1 frame/sec and scores each frame against
"an octopus underwater". Returns a (timestamps, scores) pair so the
caller can decide which segments to keep.

Usage:
    from phase2.scanner import load_clip, scan_video, save_scores
    model, processor, device = load_clip()
    timestamps, scores = scan_video("data/videos/foo.mp4", model, processor, device)
    save_scores("foo", timestamps, scores)
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

SCORES_DIR = Path("data/phase2/scores")
log = logging.getLogger(__name__)

TEXT_PROMPTS = [
    "an octopus in an aquarium tank",
    "empty aquarium tank with rocks and no animals",
]


def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    log.info("Loading CLIP on %s ...", device)
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # pre-encode text prompts once
    text_inputs = processor(text=TEXT_PROMPTS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    log.info("CLIP loaded in %.1fs", time.perf_counter() - t0)
    return model, processor, text_features, device


def _read_frames_at_1fps(video_path: str) -> tuple[list[np.ndarray], list[float]]:
    """Read one frame per second from a video using cv2. Returns (frames, timestamps)."""
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    log.info("Video: %.1fs  |  %.1f fps  |  %d total frames", duration, fps, total_frames)

    frames, timestamps = [], []
    second = 0
    while second < duration:
        frame_idx = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamps.append(float(second))
        second += 1

    cap.release()
    log.info("Frame read: %d frames in %.1fs", len(frames), time.perf_counter() - t0)
    return frames, timestamps


def scan_video(
    video_path: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    text_features: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan a video at 1fps and return P(octopus) for each second.

    Returns:
        timestamps : float array, shape (N,)  — seconds from start
        scores     : float array, shape (N,)  — P(octopus present)
    """
    t_total = time.perf_counter()
    log.info("Scanning %s ...", video_path)
    frames, timestamps = _read_frames_at_1fps(video_path)

    t_score = time.perf_counter()
    scores = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        pil_images = [Image.fromarray(f) for f in batch_frames]

        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        logits = (img_features @ text_features.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)
        scores.extend(probs[:, 0].cpu().tolist())  # P(octopus)

    scores = np.array(scores, dtype=np.float32)
    log.info(
        "Scoring: %d frames in %.1fs  (%.1f frames/sec)",
        len(frames),
        time.perf_counter() - t_score,
        len(frames) / max(time.perf_counter() - t_score, 1e-6),
    )
    log.info(
        "Total scan time: %.1fs  |  score range: %.3f – %.3f  |  mean: %.3f",
        time.perf_counter() - t_total,
        scores.min(), scores.max(), scores.mean(),
    )
    return np.array(timestamps, dtype=np.float32), scores


def save_scores(video_id: str, timestamps: np.ndarray, scores: np.ndarray):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORES_DIR / f"{video_id}.npz"
    np.savez_compressed(out, timestamps=timestamps, scores=scores)
    log.info("Saved scores → %s", out)


def load_scores(video_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = SCORES_DIR / f"{video_id}.npz"
    data = np.load(path)
    return data["timestamps"], data["scores"]


def detect_segments(
    timestamps: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.6,
    min_duration: float = 5.0,
) -> list[tuple[float, float]]:
    """
    Return list of (start_sec, end_sec) segments where score >= threshold
    for at least min_duration seconds.
    """
    above = scores >= threshold
    segments, in_seg, start = [], False, 0.0

    for i, (t, flag) in enumerate(zip(timestamps, above)):
        if flag and not in_seg:
            in_seg, start = True, float(t)
        elif not flag and in_seg:
            if float(t) - start >= min_duration:
                segments.append((start, float(t)))
            in_seg = False

    # close final segment
    if in_seg:
        end = float(timestamps[-1]) + 1.0
        if end - start >= min_duration:
            segments.append((start, end))

    return segments
