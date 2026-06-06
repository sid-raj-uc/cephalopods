"""
DINOv2-based feature extractor.

For each detected segment (start_sec, end_sec), samples N frames uniformly
from the source MP4, runs DINOv2, and mean-pools the CLS tokens across frames
into a single 768-d appearance vector.

Usage:
    from phase2.extractor import load_dinov2, extract_video
    model, transform, device = load_dinov2()
    extract_video("data/videos/foo.mp4", "foo", model, transform, device)
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

FEATURES_DIR = Path("data/phase2/features")
N_FRAMES = 16  # frames sampled per segment

log = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_dinov2(model_name: str = "dinov2_vitb14", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading DINOv2 (%s) on %s ...", model_name, device)
    t0 = time.perf_counter()

    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    log.info("DINOv2 loaded in %.1fs", time.perf_counter() - t0)
    return model, transform, device


def _sample_frames(video_path: str, start_sec: float, end_sec: float, n: int) -> list[np.ndarray]:
    """Sample n frames uniformly from [start_sec, end_sec] in the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = np.linspace(start_sec, end_sec, n, endpoint=False)
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def motion_score(frames: list[np.ndarray]) -> float:
    """
    Mean absolute difference between consecutive frames, normalised to [0, 1].
    Low score = still / camouflaged. High score = active movement.
    """
    if len(frames) < 2:
        return 0.0
    diffs = [
        np.mean(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)))
        for i in range(len(frames) - 1)
    ]
    return float(np.mean(diffs)) / 255.0


def filter_by_motion(
    segments: list[tuple[float, float]],
    video_path: str,
    threshold: float = 0.02,
    n_frames: int = N_FRAMES,
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Return only segments whose motion score >= threshold.
    Also returns the motion scores for all segments (for logging/inspection).
    """
    kept, scores_all = [], []
    for start, end in segments:
        frames = _sample_frames(video_path, start, end, n_frames)
        score  = motion_score(frames)
        scores_all.append(score)
        if score >= threshold:
            kept.append((start, end))
        else:
            log.info("  Dropped still segment %.1f–%.1fs  (motion=%.4f)", start, end, score)

    log.info(
        "Motion filter: %d / %d segments kept  (threshold=%.3f)",
        len(kept), len(segments), threshold,
    )
    return kept, scores_all


def _frames_to_vector(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: str,
) -> np.ndarray:
    """Run DINOv2 on each frame, mean-pool CLS tokens → 768-d vector."""
    tensors = torch.stack([transform(Image.fromarray(f)) for f in frames]).to(device)

    with torch.no_grad():
        cls_tokens = model(tensors)  # (N, 768)

    return cls_tokens.mean(dim=0).cpu().numpy()  # (768,)


def extract_video(
    video_path: str,
    video_id: str,
    segments: list[tuple[float, float]],
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: str,
    n_frames: int = N_FRAMES,
) -> Path:
    """
    Extract DINOv2 features for all segments in one video.
    Saves a .npz to data/phase2/features/<video_id>.npz and returns the path.

    The .npz contains:
        features   : float32 (S, 768)  — one vector per segment
        start_secs : float32 (S,)
        end_secs   : float32 (S,)
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if not segments:
        log.warning("No segments for %s — skipping", video_id)
        return None

    t_total = time.perf_counter()
    log.info("Extracting features for %s  (%d segments) ...", video_id, len(segments))

    features, starts, ends = [], [], []

    for i, (start, end) in enumerate(segments):
        t_seg = time.perf_counter()
        frames = _sample_frames(video_path, start, end, n_frames)

        if len(frames) < 2:
            log.warning("  Segment %.1f–%.1fs: too few frames, skipping", start, end)
            continue

        vec = _frames_to_vector(frames, model, transform, device)
        features.append(vec)
        starts.append(start)
        ends.append(end)

        log.info(
            "  [%d/%d]  %.1f–%.1fs  →  %d frames  (%.1fs)",
            i + 1, len(segments), start, end, len(frames),
            time.perf_counter() - t_seg,
        )

    features = np.stack(features).astype(np.float32)   # (S, 768)
    starts   = np.array(starts,   dtype=np.float32)
    ends     = np.array(ends,     dtype=np.float32)

    out_path = FEATURES_DIR / f"{video_id}.npz"
    np.savez_compressed(out_path, features=features, start_secs=starts, end_secs=ends)

    elapsed = time.perf_counter() - t_total
    log.info(
        "Done: %d vectors saved → %s  (%.1fs total, %.1fs/segment)",
        len(features), out_path, elapsed, elapsed / max(len(features), 1),
    )
    return out_path


def load_features(video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved features for a video. Returns (features, start_secs, end_secs)."""
    data = np.load(FEATURES_DIR / f"{video_id}.npz")
    return data["features"], data["start_secs"], data["end_secs"]


def load_all_features() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Load and stack features from all processed videos.
    Returns (features, video_ids, start_secs, end_secs) — all aligned by row.
    """
    all_features, all_ids, all_starts, all_ends = [], [], [], []

    for path in sorted(FEATURES_DIR.glob("*.npz")):
        vid = path.stem
        feats, starts, ends = load_features(vid)
        all_features.append(feats)
        all_ids.extend([vid] * len(feats))
        all_starts.append(starts)
        all_ends.append(ends)

    if not all_features:
        raise RuntimeError("No features found in data/phase2/features/")

    return (
        np.vstack(all_features),
        all_ids,
        np.concatenate(all_starts),
        np.concatenate(all_ends),
    )
