"""
Experiment 16: Match ethogram events to video timestamps via motion + VLM.

For each indexed event in ethogram_index.json:
  1. Motion-scan the reference camera (Right Front preferred, Right Back fallback)
     across the full 30-min video to find activity peaks
  2. For each top peak: extract 3 frames from every available camera
  3. Ask Qwen2-VL-2B: does this frame match the event? (score 0-3)
  4. Write best_match (peak time, camera, confidence) and all peak_results
     back into ethogram_index.json after every entry

Usage:
    python3 phase2/exp16_event_matcher.py          # process first 5 indexed events
    python3 phase2/exp16_event_matcher.py --n 10   # process first 10
"""

import json, subprocess, os, re, sys, argparse
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).parent))
from motion_detector import scan_motion

PROJECT    = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT / "data" / "ethogram_index.json"
FRAMES_DIR = Path("/tmp/nity_match_frames")
FRAMES_DIR.mkdir(exist_ok=True)

USER, PASS     = "octopus", "communication42"
N_PEAKS        = 5    # top motion peaks to evaluate per video
PEAK_MIN_GAP   = 90   # seconds — minimum gap between distinct peaks
FRAME_OFFSETS  = [-20, 0, 20]  # seconds around each peak to sample
REF_CAMERAS    = ["Right Front", "Right Back"]  # preference order for motion scan


# ── utilities ─────────────────────────────────────────────────────────────────

def embed_auth(url: str) -> str:
    p = urlparse(url)
    netloc = f"{USER}:{PASS}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse(p._replace(netloc=netloc))


def extract_frame(video_url: str, t_sec: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-ss", str(max(0, int(t_sec))),
        "-i", embed_auth(video_url),
        "-frames:v", "1", "-q:v", "3", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        return os.path.exists(out_path)
    except Exception:
        return False


def find_peaks(timestamps: np.ndarray, scores: np.ndarray,
               n: int = N_PEAKS, min_gap: int = PEAK_MIN_GAP) -> list[tuple[float, float]]:
    """Return top-n (time, score) pairs with at least min_gap seconds between them."""
    peaks, used = [], set()
    for idx in np.argsort(scores)[::-1]:
        t = float(timestamps[idx])
        if any(abs(t - p) < min_gap for p in used):
            continue
        peaks.append((t, float(scores[idx])))
        used.add(t)
        if len(peaks) >= n:
            break
    return sorted(peaks, key=lambda x: x[0])


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ── VLM ───────────────────────────────────────────────────────────────────────

def build_prompt(event: str, details: str) -> str:
    detail_line = f'Details: "{details}"\n' if details.strip() else ""
    return (
        f'Aquarium security camera frame. An octopus named Nity lives in this tank.\n'
        f'Ethogram event: "{event}"\n'
        f'{detail_line}'
        f'Does this frame show Nity doing something related to this event?\n'
        f'Score: 0=Nity not visible, 1=Nity visible but unrelated, '
        f'2=possibly related, 3=clear match.\n'
        f'Respond with exactly: SCORE: [0-3] — [one sentence explanation]'
    )


SCORE_RE = re.compile(r"SCORE\s*:\s*([0-3])", re.IGNORECASE)

def vlm_score(model, processor, config, generate_fn,
              frame_path: str, prompt: str) -> tuple[int, str]:
    from mlx_vlm.prompt_utils import apply_chat_template
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    result = generate_fn(model, processor, formatted, image=frame_path,
                         max_tokens=80, verbose=False)
    text = result.text if hasattr(result, "text") else str(result)
    m = SCORE_RE.search(text)
    return (int(m.group(1)) if m else 0), text.strip()


# ── core logic ────────────────────────────────────────────────────────────────

def process_entry(entry: dict, model, processor, config, generate_fn) -> dict:
    event   = entry.get("event", "").strip()
    details = entry.get("details", "").strip()
    prompt  = build_prompt(event, details)

    avail = {c["name"]: c["video_url"]
             for c in entry.get("cameras", []) if c.get("available")}
    if not avail:
        print("  No available cameras — skip")
        return entry

    # Pick motion reference
    ref_cam = next((c for c in REF_CAMERAS if c in avail), next(iter(avail)))
    print(f"  Motion scan → {ref_cam} …", flush=True)
    timestamps, scores = scan_motion(embed_auth(avail[ref_cam]), fps=1.0)

    if len(scores) == 0:
        print("  No motion data returned")
        return entry

    peaks = find_peaks(timestamps, scores)
    print(f"  {len(peaks)} peaks: {[fmt(t) for t, _ in peaks]}", flush=True)

    best_score   = -1.0
    best_peak_t  = None
    best_camera  = None
    peak_results = []

    for peak_t, motion_score in peaks:
        frame_times = [max(0, peak_t + off) for off in FRAME_OFFSETS]
        cam_results = []

        for cam_name, cam_url in avail.items():
            frame_scores, frame_texts = [], []

            for ft in frame_times:
                fpath = str(FRAMES_DIR / f"{cam_name.replace(' ','_')}_{int(ft)}.jpg")
                ok = extract_frame(cam_url, ft, fpath)
                if not ok:
                    frame_scores.append(0)
                    frame_texts.append("extraction_failed")
                    continue
                sc, txt = vlm_score(model, processor, config, generate_fn, fpath, prompt)
                frame_scores.append(sc)
                frame_texts.append(txt)
                os.remove(fpath)

            avg = sum(frame_scores) / len(frame_scores)
            cam_results.append({
                "camera": cam_name,
                "avg_score": round(avg, 2),
                "frame_scores": frame_scores,
                "frame_descriptions": frame_texts,
            })

            if avg > best_score:
                best_score  = avg
                best_peak_t = peak_t
                best_camera = cam_name

        peak_results.append({
            "peak_time":     fmt(peak_t),
            "motion_score":  round(motion_score, 3),
            "cameras":       sorted(cam_results, key=lambda c: -c["avg_score"]),
        })
        top = max(cam_results, key=lambda c: c["avg_score"])
        print(f"    {fmt(peak_t)} → best: {top['camera']} score={top['avg_score']:.2f}", flush=True)

    entry["best_match"] = {
        "peak_time":        fmt(best_peak_t) if best_peak_t is not None else None,
        "camera":           best_camera,
        "confidence_score": round(best_score / 3, 2),
    }
    entry["peak_results"] = peak_results
    print(f"  ✓ Best: {best_camera} @ {fmt(best_peak_t)} (conf {best_score/3:.2f})", flush=True)
    return entry


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                        help="Number of indexed entries to process (default 5)")
    args = parser.parse_args()

    with open(INDEX_PATH) as f:
        index = json.load(f)

    print("Loading Qwen2-VL-2B-Instruct-4bit …", flush=True)
    from mlx_vlm import load, generate as _generate
    from mlx_vlm.utils import load_config
    MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print("Model ready.\n", flush=True)

    done, total = 0, 0
    for i, entry in enumerate(index):
        if entry.get("status") != "indexed":
            continue
        if "best_match" in entry:
            print(f"  already matched: {entry['date']} {entry['time']} — skip")
            continue
        if done >= args.n:
            break

        total += 1
        print(f"\n[{done+1}/{args.n}] {entry['date']} {entry['time']} — {entry['event'][:55]}")
        index[i] = process_entry(entry, model, processor, config, _generate)
        done += 1

        with open(INDEX_PATH, "w") as f:
            json.dump(index, f, indent=2)
        print("  → saved to ethogram_index.json")

    print(f"\nFinished. Processed {done} entries.")


if __name__ == "__main__":
    main()
