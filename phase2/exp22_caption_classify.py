"""
Exp 22 — Caption + ethogram-classify octopus clips with Qwen2-VL-2B (local).

For each clip in data/octopus_clips/*.mp4:
  1. Sample 4 frames spread across the 20s clip (t = 2, 7, 13, 18s)
  2. Ask Qwen2-VL-2B for, in one pass:
       - CAPTION   : what Nity (the octopus) is doing
       - BEHAVIOR  : the single best ethogram label (from data/ethogram_list.json)
       - CONFIDENCE: high / medium / low
     If the octopus isn't clearly visible / the model is unsure, the caption
     says so and BEHAVIOR is set to "not_visible".
  3. Save everything to data/octopus_clips/captions.json

Resource-light by design (keeps a 16GB Mac responsive):
  - 4-bit 2B model (~2GB RAM), loaded ONCE
  - one clip processed at a time (no batching, no parallelism)
  - saves after every clip → safe to Ctrl-C and resume (skips done clips)

Usage:
  python3 phase2/exp22_caption_classify.py                 # all clips
  python3 phase2/exp22_caption_classify.py --max-clips 3   # quick smoke test
"""
import argparse, json, re, subprocess, tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT       = Path(__file__).resolve().parent.parent
CLIPS_DIR     = PROJECT / "data" / "octopus_clips"
ETHOGRAM_PATH = PROJECT / "data" / "ethogram_list.json"
CAPTIONS_PATH = CLIPS_DIR / "captions.json"
MODEL         = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

FRAME_TIMES   = [2, 7, 13, 18]   # seconds into the 20s clip to sample
MAX_TOKENS    = 160


def build_prompt(behaviors: list[dict]) -> str:
    """Group ethogram labels by category so the model sees the taxonomy clearly."""
    by_cat: dict[str, list[str]] = {}
    for b in behaviors:
        by_cat.setdefault(b["category"], []).append(b["label"])
    taxonomy = "\n".join(
        f"  {cat}: " + ", ".join(labels) for cat, labels in by_cat.items()
    )
    return (
        "You are watching 4 still frames sampled in time order from a 20-second "
        "aquarium security-camera clip. The subject is Nity, an octopus "
        "(Octopus vulgaris). Octopuses crawl, swim, hide in a den, change color, "
        "extend arms to explore, manipulate objects, and interact with people.\n\n"
        "Look across all 4 frames to judge motion and what changes between them.\n\n"
        "If you cannot clearly see an octopus, or you are not confident an octopus "
        "is present, do NOT guess a behavior — report it as not visible.\n\n"
        "Choose the single BEST behavior from this ethogram:\n"
        f"{taxonomy}\n\n"
        "Respond in EXACTLY these three lines and nothing else:\n"
        "CAPTION: <one plain sentence: what Nity is doing across the frames; "
        "if no octopus is clearly visible, write 'Octopus is not clearly visible.'>\n"
        "BEHAVIOR: <one exact label from the ethogram above, or 'not_visible'>\n"
        "CONFIDENCE: <high | medium | low>"
    )


def sample_frames(clip: Path, tmpdir: str) -> list[str]:
    """Extract frames at FRAME_TIMES; returns list of jpg paths that succeeded."""
    paths = []
    for t in FRAME_TIMES:
        out = str(Path(tmpdir) / f"f_{t:02d}.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(t), "-i", str(clip),
             "-vframes", "1", "-q:v", "3", out],
            capture_output=True,
        )
        if r.returncode == 0 and Path(out).exists():
            paths.append(out)
    return paths


def parse_response(text: str, label_lookup: dict[str, str]) -> dict:
    """label_lookup maps lowercased label -> canonical label."""
    caption, behavior, confidence = "", "not_visible", "low"
    for line in text.splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith("CAPTION:"):
            caption = s.split(":", 1)[1].strip().strip("'\"")
        elif up.startswith("BEHAVIOR:"):
            raw = s.split(":", 1)[1].strip().strip("'\"").lower()
            if "not" in raw and ("visible" in raw or "present" in raw):
                behavior = "not_visible"
            else:
                behavior = "unknown"
                # exact match first, then substring either direction
                if raw in label_lookup:
                    behavior = label_lookup[raw]
                else:
                    for low, canon in label_lookup.items():
                        if low in raw or raw in low:
                            behavior = canon
                            break
        elif up.startswith("CONFIDENCE:"):
            c = s.split(":", 1)[1].strip().lower()
            confidence = next((k for k in ("high", "medium", "low") if k in c), "low")
    # If model hedged in the caption, normalize behavior to not_visible
    if re.search(r"not (clearly )?(visible|present)|no octopus", caption.lower()):
        behavior = "not_visible"
    return {"caption": caption, "behavior": behavior, "confidence": confidence}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=None,
                    help="process at most this many (smoke test)")
    args = ap.parse_args()

    ethogram     = json.load(open(ETHOGRAM_PATH))["behaviors"]
    label_lookup = {b["label"].lower(): b["label"] for b in ethogram}
    prompt       = build_prompt(ethogram)

    results   = json.load(open(CAPTIONS_PATH)) if CAPTIONS_PATH.exists() else []
    done      = {r["file"] for r in results}
    clips     = sorted(CLIPS_DIR.glob("*.mp4"))
    todo      = [c for c in clips if c.name not in done]
    if args.max_clips:
        todo = todo[: args.max_clips]

    print(f"{len(clips)} clips total | {len(done)} already done | {len(todo)} to process\n")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading {MODEL} (one-time) …", flush=True)
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(MODEL)
    config           = load_config(MODEL)
    n_imgs           = len(FRAME_TIMES)
    formatted        = apply_chat_template(processor, config, prompt, num_images=n_imgs)
    print("Model ready.\n" + "-" * 64, flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        for i, clip in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {clip.name}", flush=True)
            frames = sample_frames(clip, tmp)
            if len(frames) < n_imgs:
                print(f"  ! only {len(frames)} frames extracted, skipping", flush=True)
                continue
            try:
                out = generate(model, processor, formatted,
                               image=frames, max_tokens=MAX_TOKENS, verbose=False)
                raw = out.text if hasattr(out, "text") else str(out)
            except Exception as e:
                print(f"  ! inference failed: {e}", flush=True)
                continue

            parsed = parse_response(raw, label_lookup)
            print(f"  caption   : {parsed['caption']}")
            print(f"  behavior  : {parsed['behavior']}  ({parsed['confidence']})", flush=True)

            results.append({
                "file":         clip.name,
                "caption":      parsed["caption"],
                "behavior":     parsed["behavior"],
                "confidence":   parsed["confidence"],
                "raw_response": raw.strip(),
                "captioned_at": datetime.now().isoformat(timespec="seconds"),
            })
            with open(CAPTIONS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print("-" * 64)
    print(f"Done. {len(results)} results -> {CAPTIONS_PATH}\n")
    dist = Counter(r["behavior"] for r in results)
    print("Behavior distribution:")
    for label, n in dist.most_common():
        print(f"  {n:3d}  {label}")


if __name__ == "__main__":
    main()
