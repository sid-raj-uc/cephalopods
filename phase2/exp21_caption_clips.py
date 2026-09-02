"""
Exp 21 — Caption octopus clips with Qwen2-VL + ethogram mapping.

For each clip in data/octopus_clips/:
  1. Extract 1 frame at the midpoint (t=10s)
  2. Ask Qwen2-VL-2B to write a caption AND pick the best ethogram label
  3. Save all results to data/octopus_clips/captions.json

Resumable — skips clips already in captions.json.

Usage:
  python3 phase2/exp21_caption_clips.py
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime

PROJECT        = Path(__file__).resolve().parent.parent
CLIPS_DIR      = PROJECT / "data" / "octopus_clips"
ETHOGRAM_PATH  = PROJECT / "data" / "ethogram_list.json"
CAPTIONS_PATH  = CLIPS_DIR / "captions.json"
MODEL          = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
MIDPOINT_SEC   = 10  # frame extracted at this offset in the 20s clip


def build_prompt(_ethogram_labels: list[str]) -> str:
    return (
        "This is an aquarium security camera frame. The subject is Nity, an octopus (Octopus vulgaris). "
        "Octopuses change color, extend arms, hide in dens, manipulate objects, and interact with humans. "
        "Describe in ONE sentence exactly what Nity is doing — include her posture, arm position, color, "
        "and anything she is touching or interacting with. "
        "If Nity is not visible in the frame, say 'Nity is not visible'."
    )


# Keyword → ethogram label mapping for post-hoc classification
_ETHOGRAM_KEYWORDS = [
    (["crawl", "walking on arms", "moving across"],             "Crawling"),
    (["swim", "jet", "propel", "water column"],                 "Swimming / jetting"),
    (["arm walk", "two arm", "bipedal"],                        "Arm walking"),
    (["hunt", "stalk", "pursuit", "chasing"],                   "Hunting"),
    (["captur", "pounce", "grab", "catch", "seiz"],             "Capturing prey"),
    (["eat", "feeding", "consuming", "tearing food", "food"],   "Manipulating food"),
    (["entering den", "into den", "into shelter", "retreating into"],  "Entering den"),
    (["exiting den", "emerging", "leaving den", "coming out"],  "Exiting den"),
    (["rearrang", "piling", "moving shells", "moving rocks", "den entrance"], "Rearranging den"),
    (["extend", "probing", "reaching out", "arm out", "tentacle out"], "Arm extension / probing"),
    (["manipulat", "picking up", "holding object", "playing with"], "Object manipulation"),
    (["above water", "out of water", "water surface"],          "Reaching out of water"),
    (["human", "person", "hand", "researcher", "respond"],      "Responding to human"),
    (["joystick", "toy", "enrichment", "device", "screen"],     "Enrichment interaction"),
    (["color", "colour", "blanch", "darken", "chromatophore", "texture", "camouflage"], "Color / texture change"),
    (["ink", "cloud"],                                          "Ink release"),
    (["hid", "flatten", "conceal", "press"],                    "Hiding / flattening"),
    (["stationary", "resting", "motionless", "still", "not moving", "sitting in den", "inside den"], "Stationary in den"),
    (["stationary", "resting", "motionless", "still", "not moving", "open area", "tank floor"], "Stationary in open"),
]


def match_ethogram(text: str) -> str:
    t = text.lower()
    for keywords, label in _ETHOGRAM_KEYWORDS:
        if any(k in t for k in keywords):
            return label
    return "unknown"


def extract_midpoint_frame(clip_path: Path, tmpdir: str) -> str:
    out = str(Path(tmpdir) / "mid.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(MIDPOINT_SEC), "-i", str(clip_path),
         "-vframes", "1", "-q:v", "2", out],
        check=True,
    )
    return out


def parse_response(text: str, valid_labels: set) -> tuple[str, str]:
    caption, ethogram = "", "unknown"
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CAPTION:"):
            caption = line[len("CAPTION:"):].strip().strip("'\"")
        elif line.upper().startswith("ETHOGRAM:"):
            raw = line[len("ETHOGRAM:"):].strip().strip("'\"")
            # Find best matching label (case-insensitive substring match)
            raw_lower = raw.lower()
            for label in valid_labels:
                if label.lower() in raw_lower or raw_lower in label.lower():
                    ethogram = label
                    break
            else:
                ethogram = raw  # keep raw if no match
    return caption, ethogram


def main():
    ethogram  = json.load(open(ETHOGRAM_PATH))
    behaviors = ethogram["behaviors"]
    labels    = [b["label"] for b in behaviors]
    valid_set = set(labels)
    prompt    = build_prompt(labels)

    # Load existing captions (for resuming)
    if CAPTIONS_PATH.exists():
        results = json.load(open(CAPTIONS_PATH))
    else:
        results = []
    done_files = {r["file"] for r in results}

    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    todo  = [c for c in clips if c.name not in done_files]
    print(f"{len(clips)} clips total, {len(todo)} to caption\n")

    if not todo:
        print("All clips already captioned.")
        return

    # Load model once
    print(f"Loading {MODEL} …")
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(MODEL)
    config = load_config(MODEL)
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    print("Model ready.\n" + "─" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, clip in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {clip.name}", flush=True)

            try:
                frame = extract_midpoint_frame(clip, tmpdir)
            except Exception as e:
                print(f"  frame extraction failed: {e}")
                continue

            try:
                out     = generate(model, processor, formatted_prompt,
                                   image=frame, max_tokens=150, verbose=False)
                raw_text = out.text if hasattr(out, "text") else str(out)
            except Exception as e:
                print(f"  inference failed: {e}")
                continue

            caption       = raw_text.strip().strip("'\"")
            ethogram_label = match_ethogram(caption)
            print(f"  caption  : {caption}")
            print(f"  ethogram : {ethogram_label}", flush=True)

            results.append({
                "file":          clip.name,
                "caption":       caption,
                "ethogram":      ethogram_label,
                "raw_response":  raw_text,
                "captioned_at":  datetime.now().isoformat(timespec="seconds"),
            })

            # Save after every clip so progress isn't lost
            with open(CAPTIONS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    print("\n" + "─" * 60)
    print(f"Done. {len(results)} captions saved → {CAPTIONS_PATH}")

    # Summary by ethogram
    from collections import Counter
    counts = Counter(r["ethogram"] for r in results)
    print("\nEthogram distribution:")
    for label, n in counts.most_common():
        print(f"  {n:3d}  {label}")


if __name__ == "__main__":
    main()
