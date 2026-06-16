"""
Light version: extract 1 frame every 30s, describe each frame with Qwen2-VL-2B-4bit.
Low memory — processes one frame at a time, model stays ~1.5GB.
"""
import subprocess, os, sys, json
from pathlib import Path

VIDEO  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nity_event_clip.mp4"
FRAMES_DIR = "/tmp/nity_frames"
INTERVAL   = 30   # seconds between frames
os.makedirs(FRAMES_DIR, exist_ok=True)

# 1. Get duration
dur = float(subprocess.check_output(
    ["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0", VIDEO],
    text=True).strip())
print(f"Video duration: {dur:.0f}s, extracting 1 frame every {INTERVAL}s", flush=True)

# 2. Extract frames
timestamps = list(range(0, int(dur), INTERVAL))
frame_paths = []
for t in timestamps:
    out = f"{FRAMES_DIR}/frame_{t:05d}.jpg"
    subprocess.run(["ffmpeg","-loglevel","error","-y","-ss",str(t),"-i",VIDEO,
                    "-vframes","1","-q:v","3", out], check=True)
    frame_paths.append((t, out))
print(f"Extracted {len(frame_paths)} frames", flush=True)

# 3. Load model once
print("Loading Qwen2-VL-2B-4bit (~1.5GB)...", flush=True)
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(MODEL)
config = load_config(MODEL)

PROMPT = ("Aquarium security camera frame. Subject: an octopus named Nity. "
          "Octopuses change color, extend arms, hide in dens, interact with objects and humans. "
          "State ONE of these: "
          "(A) 'Nity is [doing X]' — describe her exact posture, arm position, color, and what she is touching or interacting with. "
          "(B) 'Nity is not visible' — only if she is clearly absent from the frame. "
          "Do not hedge. Give a direct observation.")

# 4. Describe each frame
print("\n" + "="*60, flush=True)
results = []
for t, path in frame_paths:
    mins, secs = divmod(t, 60)
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)
    out = generate(model, processor, formatted, image=path, max_tokens=120, verbose=False)
    label = f"[{mins:02d}:{secs:02d}]"
    print(f"{label} {out}", flush=True)
    results.append({"time": f"{mins:02d}:{secs:02d}", "t_sec": t, "description": out.text})

# 5. Save JSON
out_json = "/tmp/nity_captions.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print("="*60)
print(f"Saved to {out_json}", flush=True)
