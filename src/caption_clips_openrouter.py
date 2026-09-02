"""
Caption extracted octopus clips via OpenRouter (Qwen3-VL-235B) — local, no GPU.

Reads a clip index (e.g. octopus_clips_verified.json), samples a few frames from
each already-extracted local clip, sends them to Qwen3-VL-235B-A22B on OpenRouter,
and writes a NEW caption JSON (same schema + `caption` / `ethogram_label`).

- Frames are brightness/contrast-lifted (ffmpeg `eq`) to help the dim IR footage.
- Ethogram labels come from the 7-class ethogram_list_v2.json (or "octopus not
  present" / "uncertain").
- Resumable: skips clips already captioned in the output; saves after every clip.
- Prints running token + $ totals (rates: $0.20/M in, $0.88/M out).

Setup:
  pip install requests            # + ffmpeg on PATH
  export OPENROUTER_API_KEY=...   # or put it in src/.env  (never commit it)

Usage:
  python3 caption_clips_openrouter.py                     # caption all clips in the index
  python3 caption_clips_openrouter.py --limit 20          # calibration batch
  python3 caption_clips_openrouter.py --input ../data/octopus_clips_verified.json
"""
import os, sys, json, base64, subprocess, tempfile, argparse, datetime, time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                  # clip_path values are repo-root-relative

MODEL       = "qwen/qwen3-vl-235b-a22b-instruct"
IN_PRICE    = 0.20 / 1_000_000                      # $/token, OpenRouter (verify current)
OUT_PRICE   = 0.88 / 1_000_000
N_FRAMES    = 6
MAX_SIDE    = 512                                   # long-side px per frame (drives cost)
MAX_TOKENS  = 200
ETHOGRAM    = REPO / "data" / "ethogram_list_v2.json"     # the existing ethogram sheet
API_URL     = "https://openrouter.ai/api/v1/chat/completions"


# ── creds (env, then src/.env) ────────────────────────────────────────────────
def _load_env(p: Path):
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
# repo-root .env first — it is canonical; src/.env is the standalone-package fallback (see
# caption_openrouter.py for the 401 trap this ordering prevents)
_load_env(HERE.parent / ".env")
_load_env(HERE / ".env")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ── ethogram + prompt ─────────────────────────────────────────────────────────
_e = json.load(open(ETHOGRAM))
LABELS = [b["label"] for b in _e["behaviors"]]
VALID = set(LABELS)
_LABEL_BLOCK = "\n".join(f"- {b['label']}: {b['description']}" for b in _e["behaviors"])

def build_prompt() -> str:
    return (
        "These frames are sampled in order from one short aquarium clip of Nity, an "
        "octopus (Octopus vulgaris) in a dim IR-lit tank. Describe ONLY what you can see.\n\n"
        "If no octopus is visible in any frame, respond EXACTLY:\n"
        "  CAPTION: octopus not present\n  ETHOGRAM: octopus not present\n"
        "Otherwise write ONE caption of what the octopus does across the clip, then pick the "
        "single best behavior label below (or 'uncertain' if genuinely unclear):\n"
        f"{_LABEL_BLOCK}\n\n"
        "Respond in EXACTLY this format:\n"
        "CAPTION: <one sentence>\nETHOGRAM: <one label verbatim, or 'uncertain', or 'octopus not present'>"
    )

def parse_response(text: str):
    caption, etho = "", None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("CAPTION:"):
            caption = s[8:].strip().strip("'\"")
        elif s.upper().startswith("ETHOGRAM:"):
            raw = s[9:].strip().strip("'\""); rl = raw.lower()
            if "not present" in rl:
                etho = "octopus not present"
            elif rl in ("uncertain", "unclear", "unknown"):
                etho = "uncertain"
            else:
                for lab in VALID:
                    if lab.lower() == rl or lab.lower() in rl or rl in lab.lower():
                        etho = lab; break
                else:
                    etho = raw or "uncertain"
    if not caption:
        caption = text.strip().strip("'\"")
    if "not present" in caption.lower():
        return "octopus not present", "octopus not present"
    return caption, (etho or "uncertain")


# ── frames -> base64 data URIs ────────────────────────────────────────────────
def frame_data_uris(clip_file: Path):
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_file),
                        "-vf", f"fps=1,scale='min({MAX_SIDE},iw)':-2,eq=contrast=1.25:brightness=0.06",
                        "-q:v", "3", f"{t}/f_%03d.jpg"], capture_output=True)
        fs = sorted(Path(t).glob("f_*.jpg"))
        if not fs:
            return []
        import numpy as np
        idx = np.linspace(0, len(fs) - 1, min(N_FRAMES, len(fs))).round().astype(int)
        uris = []
        for i in idx:
            b = fs[i].read_bytes()
            uris.append("data:image/jpeg;base64," + base64.b64encode(b).decode())
        return uris


def caption_via_openrouter(uris, prompt):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in uris]
    content.append({"type": "text", "text": prompt})
    body = {"model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(4):
        r = requests.post(API_URL, headers=headers, json=body, timeout=120)
        if r.status_code == 200:
            j = r.json()
            usage = j.get("usage", {})
            return j["choices"][0]["message"]["content"].strip(), usage
        if r.status_code in (429, 502, 503):        # rate-limit / transient -> backoff
            time.sleep(2 * (attempt + 1)); continue
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
    raise RuntimeError("OpenRouter: repeated failures")


def resolve_clip(clip_path: str) -> Path:
    for cand in (Path(clip_path), REPO / clip_path, HERE / clip_path):
        if cand.exists():
            return cand
    return REPO / clip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO / "data" / "octopus_clips_verified.json"))
    ap.add_argument("--output", default=str(REPO / "data" / "octopus_clips_captioned.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("Set OPENROUTER_API_KEY (env or src/.env).")

    src = json.load(open(args.input))
    src_clips = src["clips"] if isinstance(src, dict) else src

    out_path = Path(args.output)
    if out_path.exists():
        out = json.load(open(out_path))
    else:
        out = {"description": f"OpenRouter {MODEL} captions of extracted octopus clips.",
               "model": MODEL, "source_index": args.input,
               "updated_at": None, "count": 0, "clips": []}
    done = {c["clip_path"] for c in out["clips"] if c.get("caption")}

    todo = [c for c in src_clips if c["clip_path"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    prompt = build_prompt()
    print(f"{len(src_clips)} clips in index | {len(done)} already captioned | {len(todo)} to do", flush=True)
    print(f"model: {MODEL}\n" + "-" * 60, flush=True)

    tot_in = tot_out = 0; cost = 0.0; n = 0
    for i, c in enumerate(todo, 1):
        clip_file = resolve_clip(c["clip_path"])
        if not clip_file.exists():
            print(f"[{i}/{len(todo)}] MISSING {c['clip_path']}"); continue
        try:
            uris = frame_data_uris(clip_file)
            if not uris:
                print(f"[{i}/{len(todo)}] no frames {c['clip_path']}"); continue
            text, usage = caption_via_openrouter(uris, prompt)
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED: {e}"); continue

        caption, etho = parse_response(text)
        pt = usage.get("prompt_tokens", 0); ct = usage.get("completion_tokens", 0)
        tot_in += pt; tot_out += ct
        cost += pt * IN_PRICE + ct * OUT_PRICE; n += 1

        entry = {k: c.get(k) for k in ("video", "video_url", "date", "segment", "camera",
                                       "start_sec", "end_sec", "video_timeline",
                                       "visible_frac", "mean_motion", "clip_path")}
        entry.update({"caption": caption, "ethogram_label": etho,
                      "caption_model": MODEL,
                      "captioned_at": datetime.datetime.now().isoformat(timespec="seconds")})
        out["clips"] = [x for x in out["clips"] if x["clip_path"] != c["clip_path"]] + [entry]
        out["count"] = len(out["clips"])
        out["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        json.dump(out, open(out_path, "w"), indent=2)

        print(f"[{i}/{len(todo)}] {etho:26s} | {caption[:60]}", flush=True)
        if i % 25 == 0:
            print(f"    ... {n} done | tokens {tot_in}+{tot_out} | est ${cost:.3f} "
                  f"| ${cost/max(1,n):.5f}/clip", flush=True)

    print("-" * 60)
    print(f"DONE. captioned {n} clips this run.")
    print(f"tokens: {tot_in} in + {tot_out} out | est cost ${cost:.4f} "
          f"({'$%.5f' % (cost/n) if n else '-'}/clip)")
    print(f"output -> {out_path}  ({out['count']} clips total)")


if __name__ == "__main__":
    main()
