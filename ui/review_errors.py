"""
Misclassification reviewer — move mislabeled frames to the correct folder.

Shows FP (hidden predicted as visible) and FN (visible predicted as hidden)
one frame at a time. Click "Label was WRONG" to move the image to the correct
folder and update manifest.csv. patches.json is never touched.

Usage: venv/bin/python3 ui/review_errors.py  →  http://localhost:8002
"""
import csv, shutil, threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT      = Path(__file__).resolve().parent.parent
FRAMES_DIR   = PROJECT / "data" / "frames"
MISC_DIR     = FRAMES_DIR / "misclassified"
MANIFEST     = FRAMES_DIR / "manifest.csv"

app   = FastAPI()
_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))

def save_manifest(rows: list[dict]):
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(rows)

def build_frame_list():
    """Return list of dicts: {path, kind, true_label, conf, patch_prefix}"""
    frames = []
    for kind in ("fp", "fn"):
        folder = MISC_DIR / kind
        if not folder.exists():
            continue
        true_label = "hidden" if kind == "fp" else "visible"
        pred_label = "visible" if kind == "fp" else "hidden"
        for f in sorted(folder.glob("*.jpg"), reverse=True):
            # filename: {conf}_{original_name}
            original = f.name.split("_", 1)[1]          # strip conf prefix
            prefix   = original.rsplit("_", 1)[0]        # strip _XXXX frame num
            frames.append({
                "path":        str(f),
                "filename":    f.name,
                "kind":        kind,
                "true_label":  true_label,
                "pred_label":  pred_label,
                "conf":        f.name.split("_")[0],
                "prefix":      prefix,
            })
    return frames

# ── state ─────────────────────────────────────────────────────────────────────

_frames  = build_frame_list()
_skipped = set()   # filenames skipped this session
_flipped = set()   # filenames flipped this session


def pending():
    return [f for f in _frames if f["filename"] not in _skipped and f["filename"] not in _flipped]


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/frame-image/{kind}/{filename}")
def serve_image(kind: str, filename: str):
    path = MISC_DIR / kind / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="image/jpeg")

@app.get("/status")
def status():
    queue = pending()
    return {
        "total":   len(_frames),
        "pending": len(queue),
        "skipped": len(_skipped),
        "flipped": len(_flipped),
        "fp_total": sum(1 for f in _frames if f["kind"] == "fp"),
        "fn_total": sum(1 for f in _frames if f["kind"] == "fn"),
    }

@app.get("/next")
def next_frame(kind: str = "all", offset: int = 0):
    queue = pending()
    if kind != "all":
        queue = [f for f in queue if f["kind"] == kind]
    if offset >= len(queue):
        return JSONResponse({"done": True})
    f = queue[offset]
    return {**f, "queue_len": len(queue), "done": False}

class Action(BaseModel):
    filename: str
    kind: str
    action: str   # "flip" or "skip"

@app.post("/action")
def do_action(body: Action):
    with _lock:
        if body.action == "skip":
            _skipped.add(body.filename)
            return {"ok": True}

        # flip: move image file to the correct folder + update manifest
        src_dir  = MISC_DIR / body.kind
        src_file = src_dir / body.filename

        # original filename without conf prefix
        original  = body.filename.split("_", 1)[1]
        old_label = "hidden" if body.kind == "fp" else "visible"
        new_label = "visible" if old_label == "hidden" else "hidden"

        # move from current label folder to correct label folder
        old_path = FRAMES_DIR / old_label / original
        new_path = FRAMES_DIR / new_label / original

        if not old_path.exists():
            return JSONResponse({"error": f"source not found: {old_path}"}, status_code=404)

        shutil.move(str(old_path), str(new_path))

        # update manifest
        rows = load_manifest()
        old_rel = str(old_path.relative_to(PROJECT))
        new_rel = str(new_path.relative_to(PROJECT))
        for row in rows:
            if row["path"] == old_rel:
                row["path"]  = new_rel
                row["label"] = new_label
                break
        save_manifest(rows)

        _flipped.add(body.filename)
        return {"ok": True, "old": old_label, "new": new_label}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Error Reviewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }

  #header { padding: 12px 20px; background: #1a1a1a; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 20px; flex-shrink: 0; }
  #header h1 { font-size: 16px; font-weight: 600; }
  #progress-bar { flex: 1; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
  #progress-fill { height: 100%; background: #4ade80; transition: width 0.3s; }
  #counter { font-size: 13px; color: #888; white-space: nowrap; }

  #filter-bar { padding: 8px 20px; background: #161616; border-bottom: 1px solid #2a2a2a; display: flex; gap: 8px; flex-shrink: 0; }
  .filter-btn { padding: 4px 14px; border-radius: 20px; border: 1px solid #444; background: transparent; color: #aaa; cursor: pointer; font-size: 13px; }
  .filter-btn.active { background: #2563eb; border-color: #2563eb; color: #fff; }

  #main { flex: 1; display: flex; overflow: hidden; }

  #img-panel { flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; min-width: 0; }
  #img-panel img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }

  #side { width: 280px; flex-shrink: 0; border-left: 1px solid #222; display: flex; flex-direction: column; padding: 20px; gap: 16px; overflow-y: auto; }

  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-fp { background: #7f1d1d; color: #fca5a5; }
  .badge-fn { background: #1e3a5f; color: #93c5fd; }

  .meta { font-size: 12px; color: #666; line-height: 1.8; }
  .meta span { color: #aaa; }

  .label-row { display: flex; gap: 8px; align-items: center; font-size: 13px; }
  .tag-hidden { background: #292929; color: #aaa; padding: 2px 8px; border-radius: 4px; }
  .tag-visible { background: #14532d; color: #4ade80; padding: 2px 8px; border-radius: 4px; }

  .divider { border-top: 1px solid #222; }

  #btn-flip { width: 100%; padding: 14px; border-radius: 8px; border: none; background: #dc2626; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
  #btn-flip:hover { background: #b91c1c; }
  #btn-skip { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #333; background: transparent; color: #888; font-size: 14px; cursor: pointer; }
  #btn-skip:hover { border-color: #555; color: #ccc; }

  #toast { position: fixed; bottom: 24px; right: 24px; padding: 10px 18px; border-radius: 8px; font-size: 13px; font-weight: 500; opacity: 0; transition: opacity 0.2s; pointer-events: none; }
  #toast.show { opacity: 1; }
  #toast.ok { background: #14532d; color: #4ade80; }
  #toast.err { background: #7f1d1d; color: #fca5a5; }

  #done-screen { display: none; flex: 1; align-items: center; justify-content: center; font-size: 20px; color: #4ade80; }
</style>
</head>
<body>

<div id="header">
  <h1>Error Reviewer</h1>
  <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
  <div id="counter">–</div>
</div>

<div id="filter-bar">
  <button class="filter-btn active" onclick="setFilter('all')">All</button>
  <button class="filter-btn" onclick="setFilter('fp')">FP — hidden predicted as visible</button>
  <button class="filter-btn" onclick="setFilter('fn')">FN — visible predicted as hidden</button>
</div>

<div id="main">
  <div id="img-panel"><img id="frame-img" src="" alt="frame"></div>
  <div id="side">
    <div>
      <span id="badge" class="badge">–</span>
    </div>
    <div class="meta">
      <div>Confidence: <span id="meta-conf">–</span></div>
      <div>True label: <span id="meta-true">–</span></div>
      <div>Model said: <span id="meta-pred">–</span></div>
    </div>
    <div class="divider"></div>
    <div class="label-row">
      Current label: <span id="label-tag" class="tag-hidden">–</span>
    </div>
    <div style="font-size:12px;color:#555;">If flipped → <span id="flip-preview">–</span></div>
    <div class="divider"></div>
    <button id="btn-flip" onclick="act('flip')">Label was WRONG — flip it</button>
    <button id="btn-skip" onclick="act('skip')">Label is correct — skip</button>
  </div>
  <div id="done-screen">All frames reviewed ✓</div>
</div>

<div id="toast"></div>

<script>
let filter = 'all';
let offset = 0;
let current = null;

function setFilter(f) {
  filter = f;
  offset = 0;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  loadNext();
}

async function loadNext() {
  const res = await fetch(`/next?kind=${filter}&offset=${offset}`);
  const data = await res.json();

  if (data.done) {
    document.getElementById('main').style.display = 'none';
    document.getElementById('done-screen').style.display = 'flex';
    document.getElementById('done-screen').style.flex = '1';
    updateProgress(0, 0);
    return;
  }

  current = data;
  document.getElementById('frame-img').src = `/frame-image/${data.kind}/${data.filename}`;
  document.getElementById('badge').textContent = data.kind.toUpperCase();
  document.getElementById('badge').className = `badge badge-${data.kind}`;
  document.getElementById('meta-conf').textContent = (parseFloat(data.conf) * 100).toFixed(1) + '%';
  document.getElementById('meta-true').textContent = data.true_label;
  document.getElementById('meta-pred').textContent = data.pred_label;

  const labelTag = document.getElementById('label-tag');
  labelTag.textContent = data.true_label;
  labelTag.className = `tag-${data.true_label}`;

  const flipped = data.true_label === 'visible' ? 'hidden' : 'visible';
  document.getElementById('flip-preview').textContent = flipped;

  updateProgress(data.queue_len, offset);
}

function updateProgress(total, done) {
  const statusRes = fetch('/status').then(r => r.json()).then(s => {
    const pct = s.total > 0 ? ((s.skipped + s.flipped) / s.total * 100) : 0;
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('counter').textContent =
      `${s.skipped + s.flipped} / ${s.total}  ·  ${s.flipped} flipped`;
  });
}

async function act(action) {
  if (!current) return;
  const res = await fetch('/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: current.filename, kind: current.kind, action}),
  });
  const data = await res.json();

  if (!res.ok) {
    showToast('Error: ' + (data.error || 'unknown'), 'err');
    return;
  }

  if (action === 'flip') {
    showToast(`Flipped: ${data.old} → ${data.new}`, 'ok');
  } else {
    showToast('Skipped', 'ok');
    offset++;
  }

  loadNext();
  updateProgress();
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `show ${type}`;
  setTimeout(() => t.className = '', 1800);
}

// keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.key === 'f' || e.key === 'F') act('flip');
  if (e.key === 's' || e.key === 'S' || e.key === ' ') { e.preventDefault(); act('skip'); }
  if (e.key === 'ArrowRight') act('skip');
});

loadNext();
updateProgress();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Loaded {len(_frames)} misclassified frames")
    print("  FP (hidden→visible):", sum(1 for f in _frames if f["kind"] == "fp"))
    print("  FN (visible→hidden):", sum(1 for f in _frames if f["kind"] == "fn"))
    print("\nOpen: http://localhost:8002")
    print("Shortcuts: F = flip label  |  S / Space / → = skip\n")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")
