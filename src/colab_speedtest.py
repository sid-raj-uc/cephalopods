"""
Footage-server speed test — run on Colab (or any cloud box) to decide the pipeline architecture.

Measures 3 things against repo.octopus-intelligence.org:
  1. time-to-first-byte (latency)
  2. raw HTTP download throughput (MB/s) over the first ~150 MB
  3. ffmpeg STREAMING-SCAN throughput — the real pipeline pattern (HTTP -> 1fps frames),
     i.e. "how fast can we sip frames without downloading the whole file"

Setup on Colab (one cell):
    !pip -q install requests
    import os; os.environ["OCTOPUS_USER"]="..."; os.environ["OCTOPUS_PASS"]="..."   # or getpass below
    !ffmpeg -version >/dev/null 2>&1 || apt-get -qq install ffmpeg
Then run this file.
"""
import base64, os, subprocess, sys, time, getpass, shutil

# --- creds (env first, else prompt) ---
USER = os.environ.get("OCTOPUS_USER") or getpass.getpass("OCTOPUS_USER: ")
PASS = os.environ.get("OCTOPUS_PASS") or getpass.getpass("OCTOPUS_PASS: ")
AUTH = "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()

# a few real den-camera videos from the index (override with argv[1])
URLS = [
    "https://repo.octopus-intelligence.org/public/O-vulgaris-Nity-2026-2-20--/Right%20Top/Local/2026-02-24/163003--vv-1.mp4",
    "https://repo.octopus-intelligence.org/public/O-vulgaris-Nity-2026-2-20--/Right%20Left/Local/2026-02-20/092420--av-1.mp4",
]
URL = sys.argv[1] if len(sys.argv) > 1 else URLS[0]
RAW_MB = 150            # how many MB to pull for the raw-throughput test
STREAM_SECONDS = 120    # how many seconds of video to stream-scan at 1 fps

import requests


def human(n):  # bytes -> MB
    return n / (1024 * 1024)


def test_headers():
    print(f"URL: {URL}")
    t0 = time.time()
    r = requests.head(URL, headers={"Authorization": AUTH}, allow_redirects=True, timeout=30)
    size = int(r.headers.get("Content-Length", 0))
    print(f"  HEAD {r.status_code} in {time.time()-t0:.2f}s | size {human(size):.0f} MB | "
          f"range-support={r.headers.get('Accept-Ranges','?')}")
    return size


def test_raw():
    """time-to-first-byte + throughput over the first RAW_MB."""
    hdr = {"Authorization": AUTH, "Range": f"bytes=0-{RAW_MB*1024*1024-1}"}
    t0 = time.time(); got = 0; ttfb = None
    with requests.get(URL, headers=hdr, stream=True, timeout=60) as r:
        for chunk in r.iter_content(1 << 20):
            if ttfb is None:
                ttfb = time.time() - t0
            got += len(chunk)
    dt = time.time() - t0
    print(f"\n[RAW DOWNLOAD]")
    print(f"  time-to-first-byte: {ttfb:.2f}s")
    print(f"  downloaded {human(got):.0f} MB in {dt:.1f}s  ->  {human(got)/dt:.1f} MB/s")
    return human(got) / dt


def test_stream_scan():
    """ffmpeg HTTP -> 1fps 224x224 rgb frames; read STREAM_SECONDS frames; measure wall time.
    This is what the fused pipeline does (sip frames, early-exit) — the number that matters."""
    if not shutil.which("ffmpeg"):
        print("\n[STREAM SCAN] ffmpeg not found — skip (apt-get install ffmpeg)"); return None
    cmd = ["ffmpeg", "-loglevel", "error",
           "-headers", f"Authorization: {AUTH}\r\n",
           "-i", URL, "-vf", "fps=1,scale=224:224",
           "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"]
    fsize = 224 * 224 * 3
    t0 = time.time(); ttff = None; n = 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while n < STREAM_SECONDS:
        raw = p.stdout.read(fsize)
        if len(raw) < fsize:
            break
        if ttff is None:
            ttff = time.time() - t0
        n += 1
    dt = time.time() - t0
    p.kill(); p.wait()
    print(f"\n[STREAM SCAN]  (the pipeline pattern: sip frames, no full download)")
    print(f"  time-to-first-frame: {ttff:.2f}s" if ttff else "  no frames received")
    if n:
        print(f"  scanned {n} s of video (1fps) in {dt:.1f}s  ->  {n/dt:.1f} video-seconds/s")
        print(f"  => a 30-min video would scan in ~{1800/(n/dt)/60:.1f} min "
              f"(less with early-exit at 2 clips)")
    return n / dt if n else 0


if __name__ == "__main__":
    print("=" * 60)
    test_headers()
    raw = test_raw()
    stream = test_stream_scan()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  raw download   : {raw:.1f} MB/s")
    if stream:
        print(f"  stream-scan    : {stream:.1f} video-sec/s  "
              f"({'FAST — fused streaming pipeline viable' if stream >= 3 else 'slow — may prefer download-then-scan'})")
    print("  (compare to the Mac's ~10-13 MB/s over the IAP tunnel we measured earlier)")
