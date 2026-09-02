"""Modal app — run the streaming footage harvester on a CPU container.

Downloads are network-bound (~5 MB/s server cap) and empties stream ~fully, so this is a
long, cheap CPU job. Writes clips + ledger to a Modal Volume; commits periodically so it's
resumable across timeouts/re-runs (the ledger skips already-scanned video_urls).

Setup (once):
  MODAL_PROFILE=sidraj modal secret create octopus-creds \
      OCTOPUS_USER=octopus OCTOPUS_PASS=communication42

Validate (small):
  MODAL_PROFILE=sidraj modal run src/modal_harvest.py --limit 15
Full run (Nity colour, ~209 days):
  MODAL_PROFILE=sidraj modal run --detach src/modal_harvest.py
Fetch results:
  MODAL_PROFILE=sidraj modal volume get octopus-harvest /harvest ./harvest_dl
"""
import modal

HERE = __file__.rsplit("/", 1)[0]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch", "torchvision", "numpy", "pillow", "requests", "opencv-python-headless",
                 "ftfy", "regex", "tqdm", "setuptools<81", "packaging", "openai-clip")
    .add_local_file(f"{HERE}/clip_mlp_hardneg_v2.pt", "/root/clip_mlp_hardneg_v2.pt")
    .add_local_file(f"{HERE}/harvest_stream.py", "/root/harvest_stream.py")
)
app = modal.App("octopus-harvest")
vol = modal.Volume.from_name("octopus-harvest-vol", create_if_missing=True)


@app.function(image=image, cpu=4.0, timeout=86400, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("octopus-creds")])
def harvest(limit: int = 0, workers: int = 2, max_seg: int = 3, max_scan_sec: int = 0):
    import sys
    sys.path.insert(0, "/root")
    import harvest_stream as H
    # module reads creds from env at import; secret injects them before this runs
    return H.run(out="/data/harvest", ckpt="/root/clip_mlp_hardneg_v2.pt",
                 collections=H.NITY_COLLECTIONS, workers=workers, max_seg=max_seg,
                 limit=limit, max_scan_sec=max_scan_sec, commit_cb=vol.commit)


@app.local_entrypoint()
def main(limit: int = 0, workers: int = 2, max_seg: int = 3, max_scan_sec: int = 0):
    res = harvest.remote(limit=limit, workers=workers, max_seg=max_seg, max_scan_sec=max_scan_sec)
    print("RESULT:", res)
