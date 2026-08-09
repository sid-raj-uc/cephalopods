"""Modal GPU app — auto-label the harvested clips (GroundingDINO+SAM2 teacher), then train the
tiny octopus segmenter. The 530 harvested clips (276 new videos / 149 dates) already live on the
`octopus-harvest-vol` volume at /harvest, so we compute where the data is instead of moving ~20 GB.

Layout on the volume (/data == volume root):
  /data/harvest/<collection>/<date>/<segment>/<Camera>_s-e.mp4   the harvested clips (input)
  /data/dataset_seg_harvest/{images,masks,manifest.jsonl}        auto-labeled pairs (autolabel output)
  /data/hf_cache/                                                cached GroundingDINO/SAM2 weights
  /data/weights/octo_seg_<ver>_<arch>.pt                         trained segmenter (train output)

Run (sidraj profile — that's where the volume lives):
  # 1) validate the teacher on a few clips (also builds the image + caches HF models)
  MODAL_PROFILE=sidraj modal run src/modal_seg_train.py::autolabel --limit 5
  # 2) full auto-label of all harvested clips (detached — ~2-3 h)
  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::autolabel
  # 3) train the segmenter on the new (diverse) labels, split BY SOURCE VIDEO
  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::train --epochs 60

Fetch the trained model:
  MODAL_PROFILE=sidraj modal volume get octopus-harvest-vol /weights ./weights_dl
"""
import modal

HERE = __file__.rsplit("/", 1)[0]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", "torchvision", "numpy", "pillow", "opencv-python-headless",
        "transformers", "tqdm", "hydra-core", "iopath",
    )
    # sam2 has no nvcc on the box -> skip its optional CUDA extension (benign _C import warning)
    .run_commands("SAM2_BUILD_CUDA=0 pip install 'git+https://github.com/facebookresearch/sam2.git'")
    .add_local_file(f"{HERE}/auto_segment.py", "/root/auto_segment.py")
    .add_local_file(f"{HERE}/train_segmenter.py", "/root/train_segmenter.py")
)

app = modal.App("octopus-seg-train")
vol = modal.Volume.from_name("octopus-harvest-vol", create_if_missing=True)
GPU = "A10G"  # GD-tiny + SAM2-tiny are small; A10G (24 GB) is plenty and cheap


@app.function(image=image, gpu=GPU, timeout=86400, volumes={"/data": vol})
def autolabel(limit: int = 0, min_seed_conf: float = 0.60,
              cameras: str = "Right_Front,Right_Back", out: str = "/data/dataset_seg_harvest",
              clips_root: str = "/data/harvest", gd_model: str = "tiny", sam2_model: str = "tiny",
              seed_mode: str = "gd", debug_n: int = 0):
    """Teacher over clips_root -> (image, mask) pairs on the volume. Resumable.

    gd_model/sam2_model pick the teacher size: 'base'+'large' = HQ teacher (better seed boxes + sharper
    masks) to raise the label-quality ceiling the student is capped by. debug_n dumps N seed overlays.
    """
    import os, subprocess, glob
    os.environ["HF_HOME"] = "/data/hf_cache"      # cache GD/SAM2 weights on the volume (persist across runs)
    cmd = ["python", "/root/auto_segment.py",
           "--clips-root", clips_root, "--out", out,
           "--cameras", *cameras.split(","),
           "--min-seed-conf", str(min_seed_conf), "--seed-mode", seed_mode,
           "--gd-model", gd_model, "--sam2-model", sam2_model]
    if limit:
        cmd += ["--limit", str(limit)]
    if debug_n:
        cmd += ["--debug-dir", f"{out}/dbg", "--debug-n", str(debug_n)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    n = len(glob.glob(f"{out}/images/*.jpg"))
    print(f"TOTAL PAIRS in {out}: {n}", flush=True)
    return n


@app.function(image=image, gpu=GPU, timeout=86400, volumes={"/data": vol})
def train(epochs: int = 60, arch: str = "lraspp", base_ch: int = 16,
          ds: str = "/data/dataset_seg_harvest", ver: str = "harvest", sources: str = "",
          in_size: int = 256, loss: str = "dice_bce", holdout: str = ""):
    """Train the tiny segmenter on a labeled dataset dir (split BY SOURCE VIDEO). -> /data/weights.

    `ds` may be a comma-separated list of dataset dirs — they're merged (symlinked images/masks +
    concatenated manifests) into /data/dataset_seg_<ver> before training. Old/new filenames don't
    collide (old: Right_Front_s-e_..., new: date_segment_...), and source_video() still groups by
    date/segment so the train/val split stays leakage-free across the merged set.
    """
    import subprocess, os, glob
    ds_dirs = [d.strip() for d in ds.split(",") if d.strip()]
    if len(ds_dirs) > 1:
        merged = f"/data/dataset_seg_{ver}"
        os.makedirs(f"{merged}/images", exist_ok=True)
        os.makedirs(f"{merged}/masks", exist_ok=True)
        n = 0
        with open(f"{merged}/manifest.jsonl", "w") as out_mf:
            for d in ds_dirs:
                for sub, ext in (("images", "*.jpg"), ("masks", "*.png")):
                    for f in glob.glob(f"{d}/{sub}/{ext}"):
                        dst = f"{merged}/{sub}/{os.path.basename(f)}"
                        if not os.path.exists(dst):
                            os.symlink(f, dst)
                with open(f"{d}/manifest.jsonl") as mf:
                    for line in mf:
                        if line.strip():
                            out_mf.write(line); n += 1
        print(f"MERGED {ds_dirs} -> {merged}  ({n} manifest rows)", flush=True)
        ds = merged
    else:
        ds = ds_dirs[0]
    os.makedirs("/data/weights", exist_ok=True)
    out = f"/data/weights/octo_seg_{ver}_{arch}.pt"
    cmd = ["python", "/root/train_segmenter.py", "--ds", ds, "--ver", ver,
           "--arch", arch, "--base-ch", str(base_ch), "--epochs", str(epochs), "--out", out,
           "--in-size", str(in_size), "--loss", loss]
    if sources:
        cmd += ["--sources", sources]
    if holdout:
        cmd += ["--holdout-videos", holdout]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    print("SAVED:", out, flush=True)
    return out


@app.function(image=image, timeout=1200, volumes={"/data": vol})
def montage(ds: str = "/data/dataset_seg_harvest_hq", n: int = 30, cols: int = 6,
            out: str = "/data/gt_montage.jpg", th: int = 240, seed: int = 0):
    """Tile n (image, teacher-mask) pairs with the mask overlaid green -> a single montage image on the
    volume, so a human can eyeball whether the auto-labeled 'ground truth' is actually correct."""
    import glob, cv2, numpy as np
    imgs = sorted(glob.glob(f"{ds}/images/*.jpg"))
    if not imgs:
        print("NO IMAGES in", ds); return None
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    sel = [imgs[i] for i in idx]
    tiles, TW = [], int(th * 16 / 9)  # assume ~16:9; letterbox to fixed cell
    for ip in sel:
        mp = ip.replace("/images/", "/masks/").rsplit(".", 1)[0] + ".png"
        im = cv2.imread(ip); m = cv2.imread(mp, 0)
        if im is None or m is None:
            continue
        m = m > 127
        ov = im.astype(np.float32)
        ov[m] = 0.5 * ov[m] + 0.5 * np.array([0.0, 235.0, 120.0])  # green (BGR)
        # red mask contour for precise boundary read
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ov = ov.astype(np.uint8); cv2.drawContours(ov, cnts, -1, (0, 0, 255), 2)
        h, w = ov.shape[:2]; s = min(TW / w, th / h)
        rw, rh = int(w * s), int(h * s)
        cell = np.full((th, TW, 3), 30, np.uint8)
        r = cv2.resize(ov, (rw, rh)); y0 = (th - rh) // 2; x0 = (TW - rw) // 2
        cell[y0:y0 + rh, x0:x0 + rw] = r
        tag = "/".join(ip.split("/")[-1].split("_")[:2])  # date_segment-ish
        cv2.putText(cell, f"{int(m.mean()*1000)/10}%", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        tiles.append(cell)
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * th, cols * TW, 3), 20, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[r*th:(r+1)*th, c*TW:(c+1)*TW] = t
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    vol.commit()
    print(f"MONTAGE {len(tiles)} tiles -> {out}  ({grid.shape[1]}x{grid.shape[0]})")
    return out


@app.local_entrypoint()
def main():
    print("Call a function directly, e.g.:")
    print("  MODAL_PROFILE=sidraj modal run src/modal_seg_train.py::autolabel --limit 5")
    print("  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::autolabel")
    print("  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::train --epochs 60")
