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
              cameras: str = "Right_Front,Right_Back", out: str = "/data/dataset_seg_harvest"):
    """Teacher over /data/harvest clips -> (image, mask) pairs on the volume. Resumable."""
    import os, subprocess, glob
    os.environ["HF_HOME"] = "/data/hf_cache"      # cache GD/SAM2 weights on the volume (persist across runs)
    cmd = ["python", "/root/auto_segment.py",
           "--clips-root", "/data/harvest", "--out", out,
           "--cameras", *cameras.split(","),
           "--min-seed-conf", str(min_seed_conf)]
    if limit:
        cmd += ["--limit", str(limit)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    n = len(glob.glob(f"{out}/images/*.jpg"))
    print(f"TOTAL PAIRS in {out}: {n}", flush=True)
    return n


@app.function(image=image, gpu=GPU, timeout=86400, volumes={"/data": vol})
def train(epochs: int = 60, arch: str = "lraspp", base_ch: int = 16,
          ds: str = "/data/dataset_seg_harvest", ver: str = "harvest"):
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
           "--arch", arch, "--base-ch", str(base_ch), "--epochs", str(epochs), "--out", out]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    print("SAVED:", out, flush=True)
    return out


@app.local_entrypoint()
def main():
    print("Call a function directly, e.g.:")
    print("  MODAL_PROFILE=sidraj modal run src/modal_seg_train.py::autolabel --limit 5")
    print("  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::autolabel")
    print("  MODAL_PROFILE=sidraj modal run --detach src/modal_seg_train.py::train --epochs 60")
