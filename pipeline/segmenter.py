"""
Run Grounded SAM 2 on each clip:
  1. Scan frames for the best DINO detection of 'octopus'.
  2. Initialise SAM 2 VideoPredictor on that keyframe.
  3. Propagate mask through the full clip.
  4. Save masks + confidence proxy as .npz in data/masks/.
"""

import logging
from pathlib import Path

import numpy as np
import torch

MASKS_DIR = Path("data/masks")
log = logging.getLogger(__name__)

# Frames to probe for the best DINO detection (as fractions of clip length)
_PROBE_FRACTIONS = [0.1, 0.25, 0.5, 0.75]


def _load_models(sam2_cfg: str, sam2_ckpt: str, gdino_cfg: str, gdino_ckpt: str, device: str):
    from groundingdino.util.inference import load_model as load_gdino
    from sam2.build_sam import build_sam2_video_predictor
    import groundingdino.datasets.transforms as GDT
    from groundingdino.util import box_ops

    gdino = load_gdino(gdino_cfg, gdino_ckpt).to(device)

    transform = GDT.Compose([
        GDT.RandomResize([800], max_size=1333),
        GDT.ToTensor(),
        GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sam2 = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)

    return gdino, transform, box_ops, sam2


def _best_detection(frames: list, gdino, transform, box_ops, device: str,
                    text_prompt: str, box_thresh: float, text_thresh: float,
                    probe_fractions: list):
    """Return (frame_idx, box_xyxy) for the highest-confidence detection."""
    from groundingdino.util.inference import predict as gdino_predict
    from PIL import Image

    n = len(frames)
    probe_idxs = sorted(set(int(f * (n - 1)) for f in probe_fractions))

    best = None  # (score, frame_idx, box)
    for idx in probe_idxs:
        img = Image.open(frames[idx]).convert("RGB")
        W, H = img.size
        tensor, _ = transform(img, None)
        boxes, logits, _ = gdino_predict(
            model=gdino, image=tensor, caption=text_prompt,
            box_threshold=box_thresh, text_threshold=text_thresh, device=device,
        )
        if len(boxes) == 0:
            continue
        top = logits.argmax()
        score = float(logits[top])
        if best is None or score > best[0]:
            box_abs = (box_ops.box_cxcywh_to_xyxy(boxes[top:top+1])
                       * torch.tensor([W, H, W, H])).squeeze(0)
            best = (score, idx, box_abs.numpy())

    return best  # None if nothing detected


def segment_clips(
    sam2_cfg: str   = "configs/sam2/sam2_hiera_l",
    sam2_ckpt: str  = "weights/sam2_hiera_large.pt",
    gdino_cfg: str  = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    gdino_ckpt: str = "weights/groundingdino_swint_ogc.pth",
    text_prompt: str = "octopus",
    box_thresh: float = 0.3,
    text_thresh: float = 0.25,
    device: str = None,
):
    from .manifest import mark_clip, pending_clips
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    clips = pending_clips()
    if not clips:
        log.info("No pending clips.")
        return

    log.info("Loading models on %s ...", device)
    gdino, transform, box_ops, sam2 = _load_models(
        sam2_cfg, sam2_ckpt, gdino_cfg, gdino_ckpt, device
    )

    cast = torch.bfloat16 if device == "cuda" else torch.float32

    for row in tqdm(clips, desc="Segmenting clips", unit="clip"):
        clip_id   = row["id"]
        frames_dir = Path(row["frames_dir"])
        frames    = sorted(frames_dir.glob("*.jpg"))

        if len(frames) < 2:
            mark_clip(clip_id, "skipped", error="too few frames")
            continue

        try:
            detection = _best_detection(
                frames, gdino, transform, box_ops, device,
                text_prompt, box_thresh, text_thresh, _PROBE_FRACTIONS,
            )
            if detection is None:
                mark_clip(clip_id, "skipped", error="no detection")
                log.info("No octopus detected in %s", clip_id)
                continue

            dino_score, keyframe_idx, box_xyxy = detection

            with torch.inference_mode(), torch.autocast(device, dtype=cast):
                state = sam2.init_state(video_path=str(frames_dir))
                sam2.reset_state(state)

                sam2.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=keyframe_idx,
                    obj_id=1,
                    box=torch.tensor(box_xyxy, dtype=torch.float32, device=device),
                )

                all_masks    = {}
                all_logit_max = {}
                for fidx, obj_ids, logits in sam2.propagate_in_video(state):
                    mask = (logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(logits[0, 0].max().cpu())
                    all_masks[fidx]     = mask
                    all_logit_max[fidx] = conf

            n = len(frames)
            masks_arr = np.stack([all_masks.get(i, np.zeros_like(list(all_masks.values())[0]))
                                  for i in range(n)])
            confs_arr = np.array([all_logit_max.get(i, 0.0) for i in range(n)], dtype=np.float32)

            out_path = MASKS_DIR / f"{clip_id}.npz"
            np.savez_compressed(
                out_path,
                masks=masks_arr,
                logit_confidence=confs_arr,
                box_xyxy=box_xyxy,
                dino_score=np.float32(dino_score),
                keyframe_idx=np.int32(keyframe_idx),
            )
            mark_clip(clip_id, "segmented")
            log.info("Saved %s (%d frames)", clip_id, n)

        except Exception as exc:
            mark_clip(clip_id, "failed", error=str(exc))
            log.warning("Failed %s: %s", clip_id, exc)
