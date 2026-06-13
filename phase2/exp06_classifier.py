"""
Experiment 6: Train a binary octopus-presence classifier.

Pipeline:
1. Scan all 5 sessions at 1fps, extract tight-patch DINOv2 features (bgsub)
2. Auto-label: positive = sim_sub > 0.55 from confirmed sessions;
               negative = 112421 (Nity absent) + low-sim frames from other sessions
3. Train logistic regression (L2) on 768-d DINOv2 features
4. Leave-one-session-out cross-validation
5. Apply trained model to all sessions, save top detection frames per session

Memory: DINOv2 ~300MB + feature cache to disk — safe on 16GB.
"""

import sys, os, subprocess, pickle
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

PROJECT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR  = os.path.join(PROJECT, "data/aquarium/full/2026-02-20")
OUT_DIR   = os.path.join(PROJECT, "report/experiments/exp06_frames")
FEAT_DIR  = os.path.join(PROJECT, "data/phase2/exp06_features")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

TIMESTAMPS = ["095420", "102421", "112421", "122421", "132421"]
CAMERA     = "Right_Front"
PROC_W     = 800
SAMPLE_FPS = 1.0   # 1fps for denser coverage
OX, OY     = 0.614, 0.640
HALF       = 0.04

# Labeling thresholds (from Exp 5 results)
POS_THRESH = 0.55   # sim_sub above this → likely positive
NEG_THRESH = 0.49   # sim_sub below this → likely negative
# 112421 is always negative (Nity not at den)
ALWAYS_NEG = {"112421"}
# Confirmed positive sessions
CONFIRMED_POS_SESSIONS = {"095420", "102421", "122421", "132421"}

transform_dino = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_frame(video_path, t_sec, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0])); h += h % 2
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(t_sec), "-i", video_path, "-vframes", "1",
           "-vf", f"scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read(); proc.wait()
    if len(raw) < width * h * 3: return None, h
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy(), h


def stream_frames(video_path, fps=1.0, width=PROC_W):
    probe = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path]
    ).decode().strip().split(",")
    h = int(int(probe[1]) * width / int(probe[0])); h += h % 2
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", video_path, "-vf", f"fps={fps},scale={width}:{h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * h * 3
    t, step = 0.0, 1.0 / fps
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes: break
            yield t, np.frombuffer(raw, dtype=np.uint8).reshape(h, width, 3).copy(), h
            t += step
    finally:
        proc.kill(); proc.wait()


def tight_crop(frame, h):
    cx = int(OX * PROC_W); cy = int(OY * h)
    hw = int(HALF * PROC_W); hh = int(HALF * h)
    return frame[max(0,cy-hh):min(h,cy+hh), max(0,cx-hw):min(PROC_W,cx+hw)]


@torch.no_grad()
def dino_embed(model, bgr_patch):
    rgb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform_dino(pil).unsqueeze(0)
    out = model.forward_features(x)
    v = out['x_norm_clstoken'][0].cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b))


def build_background(ts_list, n_frames=40):
    samples = []
    for ts in ts_list:
        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid): continue
        for t in np.linspace(60, 1200, n_frames // len(ts_list)):
            frame, h = get_frame(vid, int(t))
            if frame is None: continue
            patch = tight_crop(frame, h).astype(np.float32)
            if patch.size > 0: samples.append(patch)
    if not samples: return None
    target_h, target_w = samples[0].shape[:2]
    resized = [cv2.resize(s, (target_w, target_h)) for s in samples]
    return np.median(np.stack(resized, axis=0), axis=0).astype(np.float32)


def extract_features(model, bg, query_vec):
    """Extract features for all sessions, return dict of {ts: (features, sim_subs, timestamps)}."""
    all_feats = {}
    for ts in TIMESTAMPS:
        cache_path = os.path.join(FEAT_DIR, f"{ts}_feats.npz")
        if os.path.exists(cache_path):
            print(f"  [{ts}] Loading from cache ...")
            d = np.load(cache_path)
            all_feats[ts] = (d['features'], d['sim_subs'], d['timestamps'])
            print(f"    {len(d['timestamps'])} frames, max_sim={d['sim_subs'].max():.4f}")
            continue

        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid):
            print(f"  [{ts}] SKIP: not found"); continue

        print(f"  [{ts}] Extracting features at {SAMPLE_FPS}fps ...")
        feats, sims, t_arr = [], [], []

        for t, frame, h in stream_frames(vid, fps=SAMPLE_FPS, width=PROC_W):
            patch = tight_crop(frame, h)
            patch_f = cv2.resize(patch.astype(np.float32), (int(bg.shape[1]), int(bg.shape[0])))
            patch_sub = np.clip(patch_f - bg + 128, 0, 255).astype(np.uint8)
            vec = dino_embed(model, patch_sub)
            sim = cosine(query_vec, vec)
            feats.append(vec)
            sims.append(sim)
            t_arr.append(t)

        feats = np.array(feats, dtype=np.float32)
        sims  = np.array(sims,  dtype=np.float32)
        t_arr = np.array(t_arr, dtype=np.float32)
        np.savez(cache_path, features=feats, sim_subs=sims, timestamps=t_arr)
        print(f"    Done: {len(t_arr)} frames, max_sim={sims.max():.4f}")
        all_feats[ts] = (feats, sims, t_arr)

    return all_feats


def make_labels(all_feats):
    """Auto-label frames. Returns X (features), y (0/1), session_ids, timestamps."""
    X, y, sessions, times = [], [], [], []

    for ts, (feats, sims, t_arr) in all_feats.items():
        for i, (feat, sim, t) in enumerate(zip(feats, sims, t_arr)):
            if ts in ALWAYS_NEG:
                label = 0
            elif ts in CONFIRMED_POS_SESSIONS and sim >= POS_THRESH:
                label = 1
            elif sim <= NEG_THRESH:
                label = 0
            else:
                continue   # ambiguous zone — skip
            X.append(feat)
            y.append(label)
            sessions.append(ts)
            times.append(t)

    return np.array(X), np.array(y), sessions, np.array(times)


def main():
    print("Loading DINOv2 ViT-B/14 ...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    model.eval()
    print("Loaded.\n")

    # Background + query
    print("Building background ...")
    bg = build_background(TIMESTAMPS, n_frames=40)
    print(f"  Background patch: {bg.shape}")

    print("\nBuilding query vector from 095420 t=1024s ...")
    qvid = os.path.join(FULL_DIR, "095420", f"{CAMERA}.mp4")
    qframe, qh = get_frame(qvid, 1024)
    qpatch = tight_crop(qframe, qh)
    qpatch_f = cv2.resize(qpatch.astype(np.float32), (bg.shape[1], bg.shape[0]))
    q_sub = np.clip(qpatch_f - bg + 128, 0, 255).astype(np.uint8)
    query_vec = dino_embed(model, q_sub)

    # Feature extraction (cached to disk)
    print("\nExtracting / loading features ...")
    all_feats = extract_features(model, bg, query_vec)

    # Label
    print("\nBuilding training set ...")
    X, y, session_ids, times = make_labels(all_feats)
    n_pos = y.sum(); n_neg = (y==0).sum()
    print(f"  Total samples: {len(y)}  (pos={n_pos}, neg={n_neg})")

    # ── Leave-one-session-out cross-validation ────────────────────────────────
    print("\nLeave-one-session-out CV ...")
    cv_results = {}
    for held_out in TIMESTAMPS:
        mask_train = np.array([s != held_out for s in session_ids])
        mask_test  = np.array([s == held_out for s in session_ids])
        if mask_test.sum() == 0:
            continue

        X_train, y_train = X[mask_train], y[mask_train]
        X_test,  y_test  = X[mask_test],  y[mask_test]
        if len(np.unique(y_train)) < 2:
            print(f"  [{held_out}] skip — only one class in train"); continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        clf = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
        clf.fit(X_tr, y_train)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
        else:
            auc = float('nan')

        print(f"\n  [{held_out}] hold-out:  n={len(y_test)} pos={y_test.sum()}")
        print(f"    AUC={auc:.3f}")
        labels_present = sorted(np.unique(np.concatenate([y_test, preds])))
        names = [['empty','octopus'][l] for l in labels_present]
        print(f"    {classification_report(y_test, preds, labels=labels_present, target_names=names, zero_division=0)}")
        cv_results[held_out] = (auc, probs, y_test, times[mask_test])

    # ── Train final model on all data ─────────────────────────────────────────
    print("\nTraining final model on all labeled data ...")
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    clf_final.fit(X_all, y)

    model_path = os.path.join(FEAT_DIR, "classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'clf': clf_final, 'scaler': scaler_final,
                     'bg': bg, 'query_vec': query_vec}, f)
    print(f"  Saved: {model_path}")

    # ── Apply final model to all sessions, save top frames ────────────────────
    print("\nApplying classifier to all sessions ...")
    session_top = {}
    for ts, (feats, sims, t_arr) in all_feats.items():
        X_s = scaler_final.transform(feats)
        probs = clf_final.predict_proba(X_s)[:, 1]
        # Find top frames
        top_idx = np.argsort(-probs)[:6]
        session_top[ts] = [(probs[i], t_arr[i]) for i in top_idx]
        peak = probs.max()
        print(f"  {ts}: peak_prob={peak:.3f}  top_t=[{', '.join(f'{t_arr[i]:.0f}s' for i in top_idx[:3])}]")

    # ── Timeline plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(all_feats), 1,
                             figsize=(16, 3*len(all_feats)), squeeze=False)
    for i, (ts, (feats, sims, t_arr)) in enumerate(all_feats.items()):
        X_s = scaler_final.transform(feats)
        probs = clf_final.predict_proba(X_s)[:, 1]
        ax = axes[i][0]
        ax.plot(t_arr/60, probs, color='steelblue', lw=1.2, label='P(octopus)')
        ax.axhline(0.5, color='red', linestyle='--', lw=1)
        ax.fill_between(t_arr/60, 0, probs, where=(probs>=0.5), color='red', alpha=0.25)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{ts} — classifier P(octopus)")
        ax.set_xlabel("time (min)"); ax.set_ylabel("probability")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "classifier_timeline.png"), dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Save top frames per session
    print("\nSaving top detection frames ...")
    for ts, top in session_top.items():
        vid = os.path.join(FULL_DIR, ts, f"{CAMERA}.mp4")
        if not os.path.exists(vid): continue
        for rank, (prob, t) in enumerate(top[:3]):
            if prob < 0.5: break
            frame, h = get_frame(vid, int(t), width=1280)
            if frame is None: continue
            # Draw detection patch boundary
            cx = int(OX * 1280); cy = int(OY * h)
            hw = int(HALF * 1280); hh = int(HALF * h)
            cv2.rectangle(frame, (cx-hw*2, cy-hh*2), (cx+hw*2, cy+hh*2), (0,255,0), 2)
            cv2.putText(frame, f"P={prob:.3f}", (cx-hw*2, cy-hh*2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            fname = os.path.join(OUT_DIR, f"{ts}_rank{rank+1:02d}_t{int(t):04d}s_p{prob:.3f}.jpg")
            cv2.imwrite(fname, frame)
            print(f"  Saved: {os.path.basename(fname)}")

    print(f"\nAll outputs in: {OUT_DIR}")
    print(f"Classifier saved to: {model_path}")


if __name__ == "__main__":
    main()
