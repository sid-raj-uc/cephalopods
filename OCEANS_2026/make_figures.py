"""Generate the 3 paper figures into OCEANS_2026/assets/ (print-clean, dataviz principles)."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.image as mpimg

REPO = Path("/Users/siddharthraj/Documents/my-projects/sentiment-analysis")
OUT = REPO / "OCEANS_2026" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTE, GRID = "#222222", "#666666", "#dddddd"
HUE = "#2a78d6"                      # single hue (magnitude; validated, chroma>=0.1)
C_NONE, C_HUMAN = "#2a78d6", "#eb6834"   # categorical pair (validator: ALL PASS, light)
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK,
                     "axes.linewidth": 0.6, "font.family": "sans-serif"})

s = json.load(open(REPO / "data" / "behaviour_stats.json"))

# ── Fig 1: behaviour findings (3 panels) ─────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15))

# (a) activity budget — horizontal bars, magnitude, direct % labels
ab = {k: v for k, v in s["activity_budget"].items() if k != "uncertain"}
tot = sum(ab.values())
items = sorted(ab.items(), key=lambda kv: kv[1])
labels = [k.replace(" / ", "/\n") for k, _ in items]
vals = [v / tot * 100 for _, v in items]
y = np.arange(len(vals))
ax[0].barh(y, vals, color=HUE, height=0.66)
ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=6.3)
for yi, v in zip(y, vals):
    ax[0].text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=6.5, color=INK)
ax[0].set_xlim(0, max(vals) * 1.18); ax[0].set_xlabel("share of present clips (%)", fontsize=7)
ax[0].set_title("(a) Activity budget", fontsize=8, loc="left")
for sp in ("top", "right"): ax[0].spines[sp].set_visible(False)
ax[0].tick_params(length=0)

# (b) circadian — line of present_rate by hour, annotate peak
cir = s["circadian"]
hrs = sorted(int(h) for h in cir)
rate = [cir[str(h)]["present_rate"] * 100 for h in hrs]
ax[1].plot(hrs, rate, color=HUE, lw=1.6)
ax[1].fill_between(hrs, rate, color=HUE, alpha=0.10)
pk = int(np.argmax(rate))
ax[1].scatter([hrs[pk]], [rate[pk]], s=18, color=HUE, zorder=5)
ax[1].annotate(f"peak {rate[pk]:.0f}% @ {hrs[pk]:02d}:00", (hrs[pk], rate[pk]),
               textcoords="offset points", xytext=(-4, 6), fontsize=6.3, color=INK, ha="right")
ax[1].set_xlim(0, 23); ax[1].set_xticks([0, 6, 12, 18, 23])
ax[1].set_xlabel("hour of day", fontsize=7); ax[1].set_ylabel("visible-activity rate (%)", fontsize=7)
ax[1].set_title("(b) Circadian rhythm", fontsize=8, loc="left")
ax[1].grid(True, color=GRID, lw=0.5); ax[1].set_axisbelow(True)
for sp in ("top", "right"): ax[1].spines[sp].set_visible(False)
ax[1].tick_params(length=0)

# (c) stimulus response — motion & arousal, two conditions (different scales -> 2 groups, no dual axis)
sr = s["stimulus_response"]
none, hum = sr["none"], sr["human_present"]
groups = ["Motion", "Arousal"]
none_v = [none["mean_motion"], none["mean_arousal"]]
hum_v = [hum["mean_motion"], hum["mean_arousal"]]
# normalise each measure to its own 0..1 so both fit one panel; print true values as labels
scale = [max(none_v[0], hum_v[0]), 1.0]   # motion by its own max, arousal already 0..1
xn = np.arange(2); w = 0.36
bn = ax[2].bar(xn - w/2, [none_v[i]/scale[i] for i in range(2)], w, color=C_NONE, label="no human")
bh = ax[2].bar(xn + w/2, [hum_v[i]/scale[i] for i in range(2)], w, color=C_HUMAN, label="human present")
for i in range(2):
    ax[2].text(xn[i]-w/2, none_v[i]/scale[i]+0.02, f"{none_v[i]:.2f}", ha="center", fontsize=6, color=INK)
    ax[2].text(xn[i]+w/2, hum_v[i]/scale[i]+0.02, f"{hum_v[i]:.2f}", ha="center", fontsize=6, color=INK)
ax[2].set_xticks(xn); ax[2].set_xticklabels(groups, fontsize=7); ax[2].set_yticks([])
ax[2].set_ylim(0, 1.18); ax[2].set_title("(c) Stimulus response", fontsize=8, loc="left")
ax[2].legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.1)
for sp in ("top", "right", "left"): ax[2].spines[sp].set_visible(False)
ax[2].tick_params(length=0)

fig.tight_layout(pad=0.6, w_pad=1.4)
fig.savefig(OUT / "behaviour_findings.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote behaviour_findings.pdf")

# ── Fig 2: segmentation overlays on colour + IR ──────────────────────────────────
demo = REPO / "data" / "segmentation_demo"
back = sorted(glob.glob(str(demo / "Right_Back_*still*.jpg")))[:3]
top = sorted(glob.glob(str(demo / "Right_Top_*still*.jpg")))[:3]
if back and top:
    fig2, axs = plt.subplots(2, 3, figsize=(7.0, 2.7))
    for j, p in enumerate(back):
        axs[0, j].imshow(mpimg.imread(p)); axs[0, j].axis("off")
    for j, p in enumerate(top):
        axs[1, j].imshow(mpimg.imread(p)); axs[1, j].axis("off")
    axs[0, 0].set_title("Colour camera (Right\\_Back)", fontsize=8, loc="left")
    axs[1, 0].set_title("Infrared camera (Right\\_Top)", fontsize=8, loc="left")
    fig2.tight_layout(pad=0.3, h_pad=0.8)
    fig2.savefig(OUT / "segmentation_demo.pdf", bbox_inches="tight"); plt.close(fig2)
    print("wrote segmentation_demo.pdf")
else:
    print("WARN: segmentation stills missing")

# ── Fig 3: pipeline block diagram ────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7.0, 2.0)); ax3.set_xlim(0, 100); ax3.set_ylim(0, 34); ax3.axis("off")
def box(x, y, w, h, text, fc="#EAF1F6", ec=HUE):
    ax3.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                 fc=fc, ec=ec, lw=1.1))
    ax3.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7, color=INK)
def arrow(x1, y1, x2, y2):
    ax3.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                  color=MUTE, lw=1.0))
stages = [("Multi-camera\nfootage", 1), ("Visibility\ndetection\n(CLIP+MLP)", 20),
          ("Clip\nextraction\n(vis+motion)", 39), ("VLM structured\nextraction\n(behaviour JSON)", 58)]
for t, x in stages: box(x, 18, 16, 12, t)
for i in range(3): arrow(stages[i][1]+16, 24, stages[i+1][1], 24)
# distillation branch
box(78, 24, 20, 8, "Caption student\n(4-bit, on-device)", fc="#FBEEE0", ec=C_HUMAN)
box(78, 12, 20, 8, "Segmentation student\n(3.2M param)", fc="#FBEEE0", ec=C_HUMAN)
arrow(74, 24, 78, 28); arrow(74, 24, 78, 16)
ax3.text(88, 33, "distilled local models", ha="center", fontsize=6.5, color=MUTE, style="italic")
# aggregate output
box(58, 2, 16, 10, "Behavioural\ntime-series\n(ethogram)", fc="#EAF1F6")
arrow(66, 18, 66, 12)
# skeleton & kinematics stage (2026-08): masks -> anatomical graph -> gated kinematics
box(78, 1, 20, 8, "Skeleton + arm\nkinematics", fc="#EAF1F6")
arrow(88, 12, 88, 9.4)     # seg student -> skeleton
arrow(78, 5, 74, 6)        # kinematics -> behavioural time-series
fig3.savefig(OUT / "pipeline_behaviour.pdf", bbox_inches="tight"); plt.close(fig3)
print("wrote pipeline_behaviour.pdf")
print("\nassets now:", sorted(p.name for p in OUT.glob("*.pdf")))

# ── Fig 4: kinematics by behaviour (print-clean; single measure across categories → one hue,
#           identity carried by axis labels, medians direct-labeled) ────────────────────────
import collections
br = json.load(open(REPO / "data" / "behaviour_records.json"))
by = collections.defaultdict(list)
for rel, rec in br.items():
    k = rec.get("kinematics") or {}
    if "occluded_frac" not in k or not k.get("activity_px_s") or "Right_Left" in rel:
        continue
    by[(rec.get("struct") or {}).get("behavior", "?")].append(k["activity_px_s"]["mean"])
by.pop("uncertain", None)
order = sorted(by, key=lambda b: np.median(by[b]))
fig4, ax4 = plt.subplots(figsize=(3.45, 2.2))
data = [by[b] for b in order]
bp = ax4.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                 medianprops=dict(color=INK, lw=1.1),
                 boxprops=dict(facecolor=HUE, alpha=0.55, edgecolor=HUE, lw=0.8),
                 whiskerprops=dict(color=MUTE, lw=0.8), capprops=dict(color=MUTE, lw=0.8),
                 flierprops=dict(marker="o", ms=2.5, mfc=MUTE, mec=MUTE))
ax4.set_yticks(range(1, len(order) + 1))
ax4.set_yticklabels([f"{b.split(' / ')[0].split(' out')[0]} (n={len(by[b])})" for b in order],
                    fontsize=6.4)
for i, b in enumerate(order, 1):
    med = float(np.median(by[b]))
    ax4.text(med, i + 0.34, f"{med:.0f}", ha="center", fontsize=6, color=INK)
ax4.set_xlabel("state-gated arm-tip speed (px s$^{-1}$)", fontsize=7)
ax4.grid(True, axis="x", color=GRID, lw=0.5); ax4.set_axisbelow(True)
for sp in ("top", "right"): ax4.spines[sp].set_visible(False)
ax4.tick_params(length=0)
fig4.tight_layout(pad=0.4)
fig4.savefig(OUT / "kinematics_by_behaviour.pdf", bbox_inches="tight"); plt.close(fig4)
print("wrote kinematics_by_behaviour.pdf")

# ── Fig 5: skeleton qualitative example (image panels: base vs SAM2-refined) ───────────────
# Figure sources are FROZEN under assets/frozen/: data/skel_diag/ is a scratch dir that every
# experiment overwrites, so reading it here silently changed a published figure once.
src = OUT / "frozen" / "skeleton_example_src.jpg"
if src.exists():
    im = mpimg.imread(str(src))
    W = im.shape[1]; w = (W - 6) // 2          # frozen source is a 2-panel strip: base | refined
    base_p, sam_p = im[:, :w], im[:, W - w:]
    fig5, ax5 = plt.subplots(1, 2, figsize=(7.0, 1.75))
    ax5[0].imshow(base_p); ax5[0].axis("off")
    ax5[1].imshow(sam_p); ax5[1].axis("off")
    fig5.tight_layout(pad=0.3)
    fig5.savefig(OUT / "skeleton_example.pdf", bbox_inches="tight", dpi=200); plt.close(fig5)
    print("wrote skeleton_example.pdf")
else:
    print("WARN: skel_diag/004.jpg missing")
