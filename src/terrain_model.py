"""
Rocket downward-camera terrain + landing-zone model.

Per frame:
  1. Ignore mask   : HUD, rocket body/fins, altitude tape (learned once from the whole video)
  2. Superpixels   : SLIC splits the ground into ~1200 regions, each gets color + texture features
  3. Terrain class : grass, crop/bare field, plowed field, trees, shadow, water, smoke
  4. Thin hazards  : roads (ridge filter) and buildings / vehicles (bright compact blobs)
  5. YOLOE         : open-vocabulary detector for smoke, buildings, vehicles, people, water
  6. Landing score : per-class risk + distance to hard hazards, averaged over a landing circle,
                     top 3 spots returned

python src/terrain_model.py data/frames results [--no-yolo]
"""

import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.filters import sato
from skimage.segmentation import slic

W, H = 960, 600
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (src/..)

CLASSES = ["grass", "crop/bare field", "plowed field", "trees", "cloud shadow", "water", "smoke",
           "road", "structure"]
COLORS = np.array([
    [60, 200, 60],     # grass
    [230, 200, 90],    # crop/bare
    [200, 160, 220],   # plowed
    [0, 90, 20],       # trees
    [70, 70, 110],     # shadow
    [30, 110, 240],    # water
    [240, 240, 240],   # smoke
    [255, 120, 0],     # road
    [230, 20, 20],     # structure
], np.uint8)
RISK = np.array([0.05, 0.12, 0.35, 0.9, 0.55, 1.0, 0.85, 0.95, 1.0])
HARD = [3, 5, 7, 8]            # trees, water, road, structure: keep distance from these
FOOT = 22                      # landing circle radius, px at 960 wide
BUFFER = 30                    # px: risk fades out this far from a hard hazard
EDGE_PEN = 0.5                 # penalty for spots near the blurry fisheye edge


# ---- 1. load -----------------------------------------------------------------
def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255


# ---- 3. terrain classes on superpixels ------------------------------------------
def classify(a, ignore):
    hsv = rgb2hsv(a)
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    g = ndimage.gaussian_filter(a.mean(2), 1.0)
    tex = np.sqrt(np.maximum(ndimage.uniform_filter(g ** 2, 9) - ndimage.uniform_filter(g, 9) ** 2, 0))
    blue_bias = a[..., 2] - a[..., 1]
    blue_red = a[..., 2] - a[..., 0]           # smoke is blue-grey, dirt/plowed is warm

    seg = slic(a, n_segments=1200, compactness=12, start_label=0, mask=~ignore)
    n = seg.max() + 1
    idx = np.arange(n)
    valid = seg >= 0

    def m(x):
        return ndimage.mean(x, seg, idx)

    H_, S_, V_, T_, B_, BR_ = m(h), m(s), m(v), m(tex), m(blue_bias), m(blue_red)
    cls = np.full(n, 0)
    for i in range(n):
        hh, ss, vv, tt, bb, br = H_[i], S_[i], V_[i], T_[i], B_[i], BR_[i]
        if vv < 0.25 and bb > 0.06 and tt < 0.02:
            c = 5                                   # water: dark, clearly blue, glassy (cloud shadow is only a bit blue)
        elif vv < 0.30 and tt < 0.035:
            c = 4                                   # shadow: dark, smooth
        elif vv < 0.45 and tt >= 0.035 and 50 <= hh <= 160:
            c = 3                                   # trees: darkish green, clumpy
        elif br > 0.06 and vv > 0.45 and ss < 0.4:
            c = 6                                   # smoke: bright blue-grey (dirt is warm-toned)
        elif ss < 0.22 and vv > 0.5:
            c = 2                                   # plowed / striped field
        elif 25 <= hh < 58 and vv > 0.5:
            c = 1                                   # crop / bare / cut hay
        elif vv < 0.4 and tt >= 0.03:
            c = 3
        else:
            c = 0                                   # grass
        cls[i] = c
    lab = np.where(valid, cls[np.clip(seg, 0, None)], -1)
    return lab, dict(h=h, s=s, v=v, tex=tex, br=blue_red)


# ---- 4. thin hazards: roads + structures ------------------------------------------
def thin_hazards(a, ignore, f):
    v = f["v"]
    # roads: thin bright ridges, long and skinny
    r = sato(v, sigmas=[1.5, 2.5], black_ridges=False)
    road = r > np.percentile(r[~ignore], 97)
    road &= f["s"] < 0.35
    lab, n = ndimage.label(road)
    road_keep = np.zeros_like(road)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == i
        hgt, wid = comp.shape
        if comp.sum() > 120 and max(hgt, wid) > 60 and comp.sum() / (hgt * wid) < 0.35:
            road_keep[sl] |= comp
    # structures: small bright compact blobs brighter than their surroundings
    th = v - ndimage.grey_opening(v, size=(15, 15))
    blob = (th > 0.18) & (v > 0.7)
    blob = ndimage.binary_opening(blob)
    lab, n = ndimage.label(blob)
    sizes = ndimage.sum(blob, lab, range(1, n + 1))
    struct = np.isin(lab, 1 + np.where((sizes >= 6) & (sizes <= 900))[0]) & ~road_keep
    return road_keep & ~ignore, ndimage.binary_dilation(struct, iterations=2) & ~ignore


# ---- 5. YOLOE open-vocabulary detector ----------------------------------------------
YOLO_NAMES = ["smoke", "building", "house", "barn", "shed", "car", "truck", "trailer",
              "tent", "person", "pond", "water"]
YOLO_TO_CLASS = {"smoke": 6, "pond": 5, "water": 5}   # everything else -> structure


class Detector:
    def __init__(self, weights="models/yoloe-11l-seg.pt"):
        from ultralytics import YOLOE
        self.m = YOLOE(os.path.join(ROOT, weights))
        self.m.set_classes(YOLO_NAMES, self.m.get_text_pe(YOLO_NAMES))

    def __call__(self, path, conf=0.12):
        r = self.m.predict(path, conf=conf, imgsz=1280, verbose=False)[0]
        out = []
        if r.masks is None:
            return out
        for c, p, mk in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.data.cpu().numpy()):
            mk = np.asarray(Image.fromarray((mk * 255).astype(np.uint8)).resize((W, H))) > 127
            name = YOLO_NAMES[int(c)]
            out.append((name, p, mk, YOLO_TO_CLASS.get(name, 8)))
        return out


def keep_det(d, f, ignore):
    """Sanity-check YOLOE masks against what the pixels actually look like."""
    name, p, mk, c = d
    mk = mk & ~ignore
    area = mk.sum() / (~ignore).sum()
    if area == 0:
        return False
    if c == 8 and area > 0.03:                     # a "building" can't cover 3%+ of the view up here
        return False
    if c == 5 and (area > 0.15 or f["v"][mk].mean() > 0.35):   # water has to be dark
        return False
    if c == 6:                                     # smoke has to be grey/white and soft
        looks = ((f["br"] > 0.03) & (f["v"] > 0.45) & (f["s"] < 0.45))[mk].mean()
        if looks < 0.4:
            return False
    return True


# ---- 6. landing score ---------------------------------------------------------------
def landing(lab, ignore, k=3, sep=90):
    risk = np.where(lab >= 0, RISK[np.clip(lab, 0, None)], 1.0)
    hard = np.isin(lab, HARD)
    near = np.exp(-ndimage.distance_transform_edt(~hard) / BUFFER)
    risk = np.clip(0.8 * risk + 0.4 * near * (~hard), 0, 1)
    # fisheye edge: far away, blurry, low detail -> less trustworthy, prefer the middle
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    risk = np.clip(risk + EDGE_PEN * np.clip(rr - 0.35, 0, None) ** 2, 0, 1)
    risk[ignore] = 1.0
    disk = np.hypot(*np.mgrid[-FOOT:FOOT + 1, -FOOT:FOOT + 1]) <= FOOT
    mean_r = ndimage.convolve(risk, disk / disk.sum(), mode="constant", cval=1.0)
    max_r = ndimage.maximum_filter(risk, footprint=disk, mode="constant", cval=1.0)
    zone = 0.7 * mean_r + 0.3 * max_r
    # circle must not touch the ignore mask (can't see under the HUD/rocket)
    blocked = ndimage.binary_dilation(ignore, structure=disk)
    zone[blocked] = np.inf
    zone[:FOOT, :] = zone[-FOOT:, :] = zone[:, :FOOT] = zone[:, -FOOT:] = np.inf
    spots, z = [], zone.copy()
    yy, xx = np.ogrid[:H, :W]
    for _ in range(k):
        r, c = np.unravel_index(np.argmin(z), z.shape)
        if not np.isfinite(z[r, c]):
            break
        spots.append((int(r), int(c), float(z[r, c])))
        z[(yy - r) ** 2 + (xx - c) ** 2 < sep ** 2] = np.inf
    return spots, risk


# ---- run ------------------------------------------------------------------------------
def analyze(path, ignore, det=None):
    a = load(path)
    lab, f = classify(a, ignore)
    road, struct = thin_hazards(a, ignore, f)
    lab[road & (lab != 6)] = 7
    lab[struct & (lab != 2) & (lab != 6)] = 8          # furrows in plowed fields look like blobs
    dets = [d for d in (det(path) if det else []) if keep_det(d, f, ignore)]
    for name, p, mk, c in dets:
        lab[mk & ~ignore] = c
    spots, risk = landing(lab, ignore)
    return a, lab, risk, spots, dets


def render(a, lab, risk, spots, dets, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    seg = np.where(lab[..., None] >= 0, COLORS[np.clip(lab, 0, None)] / 255, a * 0.35)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
    ax[0].imshow(a); ax[0].set_title(title)
    ax[1].imshow(a); ax[1].imshow(seg, alpha=0.6); ax[1].set_title("terrain classes")
    r = np.ma.masked_where(lab < 0, risk)
    ax[2].imshow(a); ax[2].imshow(r, cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.55)
    for j, (y, x, s) in enumerate(spots):
        col = ["cyan", "yellow", "orange"][j]
        for axx in (ax[0], ax[2]):
            axx.add_patch(plt.Circle((x, y), FOOT, fill=False, ec=col, lw=2.5))
        ax[2].text(x + FOOT + 4, y, f"#{j + 1} {s:.2f}", color=col, fontsize=11, weight="bold")
    for name, p, mk, c in dets:
        ys, xs = np.where(mk)
        if len(xs):
            ax[1].text(xs.mean(), ys.min() - 4, f"{name} {p:.2f}", color="white", fontsize=8,
                       bbox=dict(fc="black", alpha=0.6, pad=1))
    ax[2].set_title("landing risk + best spots")
    for axx in ax:
        axx.axis("off")
    present = sorted(set(np.unique(lab[lab >= 0]).tolist()))
    ax[1].legend(handles=[Patch(color=COLORS[i] / 255, label=CLASSES[i]) for i in present],
                 loc="lower left", fontsize=7, ncol=2, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ignore = np.load(os.path.join(ROOT, "data", "ignore_mask.npy"))
    det = None if "--no-yolo" in sys.argv else Detector()
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "*.jpg"))):
        a, lab, risk, spots, dets = analyze(p, ignore, det)
        name = os.path.splitext(os.path.basename(p))[0]
        render(a, lab, risk, spots, dets, name, os.path.join(out, name + ".png"))
        vis = lab[lab >= 0]
        mix = {CLASSES[i]: round(float((vis == i).mean() * 100), 1) for i in range(len(CLASSES)) if (vis == i).any()}
        rows.append((name, spots, mix, [(d[0], round(d[1], 2)) for d in dets]))
        print(name, "best:", spots[0] if spots else None, "| mix %:", mix, "| yolo:", rows[-1][3])

    import csv
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "best_x", "best_y", "best_risk", "verdict", *CLASSES, "yolo"])
        for name, spots, mix, yd in rows:
            y, x, r = spots[0] if spots else (None, None, None)
            verdict = "GOOD" if r is not None and r < 0.08 else "OK" if r is not None and r < 0.15 else "RISKY"
            w.writerow([name, x, y, round(r, 3) if r is not None else "", verdict,
                        *[mix.get(c, 0) for c in CLASSES], "; ".join(f"{n} {p}" for n, p in yd)])
    # contact sheet: original + best spot for every frame
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    cols = 4
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.6 * cols, 3.1 * ((n + cols - 1) // cols)))
    ax = np.atleast_1d(ax).ravel()
    for i, (name, spots, mix, yd) in enumerate(rows):
        a = load(os.path.join(src, name + ".jpg"))
        ax[i].imshow(a)
        for j, (y, x, r) in enumerate(spots):
            ax[i].add_patch(plt.Circle((x, y), FOOT, fill=False, ec=["cyan", "yellow", "orange"][j], lw=2))
        top = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(mix.items(), key=lambda t: -t[1])[:3])
        ax[i].set_title(f"{name}  risk {spots[0][2]:.2f}\n{top}", fontsize=9)
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=70)
