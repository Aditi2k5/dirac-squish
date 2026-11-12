import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--video', default='data/input.mp4'); ap.add_argument('--targets', default='outputs/clip1/targets.npz'); ap.add_argument('--out', default='outputs/sway/dots')
ap.add_argument('--xmin', type=int, default=170); ap.add_argument('--xmax', type=int, default=620); ap.add_argument('--ycut', type=int, default=381)
ap.add_argument('--t0', type=float, default=8.3)
ap.add_argument('--t1', type=float, default=10.5)
ap.add_argument('--max_step', type=float, default=14.0, help='max dot motion between frames (px)')
a = ap.parse_args()

T = np.load(a.targets); s = float(T['px_per_mm']); fps = float(T['fps']); K = s / 4.12     # K: pixel sizes below were tuned at 4.12 px/mm
cap = cv2.VideoCapture(a.video)
f0, f1 = int(a.t0 * fps), int(a.t1 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)


def detect(fr):
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    ball = cv2.inRange(hsv, (10, 90, 35), (35, 255, 255)); ball[:, :a.xmin] = 0; ball[:, a.xmax:] = 0; ball[a.ycut:] = 0
    ball = cv2.morphologyEx(ball, cv2.MORPH_OPEN, np.ones((max(3, int(5 * K)), max(3, int(5 * K))), np.uint8))
    cs, _ = cv2.findContours(ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None, []
    c = max(cs, key=cv2.contourArea)
    fill = np.zeros_like(ball); cv2.drawContours(fill, [c], -1, 255, -1)
    inner = cv2.erode(fill, np.ones((int(25 * K), int(25 * K)), np.uint8))            # stay away from the silhouette edge (dots foreshorten there)
    v = hsv[..., 2]
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(15 * K) | 1, int(15 * K) | 1)))
    dark = ((bh > 40) & (inner > 0)).astype(np.uint8)
    dark[int(T['table_px'] - 40 * K):] = 0                              # table-contact shadow is not a dot
    n, _, st, cen = cv2.connectedComponentsWithStats(dark)
    d = [cen[i] for i in range(1, n) if 25 * K * K <= st[i, 4] <= 150 * K * K and max(st[i, 2], st[i, 3]) <= 16 * K]
    M = cv2.moments(c); return (M['m10'] / M['m00'], M['m01'] / M['m00']), d


tracks, ts, cents = [], [], []                                       # tracks: list of dict frame->xy
for k in range(f0, f1):
    ok, fr = cap.read()
    if not ok: break
    cen, dots = detect(fr)
    ts.append(k / fps); cents.append(cen if cen else (np.nan, np.nan))
    used = set()
    for tr in tracks:
        if tr['last'] != k - 1: continue
        p = tr['xy'][-1]
        best, bd = None, a.max_step
        for j, q in enumerate(dots):
            dd = np.hypot(q[0] - p[0], q[1] - p[1])
            if j not in used and dd < bd: best, bd = j, dd
        if best is not None: tr['xy'].append(dots[best]); tr['last'] = k; tr['f'].append(k); used.add(best)
    for j, q in enumerate(dots):
        if j not in used: tracks.append(dict(xy=[q], f=[k], last=k))
ts = np.array(ts); cents = np.array(cents)
nf = len(ts)
long = [t for t in tracks if len(t['f']) >= 0.8 * nf]
print(f'{nf} frames, {len(tracks)} tracks started, {len(long)} dots tracked through >=80% of the window')
if len(long) < 3:
    raise SystemExit('too few long tracks - try --max_step or a shorter window')
# keep only frames where all long tracks exist, compute span signals
fr_all = sorted(set.intersection(*[set(t['f']) for t in long]))
idx = [f - f0 for f in fr_all]
P = np.array([[t['xy'][t['f'].index(f)] for f in fr_all] for t in long])   # (dots, frames, 2)
ct = P.mean(0)
span = np.mean(np.hypot(P[..., 0] - ct[None, :, 0], P[..., 1] - ct[None, :, 1]), axis=0) / s     # mm
sx = (P[..., 0].max(0) - P[..., 0].min(0)) / s                       # horizontal extent of the dot cloud, mm
sy = (P[..., 1].max(0) - P[..., 1].min(0)) / s
tt = ts[idx]
Tt = T['t']; W = np.interp(tt, Tt, T['width_mm']); A = np.interp(tt, Tt, T['area_mm2'] / T['area_rest_mm2'])
ref = tt > tt[-1] - 0.5
rel = lambda x: x - np.median(x[ref])
np.savez(a.out + '.npz', t=tt, span=span, sx=sx, sy=sy, width=W, area=A, P=P)
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(tt, rel(W), 'k', label='silhouette width'); ax[0].set_ylabel('width - rest (mm)')
ax[1].plot(tt, rel(sx), 'C0', label='dot cloud horizontal extent'); ax[1].plot(tt, rel(sy), 'C3', label='dot cloud vertical extent')
ax[1].set_ylabel('mm'); ax[2].plot(tt, rel(span), 'C2', label='mean dot distance from centroid'); ax[2].set_ylabel('mm'); ax[2].set_xlabel('s')
for x_ in ax: x_.axhline(0, c='gray', lw=.5); x_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(a.out + '.png', dpi=95)
for name, x in [('silhouette width', W), ('dot extent x', sx), ('dot extent y', sy), ('dot span', span)]:
    r = rel(x); print(f'{name:18s} peak-to-peak {r.max() - r.min():.2f} mm   std {r.std():.2f} mm')
print('wrote', a.out + '.npz', a.out + '.png')