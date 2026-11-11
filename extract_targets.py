"""Step 1 - measure the video.

Reads the press/release clip and writes outputs/clip1/targets.npz with, per frame:
  plate_mm      height of the book's underside above the table (NaN when the book is not in frame)
  ball_height_mm height of the ball (ball top is the book underside while they touch)
  width_mm, area_mm2   visible silhouette of the ball (yellow mask, below the book)
Scale: the ball's rest width in pixels is matched to its measured diameter.
NOTE: pass --circ_in if you measured the CIRCUMFERENCE with a tape (diameter = circ / pi), or --diam_mm for a diameter.
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--video', default='data/input.mp4')
ap.add_argument('--out', default='outputs/clip1/targets.npz')
ap.add_argument('--circ_in', type=float, default=8.5, help='measured circumference in inches (default 8.5)')
ap.add_argument('--diam_mm', type=float, default=None, help='or give the diameter directly (mm)')
ap.add_argument('--xmin', type=int, default=170, help='ignore pixels left of this (keeps hands out)')
ap.add_argument('--xmax', type=int, default=620)
ap.add_argument('--ycut', type=int, default=381, help='pixel row of the table contact line; mask below it (reflection) is ignored')
ap.add_argument('--tilt_half_px', type=int, default=0, help='>0: also measure the book tilt (underside height at +-this many px from the ball centre) and store it as slope/left_mm/right_mm; run_3d.py --tilt uses it')
ap.add_argument('--rest_frames', type=int, default=60, help='first N frames show the undisturbed ball')
a = ap.parse_args()

D = a.diam_mm or a.circ_in * 25.4 / np.pi
cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f'cannot open {a.video}')
fps = cap.get(cv2.CAP_PROP_FPS)
t, W, A, BB, TOP = [], [], [], [], []
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    ball = cv2.inRange(hsv, (10, 90, 35), (35, 255, 255))           # yellow ball (incl. its shaded parts)
    ball[:, :a.xmin] = 0
    ball[:, a.xmax:] = 0
    ball = cv2.morphologyEx(ball, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    book = cv2.inRange(hsv, (100, 60, 20), (135, 255, 255))         # blue book
    book = cv2.morphologyEx(book, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    t.append(i / fps)
    i += 1
    ys, xs = np.nonzero(ball)
    if len(ys) < 300:
        W.append(np.nan); A.append(np.nan); BB.append(np.nan); TOP.append(np.nan)
        continue
    cx = int(np.median(xs))
    col = book[:, max(0, cx - 25):cx + 25].any(axis=1)
    idx = np.nonzero(col)[0]
    BB.append(idx.max() if len(idx) else np.nan)                    # book underside (pixel row) above the ball
    TOP.append(np.percentile(ys, 0.5))
    below = ball.copy()
    below[:int(BB[-1]) + 1 if len(idx) else 0] = 0                   # visible ball = below the book underside
    ys2, xs2 = np.nonzero(below[:a.ycut])                               # ycut ~ table row (reflection below is ignored)
    if len(ys2) < 100:
        W.append(np.nan); A.append(0.0)
    else:
        W.append(np.percentile(xs2, 99.5) - np.percentile(xs2, 0.5)); A.append(len(ys2))

t, W, A, BB, TOP = (np.array(v, float) for v in (t, W, A, BB, TOP))
r = slice(0, a.rest_frames)
W0 = np.nanmedian(W[r]); table = np.nanmedian(TOP[r]) + W0           # ball sits on the table: bottom = top + width
s = W0 / D                                                           # pixels per mm
plate = (table - BB) / s
topc = np.fmax(TOP, np.where(np.isnan(BB), -1, BB))                  # ball top = book underside while in contact
H = (table - topc) / s
np.savez(a.out, t=t, plate_mm=plate, ball_height_mm=H, width_mm=W / s, area_mm2=A / s ** 2,
         area_rest_mm2=np.nanmedian(A[r]) / s ** 2, D_mm=D, px_per_mm=s, fps=fps, table_px=table)
print(f'{len(t)} frames @ {fps:.1f} fps | ball diameter {D:.1f} mm | {s:.2f} px/mm | table row {table:.0f}')
print(f'wrote {a.out}')
if a.tilt_half_px > 0:                                              # second pass: book underside at two columns -> slope (mm per mm, + = right side higher)
    cap = cv2.VideoCapture(a.video); cxs, Lr, Rr = [], [], []
    def under(hsv, x):
        col = cv2.inRange(hsv[:, x - 10:x + 10], (100, 60, 20), (135, 255, 255)).sum(axis=1) > 6 * 255      # row counts only if >= 6 of 20 columns are blue (robust to stray pixels)
        ii = np.nonzero(col)[0]; return (table - ii.max()) / s if len(ii) else np.nan
    for i in range(len(t)):
        ok, fr = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        if i < a.rest_frames:
            m_ = cv2.inRange(hsv, (10, 90, 35), (35, 255, 255)); m_[:, :a.xmin] = 0; m_[:, a.xmax:] = 0; cxs.append(np.median(np.nonzero(m_)[1])); Lr.append(np.nan); Rr.append(np.nan)
            if i == a.rest_frames - 1: c0 = int(np.median(cxs))
        else: Lr.append(under(hsv, c0 - a.tilt_half_px)); Rr.append(under(hsv, c0 + a.tilt_half_px))
    Lr, Rr = np.array(Lr), np.array(Rr); n_ = min(len(Lr), len(t)); slope = np.full(len(t), np.nan)
    slope[:n_] = (Rr[:n_] - Lr[:n_]) / (2 * a.tilt_half_px / s); slope[~np.isfinite(plate)] = np.nan
    d = dict(np.load(a.out)); d.update(slope=slope, left_mm=np.pad(Lr, (0, len(t) - len(Lr)), constant_values=np.nan), right_mm=np.pad(Rr, (0, len(t) - len(Rr)), constant_values=np.nan), cx_px=c0)
    np.savez(a.out, **d)
    print(f'book tilt: ball centre x={c0}px, slope while book in frame mean {np.nanmean(slope):+.3f} (min {np.nanmin(slope):+.3f}, max {np.nanmax(slope):+.3f}) mm/mm')