import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import json, cv2, numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
T = np.load('outputs/clip1/targets.npz'); s = float(T['px_per_mm']); fps = float(T['fps']); table = float(T['table_px'])
cap = cv2.VideoCapture('data/input.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, int(8.0 * fps))
ts, L, R, C = [], [], [], []
for k in range(int(8.0 * fps), int(10.8 * fps)):
    ok, fr = cap.read()
    if not ok: break
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (10, 60, 25), (35, 255, 255)); m[:, :170] = 0; m[:, 620:] = 0; m[int(table) - 2:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); ys, xs = np.nonzero(m)
    if len(ys) < 500: continue
    yc = np.percentile(ys, 50); b = (ys > yc - 40) & (ys < yc + 40)
    ts.append(k / fps); L.append(np.percentile(xs[b], 1) / s); R.append(np.percentile(xs[b], 99) / s); C.append(np.mean(xs) / s)
t, L, R, C = map(np.array, (ts, L, R, C)); ref = t > t[-1] - 0.5
rel = lambda x: x - np.median(x[ref]); Wd = rel(R - L); Cx = rel(C)
m = t >= 8.3; t0 = 8.3
def model(p, tt): A, tau, f, ph, c = p; return A * np.exp(-(tt - t0) / tau) * np.cos(2 * np.pi * f * (tt - t0) + ph) + c
def fit(y):
    best = None
    for f0 in np.arange(0.4, 3.2, 0.2):
        for ph0 in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            r = least_squares(lambda p: model(p, t[m]) - y[m], [np.ptp(y[m]) / 2, 0.5, f0, ph0, 0], bounds=([0, 0.05, 0.3, -7, -3], [30, 5, 4, 7, 3]))
            if best is None or r.cost < best.cost: best = r
    return best
out = {}
for name, y in (('sway (centroid x)', Cx), ('edge-to-edge width', Wd)):
    r = fit(y); A, tau, f, ph, c = r.x; z = 1 / np.sqrt(1 + (2 * np.pi * f * tau) ** 2)
    rm = np.sqrt(2 * r.cost / m.sum())
    print(f'{name:20s} f={f:.2f} Hz (period {1/f:.2f} s)  tau={tau:.2f} s  zeta~{z:.2f}  A0={A:.2f} mm  rmse={rm:.2f} mm   data p-p {np.ptp(y[m]):.1f} mm')
    out[name] = dict(f=f, tau=tau, zeta=z, A=A, phi=ph, c=c, rmse=rm, t0=t0)
    if name.startswith('sway'): xfit = model(r.x, t)
json.dump(out, open('outputs/sway/sway_fit.json', 'w'), indent=1); np.savez('outputs/sway/sway.npz', t=t, x=Cx, xfit=xfit, width=Wd, t0=t0)
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t, rel(L), label='left edge'); ax[0].plot(t, rel(R), label='right edge'); ax[0].plot(t, Cx, 'k', label='centroid x'); ax[0].plot(t[m], xfit[m], 'r--', label='damped-oscillation fit'); ax[0].legend(); ax[0].set_ylabel('mm')
ax[1].plot(t, Wd, 'k', label='edge-to-edge width'); ax[1].set_ylabel('mm'); ax[1].set_xlabel('s'); ax[1].legend()
for a in ax: a.axhline(0, c='gray', lw=.5)
plt.tight_layout(); plt.savefig('outputs/sway/sway.png', dpi=95); print('wrote outputs/sway/sway.png')