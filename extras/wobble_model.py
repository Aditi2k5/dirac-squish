"""Step 8 - lumped model that accounts for the slow shape wobble after release.

  height  : H' = (H0 - H)/tau  with unilateral contact against the book (as in baseline_1dof.py), tau fixed at 0.1 s
  width   : (W - W0) = a shape mode q, kicked by the recovery velocity of the height (hold state = rest, as measured):
               q'' + 2 zeta w q' + w^2 q = -b dH/dt ,   W = W0 + q
            i.e. when the ball springs back quickly the width is thrown off and rings down, like a jelly.
Free parameters: b (mm per mm/s... units 1/s), f = w/2pi (Hz), zeta. They are fitted to the measured width AFTER the book has left (t >= 7.9 s).
This is a DESCRIPTIVE model of the wobble, not a derived material law (see README: elastic foam would ring far faster).
Writes outputs/sway/wobble_model.png and outputs/sway/wobble_model.json. Prediction for a new, faster release is panel (c).
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import json
import numpy as np
from scipy.signal import lsim
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = np.load('outputs/clip1/targets.npz'); t = T['t']; plate = T['plate_mm']; W = T['width_mm']; H0 = float(T['D_mm']); W0 = H0
TAU = 0.1


def height(hplate, tg, tau=TAU):
    out, h = np.empty(len(tg)), H0
    for k in range(len(tg)):
        if k:
            dt = tg[k] - tg[k - 1]
            for _ in range(20):
                h += dt / 20 * (H0 - h) / tau; h = min(h, hplate[k])
        out[k] = h = min(h, hplate[k])
    return out


def width(p, tg, d):
    b, f, z = p; w = 2 * np.pi * f
    Hd = np.gradient(H0 - d, tg)                                   # dH/dt from the deficit d = H0 - H
    _, y, _ = lsim(([-b, 0.0], [1, 2 * z * w, w * w]) if False else ([-b], [1, 2 * z * w, w * w]), U=Hd, T=tg - tg[0]); return W0 + y


tg = t[(t >= 4.75) & (t <= 10.8)]
good = np.isfinite(plate)
hp = np.where(tg > t[good][-1], 200.0, np.interp(tg, t[good], plate[good]))
H = height(hp, tg); d = H0 - H
Wm = np.interp(tg, t, W)
fitm = tg >= 7.9
best = None
for f0 in (0.8, 1.2, 1.6, 2.0, 2.6):
    for z0 in (0.15, 0.3, 0.6):
        r = least_squares(lambda p: (width(p, tg, d) - Wm)[fitm], [0.5, f0, z0], bounds=([-5, 0.4, 0.02], [5, 5, 2]))
        if best is None or r.cost < best.cost: best = r
b, f, z = best.x; K = b
rm_fit = np.sqrt(np.mean((width(best.x, tg, d) - Wm)[fitm] ** 2))
base = np.sqrt(np.mean((W0 - Wm)[fitm] ** 2))
flat = np.sqrt(np.mean((W0 - Wm)[fitm] ** 2))
pre = (tg >= 4.75) & (tg < 7.9) & np.isfinite(Wm)
rm_press = np.sqrt(np.mean((width(best.x, tg, d) - Wm)[pre] ** 2))
print(f'fitted: b={b:.3f} (1/s units: mm per mm/s)  f={f:.2f} Hz  zeta={z:.2f}')
print(f'width RMSE after release (7.9-10.8 s):  wobble model {rm_fit:.2f} mm | constant rest width {flat:.2f}')
print(f'width RMSE during press/hold (4.75-7.9 s, NOT fitted): {rm_press:.2f} mm')
json.dump(dict(b=b, f_hz=f, zeta=z, tau=TAU, rmse_release=rm_fit, rmse_nodyn=base, rmse_const=flat, rmse_press=rm_press), open('outputs/sway/wobble_model.json', 'w'), indent=1)

# fresh instances: same press depth as the clip but release 0.1 s (fast) and 0.5 s (slow)
tf = np.linspace(0, 4.0, 1200)
def path(rel):
    return np.where(tf < 1.0, H0 + 3 - (H0 + 3 - 8.0) * tf, np.where(tf < 1.5, 8.0, np.where(tf < 1.5 + rel, 8.0 + (tf - 1.5) / rel * (H0 + 3 - 8.0) * 1.0, 300.0)))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
ax[0].plot(tg, Wm - W0, 'k', lw=2, label='measured width - rest'); ax[0].plot(tg, width(best.x, tg, d) - W0, 'C3', label='wobble model')
ax[0].axvline(7.9, c='gray', ls='--', lw=.7)
ax[0].set(xlim=(4.7, 10.8), xlabel='s', ylabel='mm', title=f'(a) fit: f={f:.2f} Hz, zeta={z:.2f}, b={b:.3f}'); ax[0].legend(fontsize=8)
m = tg >= 7.5
ax[1].plot(tg[m], Wm[m] - W0, 'k', lw=2, label='measured'); ax[1].plot(tg[m], (width(best.x, tg, d) - W0)[m], 'C3', label='model'); ax[1].axhline(0, c='gray', lw=.5)
ax[1].set(xlabel='s', ylabel='mm', title='(b) zoom: release and wobble'); ax[1].legend(fontsize=8)
for rel, c in ((0.5, 'C2'), (0.1, 'C1'), (0.05, 'C4')):
    hf = path(rel); dd = H0 - height(hf, tf); ax[2].plot(tf, width(best.x, tf, dd) - W0, c, label=f'release over {rel} s')
ax[2].axhline(0, c='gray', lw=.5); ax[2].set(xlabel='s', ylabel='width - rest (mm)', title='(c) prediction: fresh press, different release speeds'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('outputs/sway/wobble_model.png', dpi=95); print('wrote outputs/sway/wobble_model.png')
np.savez('outputs/sway/wobble_model.npz', t=tg, q=width(best.x, tg, d) - W0, H=H)   # used by render_demo.py (WOBBLE=1)