import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import numpy as np
from scipy.optimize import least_squares
D = np.load('outputs/sway/dots.npz'); t = D['t']; t0 = t[0]
def model(p, tt): A, tau, f, ph, c = p; return A * np.exp(-(tt - t0) / tau) * np.cos(2 * np.pi * f * (tt - t0) + ph) + c
out = {}
for name in ['span', 'sx', 'sy', 'width']:
    y = D[name]; best = None
    for f0 in np.arange(1.0, 3.6, 0.2):                      # multi-start over frequency (1-3.5 Hz)
        for ph0 in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            try:
                r = least_squares(lambda p: model(p, t) - y, [np.ptp(y) / 2, 0.5, f0, ph0, np.median(y[-20:])],
                                  bounds=([0, 0.05, 0.8, -7, -5], [10, 5, 4.0, 7, 80]))
            except Exception: continue
            if best is None or r.cost < best.cost: best = r
    A, tau, f, ph, c = best.x; rmse = np.sqrt(2 * best.cost / len(t))
    zeta = 1 / np.sqrt(1 + (2 * np.pi * f * tau) ** 2)       # from decay rate 1/tau and damped freq f
    print(f'{name:6s} f={f:.2f} Hz  period={1/f:.2f} s  tau={tau:.2f} s  zeta~{zeta:.2f}  amp0={A:.2f} mm  rmse={rmse:.2f} mm  (signal p-p {np.ptp(y):.2f})')
    out[name] = best.x
np.savez('outputs/sway/wobble_fit.npz', **out)