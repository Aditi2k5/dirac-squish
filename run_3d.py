import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import argparse, time
import numpy as np
from scipy.signal import savgol_filter

ap = argparse.ArgumentParser()
ap.add_argument('--t0', type=float, default=4.2); ap.add_argument('--t1', type=float, default=8.6)
ap.add_argument('--stride', type=int, default=3, help='output every N-th video frame')
ap.add_argument('--e_rho', type=float, default=8.0, help='Young modulus / density (m^2/s^2)')
ap.add_argument('--nu', type=float, default=0.0); ap.add_argument('--alpha', type=float, default=100.0, help='Rayleigh damping (1/s); 100 chosen by a 2D sweep as the smallest tested value with no post-release overshoot')
ap.add_argument('--mu', type=float, default=0.5, help='Coulomb friction (plate and table)')
ap.add_argument('--gap', type=float, default=-1.0, help='grid contact gap (mm) subtracted from the plate command; -1 = auto (height-dependent, scaled from the 2D calibration)')
ap.add_argument('--gravity', type=float, default=0.0, help='mm/s^2. Default 0: at the soft reference stiffness the ball would sag ~7 mm under its own weight (an artifact); a real foam ball (E/rho in the hundreds) sags negligibly')
ap.add_argument('--targets', default='outputs/clip1/targets.npz', help='targets file of the clip whose book path drives the sim (e.g. outputs/clip2/targets.npz for the held-out test)')
ap.add_argument('--tilt', action='store_true', help='tilt the book plane by the slope stored in the targets file (extract_targets.py --tilt_half_px 150). Tested on clip 2: did NOT improve the fit (see README)')
ap.add_argument('--plate_smooth', type=int, default=11, help='Savitzky-Golay window (frames, odd) for the measured book path; the book is lowered slowly, so 31-41 is reasonable for noisy clips')
ap.add_argument('--proj_k', type=float, default=0.0, help='with --project: also compress F by proj_k*d/DX when a particle is moved down by d (0 = positions only)')
ap.add_argument('--project', action='store_true', help='particle-level contact: push any particle that gets above the book surface back down (fixes the lag of the grid-only contact during slow presses; see README)')
ap.add_argument('--out', default=None, help='default: <folder of --targets>/sim3d_alpha<alpha>.npz')
a = ap.parse_args()
if a.out is None: a.out = f'{os.path.dirname(a.targets) or "."}/sim3d_alpha{a.alpha:g}.npz'      # saved next to the targets of the clip that drives it (outputs/clip1/ or outputs/clip2/)
import mpm3d as m            # imported after parsing so MPM_DX3 / MPM_ARCH env vars apply

T = np.load(a.targets)
t, plate, D = T['t'], T['plate_mm'], float(T['D_mm'])
v = np.isfinite(plate); tv, ps = t[v], savgol_filter(plate[v], a.plate_smooth, 2)
vend = np.polyfit(tv[-8:], ps[-8:], 1)[0]
def h(tt):
    if tt <= tv[0]: return ps[0]
    if tt >= tv[-1]: return ps[-1] + vend * (tt - tv[-1])
    return float(np.interp(tt, tv, ps))
if a.tilt:
    ok_s = np.isfinite(T['slope']); tsl = t[ok_s]; ssl = savgol_filter(T['slope'][ok_s], 21, 2)
def slope(tt): return float(np.interp(tt, tsl, ssl)) if a.tilt else 0.0       # clamps to the first/last measured value outside the visible window
_CP = np.array([9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 70.]); _CG = 1.5 * np.array([2.06, 1.97, 1.88, 1.77, 1.66, 1.54, 1.40, 1.20, 0.91, 0.44, 0.0])
def gap(hh): return 0.75 if a.project and a.gap < 0 else a.gap if a.gap >= 0 else float(np.interp(hh, _CP, _CG))   # 3 mm grid ~ 1.5 x the 2 mm 2D calibration
def vel(tt): return (h(tt + 0.02) - h(tt - 0.02)) / 0.04
fr = np.arange(np.searchsorted(t, a.t0), np.searchsorted(t, a.t1), a.stride)
tg = t[fr]
SETTLE = 0.6                                  # s of heavily damped settling before t0 (plate far away)
tg_sim = np.concatenate([[tg[0] - SETTLE], tg])
snaps = []
import time; t_wall = time.time()
def log(tk, o): print(f't={tk:5.2f}s  width {o[0]:5.1f} mm  height {o[1]:5.1f} mm  J {o[2]:.2f}   ({time.time()-t_wall:.0f}s elapsed)', flush=True)
r = m.simulate(dict(e_rho=a.e_rho, nu=a.nu, alpha=a.alpha, mu_p=a.mu, mu_t=a.mu, gravity=a.gravity, settle_s=SETTLE, settle_alpha=40.0),
               lambda s: (h(s) - gap(h(s))) if s >= tg[0] else 300.0, lambda s: vel(s) if s >= tg[0] else 0.0, tg_sim, D, snap=snaps, log=log, slope_fn=slope, proj_fn=((lambda s: h(s) - 0.75) if a.project else None), proj_strain=a.proj_k)
snaps = snaps[1:]; r = {k_: (v_[1:] if isinstance(v_, np.ndarray) and v_.ndim == 1 and len(v_) == len(tg_sim) else v_) for k_, v_ in r.items()}
rng = np.random.default_rng(0); keep = rng.choice(r['n'], size=min(9000, r['n']), replace=False)
np.savez_compressed(a.out, t=tg, width=r['width'], height=r['height'], J=r['J'], w0=r['w0'],
                    snaps=np.stack([s[keep] for s in snaps]).astype(np.float16), y_floor=m.Y_FLOOR, cx=m.CX,
                    params=np.array([a.e_rho, a.nu, a.alpha, a.mu, a.gap, m.DX, a.gravity]))
print('wrote', a.out)