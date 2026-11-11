"""Drive the MPM foam disc with the plate trajectory measured from the video and compare with the observed silhouette.
Usage:  python fit_2d.py baseline | fit [E_RHO] [MAX_EVALS] | sweep (damping sweep that sets run_3d.py's default alpha) | plot [npz ...]
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import sys, json, time, numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import mpm2d as mpm

T = np.load('outputs/clip1/targets.npz')
t, plate_mm, width_mm, area_mm2 = T['t'], T['plate_mm'], T['width_mm'], T['area_mm2']
D = float(T['D_mm']); A_rest = float(T['area_rest_mm2'])
valid = np.isfinite(plate_mm)
ta, tb = t[valid][0], t[valid][-1]
# ---- plate input: smoothed measured book underside; after the book leaves the clip it keeps lifting
tv, pv_ = t[valid], plate_mm[valid]
ps = savgol_filter(pv_, 11, 2)
_v_end = np.polyfit(tv[-8:], ps[-8:], 1)[0]


def plate_h(tt):
    if tt <= tv[0]: return ps[0]
    if tt >= tv[-1]: return ps[-1] + _v_end * (tt - tv[-1])
    return float(np.interp(tt, tv, ps))


_dt_d = 0.02


def plate_v(tt):
    return (plate_h(tt + _dt_d) - plate_h(tt - _dt_d)) / (2 * _dt_d)


# ---- grid-contact calibration (2 mm grid): (command plate) -> (particle top). See test_mpm.py
CAL_P = np.array([9, 15, 21, 27, 33, 39, 45, 51, 57, 63.])
CAL_TOP = np.array([11.1, 17.0, 22.9, 28.8, 34.7, 40.5, 46.4, 52.2, 57.9, 63.4])


def cmd_plate(h):
    """plate command so that the simulated ball top meets the measured book underside h."""
    target = h - 0.3                                  # silhouette ~ particle centre + ~half a (compressed) particle
    if target >= 66.0: return target - 0.5
    if target <= CAL_TOP[0]: return target - 2.1
    return float(np.interp(target, CAL_TOP, CAL_P))


mpm.PLATE_OFFSET = 0.0
FRAMES = np.arange(np.searchsorted(t, ta), np.searchsorted(t, 9.2), 2)
t_grid = t[FRAMES]
obs_w = width_mm[FRAMES] - 68.7                        # change in visible width vs rest (mm)
obs_a = area_mm2[FRAMES] / A_rest                      # visible area / rest
# the loss window: everything up to 1.0 s after the book has left
LOSS = t_grid <= 9.0


def predict(par, e_rho=30.0, grid=None):
    p = dict(e_rho=e_rho, nu=par[0], alpha=par[1], mu_p=par[2], mu_t=par[2], gravity=0.0)
    tg = t_grid if grid is None else grid
    # mpm.simulate adds PLATE_OFFSET inside; we feed already-calibrated commands through plate_fn
    r = mpm.simulate(p, lambda s: cmd_plate(plate_h(s)), plate_v, tg, D)
    return r['width'] - r['w0'], r['J'], r


def loss(par, e_rho=30.0, verbose=False):
    nu, alpha, mu = par
    if not (-0.45 <= nu <= 0.3 and 0 <= alpha <= 30 and 0 <= mu <= 1.5): return 1e3
    w, J, r = predict(par, e_rho)
    ew = (w - obs_w)[LOSS] / 1.0                       # 1 mm sigma
    ea = (J - obs_a)[LOSS] / 0.04                      # 4 % sigma
    L = float(np.mean(ew ** 2) + np.mean(ea ** 2))
    if verbose: print(f'nu={nu:+.3f} alpha={alpha:.2f} mu={mu:.2f} loss={L:.3f} (w {np.mean(ew**2):.2f}, A {np.mean(ea**2):.2f})', flush=True)
    return L


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    if mode == 'baseline':
        par = [0.05, 2.0, 0.3]
        t0 = time.time(); w, J, r = predict(par); print('wall', round(time.time() - t0, 1), 's, dt', r['dt'])
        np.savez('outputs/models/baseline_2d.npz', t=t_grid, w=w, J=J, obs_w=obs_w, obs_a=obs_a, h=r['height'])
        print('loss', loss(par, verbose=True))
    if mode == 'fit':
        E_RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
        hist = []
        def f(p):
            L = loss(p, E_RHO, verbose=True); hist.append((list(map(float, p)), L)); return L
        x0 = np.array([-0.15, 3.0, 0.3])
        sim = np.array([x0, x0 + [0.25, 0, 0], x0 + [0, 4.0, 0], x0 + [0, 0, 0.4]])
        res = minimize(f, x0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[3]) if len(sys.argv) > 3 else 40, initial_simplex=sim, xatol=0.01, fatol=0.05))
        w, J, r = predict(res.x, E_RHO)
        json.dump(dict(e_rho=E_RHO, nu=res.x[0], alpha=res.x[1], mu=res.x[2], loss=res.fun, hist=hist), open(f'outputs/models/fit2d_erho{E_RHO:g}.json', 'w'))
        np.savez(f'outputs/models/fit2d_erho{E_RHO:g}.npz', t=t_grid, w=w, J=J, obs_w=obs_w, obs_a=obs_a, h=r['height'])
        print('BEST', res.x, res.fun)
    if mode == 'sweep':
        Hm = np.interp(t_grid, t[np.isfinite(T['ball_height_mm'])], T['ball_height_mm'][np.isfinite(T['ball_height_mm'])]); H0 = D
        for alpha in [2, 30, 100, 300]:
            r = mpm.simulate(dict(e_rho=8.0, nu=0.0, alpha=float(alpha), mu_p=0.5, mu_t=0.5, gravity=0.0), lambda s: cmd_plate(plate_h(s)), plate_v, t_grid, D)
            Hs = r['height']; rel = (t_grid > 7.0) & (t_grid < 8.05); late = t_grid > 8.1
            print(f'alpha={alpha:4d}  release RMSE {np.sqrt(np.mean((Hs[rel]-Hm[rel])**2)):4.1f} mm   max height after release {Hs[late].max():5.1f} mm (rest {H0:.1f})  final {Hs[-1]:5.1f}', flush=True)
    if mode == 'plot':
        import os
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        paths = sys.argv[2:] or [p for p in ('outputs/models/baseline_2d.npz', 'outputs/models/fit2d_erho8.npz') if os.path.exists(p)]
        if not paths: raise SystemExit('run `python fit_2d.py baseline` or `python fit_2d.py fit 8 36` first')
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True); d0 = np.load(paths[0])
        ax[0].plot(d0['t'], d0['obs_w'], 'k', lw=2, label='video'); ax[1].plot(d0['t'], d0['obs_a'], 'k', lw=2)
        for p in paths:
            d = np.load(p); lab = os.path.basename(p).replace('.npz', ''); ax[0].plot(d['t'], d['w'], label=lab); ax[1].plot(d['t'], d['J'])
        ax[0].set_ylabel('width change vs rest (mm)'); ax[0].legend(); ax[1].set_ylabel('silhouette area / rest'); ax[1].set_xlabel('s')
        for a_ in ax: a_.grid(alpha=.3)
        plt.tight_layout(); plt.savefig('outputs/models/fit2d.png', dpi=100); print('wrote outputs/models/fit2d.png')