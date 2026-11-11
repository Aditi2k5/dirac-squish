"""Lumped (1-DOF) models of the ball height, driven by the measured book height. No mesh, no grid.
  python lumped.py                              baseline: geometry law, recovery-time scan, fresh-press prediction -> outputs/clip1/baseline_1dof.png
  python lumped.py cross [A.npz B.npz]          fit three recovery laws on one clip, score them on the other (default outputs/clip1/targets.npz, outputs/clip2/targets.npz)
Recovery laws (the book can only push, never pull):
  tau   : H' = (D-H)/tau
  tau_c : H' = (D-H)/(tau0*exp(k*c)), c=(D-H)/D          (recovery speed depends on compression depth)
  osc2  : H'' = w^2 (D-H) - 2 z w H'                      (inertia/viscous lag -> S-shaped recovery; no new physics claimed)
Score = RMSE of ball height from the press minimum to +2 s (the release).
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter


def load(path):
    T = np.load(path); t, D = T['t'], float(T['D_mm']); P, H = T['plate_mm'], T['ball_height_mm']
    ok = np.isfinite(P); Ps = P.copy(); Ps[ok] = savgol_filter(P[ok], 11, 2)
    tf = np.arange(t[0], t[-1], 1e-3)
    hp = np.where(tf < t[ok][0], 300.0, np.where(tf > t[ok][-1], 300.0, np.interp(tf, t[ok], Ps[ok])))
    Hm = np.interp(tf, t[np.isfinite(H)], H[np.isfinite(H)])
    i = np.nanargmin(np.where(t > t[ok][0], H, np.inf)); tmin = t[i]
    return dict(tf=tf, hp=hp, Hm=Hm, D=D, tmin=tmin, name='/'.join(path.split('/')[-2:]), ok_last=t[ok][-1], ok_first=t[ok][0])


def sim(model, p, c):
    tf, hp, D = c['tf'], c['hp'], c['D']; h = D; v = 0.0; out = np.empty(len(tf)); dt = 1e-3
    for k in range(len(tf)):
        if model == 'tau':   h += dt * (D - h) / p[0]
        elif model == 'tau_c': h += dt * (D - h) / (p[0] * np.exp(p[1] * (D - h) / D))
        else:
            v += dt * (p[0] ** 2 * (D - h) - 2 * p[1] * p[0] * v); h += dt * v
        if h > hp[k]: h = hp[k]; v = min(v, 0.0) if model == 'osc2' else v   # plate pushes the ball down
        if h > D: h = D; v = min(v, 0.0) if model == 'osc2' else v
        out[k] = h
    return out


def err(model, p, c, which='release'):
    o = sim(model, p, c); tf = c['tf']
    m = (tf >= c['tmin']) & (tf <= c['tmin'] + 2.0) if which == 'release' else (tf >= c['ok_first']) & (tf <= c['tmin'])
    return float(np.sqrt(np.mean((o[m] - c['Hm'][m]) ** 2)))


X0 = {'tau': [0.1], 'tau_c': [0.1, 0.0], 'osc2': [8.0, 1.0]}
def fit(model, clips):
    f = lambda q: np.mean([err(model, np.exp(q) if model != 'tau_c' else [np.exp(q[0]), q[1]], c) ** 2 for c in clips])
    q0 = np.log(X0[model]) if model != 'tau_c' else [np.log(0.1), 0.0]
    r = minimize(f, q0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-4, maxiter=150))
    return (np.exp(r.x) if model != 'tau_c' else [np.exp(r.x[0]), r.x[1]])




def baseline():
    T = np.load('outputs/clip1/targets.npz')
    t, plate, H = T['t'], T['plate_mm'], T['ball_height_mm']
    area = T['area_mm2'] / T['area_rest_mm2']
    H0 = float(T['D_mm'])

    ok = (t > 4.75) & (t < 8.2) & np.isfinite(H)                          # book in firm, visible contact
    i_min = np.nanargmin(np.where(ok, H, np.inf))
    idx = np.arange(len(t))
    load, unload = ok & (idx <= i_min), ok & (idx > i_min)
    x = H / H0


    def powfit(xx, yy):
        m = np.isfinite(xx) & np.isfinite(yy) & (xx > 0.05)
        return np.polyfit(np.log(xx[m]), np.log(yy[m]), 1)[0]


    print(f'area/rest ~ (H/H0)^p : p_loading={powfit(x[load], area[load]):.2f}  p_unloading={powfit(x[unload], area[unload]):.2f}'
          '   (p = 1: width constant, pure vertical compaction)')

    tt = t[(t >= 4.75) & (t <= 8.6)]
    good = np.isfinite(plate)
    hp = np.interp(tt, t[good], plate[good])
    hp = np.where(tt > t[good][-1], 200.0, hp)                            # book has left the frame
    Hm = np.interp(tt, t[np.isfinite(H)], H[np.isfinite(H)])


    def run(tau, hplate, tgrid, h0=H0):
        out, h = np.empty(len(tgrid)), h0
        for k in range(len(tgrid)):
            if k:
                dt, sub = tgrid[k] - tgrid[k - 1], 20
                for _ in range(sub):
                    h += (dt / sub) * (H0 - h) / tau
                    h = min(h, hplate[k])                                 # unilateral contact
            out[k] = h = min(h, hplate[k])
        return out


    win = (tt >= 7.0) & (tt <= 8.05)                                      # clean part of the release
    print('release-window RMSE (mm) vs recovery time tau (s):')
    for tau in [0.005, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.4]:
        print(f'   tau={tau:5.3f}  RMSE={np.sqrt(np.mean((run(tau, hp, tt)[win] - Hm[win]) ** 2)):.2f}')
    print(f'book lift speed in the release: up to {np.nanmax(np.where(win, np.gradient(hp, tt), np.nan)):.0f} mm/s')

    tf = np.linspace(0, 3.0, 600)                                         # fresh path: press to 50 %, hold, 0.1 s release
    hf = np.where(tf < 1.0, H0 + 3 - (H0 + 3 - 0.5 * H0) * tf,
                  np.where(tf < 1.5, 0.5 * H0, np.where(tf < 1.6, 0.5 * H0 + (tf - 1.5) * 10 * H0, 300.0)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(x[load], area[load], 'b.', ms=3, label='loading')
    ax[0].plot(x[unload], area[unload], 'r.', ms=3, label='unloading')
    ax[0].plot([0, 1], [0, 1], 'k--', lw=.8, label='area = H/H0')
    ax[0].set(xlabel='H / H0', ylabel='visible area / rest', title='(a) geometry law'); ax[0].legend()
    ax[1].plot(tt, Hm, 'k', lw=2, label='measured ball height'); ax[1].plot(tt, np.minimum(hp, 80), 'g:', label='book underside')
    for tau in (0.02, 0.1, 0.4):
        ax[1].plot(tt, run(tau, hp, tt), label=f'tau={tau}s')
    ax[1].set(xlim=(6.8, 8.5), ylim=(0, 85), xlabel='s', ylabel='mm', title='(b) release window'); ax[1].legend(fontsize=7)
    for tau in (0.02, 0.1, 0.4):
        ax[2].plot(tf, run(tau, hf, tf), label=f'tau={tau}s')
    ax[2].plot(tf, np.minimum(hf, 80), 'g:', label='new plate path')
    ax[2].set(ylim=(0, 85), xlabel='s', title='(c) fresh press, fast release'); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig('outputs/clip1/baseline_1dof.png', dpi=95)
    print('wrote outputs/clip1/baseline_1dof.png')


def cross(A, B):
    cA, cB = load(A), load(B)
    print(f'clip A {cA["name"]}: min at t={cA["tmin"]:.2f}s   clip B {cB["name"]}: min at t={cB["tmin"]:.2f}s')
    print(f'{"model":6s} {"params (fit on A)":>26s} {"A in":>6s} {"B out":>6s} | {"params (fit on B)":>26s} {"B in":>6s} {"A out":>6s} | {"params (A+B)":>22s} {"A":>5s} {"B":>5s}')
    for m in ('tau', 'tau_c', 'osc2'):
        pa, pb, pj = fit(m, [cA]), fit(m, [cB]), fit(m, [cA, cB])
        f = lambda p: '(' + ', '.join(f'{x:.3g}' for x in p) + ')'
        print(f'{m:6s} {f(pa):>26s} {err(m, pa, cA):6.2f} {err(m, pa, cB):6.2f} | {f(pb):>26s} {err(m, pb, cB):6.2f} {err(m, pb, cA):6.2f} | {f(pj):>22s} {err(m, pj, cA):5.2f} {err(m, pj, cB):5.2f}')
    print('(B out / A out = held-out clip, parameters NOT refitted. In-sample numbers are optimistic.)')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'cross':
        cross(sys.argv[2] if len(sys.argv) > 3 else 'outputs/clip1/targets.npz', sys.argv[3] if len(sys.argv) > 3 else 'outputs/clip2/targets.npz')
    else:
        baseline()