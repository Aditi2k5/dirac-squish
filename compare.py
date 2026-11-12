import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import os, sys, csv
import numpy as np
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def heldout(sim, tgt, png):
    S, T = np.load(sim), np.load(tgt); t, Hm, Wm, P, D = T['t'], T['ball_height_mm'], T['width_mm'], T['plate_mm'], float(T['D_mm'])
    ts, Hs, Ws = S['t'], S['height'], S['width']
    ok = np.isfinite(P); t_on = t[ok][0]; imin = np.nanargmin(np.where(t > t_on, Hm, np.inf)); tmin = t[imin]
    tg = ts[(ts >= t_on) & (ts <= min(ts[-1], tmin + 2.0))]
    Hm_g = np.interp(tg, t, Hm); Hs_g = np.interp(tg, ts, Hs)
    rms = lambda m: float(np.sqrt(np.mean((Hs_g[m] - Hm_g[m]) ** 2))) if m.any() else np.nan
    print(f'sim {sim}  vs  {tgt}   sim covers {ts[0]:.1f}-{ts[-1]:.1f} s; press minimum at t={tmin:.2f}s (measured {Hm[imin]:.1f} mm)')
    print(f'height RMSE  press {rms(tg <= tmin):.2f} mm   release {rms(tg > tmin):.2f} mm   (n={len(tg)})')
    for dt in (0.0, 0.2, 0.4, 0.7, 1.0, 2.0):
        s_ = f'{np.interp(tmin + dt, ts, Hs):5.1f}' if tmin + dt <= ts[-1] else '  n/a (sim ends earlier)'
        print(f'  +{dt:.1f}s  measured {np.interp(tmin + dt, t, Hm):5.1f}   sim {s_}')
    dp = (ts > tmin - 1.0) & (ts <= tmin)
    print(f'width at deep squash (last 1 s before min): sim - measured = {np.mean(Ws[dp] - np.interp(ts[dp], t, Wm)):+.1f} mm')
    tf = np.arange(t[0], t[-1], 1e-3); hp = np.interp(tf, t[ok], P[ok], left=300.0, right=300.0)
    h, o = D, np.empty(len(tf))
    for k in range(len(tf)): h += 1e-3 * (D - h) / 0.114; h = min(h, hp[k]); o[k] = h
    print(f'lumped (tau 0.114 s from clip 1, frozen) release RMSE {np.sqrt(np.mean((np.interp(tg[tg > tmin], tf, o) - Hm_g[tg > tmin]) ** 2)):.2f} mm')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for a_, (lo, hi) in zip(ax, ((t_on - 0.5, tmin + 3), (tmin - 0.5, tmin + 2.5))):
        a_.plot(t, Hm, 'k', lw=2.5, label='measured'); a_.plot(t, np.minimum(P, 80), 'g:', label='book (input)')
        a_.plot(ts, Hs, 'C0', lw=1.6, label='3D MPM (frozen)'); a_.plot(tf, o, 'C4', lw=1, label='lumped tau=0.114 (frozen)')
        a_.set(xlim=(lo, hi), ylim=(0, 85), xlabel='s', ylabel='ball height (mm)')
    ax[0].legend(fontsize=8); ax[0].set_title('held-out clip, no refit'); ax[1].set_title('zoom: release')
    plt.tight_layout(); plt.savefig(png, dpi=90); print('wrote', png)


def band(tgt, sims):
    T = np.load(tgt); t, Hm, P = T['t'], T['ball_height_mm'], T['plate_mm']; ok = np.isfinite(P); t_on = t[ok][0]
    tmin = t[np.nanargmin(np.where(t > t_on, Hm, np.inf))]; R = [np.load(f) for f in sims]
    t0, t1 = max(r['t'][0] for r in R), min(r['t'][-1] for r in R); tg = np.linspace(max(t0, t_on), min(t1, tmin + 2.0), 300)
    Hs = np.array([np.interp(tg, r['t'], r['height']) for r in R]); Hm_g = np.interp(tg, t, Hm)
    for f, h in zip(sims, Hs): print(f'{os.path.basename(f):32s} RMSE all {np.sqrt(np.mean((h - Hm_g) ** 2)):.2f} mm   press {np.sqrt(np.mean((h - Hm_g)[tg <= tmin] ** 2)):.2f}   release {np.sqrt(np.mean((h - Hm_g)[tg > tmin] ** 2)):.2f}')
    sp = Hs.max(0) - Hs.min(0); print(f'spread across runs (max-min): mean {sp.mean():.2f} mm, max {sp.max():.2f} mm; at the press minimum {np.interp(tmin, tg, sp):.2f} mm; 0.5 s after {np.interp(min(tmin + 0.5, tg[-1]), tg, sp):.2f} mm')
    fig, a_ = plt.subplots(figsize=(8, 4.3)); a_.fill_between(tg, Hs.min(0), Hs.max(0), color='C0', alpha=0.25, label='min-max over runs')
    a_.plot(t, Hm, 'k', lw=2.5, label='measured'); a_.plot(t, np.minimum(P, 80), 'g:', label='book (input)')
    for f, h in zip(sims, Hs): a_.plot(tg, h, lw=0.8, label=os.path.basename(f).replace('sim3d_', '').replace('.npz', ''))
    a_.set(xlim=(t_on - 0.3, tg[-1] + 0.2), ylim=(0, 85), xlabel='s', ylabel='ball height (mm)'); a_.legend(fontsize=7)
    png = os.path.join(os.path.dirname(sims[0]), 'band.png'); plt.tight_layout(); plt.savefig(png, dpi=90); print('wrote', png)


if len(sys.argv) > 2 and sys.argv[1] == 'band':
    band(sys.argv[2], sys.argv[3:]); sys.exit(0)


if len(sys.argv) > 1 and sys.argv[1] == 'heldout':
    _a = sys.argv[2:5] + ['outputs/clip2/sim3d_alpha100.npz', 'outputs/clip2/targets.npz'][len(sys.argv[2:5]):]
    heldout(_a[0], _a[1], _a[2] if len(_a) > 2 else os.path.join(os.path.dirname(_a[0]), 'heldout.png'))
    sys.exit(0)


T = np.load('outputs/clip1/targets.npz'); t_all = T['t']; D = float(T['D_mm'])
Hm_all, Wm_all, plate = T['ball_height_mm'], T['width_mm'], T['plate_mm']
ok = np.isfinite(plate); tv, ps = t_all[ok], savgol_filter(plate[ok], 11, 2); vend = np.polyfit(tv[-8:], ps[-8:], 1)[0]
def hp(tt): return np.where(tt <= tv[0], ps[0], np.where(tt >= tv[-1], ps[-1] + vend * (tt - tv[-1]), np.interp(tt, tv, ps)))

tg = t_all[np.arange(np.searchsorted(t_all, 4.2), np.searchsorted(t_all, 8.8), 3)]           # same output grid as the sims
Hm = np.interp(tg, t_all[np.isfinite(Hm_all)], Hm_all[np.isfinite(Hm_all)]); Wm = np.interp(tg, t_all[np.isfinite(Wm_all)], Wm_all[np.isfinite(Wm_all)])
P = hp(tg)

# ---- geometric proxy: first-order recovery, unilateral contact (see baseline_1dof.py), fine time grid then sampled
def lumped(tau=0.1):
    tf = np.arange(tg[0], tg[-1] + 1e-9, 1e-3); hpf = hp(tf); h = D; out = np.empty(len(tf))
    for k in range(len(tf)):
        if k: h += 1e-3 * (D - h) / tau
        out[k] = h = min(h, hpf[k])
    return np.interp(tg, tf, out)
H_proxy = lumped()
press = (tg > 5.0) & (tg < 7.4)
q = -np.polyfit(np.log(np.clip(H_proxy[press] / D, 1e-3, 1)), np.log(Wm[press] / D), 1)[0]    # width law fitted to the measured press
W_proxy = D * np.clip(H_proxy / D, 1e-3, 1) ** (-q)

# ---- learned black box: ridge ARX, ball height from the history of the book height (book clipped to 100 mm once it leaves)
def arx_features(Pk, Hk, na=6, nb=6):
    rows = []
    for k in range(max(na, nb), len(Pk)): rows.append(np.concatenate([Hk[k - na:k], Pk[k - nb + 1:k + 1]]))
    return np.array(rows)
Pc = np.minimum(P, 100.0); na = nb = 6; X = arx_features(Pc, Hm, na, nb); y = Hm[max(na, nb):]; tt = tg[max(na, nb):]
def ridge(Xa, ya, lam=1.0): return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya)
def rollout(w, Pk, H_init):
    H = list(H_init)
    for k in range(len(H_init), len(Pk)): H.append(float(np.concatenate([H[k - na:k], Pk[k - nb + 1:k + 1]]) @ w))
    return np.array(H)
w_all = ridge(X, y); H_arx = rollout(w_all, Pc, Hm[:max(na, nb)])                                      # trained on everything, free-running rollout
w_tr = ridge(X[tt < 7.4], y[tt < 7.4]); H_arx_split = rollout(w_tr, Pc, Hm[:max(na, nb)])              # trained on press+hold only, asked to predict the release

models = {}
def load(name, path):
    if os.path.exists(path):
        S = np.load(path); n = min(len(S['t']), len(tg))
        models[name] = dict(H=np.full(len(tg), np.nan), W=np.full(len(tg), np.nan), J=np.full(len(tg), np.nan), inv=int(np.max(S['inverted'])) if 'inverted' in S.files else 0,
                            wall=float(S['wall_s']) if 'wall_s' in S.files else np.nan)
        models[name]['H'][:n] = S['height'][:n]; models[name]['W'][:n] = S['width'][:n]; models[name]['J'][:n] = S['J'][:n]
load('mpm', next((f for f in ('outputs/clip1/sim3d_alpha100_long.npz', 'outputs/clip1/sim3d_alpha100.npz') if os.path.exists(f)), ''))   # the default run_3d.py output name also works
for mm in ('fem', 'ms', 'xpbd', 'mujoco', 'pybullet'): load(mm, f'outputs/models/mesh_{mm}_h4.5.npz')
models['proxy'] = dict(H=H_proxy, W=W_proxy, J=np.clip(H_proxy / D, 0, 1) * (W_proxy / D) ** 2, inv=0, wall=0.0)
models['arx (in-sample)'] = dict(H=H_arx, W=np.full(len(tg), np.nan), J=np.full(len(tg), np.nan), inv=0, wall=0.0)
models['arx (trained on press only)'] = dict(H=H_arx_split, W=np.full(len(tg), np.nan), J=np.full(len(tg), np.nan), inv=0, wall=0.0)

W_ = (tg >= 5.0) & (tg <= 8.8); R_ = tg >= 8.1; DP = (tg >= 6.0) & (tg <= 7.3)
rms = lambda a, m_: float(np.sqrt(np.nanmean((a[m_] - Hm[m_]) ** 2))) if np.isfinite(a[m_]).any() else np.nan
rows = []
for name, d in models.items():
    rows.append(dict(model=name, H_rmse=rms(d['H'], W_), H_rel=rms(d['H'], R_), W_deep=float(np.nanmean(d['W'][DP] - Wm[DP])) if np.isfinite(d['W'][DP]).any() else np.nan,
                     Jmin=float(np.nanmin(d['J'])) if np.isfinite(d['J']).any() else np.nan, inverted=d['inv'], wall_s=d['wall']))
print(f'width-law exponent q = {q:.2f}   (measured width at deep squash: {np.mean(Wm[DP]) - D:+.1f} mm vs rest)')
print(f'{"model":30s} {"H_rmse":>7s} {"H_rel":>7s} {"W_deep":>7s} {"Jmin":>6s} {"inverted":>8s} {"wall_s":>7s}')
for r in rows: print(f'{r["model"]:30s} {r["H_rmse"]:7.2f} {r["H_rel"]:7.2f} {r["W_deep"]:7.1f} {r["Jmin"]:6.2f} {r["inverted"]:8d} {r["wall_s"]:7.0f}')
with open('outputs/models/compare.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8)); sty = {'mpm': 'C0', 'fem': 'C1', 'ms': 'C2', 'xpbd': 'C3', 'mujoco': 'C7', 'pybullet': 'C8', 'proxy': 'C4', 'arx (in-sample)': 'C5', 'arx (trained on press only)': 'C6'}
for a_, (lo, hi, ttl) in zip(ax[:2], ((4.2, 8.8, 'ball height, press and release'), (7.6, 8.8, 'zoom: release'))):
    a_.plot(tg, Hm, 'k', lw=2.5, label='measured'); a_.plot(tg, np.minimum(P, 80), 'g:', label='book (input)')
    for name, d in models.items(): a_.plot(tg, d['H'], color=sty.get(name), lw=1.4, label=name)
    a_.set(xlim=(lo, hi), ylim=(0, 85), xlabel='s', ylabel='mm', title=ttl)
ax[0].legend(fontsize=7, ncol=2)
ax[2].plot(tg, Wm - D, 'k', lw=2.5, label='measured width - rest')
for name, d in models.items():
    if np.isfinite(d['W']).any(): ax[2].plot(tg, d['W'] - D, color=sty.get(name), lw=1.4, label=name)
ax[2].set(xlim=(4.2, 8.8), ylim=(-10, 30), xlabel='s', ylabel='mm', title='width - rest (measured width is suspect, see README)'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('outputs/models/compare.png', dpi=90); print('wrote outputs/models/compare.png, outputs/models/compare.csv')