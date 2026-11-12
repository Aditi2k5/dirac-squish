import argparse, time, re
import numpy as np
import mujoco
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=float, default=4.5); ap.add_argument('--e_rho', type=float, default=8.0); ap.add_argument('--nu', type=float, default=0.0)
ap.add_argument('--damp', type=float, default=0.0, help='flex elasticity damping coefficient (unstable above 0 with Euler here)'); ap.add_argument('--alpha', type=float, default=100.0, help='mass-proportional damping (1/s), same as the other models'); ap.add_argument('--mu', type=float, default=0.5)
ap.add_argument('--t0', type=float, default=4.2); ap.add_argument('--t1', type=float, default=8.8); ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--lift', type=float, default=0.2); ap.add_argument('--dt', type=float, default=2e-4); ap.add_argument('--out', default='outputs/mesh_mujoco_h4.5.npz')
a = ap.parse_args()
T = np.load('outputs/targets.npz'); t_vid, plate, D = T['t'], T['plate_mm'], float(T['D_mm']); R = D / 2

def build(h, R, seed=0):                                                   # identical to mesh_models.build
    g = np.arange(-R, R + 1e-9, h); X = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    rng = np.random.default_rng(seed); inner = X[np.linalg.norm(X, axis=1) < R - 0.55 * h]; inner = inner + rng.normal(0, 0.03 * h, inner.shape)
    n = int(4 * np.pi * R * R / (0.85 * h * h)); i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    S = R * np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    P = np.vstack([inner, S]); tet = Delaunay(P).simplices
    Dm = np.stack([P[tet[:, 1]] - P[tet[:, 0]], P[tet[:, 2]] - P[tet[:, 0]], P[tet[:, 3]] - P[tet[:, 0]]], 2); sw = np.linalg.det(Dm) < 0; tet[sw] = tet[sw][:, [0, 2, 1, 3]]
    Dm = np.stack([P[tet[:, 1]] - P[tet[:, 0]], P[tet[:, 2]] - P[tet[:, 0]], P[tet[:, 3]] - P[tet[:, 0]]], 2); vol = np.linalg.det(Dm) / 6
    tet = tet[vol > 0.05 * np.median(vol)]; used = np.unique(tet); remap = -np.ones(len(P), int); remap[used] = np.arange(len(used))
    return P[used], remap[tet]
P0, tet = build(a.h, R); tet = tet[:, [0, 2, 1, 3]]    # (x,y,z)->(x,z,y) is a reflection: swap two nodes to keep positive orientation in MuJoCo's frame
n, nt = len(P0), len(tet)
# MuJoCo world: z up. our y -> z. Positions in metres.
pts = np.stack([P0[:, 0], P0[:, 2], P0[:, 1] + R + a.lift], 1) / 1000.0        # +0.2 mm lift: starts just above the table
rho = 100.0; E = a.e_rho * rho
xml = f'''<mujoco><size memory="1500M"/><option timestep="{a.dt}" gravity="0 0 0" integrator="Euler"/>
<worldbody>
 <geom name="table" type="plane" size="1 1 0.1" friction="{a.mu} 0.005 0.0001"/>
 <body name="book" mocap="true" pos="0 0 0.3"><geom type="box" size="0.075 0.075 0.007" friction="{a.mu} 0.005 0.0001" rgba="0.3 0.4 0.7 1"/></body>
 <flexcomp type="direct" name="ball" dim="3" radius="0.0006" mass="{rho * 4 / 3 * np.pi * (R / 1000) ** 3}"
   point="{' '.join(f'{c:.6f}' for c in pts.ravel())}" element="{' '.join(map(str, tet.ravel()))}">
   <contact friction="{a.mu} 0.005 0.0001" solref="0.004 1"/>
   <edge equality="false" damping="{a.damp}"/>
   <elasticity young="{E}" poisson="{a.nu}" damping="{a.damp}"/>
 </flexcomp>
</worldbody></mujoco>'''
mdl = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(mdl)
mdl.dof_damping[:] = a.alpha * mdl.body_mass[mdl.dof_bodyid]          # Rayleigh mass-proportional damping on every vertex DOF
ok = np.isfinite(plate); tv, ps = t_vid[ok], savgol_filter(plate[ok], 11, 2); vend = np.polyfit(tv[-8:], ps[-8:], 1)[0]
def hp(tt): return float(ps[0]) if tt <= tv[0] else (float(ps[-1] + vend * (tt - tv[-1])) if tt >= tv[-1] else float(np.interp(tt, tv, ps)))
def verts(): return np.array(d.flexvert_xpos).reshape(-1, 3)
V0 = None
fr = np.arange(np.searchsorted(t_vid, a.t0), np.searchsorted(t_vid, a.t1), a.stride); tg = t_vid[fr]
def vol_ratio(X):
    dd = np.stack([X[tet[:, 1]] - X[tet[:, 0]], X[tet[:, 2]] - X[tet[:, 0]], X[tet[:, 3]] - X[tet[:, 0]]], 2); return np.linalg.det(dd) / 6
mujoco.mj_forward(mdl, d)
res = dict(height=[], width=[], J=[], inv=[]); snaps = []; tc = tg[0]; t0w = time.time(); vol0 = None
for k, tk in enumerate(tg):
    nsub = max(1, int(round((tk - tc) / a.dt))) if k else 0
    for s_ in range(nsub):
        tm = tc + (s_ + 1) * (tk - tc) / nsub; d.mocap_pos[0] = [0, 0, hp(tm) / 1000 + 0.007]; mujoco.mj_step(mdl, d)
    tc = tk; mujoco.mj_forward(mdl, d) if k == 0 else None
    X = verts()
    if vol0 is None: vol0 = vol_ratio(X).sum()
    dv = vol_ratio(X)
    res['height'].append(X[:, 2].max() * 1000); res['width'].append(np.ptp(X[:, 0]) * 1000); res['J'].append(dv.sum() / vol0); res['inv'].append(int((dv < 0).sum()))
    snaps.append(np.stack([X[:, 0], X[:, 2], X[:, 1]], 1) * 1000)
    if not np.isfinite(res['height'][-1]): print('NaN, stopped'); break
    if k % 5 == 0: print(f't={tk:5.2f}  H {res["height"][-1]:5.1f}  W {res["width"][-1]:5.1f}  J {res["J"][-1]:.2f}  inverted {res["inv"][-1]}  ({time.time()-t0w:.0f}s)', flush=True)
nk = len(res['height']); S = np.stack(snaps) - np.array([0, 0, 0]); S[..., 0] += 64 - 0; S[..., 2] += 64; S[..., 1] += 9
np.savez_compressed(a.out, t=tg[:nk], width=res['width'], height=res['height'], J=res['J'], inverted=res['inv'], minratio=np.zeros(nk), w0=D,
                    snaps=S.astype(np.float16), y_floor=9.0, cx=64.0, model='mujoco', wall_s=time.time() - t0w, params=np.array([a.e_rho, a.nu, a.damp, a.mu, a.h, n, nt]))
print('wrote', a.out)