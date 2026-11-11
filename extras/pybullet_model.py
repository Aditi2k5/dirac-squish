"""Third-party library baseline #2: PyBullet deformable world (Neo-Hookean tetrahedral FEM, Bullet's own solver and collision), on the SAME mesh,
driven by the measured book height (a kinematic box moved every step). No physics code of ours.  SI units (m, kg, s), z up.
rho = 100 kg/m^3, E = 8*100 Pa (E/rho = 8 m^2/s^2).  Output format as mesh_models.py -> outputs/mesh_pybullet_h4.5.npz"""
import argparse, time, tempfile, os
import numpy as np
import pybullet as p
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=float, default=4.5); ap.add_argument('--e_rho', type=float, default=8.0); ap.add_argument('--nu', type=float, default=0.0)
ap.add_argument('--alpha', type=float, default=100.0, help='mapped to Bullet NeoHookeanDamping (see note in README: not identical to Rayleigh)')
ap.add_argument('--mu', type=float, default=0.5); ap.add_argument('--t0', type=float, default=4.2); ap.add_argument('--t1', type=float, default=8.8); ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--dt', type=float, default=5e-4); ap.add_argument('--out', default='outputs/mesh_pybullet_h4.5.npz')
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
P0, tet = build(a.h, R); n, nt = len(P0), len(tet)
Pm = np.stack([P0[:, 0], -P0[:, 2], P0[:, 1]], 1) / 1000.0                  # proper rotation (y up -> z up), metres
vtk = os.path.join(tempfile.mkdtemp(), 'ball.vtk')
with open(vtk, 'w') as fh:
    fh.write('# vtk DataFile Version 2.0\nball\nASCII\nDATASET UNSTRUCTURED_GRID\n')
    fh.write(f'POINTS {n} float\n'); [fh.write(f'{x:.6f} {y:.6f} {z:.6f}\n') for x, y, z in Pm]
    fh.write(f'CELLS {nt} {5 * nt}\n'); [fh.write('4 %d %d %d %d\n' % tuple(t)) for t in tet]
    fh.write(f'CELL_TYPES {nt}\n'); fh.write('10\n' * nt)
rho = 100.0; E = a.e_rho * rho; mu_l = E / (2 * (1 + a.nu)); lam_l = E * a.nu / ((1 + a.nu) * (1 - 2 * a.nu))
p.connect(p.DIRECT); p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD); p.setGravity(0, 0, 0); p.setTimeStep(a.dt)
pl = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE), basePosition=[0, 0, 0])
p.changeDynamics(pl, -1, lateralFriction=a.mu)
book = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.075, 0.075, 0.007]), basePosition=[0, 0, 0.3]); p.changeDynamics(book, -1, lateralFriction=a.mu)
soft = p.loadSoftBody(vtk, basePosition=[0, 0, R / 1000 + 0.0002], scale=1.0, mass=rho * 4 / 3 * np.pi * (R / 1000) ** 3, useNeoHookean=1,
                      NeoHookeanMu=mu_l, NeoHookeanLambda=max(lam_l, 1.0), NeoHookeanDamping=a.alpha * 0.001, useSelfCollision=0, frictionCoeff=a.mu, collisionMargin=0.0006)
ok = np.isfinite(plate); tv, ps = t_vid[ok], savgol_filter(plate[ok], 11, 2); vend = np.polyfit(tv[-8:], ps[-8:], 1)[0]
def hp(tt): return float(ps[0]) if tt <= tv[0] else (float(ps[-1] + vend * (tt - tv[-1])) if tt >= tv[-1] else float(np.interp(tt, tv, ps)))
fr = np.arange(np.searchsorted(t_vid, a.t0), np.searchsorted(t_vid, a.t1), a.stride); tg = t_vid[fr]
def verts(): return np.array(p.getMeshData(soft)[1])
X0 = verts(); print('soft body loaded:', len(X0), 'vertices (mesh has', n, ')', flush=True)
def vols(X):
    dd = np.stack([X[tet[:, 1]] - X[tet[:, 0]], X[tet[:, 2]] - X[tet[:, 0]], X[tet[:, 3]] - X[tet[:, 0]]], 2); return np.linalg.det(dd) / 6
res = dict(height=[], width=[], J=[], inv=[]); snaps = []; tc = tg[0]; t0w = time.time()
for k, tk in enumerate(tg):
    nsub = max(1, int(round((tk - tc) / a.dt))) if k else 0
    for s_ in range(nsub):
        tm = tc + (s_ + 1) * (tk - tc) / nsub; p.resetBasePositionAndOrientation(book, [0, 0, hp(tm) / 1000 + 0.007], [0, 0, 0, 1]); p.stepSimulation()
    tc = tk; X = verts()
    if len(X) == n:
        dv = vols(X); res['J'].append(dv.sum() / max(vols(Pm[:len(X)]).sum(), 1e-12)); res['inv'].append(int((dv < 0).sum()))
    else: res['J'].append(np.nan); res['inv'].append(0)
    res['height'].append(X[:, 2].max() * 1000); res['width'].append(np.ptp(X[:, 0]) * 1000)
    snaps.append(np.stack([X[:, 0], X[:, 2], -X[:, 1]], 1) * 1000)
    if not np.isfinite(res['height'][-1]): print('NaN, stopped'); break
    if k % 5 == 0: print(f't={tk:5.2f}  H {res["height"][-1]:5.1f}  W {res["width"][-1]:5.1f}  J {res["J"][-1]:.2f}  inverted {res["inv"][-1]}  ({time.time()-t0w:.0f}s)', flush=True)
nk = len(res['height']); S = np.stack(snaps); S[..., 0] += 64; S[..., 2] += 64; S[..., 1] += 9
np.savez_compressed(a.out, t=tg[:nk], width=res['width'], height=res['height'], J=res['J'], inverted=res['inv'], minratio=np.zeros(nk), w0=D,
                    snaps=S.astype(np.float16), y_floor=9.0, cx=64.0, model='pybullet', wall_s=time.time() - t0w, params=np.array([a.e_rho, a.nu, a.alpha, a.mu, a.h, n, nt]))
print('wrote', a.out)