"""Mesh-based 3D soft-body models of the ball, all on ONE shared tetrahedral mesh, driven by the measured book height.

  --model fem   explicit corotated FEM   (same fixed-corotated material as mpm3d.py, discretised with elements instead of particles+grid)
  --model ms    mass-spring network      (PhysTwin / Spring-Gaus style; spring stiffness energy-matched to the continuum, see calibrate_springs)
  --model xpbd  XPBD (extended position-based dynamics), fixed-corotated deviatoric constraint per tet (+ volume constraint if nu > 0),
                graph-coloured Gauss-Seidel, small steps (1 iteration per substep)

Everything else is identical to the MPM run: ball diameter from the tape measurement, table at y=0, rigid book plane at the measured height,
Coulomb friction, mass-proportional Rayleigh damping alpha, rho = 1 reference density (only E/rho matters), gravity off.
Output: observables per video frame (height, width, volume ratio J, inverted-tet count) + node snapshots in the same file format as sim3d_*.npz,
so render_3d.py works on it.  One model per process (Taichi fields are created once):  python mesh_models.py --model fem
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import argparse, time
import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import savgol_filter

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['fem', 'ms', 'xpbd'], required=True)
ap.add_argument('--h', type=float, default=4.5, help='mesh spacing (mm)')
ap.add_argument('--e_rho', type=float, default=8.0, help='E/rho (m^2/s^2), same reference as MPM')
ap.add_argument('--nu', type=float, default=0.0)
ap.add_argument('--alpha', type=float, default=100.0, help='Rayleigh damping (1/s)')
ap.add_argument('--mu', type=float, default=0.5, help='Coulomb friction')
ap.add_argument('--t0', type=float, default=4.2); ap.add_argument('--t1', type=float, default=8.8); ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--dt', type=float, default=2.5e-4)
ap.add_argument('--out', default=None)
a = ap.parse_args()
if a.out is None: a.out = f'outputs/models/mesh_{a.model}_h{a.h:g}.npz'

T = np.load('outputs/clip1/targets.npz'); t_vid, plate, D = T['t'], T['plate_mm'], float(T['D_mm']); R = D / 2


# ---------------------------------------------------------------- mesh
def build(h, R, seed=0):
    g = np.arange(-R, R + 1e-9, h); X = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    rng = np.random.default_rng(seed)
    inner = X[np.linalg.norm(X, axis=1) < R - 0.55 * h]; inner = inner + rng.normal(0, 0.03 * h, inner.shape)   # jitter: avoids degenerate lattice Delaunay
    n = int(4 * np.pi * R * R / (0.85 * h * h)); i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    S = R * np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    P = np.vstack([inner, S]); tet = Delaunay(P).simplices
    Dm = np.stack([P[tet[:, 1]] - P[tet[:, 0]], P[tet[:, 2]] - P[tet[:, 0]], P[tet[:, 3]] - P[tet[:, 0]]], 2)   # columns = edge vectors
    det = np.linalg.det(Dm); sw = det < 0; tet[sw] = tet[sw][:, [0, 2, 1, 3]]          # make all tets positively oriented
    Dm = np.stack([P[tet[:, 1]] - P[tet[:, 0]], P[tet[:, 2]] - P[tet[:, 0]], P[tet[:, 3]] - P[tet[:, 0]]], 2); vol = np.linalg.det(Dm) / 6
    keep = vol > 0.05 * np.median(vol); tet = tet[keep]                              # drop sliver tets (flat tets on the hull)
    used = np.unique(tet); remap = -np.ones(len(P), int); remap[used] = np.arange(len(used)); P = P[used]; tet = remap[tet]
    return P, tet


P0, tet = build(a.h, R)
n, nt = len(P0), len(tet)
Dm = np.stack([P0[tet[:, 1]] - P0[tet[:, 0]], P0[tet[:, 2]] - P0[tet[:, 0]], P0[tet[:, 3]] - P0[tet[:, 0]]], 2)
V0 = np.linalg.det(Dm) / 6; Dm_inv = np.linalg.inv(Dm)
mass = np.zeros(n)
for k in range(4): np.add.at(mass, tet[:, k], V0 / 4)                              # rho = 1 reference density
pairs = np.sort(np.concatenate([tet[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]), 1)
edges, inv = np.unique(pairs, axis=0, return_inverse=True)
l0 = np.linalg.norm(P0[edges[:, 0]] - P0[edges[:, 1]], axis=1)
w_e = np.zeros(len(edges)); np.add.at(w_e, inv, np.tile(V0, 6))                      # volume of all tets sharing each edge

E = a.e_rho * 1e6                                                                    # (mm/s)^2 * rho(=1)
mu_l = E / (2 * (1 + a.nu)); lam_l = E * a.nu / ((1 + a.nu) * (1 - 2 * a.nu))


def calibrate_springs(strain=0.03):
    """Mass-spring stiffness k_e = c E w_e / l0^2; c chosen so the network's elastic energy equals the continuum (fixed-corotated)
    energy for a 3 % uniaxial affine squash of this very mesh. A mass-spring net has a fixed Poisson ratio (~0.25 for tet meshes), so only
    the stiffness, not nu, can be matched."""
    s = 1 - strain; Fm = np.diag([1, s, 1.0])
    U_fem = np.sum(V0) * (mu_l * (s - 1) ** 2 + 0.5 * lam_l * (s - 1) ** 2)
    d = P0[edges[:, 1]] - P0[edges[:, 0]]; l = np.linalg.norm(d @ Fm.T, axis=1)
    U1 = np.sum(0.5 * (E * w_e / l0 ** 2) * (l - l0) ** 2)
    return U_fem / U1


c_spring = calibrate_springs() if a.model == 'ms' else 0.0
k_e = c_spring * E * w_e / l0 ** 2

colors = None
if a.model == 'xpbd':                                                                # greedy graph colouring: no two tets of one colour share a node
    ncol = [set() for _ in range(n)]; col = np.empty(nt, int)
    for t_ in range(nt):
        used = set().union(*(ncol[i] for i in tet[t_])); c = 0
        while c in used: c += 1
        col[t_] = c
        for i in tet[t_]: ncol[i].add(c)
    order = np.argsort(col, kind='stable'); tet, Dm_inv, V0 = tet[order], Dm_inv[order], V0[order]; col = col[order]
    ptr = np.searchsorted(col, np.arange(col.max() + 2)); colors = [(int(ptr[c]), int(ptr[c + 1])) for c in range(col.max() + 1)]
print(f'{a.model}: {n} nodes, {nt} tets, {len(edges)} edges, h={a.h} mm, mesh volume {V0.sum() / (4 / 3 * np.pi * R ** 3):.3f} of sphere'
      + (f', spring scale c={c_spring:.3f}' if a.model == 'ms' else '') + (f', {len(colors)} colours' if colors else ''), flush=True)

# ---------------------------------------------------------------- taichi
import taichi as ti
import backend
backend.init()
x = ti.Vector.field(3, float, n); v = ti.Vector.field(3, float, n); f = ti.Vector.field(3, float, n); xold = ti.Vector.field(3, float, n)
m = ti.field(float, n); pen = ti.field(float, n)
tet_f = ti.Vector.field(4, int, nt); Dminv_f = ti.Matrix.field(3, 3, float, nt); V0_f = ti.field(float, nt)
ed_f = ti.Vector.field(2, int, len(edges)); l0_f = ti.field(float, len(edges)); k_f = ti.field(float, len(edges))
x.from_numpy(P0 + np.array([0, R, 0])); v.fill(0); m.from_numpy(mass)
tet_f.from_numpy(tet.astype(np.int32)); Dminv_f.from_numpy(Dm_inv); V0_f.from_numpy(V0)
ed_f.from_numpy(edges.astype(np.int32)); l0_f.from_numpy(l0); k_f.from_numpy(k_e)


@ti.func
def col(M, j):
    return ti.Vector([M[0, j], M[1, j], M[2, j]])


@ti.func
def defgrad(t):
    i0, i1, i2, i3 = tet_f[t][0], tet_f[t][1], tet_f[t][2], tet_f[t][3]
    x0 = x[i0]
    Ds = ti.Matrix.cols([x[i1] - x0, x[i2] - x0, x[i3] - x0])
    return Ds @ Dminv_f[t]


@ti.func
def cofactor(F):
    c0, c1, c2 = col(F, 0), col(F, 1), col(F, 2)
    return ti.Matrix.cols([c1.cross(c2), c2.cross(c0), c0.cross(c1)])


@ti.kernel
def clear_f():
    for i in f: f[i] = ti.Vector.zero(float, 3)


@ti.kernel
def fem_forces(mu: float, lam: float):
    for t in range(nt):
        F = defgrad(t)
        U, sig, Vv = ti.svd(F); Rr = U @ Vv.transpose()
        J = F.determinant()
        Pk = 2 * mu * (F - Rr) + lam * (J - 1) * cofactor(F)                          # fixed-corotated first Piola-Kirchhoff stress
        H = -V0_f[t] * Pk @ Dminv_f[t].transpose()
        f1, f2, f3 = col(H, 0), col(H, 1), col(H, 2)
        ti.atomic_add(f[tet_f[t][1]], f1); ti.atomic_add(f[tet_f[t][2]], f2); ti.atomic_add(f[tet_f[t][3]], f3)
        ti.atomic_add(f[tet_f[t][0]], -(f1 + f2 + f3))


@ti.kernel
def spring_forces():
    for e in range(ed_f.shape[0]):
        i, j = ed_f[e][0], ed_f[e][1]
        d = x[j] - x[i]; l = d.norm() + 1e-12
        fs = k_f[e] * (l - l0_f[e]) * d / l
        ti.atomic_add(f[i], fs); ti.atomic_add(f[j], -fs)


@ti.func
def fric(vi, dvn, mu_f):
    s = ti.sqrt(vi[0] ** 2 + vi[2] ** 2)
    if s > 1e-9:
        k = ti.max(0.0, 1.0 - mu_f * dvn / s)
        vi[0] *= k; vi[2] *= k
    return vi


@ti.kernel
def integrate(dt: float, alpha: float, ph: float, pv: float, mu_f: float):
    for i in x:
        vi = (v[i] + dt * f[i] / m[i]) * ti.exp(-alpha * dt)
        xi = x[i] + dt * vi
        if xi[1] < 0.0:                                                               # table
            xi[1] = 0.0; dvn = ti.max(0.0, -vi[1]); vi[1] = ti.max(vi[1], 0.0); vi = fric(vi, dvn, mu_f)
        if xi[1] > ph:                                                                # rigid book (can only push)
            xi[1] = ph; dvn = ti.max(0.0, vi[1] - pv); vi[1] = ti.min(vi[1], pv); vi = fric(vi, dvn, mu_f)
        x[i] = xi; v[i] = vi


@ti.kernel
def xp_predict(dt: float, alpha: float):
    for i in x:
        xold[i] = x[i]; v[i] = v[i] * ti.exp(-alpha * dt); x[i] = x[i] + dt * v[i]; pen[i] = 0.0


@ti.kernel
def xp_project(s: int, e: int, dt: float, mu: float, lam: float):
    for t in range(s, e):                                                             # one colour: no shared nodes, safe in parallel
        F = defgrad(t)
        U, sig, Vv = ti.svd(F); Rr = U @ Vv.transpose()
        Cd = (F - Rr).norm()
        i0, i1, i2, i3 = tet_f[t][0], tet_f[t][1], tet_f[t][2], tet_f[t][3]
        if Cd > 1e-9:
            G = (F - Rr) / Cd @ Dminv_f[t].transpose()
            g1, g2, g3 = col(G, 0), col(G, 1), col(G, 2); g0 = -(g1 + g2 + g3)
            w = g0.dot(g0) / m[i0] + g1.dot(g1) / m[i1] + g2.dot(g2) / m[i2] + g3.dot(g3) / m[i3]
            at = 1.0 / (2 * mu * V0_f[t]) / (dt * dt)                                 # energy mu*V0*C^2  ->  compliance 1/(2 mu V0)
            dl = -Cd / (w + at)
            x[i0] += dl * g0 / m[i0]; x[i1] += dl * g1 / m[i1]; x[i2] += dl * g2 / m[i2]; x[i3] += dl * g3 / m[i3]
        if lam > 1e-9:
            F = defgrad(t); Ch = F.determinant() - 1.0; Cf = cofactor(F)
            G = Cf @ Dminv_f[t].transpose()
            g1, g2, g3 = col(G, 0), col(G, 1), col(G, 2); g0 = -(g1 + g2 + g3)
            w = g0.dot(g0) / m[i0] + g1.dot(g1) / m[i1] + g2.dot(g2) / m[i2] + g3.dot(g3) / m[i3]
            at = 1.0 / (lam * V0_f[t]) / (dt * dt)
            dl = -Ch / (w + at)
            x[i0] += dl * g0 / m[i0]; x[i1] += dl * g1 / m[i1]; x[i2] += dl * g2 / m[i2]; x[i3] += dl * g3 / m[i3]


@ti.kernel
def xp_finish(dt: float, ph: float, pv: float, mu_f: float):
    for i in x:
        xi = x[i]; p = 0.0
        if xi[1] < 0.0: p = -xi[1]; xi[1] = 0.0
        if xi[1] > ph: p = xi[1] - ph; xi[1] = ph
        x[i] = xi
        vi = (xi - xold[i]) / dt
        if p > 0.0: vi = fric(vi, p / dt, mu_f)
        v[i] = vi


# ---------------------------------------------------------------- measured book path (same as run_3d.py)
ok = np.isfinite(plate); tv, ps = t_vid[ok], savgol_filter(plate[ok], 11, 2); vend = np.polyfit(tv[-8:], ps[-8:], 1)[0]
def hp(tt):
    if tt <= tv[0]: return float(ps[0])
    if tt >= tv[-1]: return float(ps[-1] + vend * (tt - tv[-1]))
    return float(np.interp(tt, tv, ps))
def vp(tt): return (hp(tt + 0.02) - hp(tt - 0.02)) / 0.04


def observables():
    X = x.to_numpy(); d = np.stack([X[tet[:, 1]] - X[tet[:, 0]], X[tet[:, 2]] - X[tet[:, 0]], X[tet[:, 3]] - X[tet[:, 0]]], 2)
    dv = np.linalg.det(d) / 6
    return dict(height=X[:, 1].max(), width=np.ptp(X[:, 0]), J=dv.sum() / V0.sum(), inverted=int((dv < 0).sum()), minratio=float(dv.min() / np.median(V0)), X=X)


fr = np.arange(np.searchsorted(t_vid, a.t0), np.searchsorted(t_vid, a.t1), a.stride); tg = t_vid[fr]
res = {k_: [] for k_ in ('height', 'width', 'J', 'inverted', 'minratio')}; snaps = []
tc = tg[0]; t_wall = time.time(); mu_f = float(a.mu)
for k, tk in enumerate(tg):
    nsub = max(1, int(np.ceil((tk - tc) / a.dt))) if k else 0
    for s_ in range(nsub):
        dt = (tk - tc) / nsub; tm = tc + (s_ + 1) * dt; ph, pv = hp(tm), vp(tm)
        if a.model == 'xpbd':
            xp_predict(dt, a.alpha)
            for (c0, c1) in colors: xp_project(c0, c1, dt, mu_l, lam_l)
            xp_finish(dt, ph, pv, mu_f)
        else:
            clear_f()
            fem_forces(mu_l, lam_l) if a.model == 'fem' else spring_forces()
            integrate(dt, a.alpha, ph, pv, mu_f)
    tc = tk
    o = observables(); [res[q].append(o[q]) for q in res]; snaps.append(o['X'])
    if not np.isfinite(o['height']): print('NaN - model blew up at t=%.2f' % tk); break
    if k % 5 == 0: print(f't={tk:5.2f}s  H {o["height"]:5.1f}  W {o["width"]:5.1f}  J {o["J"]:.2f}  inverted tets {o["inverted"]}  min vol ratio {o["minratio"]:.2f}  ({time.time() - t_wall:.0f}s)', flush=True)
nk = len(res['height'])
S = np.stack(snaps).astype(np.float16)
S[..., 0] += 64; S[..., 2] += 64; S[..., 1] += 9                                      # same world convention as sim3d_*.npz (table y=9, centre x=z=64)
np.savez_compressed(a.out, t=tg[:nk], width=res['width'], height=res['height'], J=res['J'], inverted=res['inverted'], minratio=res['minratio'],
                    w0=D, snaps=S, y_floor=9.0, cx=64.0, model=a.model, wall_s=time.time() - t_wall,
                    params=np.array([a.e_rho, a.nu, a.alpha, a.mu, a.h, n, nt]))
print('wrote', a.out, f'({time.time() - t_wall:.0f}s)')