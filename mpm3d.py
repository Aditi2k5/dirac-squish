"""3D MLS-MPM (APIC) foam sphere pressed between a table and a rigid plate. Same conventions as mpm.py
(mm, s, rho = 1 reference density, e_rho in m^2/s^2). Fixed-corotated elasticity, Coulomb friction, Rayleigh alpha."""
import os
import numpy as np
import taichi as ti

import backend
backend.init()
DX = float(os.environ.get('MPM_DX3', '3.0'))
L = 128.0
NG = int(round(L / DX))
Y_FLOOR = 3 * DX
CX = CZ = L / 2
GRAV = 9810.0
J_MIN = 0.03
NP_MAX = 120000

x = ti.Vector.field(3, float); v = ti.Vector.field(3, float)
C = ti.Matrix.field(3, 3, float); F = ti.Matrix.field(3, 3, float)
ti.root.dense(ti.i, NP_MAX).place(x, v, C, F)
grid_v = ti.Vector.field(3, float, (NG, NG, NG)); grid_m = ti.field(float, (NG, NG, NG))
grav_f = ti.field(float, ())
mu_f = ti.field(float, ()); lam_f = ti.field(float, ()); alpha_f = ti.field(float, ())
fric_p = ti.field(float, ()); fric_t = ti.field(float, ())
proj_y = ti.field(float, ()); proj_on = ti.field(int, ()); proj_k = ti.field(float, ()); plate_y = ti.field(float, ()); plate_s = ti.field(float, ()); plate_v = ti.field(float, ()); dt_f = ti.field(float, ())
n_p = ti.field(int, ()); p_vol_f = ti.field(float, ()); p_mass_f = ti.field(float, ())
obs = ti.field(float, 6)


@ti.kernel
def clear():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(float, 3); grid_m[I] = 0.0


@ti.kernel
def p2g():
    dt = dt_f[None]
    for p in range(n_p[None]):
        base = (x[p] / DX - 0.5).cast(int); fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Fp = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        J = Fp.determinant()
        if J < J_MIN:
            Fp = Fp * (J_MIN / ti.max(J, 1e-6)) ** (1.0 / 3.0); J = Fp.determinant()
        F[p] = Fp
        R, S = ti.polar_decompose(Fp)
        tau = 2 * mu_f[None] * (Fp - R) @ Fp.transpose() + lam_f[None] * J * (J - 1) * ti.Matrix.identity(float, 3)
        stress = (-dt * p_vol_f[None] * 4 / (DX * DX)) * tau
        affine = stress + p_mass_f[None] * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k]); dpos = (offs.cast(float) - fx) * DX
            wt = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offs] += wt * (p_mass_f[None] * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass_f[None]


@ti.kernel
def grid_op():
    dt = dt_f[None]
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            vv = grid_v[i, j, k] / grid_m[i, j, k]
            vv *= ti.exp(-alpha_f[None] * dt)
            vv[1] -= dt * grav_f[None]
            py = j * DX
            if py <= Y_FLOOR and vv[1] < 0:
                vn = vv[1]; vv[1] = 0.0
                vt = ti.Vector([vv[0], vv[2]]); sp = vt.norm()
                if sp > 1e-9:
                    vt = vt * ti.max(sp + fric_t[None] * vn, 0.0) / sp
                vv[0] = vt[0]; vv[2] = vt[1]
            if py >= plate_y[None] + plate_s[None] * (i * DX - CX) and vv[1] > plate_v[None]:
                dvn = vv[1] - plate_v[None]; vv[1] = plate_v[None]
                vt = ti.Vector([vv[0], vv[2]]); sp = vt.norm()
                if sp > 1e-9:
                    vt = vt * ti.max(sp - fric_p[None] * dvn, 0.0) / sp
                vv[0] = vt[0]; vv[2] = vt[1]
            if i < 2 and vv[0] < 0: vv[0] = 0.0
            if i > NG - 3 and vv[0] > 0: vv[0] = 0.0
            if k < 2 and vv[2] < 0: vv[2] = 0.0
            if k > NG - 3 and vv[2] > 0: vv[2] = 0.0
            grid_v[i, j, k] = vv


@ti.kernel
def g2p():
    dt = dt_f[None]
    for p in range(n_p[None]):
        base = (x[p] / DX - 0.5).cast(int); fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        nv = ti.Vector.zero(float, 3); nC = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k]); dpos = (offs.cast(float) - fx) * DX
            wt = w[i][0] * w[j][1] * w[k][2]; gv = grid_v[base + offs]
            nv += wt * gv; nC += 4 * wt * gv.outer_product(dpos) / (DX * DX)
        v[p] = nv; C[p] = nC; x[p] += dt * nv
        if proj_on[None] == 1:                                  # optional particle-level contact: no particle centre may sit above the book surface
            ysurf = proj_y[None] + plate_s[None] * (x[p][0] - CX)
            if x[p][1] > ysurf:
                d = x[p][1] - ysurf
                x[p][1] = ysurf
                # keep F consistent with the position jump: treat it as a vertical compression of d over one particle spacing (proj_k = 0 -> old behaviour)
                F[p] = (ti.Matrix.identity(float, 3) - ti.Matrix([[0.0, 0.0, 0.0], [0.0, ti.min(proj_k[None] * d / DX, 0.3), 0.0], [0.0, 0.0, 0.0]])) @ F[p]
                if v[p][1] > plate_v[None]: v[p][1] = plate_v[None]


@ti.kernel
def observe():
    for k in range(6): obs[k] = 0.0
    obs[1] = 1e9
    for p in range(n_p[None]):
        r = ti.sqrt((x[p][0] - CX) ** 2 + (x[p][2] - CZ) ** 2)
        ti.atomic_max(obs[0], r); ti.atomic_min(obs[1], x[p][1]); ti.atomic_max(obs[2], x[p][1])
        obs[3] += F[p].determinant() / ti.cast(n_p[None], float)


def init_ball(diam_mm):
    r = diam_mm / 2; sp = DX / 2
    g = np.arange(-r, r, sp) + sp / 2
    A, B, Cc = np.meshgrid(g, g, g, indexing='ij')
    m = A ** 2 + B ** 2 + Cc ** 2 <= r * r
    pts = np.stack([CX + A[m], Y_FLOOR + r + B[m], CZ + Cc[m]], 1).astype(np.float32)
    n = len(pts); assert n < NP_MAX, n
    n_p[None] = n
    xs = np.zeros((NP_MAX, 3), np.float32); xs[:n] = pts; x.from_numpy(xs)
    v.fill(0); C.fill(0)
    F.from_numpy(np.tile(np.eye(3, dtype=np.float32), (NP_MAX, 1, 1)))
    p_vol_f[None] = sp ** 3; p_mass_f[None] = sp ** 3
    return n


def simulate(params, plate_fn, plate_vel_fn, t_grid, diam_mm, cfl=0.25, max_dt=2.5e-4, snap=None, log=None, slope_fn=None, proj_fn=None, proj_strain=0.0):
    E = params['e_rho'] * 1e6; nu = params['nu']
    mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_f[None] = mu; lam_f[None] = lam; alpha_f[None] = params.get('alpha', 0.0); grav_f[None] = params.get('gravity', GRAV)
    fric_p[None] = params['mu_p']; fric_t[None] = params['mu_t']
    dt = min(cfl * DX / np.sqrt(lam + 2 * mu), max_dt)
    n = init_ball(diam_mm); out = []; t = t_grid[0]
    settle_s = params.get('settle_s', 0.0); settle_alpha = params.get('settle_alpha', 40.0); alpha_run = params.get('alpha', 0.0); t_start = t_grid[0]
    proj_on[None] = 1 if proj_fn else 0; proj_k[None] = proj_strain
    observe(); w0 = 2 * obs[0]
    for tk in t_grid:
        while t < tk - 1e-12:
            h = min(dt, tk - t); dt_f[None] = h
            alpha_f[None] = settle_alpha if (t - t_start) < settle_s else alpha_run
            plate_y[None] = Y_FLOOR + plate_fn(t + 0.5 * h); plate_v[None] = plate_vel_fn(t + 0.5 * h); plate_s[None] = slope_fn(t + 0.5 * h) if slope_fn else 0.0
            if proj_fn: proj_y[None] = Y_FLOOR + proj_fn(t + 0.5 * h)
            clear(); p2g(); grid_op(); g2p(); t += h
        observe(); out.append([2 * obs[0], obs[2] - Y_FLOOR, obs[3], obs[1] - Y_FLOOR])
        if snap is not None: snap.append(x.to_numpy()[:n].copy())
        if log: log(tk, out[-1])
    out = np.array(out)
    return dict(width=out[:, 0], height=out[:, 1], J=out[:, 2], ybot=out[:, 3], w0=w0, n=n, dt=dt)