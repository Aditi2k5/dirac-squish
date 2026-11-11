"""2D plane-strain MLS-MPM (APIC) foam disc squeezed between a table and a moving rigid plate.

Units: mm, s, and a *reference density of 1* (mass per mm^2). Only E/rho is identifiable from video, so the
material is parameterised by:
    e_rho   : Young's modulus / density, in m^2/s^2   (E = rho * e_rho; with rho=100 kg/m^3, 1 m^2/s^2 = 0.1 kPa)
    nu      : Poisson ratio (foam: ~0)
    alpha   : mass-proportional (Rayleigh) damping rate, 1/s  (explicit Kelvin-Voigt is unusably stiff here; tau_d kept optional)
    mu_p    : Coulomb friction with the plate
    mu_t    : Coulomb friction with the table
Model: fixed-corotated elasticity + linear viscous stress 2*eta*sym(grad v).
"""
import os
import numpy as np
import taichi as ti

ti.init(arch=(ti.gpu if os.environ.get('MPM_ARCH') == 'gpu' else ti.cpu), default_fp=ti.f32, cpu_max_num_threads=2, offline_cache=True)

import os
DX = float(os.environ.get('MPM_DX', '2.0'))   # grid spacing (mm)
NG = int(round(128.0 / DX))                    # grid is NG x NG (128 mm domain)
PPC_D = 2                # particles per cell per dim
Y_FLOOR = 3 * DX         # table plane (mm); grid nodes at/below are the floor
X0 = NG * DX / 2         # ball centre x (mm)
GRAV = 9810.0            # mm/s^2
J_MIN = 0.03

x = ti.Vector.field(2, float); v = ti.Vector.field(2, float)
C = ti.Matrix.field(2, 2, float); F = ti.Matrix.field(2, 2, float)
grid_v = ti.Vector.field(2, float, (NG, NG)); grid_m = ti.field(float, (NG, NG))
mu_f = ti.field(float, ()); lam_f = ti.field(float, ()); eta_f = ti.field(float, ())
fric_p = ti.field(float, ()); fric_t = ti.field(float, ()); alpha_f = ti.field(float, ())
model_f = ti.field(int, ()); ey_f = ti.field(float, ()); hard_f = ti.field(float, ())
plate_y = ti.field(float, ()); plate_v = ti.field(float, ()); dt_f = ti.field(float, ())
obs = ti.field(float, 6)  # xmin, xmax, ymax, meanJ, ymin, spare
NP_MAX = 80000
ti.root.dense(ti.i, NP_MAX).place(x, v, C, F)
n_p = ti.field(int, ())
p_vol_f = ti.field(float, ()); p_mass_f = ti.field(float, ())


@ti.kernel
def clear():
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]; grid_m[i, j] = 0.0


@ti.kernel
def p2g():
    dt = dt_f[None]
    for p in range(n_p[None]):
        base = (x[p] / DX - 0.5).cast(int); fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Fp = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        J = Fp.determinant()
        if J < J_MIN:                      # safeguard against inversion under extreme compression
            Fp = Fp * ti.sqrt(J_MIN / ti.max(J, 1e-6)); J = Fp.determinant()
        F[p] = Fp
        tau = ti.Matrix.zero(float, 2, 2)
        if model_f[None] == 0:                       # fixed-corotated
            R, S = ti.polar_decompose(Fp)
            tau = 2 * mu_f[None] * (Fp - R) @ Fp.transpose() + lam_f[None] * J * (J - 1) * ti.Matrix.identity(float, 2)
        else:                                        # Valanis-Landel / generalised-Hencky foam with compressive plateau
            U, Sg, Vt = ti.svd(Fp)
            e0 = ti.log(ti.max(Sg[0, 0], 1e-3)); e1 = ti.log(ti.max(Sg[1, 1], 1e-3))
            g0 = e0; g1 = e1
            if e0 < -ey_f[None]: g0 = -ey_f[None] + hard_f[None] * (e0 + ey_f[None])
            if e1 < -ey_f[None]: g1 = -ey_f[None] + hard_f[None] * (e1 + ey_f[None])
            t0 = 2 * mu_f[None] * g0 + lam_f[None] * (g0 + g1)
            t1 = 2 * mu_f[None] * g1 + lam_f[None] * (g0 + g1)
            tau = U @ ti.Matrix([[t0, 0.0], [0.0, t1]]) @ U.transpose()
        D = 0.5 * (C[p] + C[p].transpose())
        tau += J * 2 * eta_f[None] * D
        stress = (-dt * p_vol_f[None] * 4 / (DX * DX)) * tau
        affine = stress + p_mass_f[None] * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j]); dpos = (offs.cast(float) - fx) * DX
            wt = w[i][0] * w[j][1]
            grid_v[base + offs] += wt * (p_mass_f[None] * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass_f[None]


@ti.kernel
def grid_op():
    dt = dt_f[None]
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            vv = grid_v[i, j] / grid_m[i, j]
            vv *= ti.exp(-alpha_f[None] * dt)      # mass-proportional (Rayleigh alpha) damping
            vv[1] -= dt * GRAV
            py = j * DX
            if py <= Y_FLOOR and vv[1] < 0:           # table (unilateral, Coulomb)
                vn = vv[1]; vv[1] = 0.0
                s = 1.0 if vv[0] > 0 else -1.0
                vv[0] = s * ti.max(ti.abs(vv[0]) + fric_t[None] * vn, 0.0)
            if py >= plate_y[None] and vv[1] > plate_v[None]:   # plate (unilateral, Coulomb)
                dvn = vv[1] - plate_v[None]; vv[1] = plate_v[None]
                s = 1.0 if vv[0] > 0 else -1.0
                vv[0] = s * ti.max(ti.abs(vv[0]) - fric_p[None] * dvn, 0.0)
            if i < 2 and vv[0] < 0: vv[0] = 0.0
            if i > NG - 3 and vv[0] > 0: vv[0] = 0.0
            grid_v[i, j] = vv


@ti.kernel
def g2p():
    dt = dt_f[None]
    for p in range(n_p[None]):
        base = (x[p] / DX - 0.5).cast(int); fx = x[p] / DX - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        nv = ti.Vector.zero(float, 2); nC = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j]); dpos = (offs.cast(float) - fx) * DX
            wt = w[i][0] * w[j][1]; gv = grid_v[base + offs]
            nv += wt * gv; nC += 4 * wt * gv.outer_product(dpos) / (DX * DX)
        v[p] = nv; C[p] = nC; x[p] += dt * nv


@ti.kernel
def observe():
    for k in range(6): obs[k] = 0.0
    obs[0] = 1e9; obs[4] = 1e9
    for p in range(n_p[None]):
        ti.atomic_min(obs[0], x[p][0]); ti.atomic_max(obs[1], x[p][0])
        ti.atomic_max(obs[2], x[p][1]); ti.atomic_min(obs[4], x[p][1])
        obs[3] += F[p].determinant() / ti.cast(n_p[None], float)


def init_ball(diam_mm):
    r = diam_mm / 2; sp = DX / PPC_D
    pts = []
    for a in np.arange(-r, r, sp) + sp / 2:
        for b in np.arange(-r, r, sp) + sp / 2:
            if a * a + b * b <= r * r: pts.append((X0 + a, Y_FLOOR + r + b))
    pts = np.array(pts, np.float32); n = len(pts); assert n < NP_MAX
    n_p[None] = n
    xs = np.zeros((NP_MAX, 2), np.float32); xs[:n] = pts; x.from_numpy(xs)
    v.fill(0); C.fill(0)
    Fi = np.tile(np.eye(2, dtype=np.float32), (NP_MAX, 1, 1)); F.from_numpy(Fi)
    area = n * sp * sp
    p_vol_f[None] = sp * sp; p_mass_f[None] = sp * sp          # rho = 1
    return n, area


def simulate(params, plate_fn, plate_vel_fn, t_grid, diam_mm, cfl=0.25, max_dt=2e-4):
    """params: dict(e_rho [m^2/s^2], nu, tau_d [s], mu_p, mu_t). plate_fn(t)->plate surface height above table (mm).
    Returns dict of arrays sampled at t_grid (rows: width_mm, height_mm, J_mean, y_bottom_mm)."""
    E = params['e_rho'] * 1e6                      # mm^2/s^2 (rho = 1)
    nu = params['nu']
    mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_f[None] = mu; lam_f[None] = lam; eta_f[None] = params.get('tau_d', 0.0) * E; model_f[None] = params.get('model', 0); ey_f[None] = params.get('ey', 0.2); hard_f[None] = params.get('hard', 0.1); alpha_f[None] = params.get('alpha', 0.0)
    fric_p[None] = params['mu_p']; fric_t[None] = params['mu_t']
    c = np.sqrt((lam + 2 * mu))
    dt = min(cfl * DX / c, max_dt)
    # explicit-viscosity stability: dt < ~ DX^2 / (8 * eta)
    dt = min(dt, 0.1 * DX * DX / max(params.get('tau_d', 0.0) * E, 1e-9))
    dt_f[None] = dt
    n, area0 = init_ball(diam_mm)
    out = np.zeros((len(t_grid), 5)); t = t_grid[0]
    observe(); w0 = obs[1] - obs[0]
    for k, tk in enumerate(t_grid):
        while t < tk - 1e-12:
            h = min(dt, tk - t)
            nsub = 1
            dt_f[None] = h
            plate_y[None] = Y_FLOOR + plate_fn(t + 0.5 * h) + PLATE_OFFSET
            plate_v[None] = plate_vel_fn(t + 0.5 * h)
            clear(); p2g(); grid_op(); g2p(); t += h
        observe()
        out[k] = [obs[1] - obs[0], obs[2] - Y_FLOOR, obs[3], obs[4] - Y_FLOOR, 0]
    return dict(width=out[:, 0], height=out[:, 1], J=out[:, 2], ybot=out[:, 3], dt=dt, n=n, w0=w0)


PLATE_OFFSET = 0.0
