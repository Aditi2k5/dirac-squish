"""ONE video comparing the approaches: real clip + every available model (oblique shaded 3D view, same camera, same time).
Usage: python render_compare.py [out_basename] [--skip=ms,pybullet]      (default outputs/models/compare_demo; keys: mpm fem xpbd ms mujoco pybullet)
Uses whichever of these exist:  outputs/clip1/sim3d_alpha100_long.npz (MPM), mesh_fem_h4.5.npz, mesh_xpbd_h4.5.npz, mesh_ms_h4.5.npz, mesh_mujoco_h4.5.npz, mesh_pybullet_h4.5.npz
Each tile shows the model name and its current ball height H next to the measured height; the faint outline is the book at its measured height.
No sway layer here: these are the raw models. Needs data/input.mp4 and outputs/clip1/targets.npz (or VIDEO / TARGETS / CROP env variables, as in render_3d.py); ffmpeg optional (else the _raw.mp4 is written)."""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import os, sys, shutil, subprocess
import cv2
import numpy as np
from scipy.spatial import ConvexHull

# ---- shaded 3D ball drawing (convex hull of the particles, back-face culled; same look as render_3d.py). Coordinates: x horizontal, y up, z depth, mm.
LIGHT = np.array([-0.35, 0.8, 0.5]); LIGHT /= np.linalg.norm(LIGHT)


def project(pts, az, el, c, size, cam_d=430.0, f=1350.0):
    p = pts - c; a, e = np.radians(az), np.radians(el)
    xr = p[:, 0] * np.cos(a) - p[:, 2] * np.sin(a); zr = p[:, 0] * np.sin(a) + p[:, 2] * np.cos(a)
    yr = p[:, 1] * np.cos(e) - zr * np.sin(e); zr2 = p[:, 1] * np.sin(e) + zr * np.cos(e)
    dep = cam_d - zr2; k = size / 370.0
    return np.stack([size / 2 + k * f * xr / dep, size / 2 + 15 * k - k * f * yr / dep], 1), dep


def rot_n(n, az, el):
    a, e = np.radians(az), np.radians(el)
    xr = n[:, 0] * np.cos(a) - n[:, 2] * np.sin(a); zr = n[:, 0] * np.sin(a) + n[:, 2] * np.cos(a)
    return np.stack([xr, n[:, 1] * np.cos(e) - zr * np.sin(e), n[:, 1] * np.sin(e) + zr * np.cos(e)], 1)


def draw_ball(P, plate_h, cx0, yf, R0, title, size=320, az=35, el=24, base=(30, 190, 245)):
    c = np.array([cx0, yf + 30.0, cx0]); img = np.full((size, size, 3), 238, np.uint8)
    for g in np.arange(-70, 71, 10):
        for a_, b_ in (((cx0 + g, yf, cx0 - 70), (cx0 + g, yf, cx0 + 70)), ((cx0 - 70, yf, cx0 + g), (cx0 + 70, yf, cx0 + g))):
            uv, _ = project(np.array([a_, b_]), az, el, c, size); cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (205, 205, 205), 1, cv2.LINE_AA)
    th = np.linspace(0, 2 * np.pi, 80)
    ring = np.stack([cx0 + R0 * np.cos(th), np.full_like(th, yf), cx0 + R0 * np.sin(th)], 1)
    uv, _ = project(ring, az, el, c, size); cv2.polylines(img, [uv.astype(np.int32)], True, (60, 60, 200), 1, cv2.LINE_AA)
    ok = np.isfinite(P).all(1)
    if ok.sum() >= 8:
        try:
            hull = ConvexHull(P[ok]); Q = P[ok]; uv, dep = project(Q, az, el, c, size)
            nrm = rot_n(hull.equations[:, :3], az, el); shade = 0.35 + 0.65 * np.clip(nrm @ LIGHT, 0, 1)
            sim = hull.simplices
            front = nrm[:, 2] > 0                                  # back-face culling: a convex ball then needs no depth sorting for correctness
            for i in np.argsort(-dep[sim].mean(1)):
                if not front[i]: continue
                cv2.fillConvexPoly(img, uv[sim[i]].astype(np.int32), tuple(int(v) for v in np.clip(np.array(base, float) * shade[i], 0, 255)), cv2.LINE_AA)
        except Exception:
            pass
    if plate_h is not None:
        for yy in (yf + plate_h, yf + plate_h + 14.0):
            q = np.array([[cx0 - 75, yy, cx0 - 75], [cx0 + 75, yy, cx0 - 75], [cx0 + 75, yy, cx0 + 75], [cx0 - 75, yy, cx0 + 75]])
            uvq, _ = project(q, az, el, c, size); cv2.polylines(img, [uvq.astype(np.int32)], True, (150, 90, 40), 1, cv2.LINE_AA)
    cv2.putText(img, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


args = [x for x in sys.argv[1:] if not x.startswith('--skip=')]
SKIP = [s.strip() for x in sys.argv[1:] if x.startswith('--skip=') for s in x.split('=', 1)[1].split(',')]   # e.g. --skip=ms,pybullet
OUT = args[0] if args else 'outputs/models/compare_demo'
SRC = [('MPM (particles+grid)', 'outputs/clip1/sim3d_alpha100_long.npz'), ('FEM (corotated tets)', 'outputs/models/mesh_fem_h4.5.npz'), ('XPBD (position-based)', 'outputs/models/mesh_xpbd_h4.5.npz'),
       ('Mass-spring', 'outputs/models/mesh_ms_h4.5.npz'), ('MuJoCo flex', 'outputs/models/mesh_mujoco_h4.5.npz'), ('PyBullet deformable', 'outputs/models/mesh_pybullet_h4.5.npz')]
KEY = {'mpm': 'sim3d_alpha100_long', 'fem': 'mesh_fem', 'xpbd': 'mesh_xpbd', 'ms': 'mesh_ms', 'mujoco': 'mesh_mujoco', 'pybullet': 'mesh_pybullet'}
models = [(n, np.load(p)) for n, p in SRC if os.path.exists(p) and not any(KEY[k] in p for k in SKIP)]
if not models: raise SystemExit('no model outputs found in outputs/ - run run_3d.py / mesh_models.py first')
print('models:', [n for n, _ in models])
T = np.load(os.environ.get('TARGETS', 'outputs/clip1/targets.npz')); s, table, fps = float(T['px_per_mm']), float(T['table_px']), float(T['fps'])
plate, tt, Hm = T['plate_mm'], T['t'], T['ball_height_mm']; okp = np.isfinite(plate)
cap = cv2.VideoCapture(os.environ.get('VIDEO', 'data/input.mp4')); x0, x1, y0, y1 = 150, 700, int(table) - 330, int(table) + 40
if os.environ.get('CROP'): x0, x1, y0, y1 = (int(q) for q in os.environ['CROP'].split(','))            # same VIDEO / TARGETS / CROP variables as render_3d.py
SZ = 320; ntile = 1 + len(models); cols = 3; rows = int(np.ceil(ntile / cols))
t_grid = models[0][1]['t']; t_grid = t_grid[t_grid <= 8.8]
vw = cv2.VideoWriter(OUT + '_raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (cols * SZ, rows * SZ))
for tk in t_grid:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tk * fps))); ok, fr = cap.read()
    if not ok: continue
    real = cv2.resize(fr[y0:y1, x0:x1], (SZ, int(SZ * (y1 - y0) / (x1 - x0)))); tile = np.full((SZ, SZ, 3), 238, np.uint8); o = (SZ - real.shape[0]) // 2; tile[o:o + real.shape[0]] = real
    hm = float(np.interp(tk, tt[np.isfinite(Hm)], Hm[np.isfinite(Hm)]))
    cv2.rectangle(tile, (0, 0), (SZ, 28), (40, 40, 40), -1); cv2.putText(tile, f'real  t={tk:.2f}s  H={hm:.0f}mm', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    tiles = [tile]
    hp = float(np.interp(tk, tt[okp], plate[okp])) if tk <= tt[okp][-1] else None
    for name, S in models:
        k = int(np.argmin(np.abs(S['t'] - tk)))
        if abs(S['t'][k] - tk) > 0.2 or k >= len(S['snaps']): tiles.append(np.full((SZ, SZ, 3), 238, np.uint8)); continue          # model has no frame here (run stopped early)
        P = S['snaps'][k].astype(np.float64)
        tiles.append(draw_ball(P, hp, float(S['cx']), float(S['y_floor']), float(S['w0']) / 2, f'{name}  H={float(S["height"][k]):.0f}mm', SZ))
    while len(tiles) < rows * cols: tiles.append(np.full((SZ, SZ, 3), 238, np.uint8))
    vw.write(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]))
vw.release()
if shutil.which('ffmpeg'):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', OUT + '_raw.mp4', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', OUT + '.mp4'], check=False); print('wrote', OUT + '.mp4')
else:
    print('ffmpeg not found (brew install ffmpeg); play', OUT + '_raw.mp4')