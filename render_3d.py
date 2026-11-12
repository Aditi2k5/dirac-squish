"""3D view of the simulation: real clip | shaded simulated ball from an elevated angle | same ball seen from above.
Usage:  [WOBBLE=1] [TILT=1] [VIDEO=data/input2.mp4 TARGETS=outputs/clip2/targets.npz CROP=x0,x1,y0,y1] python render_3d.py [sim_file.npz] [out_basename]
 - the ball surface is the convex hull of the simulated particles, shaded by face normal (the squashed ball is convex)
 - the book is drawn (translucent) at its MEASURED height while it is in frame; the table has a grid and a ring marking the ball's rest footprint
   so sideways motion is easy to see
 - WOBBLE=1 adds the fitted post-release sway (sway_model.py) as a whole-ball sideways shift (an add-on layer, labelled in the video)
Particles: x = horizontal (same axis as the video), y = up, z = depth (not visible in the video).
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import os, sys, glob, shutil, subprocess
import cv2
import numpy as np
from scipy.spatial import ConvexHull

SIM = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob('outputs/*/sim3d*.npz'), key=os.path.getmtime)
WOB = os.environ.get('WOBBLE') == '1'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(SIM), 'view3d_' + os.path.basename(SIM)[:-4].replace('sim3d_', '')) + ('_sway' if WOB else '')
S = np.load(SIM); T = np.load(os.environ.get('TARGETS', 'outputs/clip1/targets.npz'))
s_px, table, fps = float(T['px_per_mm']), float(T['table_px']), float(T['fps'])
plate, tt = T['plate_mm'], T['t']
from scipy.signal import savgol_filter
_ok = np.isfinite(plate); plate = plate.copy(); plate[_ok] = savgol_filter(plate[_ok], 11, 2)      # same smoothing as run_3d.py (the raw book path jitters ~1.3 mm/frame)
if os.environ.get('TILT'):                                       # TILT=1: draw the book at the slope stored in the targets file (extract_targets.py --tilt_half_px 150)
    _o = np.isfinite(T['slope']); _ts, _ss = tt[_o], savgol_filter(T['slope'][_o], 21, 2)
else: _ts = None
if WOB: Wd = np.load('outputs/sway/sway.npz')
cx0, yf = float(S['cx']), float(S['y_floor'])
cap = cv2.VideoCapture(os.environ.get('VIDEO', 'data/input.mp4'))
x0, x1 = 150, 700; y0 = int(table) - 330; y1 = int(table) + 40
if os.environ.get('CROP'): x0, x1, y0, y1 = (int(v) for v in os.environ['CROP'].split(','))      # e.g. CROP=295,995,263,733 for clip 2
PW = 370                                                         # sim panel size (px)
LIGHT = np.array([-0.35, 0.8, 0.5]); LIGHT /= np.linalg.norm(LIGHT)


def view(pts, az, el, cam_d=430.0, f=1350.0):
    """world (mm) -> screen px (u, v) and depth. Camera looks at (cx0, yf+30, cx0) from azimuth az, elevation el (deg)."""
    p = pts - np.array([cx0, yf + 30.0, cx0])
    a, e = np.radians(az), np.radians(el)
    xr = p[:, 0] * np.cos(a) - p[:, 2] * np.sin(a); zr = p[:, 0] * np.sin(a) + p[:, 2] * np.cos(a)
    yr = p[:, 1] * np.cos(e) - zr * np.sin(e); zr2 = p[:, 1] * np.sin(e) + zr * np.cos(e)
    depth = cam_d - zr2
    return np.stack([PW / 2 + f * xr / depth, PW / 2 + 15 - f * yr / depth], 1), depth


def rot_n(n, az, el):                                           # rotate normals into view space for shading (light fixed to the camera)
    a, e = np.radians(az), np.radians(el)
    xr = n[:, 0] * np.cos(a) - n[:, 2] * np.sin(a); zr = n[:, 0] * np.sin(a) + n[:, 2] * np.cos(a)
    return np.stack([xr, n[:, 1] * np.cos(e) - zr * np.sin(e), n[:, 1] * np.sin(e) + zr * np.cos(e)], 1)


def draw_panel(P, hp, az, el, title, sl=0.0):
    img = np.full((PW, PW, 3), 238, np.uint8)
    # table: grid + rest-footprint ring
    for g in np.arange(-70, 71, 10):
        for a_, b_ in (((cx0 + g, yf, cx0 - 70), (cx0 + g, yf, cx0 + 70)), ((cx0 - 70, yf, cx0 + g), (cx0 + 70, yf, cx0 + g))):
            uv, _ = view(np.array([a_, b_]), az, el); cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (205, 205, 205), 1, cv2.LINE_AA)
    th = np.linspace(0, 2 * np.pi, 80); R0 = float(S['w0']) / 2
    ring = np.stack([cx0 + R0 * np.cos(th), np.full_like(th, yf), cx0 + R0 * np.sin(th)], 1)
    uv, _ = view(ring, az, el); cv2.polylines(img, [uv.astype(np.int32)], True, (60, 60, 200), 1, cv2.LINE_AA)
    # soft shadow: particle footprints on the table
    uvs, _ = view(np.stack([P[:, 0], np.full(len(P), yf), P[:, 2]], 1), az, el); sh = img.copy()
    for u, v_ in uvs[::6].astype(int): cv2.circle(sh, (u, v_), 2, (150, 150, 150), -1)
    img = cv2.addWeighted(sh, 0.55, img, 0.45, 0)
    # ball: shaded convex hull
    hull = ConvexHull(P); uv, dep = view(P, az, el)
    nrm = rot_n(hull.equations[:, :3], az, el); shade = 0.35 + 0.65 * np.clip(nrm @ LIGHT, 0, 1)
    sim = hull.simplices; order = np.argsort(-dep[sim].mean(1))
    base = np.array([30, 190, 245], float)
    front = nrm[:, 2] > 0                                           # back-face culling (convex ball: no depth-sort artefacts)
    for i in order:
        if not front[i]: continue
        tri = uv[sim[i]].astype(np.int32); col = tuple(int(c) for c in np.clip(base * shade[i], 0, 255))
        cv2.fillConvexPoly(img, tri, col, cv2.LINE_AA)
    # book: faint slab + outline at its measured height (outline only from above, so it does not hide the ball)
    if hp is not None:
        ov = img.copy(); hh, th_ = 75.0, 14.0
        for yy in (yf + hp, yf + hp + th_):
            q = np.array([[cx0 - hh, yy - sl * hh, cx0 - hh], [cx0 + hh, yy + sl * hh, cx0 - hh], [cx0 + hh, yy + sl * hh, cx0 + hh], [cx0 - hh, yy - sl * hh, cx0 + hh]])   # plane tilted about the ball axis (sl = measured slope, mm/mm)
            uvq, _ = view(q, az, el)
            if el < 60: cv2.fillConvexPoly(ov, uvq.astype(np.int32), (150, 90, 40), cv2.LINE_AA)
            cv2.polylines(img, [uvq.astype(np.int32)], True, (150, 90, 40), 1, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.16, img, 0.84, 0) if el < 60 else img
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return img


print(f'video={os.environ.get("VIDEO", "data/input.mp4")}  targets={os.environ.get("TARGETS", "outputs/clip1/targets.npz")}  crop x{x0}:{x1} y{y0}:{y1}  sim={SIM}')
if not cap.isOpened(): raise SystemExit('cannot open the video')
if (os.environ.get('VIDEO') is None) != (os.environ.get('TARGETS') is None): raise SystemExit('set BOTH VIDEO and TARGETS (and CROP for clip 2), or none')
cw, ch = x1 - x0, y1 - y0
out = cv2.VideoWriter(OUT + '_raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (cw + 2 * PW, max(ch, PW)))
lab = '3D MPM + sway layer (fitted)' if WOB else '3D MPM'
for k, tk in enumerate(S['t']):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tk * fps))); ok, fr = cap.read()
    if not ok: continue
    left = fr[y0:y1, x0:x1].copy(); cv2.putText(left, f'real  t={tk:.2f}s', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    P = S['snaps'][k].astype(np.float64)
    if WOB and tk >= 8.0:
        P[:, 0] += np.interp(tk, Wd['t'], Wd['xfit']) if tk >= float(Wd['t0']) else np.interp(tk, Wd['t'], Wd['x'])
    ok_p = np.isfinite(plate)
    hp = float(np.interp(tk, tt[ok_p], plate[ok_p])) if tk <= tt[ok_p][-1] else None
    sl = float(np.interp(tk, _ts, _ss)) if _ts is not None else 0.0
    a = draw_panel(P, hp, 35, 24, lab + ' - oblique', sl); b = draw_panel(P, hp, 0, 82, lab + ' - from above', sl)
    H = max(ch, PW); pad = lambda im: np.vstack([im, np.full((H - im.shape[0], im.shape[1], 3), 238, np.uint8)]) if im.shape[0] < H else im
    out.write(np.hstack([pad(left), pad(a), pad(b)]))
out.release()
if shutil.which('ffmpeg'):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', OUT + '_raw.mp4', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', OUT + '.mp4'], check=False)
    print('wrote', OUT + '.mp4')
else:
    print('ffmpeg not found (brew install ffmpeg); play', OUT + '_raw.mp4')