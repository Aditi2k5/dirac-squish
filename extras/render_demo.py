"""Step 4 - demo video: real clip (left) next to the simulated 3D ball seen from the same side view (right).
Needs outputs/sim3d.npz (from run_3d.py) and outputs/clip1/targets.npz. Writes outputs/demo.mp4.
The simulated particles are orthographically projected (x, y) and drawn at the video's pixel scale;
the book is drawn from the MEASURED plate height (it is the simulation's input, not a prediction).
"""
import os; [os.makedirs(d, exist_ok=True) for d in ('outputs/clip1', 'outputs/clip2', 'outputs/models', 'outputs/sway')]   # output folders
import subprocess, sys, shutil
import cv2
import numpy as np

T = np.load('outputs/clip1/targets.npz'); import glob, os
SIM = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob('outputs/*/sim3d*.npz'), key=os.path.getmtime)   # default: newest sim file
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(SIM), 'demo_' + os.path.basename(SIM)[:-4].replace('sim3d_', ''))   # e.g. outputs/demo_alpha100
WOB = os.environ.get('WOBBLE') == '1'                                      # WOBBLE=1: add the measured/fitted post-release SWAY (sway_model.py) on top of the MPM particles
if WOB:
    Wd = np.load('outputs/sway/sway.npz'); OUT += '_sway'
print('rendering', SIM, '->', OUT)
S = np.load(SIM)
s, table, fps = float(T['px_per_mm']), float(T['table_px']), float(T['fps'])
cap = cv2.VideoCapture('data/input.mp4'); W = int(cap.get(3)); H = int(cap.get(4))
x0, x1 = 150, 700; y0 = int(table) - 330; y1 = int(table) + 40            # crop around the ball
cw, ch = x1 - x0, y1 - y0
out = cv2.VideoWriter(OUT + '_raw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (2 * cw, ch))
cxp = 0.5 * (x0 + x1) + 10                                                 # horizontal position of the ball in the video
plate = T['plate_mm']; tt = T['t']
for k, tk in enumerate(S['t']):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tk * fps))); ok, fr = cap.read()
    if not ok: continue
    left = fr[y0:y1, x0:x1].copy()
    cv2.putText(left, f'real  t={tk:.2f}s', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    right = np.full((ch, cw, 3), 235, np.uint8)
    cv2.rectangle(right, (0, int(table) - y0 + 1), (cw, ch), (40, 40, 40), -1)   # table
    P = S['snaps'][k].astype(np.float32)
    xr = P[:, 0] - float(S['cx'])
    if WOB and tk >= 8.0:                                                      # whole-ball sideways shift: measured until 8.3 s, damped-oscillation fit after
        xr = xr + (np.interp(tk, Wd['t'], Wd['xfit']) if tk >= float(Wd['t0']) else np.interp(tk, Wd['t'], Wd['x'])) 
    px = (xr * s + cxp - x0).astype(int)
    py = (table - (P[:, 1] - float(S['y_floor'])) * s - y0).astype(int)
    for u, v_ in zip(px, py):
        if 0 <= u < cw and 0 <= v_ < ch: cv2.circle(right, (u, v_), 2, (30, 190, 245), -1)
    hp = np.interp(tk, tt[np.isfinite(plate)], plate[np.isfinite(plate)]) if tk <= tt[np.isfinite(plate)][-1] else None
    if hp is not None:
        yb = int(table - hp * s - y0); cv2.rectangle(right, (0, max(yb - 28, 0)), (cw, yb), (120, 70, 30), -1)
    cv2.putText(right, ('3D MPM + sway layer (fitted)' if WOB else '3D MPM simulation'), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out.write(np.hstack([left, right]))
out.release()
if shutil.which('ffmpeg'):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', OUT + '_raw.mp4', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', OUT + '.mp4'], check=False)
    print('wrote', OUT + '.mp4')
else:
    print('ffmpeg not found (brew install ffmpeg); the playable-in-most-players file is', OUT + '_raw.mp4')