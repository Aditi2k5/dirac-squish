"""Taichi initialisation shared by all scripts.
MPM_ARCH=gpu (or metal / cuda / vulkan) -> probe that backend in a throw-away subprocess first (Taichi can hard-crash when a
backend is missing, which no try/except can catch); use it only if the probe works, otherwise fall back to the CPU and say why.
MPM_THREADS sets the number of CPU threads (default 2)."""
import os
import subprocess
import sys
import tempfile
import taichi as ti

_PROBE = '''
import sys
import taichi as ti
name = sys.argv[1]
arch = {"gpu": ti.gpu, "metal": ti.metal, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[name]
ti.init(arch=arch, default_fp=ti.f32)
got = ti.lang.impl.current_cfg().arch
f = ti.field(float, 8)

@ti.kernel
def k():
    for i in f:
        f[i] = i * 2.0

k(); ti.sync()
assert abs(f[3] - 6.0) < 1e-6
print("ARCH", got, "CPU" if got in (ti.x64, ti.arm64, ti.cpu) else "GPU")
'''
_ARCHS = {'gpu': 'gpu', 'metal': 'metal', 'cuda': 'cuda', 'vulkan': 'vulkan'}


def probe(name):
    """Returns (ok, message). Runs the probe as a real file (Taichi needs kernel source code on disk)."""
    with tempfile.NamedTemporaryFile('w', suffix='_probe.py', delete=False) as fh:
        fh.write(_PROBE); path = fh.name
    try:
        r = subprocess.run([sys.executable, path, name], capture_output=True, text=True, timeout=240)
    finally:
        os.unlink(path)
    line = [l for l in r.stdout.splitlines() if l.startswith('ARCH')]
    if r.returncode == 0 and line and line[-1].endswith('GPU'):
        return True, line[-1]
    tail = ' | '.join([l for l in (r.stderr or '').strip().splitlines() if l.strip()][-3:])
    return False, f'probe exit code {r.returncode}; {line[-1] if line else "no ARCH line"}; {tail[:300]}'


def init():
    want = os.environ.get('MPM_ARCH', '').lower()
    if want in _ARCHS:
        ok, msg = probe(want)
        if ok:
            ti.init(arch=getattr(ti, want), default_fp=ti.f32, offline_cache=True)
            print(f'Taichi backend: {ti.lang.impl.current_cfg().arch} (GPU)  [{msg}]', flush=True)
            return ti.lang.impl.current_cfg().arch
        print(f'!! backend "{want}" not usable -> using the CPU instead. Reason: {msg}', flush=True)
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=int(os.environ.get('MPM_THREADS', '2')), offline_cache=True)
    print('Taichi backend:', ti.lang.impl.current_cfg().arch, '(CPU)', flush=True)
    return ti.lang.impl.current_cfg().arch

if __name__ == '__main__':
    # `MPM_ARCH=gpu python backend.py check` : is the GPU really used, and are the operations the solver needs correct on it?
    # (parallel float scatter-add, atomic max, 3x3 polar decomposition, kernel-launch speed). Run without MPM_ARCH to compare the CPU.
    import time
    import numpy as np
    arch = init()

    N, G = 200000, 64
    pos = ti.Vector.field(3, float, N); grid = ti.Vector.field(3, float, (G, G, G)); mx = ti.field(float, ())
    R = ti.Matrix.field(3, 3, float, 4)


    @ti.kernel
    def scatter():
        for p in range(N):
            i = ti.cast(ti.random() * (G - 1), int); j = ti.cast(ti.random() * (G - 1), int); k = ti.cast(ti.random() * (G - 1), int)
            grid[i, j, k] += ti.Vector([1.0, 2.0, 3.0])
            ti.atomic_max(mx[None], ti.cast(i, float))


    @ti.kernel
    def polar():
        A = ti.Matrix([[2.0, 0.3, 0.1], [0.2, 1.5, 0.0], [0.0, 0.1, 0.8]])
        U, S = ti.polar_decompose(A)
        R[0] = U; R[1] = S


    @ti.kernel
    def noop():
        for i in range(1):
            mx[None] += 0.0


    scatter(); tot = grid.to_numpy().sum(axis=(0, 1, 2))
    ok1 = np.allclose(tot, [N, 2 * N, 3 * N], rtol=1e-4)
    print(f'parallel scatter-add: {"OK" if ok1 else "WRONG"}  (got {tot}, expected {[N, 2*N, 3*N]})')
    print(f'atomic max:           {"OK" if mx[None] > G - 5 else "WRONG"}  ({mx[None]:.0f})')
    polar(); U = R[0].to_numpy(); S = R[1].to_numpy()
    A = np.array([[2.0, 0.3, 0.1], [0.2, 1.5, 0.0], [0.0, 0.1, 0.8]])
    print(f'polar decomposition:  {"OK" if np.allclose(U @ S, A, atol=1e-4) and np.allclose(U.T @ U, np.eye(3), atol=1e-4) else "WRONG"}')
    ti.sync(); t0 = time.time()
    for _ in range(2000): noop()
    ti.sync(); print(f'kernel launch overhead: {(time.time()-t0)/2000*1e6:.0f} us per launch (the solver launches 4 per step)')
    on_gpu = arch not in (ti.x64, ti.arm64, ti.cpu)
    print('-> GPU is working; run:  MPM_ARCH=gpu python run_3d.py' if on_gpu else '-> running on the CPU (fine, just slower): python run_3d.py')