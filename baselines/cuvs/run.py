"""Baseline: cuVS.

Reproduced verbatim from the benchmark harness used for the paper.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
HARNESS = os.path.join(os.path.dirname(__file__), "..", "cuvs", "harness_spec")

X = np.load(f"{spec}/X.npy")                       # (N,d) float32, C-order
N, d = X.shape
w0 = np.load(f"{spec}/w0.npy").astype(np.float32)
m0 = np.load(f"{spec}/m0.npy").astype(np.float32)
v0 = np.load(f"{spec}/v0.npy").astype(np.float32)
K = m0.shape[0]

with tempfile.TemporaryDirectory(dir="/tmp") as td:
    for name, arr in (("X", X), ("w0", w0), ("m0", m0), ("v0", v0)):
        np.ascontiguousarray(arr, dtype=np.float32).tofile(f"{td}/{name}.f32")
    p = subprocess.run([os.path.abspath(HARNESS), str(N), str(d), str(K), str(iters),
                        td, f"{td}/out_"], capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        sys.exit(p.returncode)
    row = json.loads(p.stdout.strip().splitlines()[-1])
    rd = lambda n, shape: np.fromfile(f"{td}/out_{n}.f32", dtype=np.float32).reshape(shape)
    np.savez(out, w=rd("w", (K,)), mu=rd("mu", (K, d)), va=rd("va", (K, d)))

print(json.dumps(row))
