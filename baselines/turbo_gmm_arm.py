"""Arm: Turbo-GMM (our fused Triton kernels). Env: core (env/setup-core.sh).

usage: python baselines/turbo_gmm_arm.py <spec_dir> <iters> <out_npz>
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from turbo_gmm.kernels import em_step  # noqa: E402

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3

X = torch.from_numpy(np.load(f"{spec}/X.npy")).cuda()
w = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32)).cuda()
m = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32)).cuda()
v = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32)).cuda()


def fit():
    ww, mm, vv = w.clone(), m.clone(), v.clone()
    for _ in range(iters):
        _, ww, mm, vv = em_step(X, ww, mm, vv)
    return ww, mm, vv


fit()                                       # warmup: Triton compile + cache
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(REPS):
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    ww, mm, vv = fit()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - _t0) * 1e3)

np.savez(out, w=ww.cpu().numpy(), mu=mm.cpu().numpy(), va=vv.cpu().numpy())
print(json.dumps(dict(
    arm="turbo-gmm", total_ms=float(np.median(times)), all_ms=times, iters=iters,
    peak_mb=torch.cuda.max_memory_allocated() / 2**20,
    versions=dict(torch=torch.__version__, triton=__import__("triton").__version__),
)))
