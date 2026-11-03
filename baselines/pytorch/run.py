"""Baseline: PyTorch.
"""
import json
import sys
import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3
LOG2PI = float(np.log(2 * np.pi))
EPS10 = 10 * torch.finfo(torch.float32).eps

X = torch.from_numpy(np.load(f"{spec}/X.npy")).cuda()
w0 = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32)).cuda()
m0 = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32)).cuda()
v0 = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32)).cuda()
N, d = X.shape


def step(w, mu, va):
    tau = 1.0 / va
    b = (torch.log(w) - 0.5 * d * LOG2PI + 0.5 * torch.log(tau).sum(1)
         - 0.5 * (mu * mu * tau).sum(1))
    Xsq = X * X                                        # materialized (N,d)
    S = X @ (mu * tau).T - 0.5 * (Xsq @ tau.T) + b     # materialized (N,K)
    L = torch.logsumexp(S, dim=1)
    resp = torch.exp(S - L[:, None])                   # a second (N,K)
    nk = resp.sum(0) + EPS10                           # 1st traversal of resp
    s = resp.T @ X                                     # 2nd
    u = resp.T @ Xsq                                   # 3rd
    mu_n = s / nk[:, None]
    return nk / N, mu_n, u / nk[:, None] - mu_n * mu_n + 1e-6


def fit():
    w, mu, va = w0.clone(), m0.clone(), v0.clone()
    for _ in range(iters):
        w, mu, va = step(w, mu, va)
    return w, mu, va


fit()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(REPS):
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    w, mu, va = fit()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - _t0) * 1e3)

np.savez(out, w=w.cpu().numpy(), mu=mu.cpu().numpy(), va=va.cpu().numpy())
print(json.dumps(dict(
    arm="pytorch", total_ms=float(np.median(times)), all_ms=times, iters=iters,
    peak_mb=torch.cuda.max_memory_allocated() / 2**20,
    versions=dict(torch=torch.__version__),
)))
