"""Baseline: Flash-GMM.
"""
import json
import os
import sys
import time

import numpy as np
import torch

IBM_DIR = "$FLASH_GMM_SRC   # set by baselines/flash_gmm/setup.sh"
sys.path.insert(0, IBM_DIR)
from flash_gmm_diag_h100 import flash_gmm_estep_diag  # noqa: E402

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3
REG_COVAR = 1e-6          # same value our finalize() uses

X = torch.from_numpy(np.load(f"{spec}/X.npy")).cuda()
w0 = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32)).cuda()
m0 = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32)).cuda()
v0 = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32)).cuda()
N, D = X.shape


def mstep(Nk, mu_acc, sig_stat, mu_in):
    """Verbatim from IBM flash_gmm_sklearn.py::_mstep, covariance_type='diag'."""
    Nk_safe = Nk.clamp_min(1e-10)
    weights = Nk / N
    means = mu_acc / Nk_safe[:, None]
    shift = (means - mu_in) ** 2
    var = (sig_stat - Nk_safe[:, None] * shift) / Nk_safe[:, None]
    cov = var.clamp_min(0.0) + REG_COVAR
    return weights, means, cov


def fit():
    w, m, v = w0.clone(), m0.clone(), v0.clone()
    for _ in range(iters):
        logZ, Nk, mu_acc, sig_acc = flash_gmm_estep_diag(
            X, m, torch.log(v), torch.log(w))
        w, m, v = mstep(Nk, mu_acc, sig_acc, m)
    return w, m, v


try:
    fit()                                    # warmup: Triton compile + autotune
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        w, m, v = fit()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - _t0) * 1e3)
    np.savez(out, w=w.cpu().numpy(), mu=m.cpu().numpy(), va=v.cpu().numpy())
    # Partition-of-unity diagnostic: sum_k pi_k must be 1. Their split-K pass 2
    # masks rows with `n_off < N` instead of `< n_end`, so rows in the final tile
    # of each N-slab are accumulated twice whenever chunk % BLOCK_N != 0.
    print(json.dumps(dict(
        arm="flash-gmm", total_ms=float(np.median(times)), all_ms=times,
        iters=iters, status="ok", weight_sum=float(w.sum().item()),
        peak_mb=torch.cuda.max_memory_allocated() / 1e6)))
except torch.cuda.OutOfMemoryError as e:
    print(json.dumps(dict(arm="flash-gmm", status="OOM", err=str(e)[:200])))
except Exception as e:
    print(json.dumps(dict(arm="flash-gmm", status="ERROR",
                          err=f"{type(e).__name__}: {str(e)[:300]}")))
