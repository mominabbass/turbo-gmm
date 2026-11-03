"""Baseline: scikit-learn.
"""
import json
import sys
import time

import numpy as np
import sklearn
import torch

sklearn.set_config(array_api_dispatch=True)
from sklearn.mixture import GaussianMixture  # noqa: E402

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3

X = torch.from_numpy(np.load(f"{spec}/X.npy")).cuda()
w0 = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32)).cuda()
m0 = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32)).cuda()
v0 = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32)).cuda()
K = m0.shape[0]


def fit():
    g = GaussianMixture(n_components=K, covariance_type="diag", max_iter=iters,
                        n_init=1, init_params="random", tol=0.0, reg_covar=1e-6,
                        weights_init=w0, means_init=m0, precisions_init=1.0 / v0)
    g.fit(X)
    return g


g = fit()
assert g.n_iter_ == iters, f"expected {iters} iterations, got {g.n_iter_}"
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(REPS):
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    g = fit()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - _t0) * 1e3)

np.savez(out,
         w=np.asarray(g.weights_.cpu()), mu=np.asarray(g.means_.cpu()),
         va=np.asarray(g.covariances_.cpu()))          # for "diag", covariances_ IS the variances
print(json.dumps(dict(
    arm="sklearn", total_ms=float(np.median(times)), all_ms=times, iters=iters,
    peak_mb=torch.cuda.max_memory_allocated() / 2**20,
    versions=dict(sklearn=sklearn.__version__, torch=torch.__version__),
)))
