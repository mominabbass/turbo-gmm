"""Baseline: gmmx.
"""
import json
from importlib.metadata import version as _pkgver
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gmmx import EMFitter, GaussianMixtureModelJax

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3


def _ver(p):
    try:
        return _pkgver(p)
    except Exception:
        return "unknown"

X = jnp.asarray(np.load(f"{spec}/X.npy"))
m0 = np.load(f"{spec}/m0.npy").astype(np.float32)
v0 = np.load(f"{spec}/v0.npy").astype(np.float32)
w0 = np.load(f"{spec}/w0.npy").astype(np.float32)
fitter = EMFitter(max_iter=iters, tol=0.0, reg_covar=1e-6)


def fit():
    gmm0 = GaussianMixtureModelJax.from_squeezed(m0, v0, w0, covariance_type="diag")
    res = fitter.fit(X, gmm0)
    res.gmm.means.block_until_ready()
    return res


res = fit()                                   # warmup: JAX trace + compile
# gmmx stops as soon as the LL change is exactly 0 even at tol=0.0, so it may run
# fewer than `iters` iterations. Report what it actually ran; the caller divides by it.
actual_iters = int(res.n_iter)
times = []
for _ in range(REPS):
    t0 = time.perf_counter()
    res = fit()
    times.append((time.perf_counter() - t0) * 1e3)

np.savez(out,
         w=np.asarray(np.squeeze(res.gmm.weights)),
         mu=np.asarray(np.squeeze(res.gmm.means)),
         va=np.asarray(np.squeeze(res.gmm.covariances.values)))
print(json.dumps(dict(
    arm="gmmx", total_ms=float(np.median(times)), all_ms=times, iters=actual_iters,
    requested_iters=iters,
    peak_mb=None,
    setup_note="fp32 only; timed with perf_counter; stops early on exact convergence",
    versions=dict(gmmx=_ver("gmmx"), jax=jax.__version__),
)))
