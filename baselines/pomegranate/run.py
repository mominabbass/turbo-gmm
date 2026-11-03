"""Baseline: pomegranate.
"""
import json
import sys
import time

import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3

X = torch.from_numpy(np.load(f"{spec}/X.npy")).cuda()
w0 = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32))
m0 = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32))
v0 = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32))
K = m0.shape[0]


def fit():
    dists = [Normal(means=m0[k].clone(), covs=v0[k].clone(),
                    covariance_type="diag", min_cov=1e-6) for k in range(K)]
    model = GeneralMixtureModel(dists, priors=w0.clone(), max_iter=iters,
                                tol=-1e9, verbose=False).cuda()
    model.fit(X)
    return model


fit()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(REPS):
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    model = fit()
    torch.cuda.synchronize()
    times.append((time.perf_counter() - _t0) * 1e3)

np.savez(out,
         w=model.priors.detach().cpu().numpy(),
         mu=torch.stack([d.means for d in model.distributions]).detach().cpu().numpy(),
         va=torch.stack([d.covs for d in model.distributions]).detach().cpu().numpy())
print(json.dumps(dict(
    arm="pomegranate", total_ms=float(np.median(times)), all_ms=times, iters=iters,
    peak_mb=torch.cuda.max_memory_allocated() / 2**20,
    setup_note="E-step loops over K in Python: K sequential kernels per iteration",
    versions=dict(pomegranate=__import__("pomegranate").__version__,
                  torch=torch.__version__),
)))
