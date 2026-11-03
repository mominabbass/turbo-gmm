"""Baseline: torchgmm.
"""
import json
from importlib.metadata import version as _pkgver
import logging
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import pytorch_lightning as pl  # noqa: E402
from torchgmm.base.data import DataLoader, collate_tensor, dataset_from_tensors  # noqa: E402
from torchgmm.bayes.gmm.lightning_module import GaussianMixtureLightningModule  # noqa: E402
from torchgmm.bayes.gmm.model import (GaussianMixtureModel,  # noqa: E402
                                      GaussianMixtureModelConfig)

spec, iters, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REPS = 3


def _ver(p):
    try:
        return _pkgver(p)
    except Exception:
        return "unknown"

Xn = np.load(f"{spec}/X.npy")
X = torch.from_numpy(Xn).cuda()
w0 = torch.from_numpy(np.load(f"{spec}/w0.npy").astype(np.float32))
m0 = torch.from_numpy(np.load(f"{spec}/m0.npy").astype(np.float32))
v0 = torch.from_numpy(np.load(f"{spec}/v0.npy").astype(np.float32))
K, d = m0.shape
loader = DataLoader(dataset_from_tensors(X), batch_size=Xn.shape[0], collate_fn=collate_tensor)


def fit():
    model = GaussianMixtureModel(GaussianMixtureModelConfig(K, d, "diag"))
    model.means.copy_(m0)
    model.component_probs.copy_(w0)
    model.precisions_cholesky.copy_(v0.rsqrt())        # diag: chol(precision) = 1/sqrt(var)
    mod = GaussianMixtureLightningModule(model, convergence_tolerance=0.0,
                                         covariance_regularization=1e-6)
    tr = pl.Trainer(max_epochs=iters, accelerator="gpu", devices=1, logger=False,
                    enable_progress_bar=False, enable_checkpointing=False,
                    enable_model_summary=False)
    tr.fit(mod, loader)
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

pc = model.precisions_cholesky.detach().cpu().numpy()
np.savez(out, w=model.component_probs.detach().cpu().numpy(),
         mu=model.means.detach().cpu().numpy(), va=1.0 / (pc ** 2))
print(json.dumps(dict(
    arm="torchgmm", total_ms=float(np.median(times)), all_ms=times, iters=iters,
    peak_mb=torch.cuda.max_memory_allocated() / 2**20,
    setup_note="includes ~0.2s fixed PyTorch-Lightning Trainer overhead per fit",
    versions=dict(torchgmm=_ver("torchgmm"), torch=torch.__version__),
)))
