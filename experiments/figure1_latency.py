"""
usage: python experiments/figure1_latency.py [--iters 10] [--out results.jsonl] [--only ARM]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ari, eval_ll, jsonl_append, make_config, provenance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.environ.get("VENV_CORE",     f"{ROOT}/.venv-core/bin/python")
T214 = os.environ.get("VENV_TORCH214", f"{ROOT}/.venv-torch214/bin/python")
JAXV = os.environ.get("VENV_JAX",      f"{ROOT}/.venv-jax/bin/python")
VENV = CORE

ARMS = [
    ("turbo-gmm",   "turbo_gmm_arm.py",     CORE, {}),
    ("flash-gmm",   "flash_gmm/run.py",     CORE, {}),
    ("cuvs",        "cuvs/run.py",          CORE, {}),
    ("pytorch",     "pytorch/run.py",       CORE, {}),
    ("sklearn",     "scikit_learn/run.py",  T214, {"SCIPY_ARRAY_API": "1"}),
    ("gmmx",        "gmmx/run.py",          JAXV,
     {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}),
    ("torchgmm",    "torchgmm/run.py",      T214, {}),
    ("pomegranate", "pomegranate/run.py",   T214, {}),
]
BASELINES = os.path.join(ROOT, "baselines")

# Four panels.  d=32 is held fixed except in panel 4, which sweeps it.
PANELS = {
    "Small N, Small K":  [(20_000, 16, 32), (20_000, 32, 32), (20_000, 64, 32)],
    "Large N, Small K":  [(1_000_000, 32, 32), (4_000_000, 32, 32),
                          (10_000_000, 32, 32), (20_000_000, 32, 32)],
    "Large N, Large K":  [(1_000_000, 128, 32), (1_000_000, 512, 32),
                          (1_000_000, 1024, 32), (1_000_000, 4096, 32)],
    "K/d crossover":     [(4_000_000, 64, 16), (4_000_000, 64, 32), (4_000_000, 32, 32),
                          (4_000_000, 32, 64), (4_000_000, 16, 64), (4_000_000, 16, 128)],
}

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--out", default=os.path.join(HERE, "results", "results.jsonl"))
ap.add_argument("--only", default=None, help="run just this arm")
ap.add_argument("--timeout", type=int, default=1800)
ap.add_argument("--seed", type=int, default=20260913)
args = ap.parse_args()

# Per-run scratch dir. 
scratch = f"/tmp/turbogmm_spec_{os.getpid()}"
env_base = dict(os.environ)   # honour CUDA_VISIBLE_DEVICES if the caller set it
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
PROV = provenance()
_gpu = env_base.get("CUDA_VISIBLE_DEVICES", "all visible")
print(f"iters={args.iters}  out={args.out}  CUDA_VISIBLE_DEVICES={_gpu}")
print(f"machine: {PROV['gpu']} driver {PROV['driver']} cuda {PROV['nvcc']}")

for panel, configs in PANELS.items():
    for (N, K, d) in configs:
        shutil.rmtree(scratch, ignore_errors=True)
        make_config(N, K, d, args.seed, scratch)
        X64 = np.load(f"{scratch}/X64.npy", mmap_mode="r")
        labels = np.load(f"{scratch}/labels.npy")
        print(f"\n=== {panel}  N={N:,} K={K} d={d} " + "=" * 20)

        for name, script, interp, extra in ARMS:
            if args.only and name != args.only:
                continue
            row = dict(panel=panel, N=N, K=K, d=d, arm=name, iters=args.iters,
                       seed=args.seed, **PROV)
            npz = f"/tmp/turbogmm_fit_{name}_{os.getpid()}.npz"
            if os.path.exists(npz):
                os.remove(npz)
            t0 = time.time()
            try:
                p = subprocess.run([interp, os.path.join(BASELINES, script),
                                    scratch, str(args.iters), npz],
                                   capture_output=True, text=True,
                                   env=dict(env_base, **extra), timeout=args.timeout)
                if p.returncode != 0:
                    err = (p.stderr or "").strip().splitlines()
                    oom = any("out of memory" in l.lower() or "OutOfMemory" in l
                              or "RESOURCE_EXHAUSTED" in l for l in err)
                    row.update(status="OOM" if oom else "ERROR",
                               error=err[-1][:300] if err else "no stderr")
                else:
                    row.update(json.loads(p.stdout.strip().splitlines()[-1]), status="ok")
                    row["ms_per_iter"] = row["total_ms"] / row["iters"]
                    z = np.load(npz)
                    row["final_ll"] = eval_ll(np.asarray(X64), z["w"], z["mu"], z["va"])
                    if K <= 4096:                    # ARI needs a K x K contingency table
                        va = z["va"] if z["va"].ndim == 2 else np.tile(z["va"][:, None], (1, d))
                        tau = 1.0 / va
                        b = (np.log(z["w"]) - 0.5 * d * np.log(2 * np.pi)
                             + 0.5 * np.log(tau).sum(1) - 0.5 * (z["mu"] ** 2 * tau).sum(1))
                        sub = np.asarray(X64[:200_000])
                        pred = (sub @ (z["mu"] * tau).T - 0.5 * (sub * sub) @ tau.T + b).argmax(1)
                        row["ari"] = ari(labels[:200_000], pred)
            except subprocess.TimeoutExpired:
                row.update(status="TIMEOUT")
            row["wall_s"] = round(time.time() - t0, 1)
            jsonl_append(args.out, row)
            if row["status"] == "ok":
                print(f"  {name:13s} {row['ms_per_iter']:10.2f} ms/iter  "
                      f"LL {row['final_ll']:.8f}  ARI {row.get('ari', float('nan')):.4f}  "
                      f"peak {row.get('peak_mb') or -1:.0f} MB")
            else:
                print(f"  {name:13s} {row['status']}  ({row.get('error','')[:90]})")

shutil.rmtree(scratch, ignore_errors=True)
print(f"\nwrote {args.out}")
