
import json
import os
import numpy as np

LOG2PI = float(np.log(2 * np.pi))


def make_config(N, K, d, seed, out_dir):
    """

    flash-kmeans' headline harness regenerates its data inside each per-arm call with
    no seed, so arms cluster different data and runs are irreproducible.  We generate
    once, seeded, and hand every arm the same bytes.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    w_true = rng.dirichlet(np.full(K, 4.0))
    mu_true = rng.normal(0.0, 3.0, (K, d))
    va_true = rng.uniform(0.5, 2.0, (K, d))
    comp = rng.choice(K, size=N, p=w_true)
    X = mu_true[comp] + rng.normal(0, 1, (N, d)) * np.sqrt(va_true[comp])

    # Initialization: random data points as means, global variance, uniform weights.
    # Deliberately NOT k-means - sklearn's Array API path rejects init_params='kmeans',
    # and a k-means init would add a full Lloyd run to every arm's timing.
    idx = rng.choice(N, size=K, replace=False)        # replace=False: no duplicate components
    m0 = X[idx].copy()
    v0 = np.tile(X.var(0), (K, 1))
    w0 = np.full(K, 1.0 / K)

    np.save(f"{out_dir}/X.npy", X.astype(np.float32))
    np.save(f"{out_dir}/X64.npy", X)
    np.save(f"{out_dir}/w0.npy", w0)
    np.save(f"{out_dir}/m0.npy", m0)
    np.save(f"{out_dir}/v0.npy", v0)
    np.save(f"{out_dir}/labels.npy", comp)
    return dict(N=N, K=K, d=d, seed=seed, dir=out_dir)


def eval_ll(X64, w, mu, va, chunk=200_000):
    """Mean log-likelihood per sample, fp64, from fitted parameters.

    One definition for every arm.  Chunked over rows so that scoring a 2e7 x 4096
    problem does not itself need an N x K allocation.
    """
    w = np.asarray(w, np.float64).ravel()
    mu = np.asarray(mu, np.float64)
    va = np.asarray(va, np.float64)
    if va.ndim == 1:                                  # spherical -> broadcast
        va = np.tile(va[:, None], (1, mu.shape[1]))
    N, d = X64.shape
    tau = 1.0 / va
    b = (np.log(w) - 0.5 * d * LOG2PI + 0.5 * np.log(tau).sum(1)
         - 0.5 * (mu * mu * tau).sum(1))
    total = 0.0
    for i in range(0, N, chunk):
        Xc = X64[i:i + chunk]
        S = Xc @ (mu * tau).T - 0.5 * (Xc * Xc) @ tau.T + b
        m = S.max(1, keepdims=True)
        total += float((m.ravel() + np.log(np.exp(S - m).sum(1))).sum())
    return total / N


def ari(labels_true, labels_pred):
    """Adjusted Rand Index, contingency-table form.  No sklearn dependency."""
    from scipy.special import comb
    a = np.asarray(labels_true); b = np.asarray(labels_pred)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(C, (ia, ib), 1)
    sij = comb(C, 2).sum()
    si = comb(C.sum(1), 2).sum()
    sj = comb(C.sum(0), 2).sum()
    n = comb(len(a), 2)
    exp = si * sj / n
    return float((sij - exp) / (0.5 * (si + sj) - exp))


def provenance():
    """GPU/driver/toolkit facts to stamp on every result row, captured once per run."""
    import subprocess

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=20).stdout.strip() or None
        except Exception:
            return None

    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0")
    name, driver = (gpu.split(", ") + [None])[:2] if gpu else (None, None)
    nvcc = sh("nvcc --version 2>/dev/null | grep release")
    return dict(
        gpu=name,
        gpu_visible=vis,                 # which physical device this run was pinned to
        driver=driver,
        nvcc=nvcc.split("release ")[-1].split(",")[0] if nvcc and "release " in nvcc else None,
        kernel=sh("uname -r"),
    )


def jsonl_append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
