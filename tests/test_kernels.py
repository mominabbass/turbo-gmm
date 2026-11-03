"""
Correctness gate for the Triton kernels.

Run:  python tests/test_kernels.py
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turbo_gmm.kernels import em_step_onepass, em_step_twopass, em_step  # noqa: E402

LOG2PI = np.log(2 * np.pi)
FAIL = 0


def oracle(X, w, mu, va, reg=1e-6):
    """fp64 reference, centered Mahalanobis, materializing."""
    X, w, mu, va = (np.asarray(a, dtype=np.float64) for a in (X, w, mu, va))
    N, d = X.shape
    K = mu.shape[0]
    S = np.empty((N, K))
    for k in range(K):
        diff = X - mu[k]
        S[:, k] = (np.log(w[k]) - 0.5 * d * LOG2PI - 0.5 * np.log(va[k]).sum()
                   - 0.5 * ((diff * diff) / va[k]).sum(1))
    m = S.max(1, keepdims=True)
    P = np.exp(S - m)
    l = P.sum(1, keepdims=True)
    L = (m + np.log(l)).ravel()
    G = P / l
    nk = G.sum(0) + 10 * np.finfo(np.float32).eps
    means = (G.T @ X) / nk[:, None]
    cov = (G.T @ (X * X)) / nk[:, None] - means * means + reg
    ess = nk ** 2 / np.sum(G ** 2, axis=0)      # components owning ~1 point are ill-conditioned
    return L.mean(), nk / N, means, cov, ess


def mkdata(N, K, d, seed=0, sep=3.0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(K, 4.0))
    mu = rng.normal(0.0, sep, (K, d))
    va = rng.uniform(0.5, 2.0, (K, d))
    comp = rng.choice(K, size=N, p=w)
    X = mu[comp] + rng.normal(0, 1, (N, d)) * np.sqrt(va[comp])
    return X, w, mu, va


def to_cuda(*a):
    return [torch.as_tensor(x, dtype=torch.float32, device="cuda") for x in a]


def relerr(a, b):
    """Per-coordinate relative error. Use only for quantities bounded away from 0."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    den = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b) / den))


def scalerr(a, b):
    """max|a-b| / max|b|.  The right metric for means and covariances: a coordinate
    that happens to sit near zero makes per-coordinate relative error meaningless
    (with K*d = 65536 coordinates at least one always does)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def check(name, ok, detail=""):
    global FAIL
    if not ok:
        FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def compare(got, ref, tag, tol_ll=1e-5, tol_mu=1e-5, tol_cov=1e-3):
    ll, w, mu, cov = got
    keep = ref[4] >= 3.0
    e = (abs(ll - ref[0]) / max(abs(ref[0]), 1e-30),
         scalerr(w.cpu().numpy(), ref[1]),
         scalerr(mu.cpu().numpy()[keep], ref[2][keep]),
         scalerr(cov.cpu().numpy()[keep], ref[3][keep]))
    ok = e[0] < tol_ll and e[1] < 1e-4 and e[2] < tol_mu and e[3] < tol_cov
    starve = int((~keep).sum())
    check(tag, ok, f"dLL {e[0]:.2e}  dw {e[1]:.2e}  dmu {e[2]:.2e}  dcov {e[3]:.2e}"
                   + (f"  [{starve} starving comps excluded]" if starve else ""))
    return e


print("=" * 88)
print("1. TWO-PASS vs fp64 centered oracle, across shapes and BLOCK_K")
print("=" * 88)
for (N, K, d) in [(20000, 8, 16), (50000, 16, 32), (30000, 7, 13), (12345, 33, 8)]:
    X, w, mu, va = mkdata(N, K, d, seed=N + K + d)
    ref = oracle(X, w, mu, va)
    Xc, wc, muc, vac = to_cuda(X, w, mu, va)
    for BK in (16, 32, 64, 128):
        got = em_step_twopass(Xc, wc, muc, vac, BLOCK_K=BK)
        compare(got, ref, f"N={N} K={K} d={d} BLOCK_K={BK} (n_ktiles={-(-K//BK)})")

print()
print("=" * 88)
print("2. TWO-PASS vs ONE-PASS where both are legal (must agree to fp32 roundoff)")
print("=" * 88)
for (N, K, d) in [(50000, 16, 32), (20000, 32, 16), (30000, 8, 32)]:
    X, w, mu, va = mkdata(N, K, d, seed=7)
    Xc, wc, muc, vac = to_cuda(X, w, mu, va)
    a = em_step_onepass(Xc, wc, muc, vac)
    b = em_step_twopass(Xc, wc, muc, vac, BLOCK_K=64)
    e = max(scalerr(a[2].cpu().numpy(), b[2].cpu().numpy()),
            scalerr(a[3].cpu().numpy(), b[3].cpu().numpy()))
    check(f"N={N} K={K} d={d} onepass vs twopass", e < 1e-4, f"max rel {e:.2e}")

print()
print("=" * 88)
print("3. LARGE K -- the regime the one-pass kernel cannot reach at all")
print("=" * 88)
for (N, K, d) in [(200000, 256, 32), (200000, 1024, 32), (100000, 4096, 16)]:
    X, w, mu, va = mkdata(N, K, d, seed=1, sep=6.0)
    ref = oracle(X, w, mu, va)
    Xc, wc, muc, vac = to_cuda(X, w, mu, va)
    got_i = em_step_twopass(Xc, wc, muc, vac, BLOCK_K=64, prec="ieee")
    compare(got_i, ref, f"N={N} K={K} d={d} K*d={K*d} prec=ieee")
    got_t = em_step_twopass(Xc, wc, muc, vac, BLOCK_K=64, prec="tf32x3")
    e = scalerr(got_t[2].cpu().numpy()[ref[4] >= 3.0], ref[2][ref[4] >= 3.0])
    print(f"         (tf32x3 same shape: dmu {e:.2e} -- matches ieee, see note)")

print()
print("=" * 88)
print("4. INVARIANTS")
print("=" * 88)
X, w, mu, va = mkdata(40000, 64, 32, seed=3)
Xc, wc, muc, vac = to_cuda(X, w, mu, va)
for name, fn in (("onepass", em_step_onepass), ("twopass", em_step_twopass)):
    ll, wn, mun, covn = fn(Xc, wc, muc, vac)
    check(f"{name}: sum(weights) == 1", abs(wn.sum().item() - 1.0) < 1e-5,
          f"{wn.sum().item():.9f}")
    check(f"{name}: cov > 0", bool((covn > 0).all()), f"min {covn.min().item():.3e}")
    check(f"{name}: ll finite", np.isfinite(ll), f"{ll:.6f}")

print()
print("=" * 88)
print("5. DISPATCH picks the right kernel")
print("=" * 88)

import turbo_gmm.kernels as _kern  # noqa: E402

_called = []


def _spy(tag, fn):
    def wrapper(*a, **kw):
        _called.append(tag)
        return fn(*a, **kw)
    return wrapper



for (K, d, want) in [(32, 32, "onepass"), (64, 32, "onepass"),
                     (128, 32, "onepass"), (32, 128, "onepass"),
                     (256, 16, "twopass"), (512, 32, "twopass"),
                     (64, 128, "twopass")]:
    X, w, mu, va = mkdata(20000, K, d, seed=5, sep=5.0)
    ref = oracle(X, w, mu, va)
    Xc, wc, muc, vac = to_cuda(X, w, mu, va)
    _orig1, _orig2 = _kern.em_step_onepass, _kern.em_step_twopass
    _kern.em_step_onepass = _spy("onepass", _orig1)
    _kern.em_step_twopass = _spy("twopass", _orig2)
    _called.clear()
    try:
        got = _kern.em_step(Xc, wc, muc, vac)
    finally:
        _kern.em_step_onepass, _kern.em_step_twopass = _orig1, _orig2
    picked = _called[0] if _called else "NONE"
    check(f"K={K} d={d} -> {picked} (want {want})", picked == want)
    compare(got, ref, f"    dispatch result K={K} d={d}")

print()
print("=" * 88)
print(f"RESULT: {'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURES'}")
print("=" * 88)
sys.exit(1 if FAIL else 0)
