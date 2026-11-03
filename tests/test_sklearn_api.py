"""TurboGaussianMixture against sklearn.mixture.GaussianMixture.

usage: python tests/test_sklearn_api.py
"""
import sys

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from turbo_gmm import TurboGaussianMixture  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def match(a_means, b_means):
    """Component permutation minimising total squared distance between mean sets."""
    C = ((a_means[:, None, :] - b_means[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    return r, c


print("=" * 88)
print("1. AGREEMENT with scikit-learn")
print("=" * 88)
for (N, K, d, ctype) in [(20_000, 8, 16, "diag"), (50_000, 16, 32, "diag"),
                         (20_000, 8, 16, "spherical"), (5_000, 3, 2, "diag")]:
    X, y = make_blobs(n_samples=N, centers=K, n_features=d, cluster_std=1.5,
                      random_state=0)
    kw = dict(n_components=K, covariance_type=ctype, random_state=0, max_iter=50,
              tol=1e-6, reg_covar=1e-6, init_params="kmeans")
    ours = TurboGaussianMixture(**kw).fit(X)
    ref = GaussianMixture(n_init=1, **kw).fit(X)

    r, c = match(ours.means_, ref.means_)
    dll = abs(ours.score(X) - ref.score(X)) / abs(ref.score(X))
    tag = f"N={N} K={K} d={d} {ctype}"

    # Fit QUALITY is always comparable, and is the claim that matters.
    check(f"{tag}: mean log-likelihood", dll < 1e-5, f"rel {dll:.2e}")

    
    a_o = adjusted_rand_score(y, ours.predict(X))
    a_r = adjusted_rand_score(y, ref.predict(X))
    same_optimum = min(a_o, a_r) > 0.95 and abs(a_o - a_r) < 1e-3
    if same_optimum:
        dmu = np.abs(ours.means_[r] - ref.means_[c]).max() / max(np.abs(ref.means_).max(), 1e-9)
        dw = np.abs(ours.weights_[r] - ref.weights_[c]).max()
        co, cr = ours.covariances_[r], ref.covariances_[c]
        dcov = np.abs(co - cr).max() / max(np.abs(cr).max(), 1e-9)
        check(f"{tag}: means", dmu < 1e-4, f"rel {dmu:.2e}")
        check(f"{tag}: weights", dw < 1e-4, f"abs {dw:.2e}")
        check(f"{tag}: covariances", dcov < 1e-3, f"rel {dcov:.2e}")
        check(f"{tag}: clustering", abs(a_o - a_r) < 1e-3, f"ARI {a_o:.4f} vs {a_r:.4f}")
    else:
        print(f"  [SKIP] {tag}: parameters -- distinct optima of equal likelihood "
              f"(ARI {a_o:.4f} vs {a_r:.4f}, log-likelihood agrees to {dll:.1e})")

print()
print("=" * 88)
print("2. API SURFACE")
print("=" * 88)
X, y = make_blobs(n_samples=10_000, centers=6, n_features=8, random_state=1)
gm = TurboGaussianMixture(n_components=6, covariance_type="diag",
                          random_state=1).fit(X)
ref = GaussianMixture(n_components=6, covariance_type="diag", random_state=1).fit(X)
for attr in ("weights_", "means_", "covariances_", "precisions_",
             "precisions_cholesky_", "n_iter_", "lower_bound_", "converged_",
             "n_features_in_"):
    check(f"attribute {attr}", hasattr(gm, attr))
for attr in ("weights_", "means_", "covariances_", "precisions_cholesky_"):
    check(f"shape of {attr} matches sklearn", getattr(gm, attr).shape ==
          getattr(ref, attr).shape,
          f"{getattr(gm, attr).shape} vs {getattr(ref, attr).shape}")
P = gm.predict_proba(X)
check("predict_proba shape", P.shape == (10_000, 6), str(P.shape))
check("predict_proba rows sum to 1", np.allclose(P.sum(1), 1.0, atol=1e-5),
      f"max dev {np.abs(P.sum(1) - 1).max():.2e}")
check("predict in range", set(np.unique(gm.predict(X))) <= set(range(6)))
check("score_samples shape", gm.score_samples(X).shape == (10_000,))
check("score equals mean of score_samples",
      abs(gm.score(X) - gm.score_samples(X).mean()) < 1e-9)
Xs, ys = gm.sample(500)
check("sample shapes", Xs.shape == (500, 8) and ys.shape == (500,),
      f"{Xs.shape} {ys.shape}")
check("bic finite", np.isfinite(gm.bic(X)))
check("aic finite", np.isfinite(gm.aic(X)))
check("bic close to sklearn", abs(gm.bic(X) - ref.bic(X)) / abs(ref.bic(X)) < 1e-5,
      f"{gm.bic(X):.1f} vs {ref.bic(X):.1f}")
import inspect
_sig = set(inspect.signature(TurboGaussianMixture.__init__).parameters) - {"self"}
_got = set(TurboGaussianMixture(n_components=3).get_params())
check("get_params covers every constructor keyword", _sig == _got,
      f"missing from get_params: {sorted(_sig - _got)}; extra: {sorted(_got - _sig)}")
check("set_params round-trip, every keyword",
      all(TurboGaussianMixture(n_components=3).set_params(**{k: v}).get_params()[k] == v
          for k, v in [("tol", 1e-9), ("init_subsample", 1234), ("max_iter", 7),
                       ("reg_covar", 1e-5), ("precision", "ieee")]))
check("fit_predict equals predict after fit",
      np.array_equal(TurboGaussianMixture(n_components=6, random_state=1)
                     .fit_predict(X), gm.predict(X)))

print()
print("=" * 88)
print("3. CONTRACTS")
print("=" * 88)
for bad, kw in (("covariance_type='full'", dict(covariance_type="full")),
                ("covariance_type='tied'", dict(covariance_type="tied")),
                ("n_init=3", dict(n_init=3))):
    try:
        TurboGaussianMixture(n_components=2, **kw)
        check(f"{bad} raises", False, "no exception")
    except (ValueError, NotImplementedError) as e:
        check(f"{bad} raises", True, type(e).__name__)
try:
    TurboGaussianMixture(n_components=50).fit(X[:10])
    check("n_samples < n_components raises", False)
except ValueError:
    check("n_samples < n_components raises", True)
try:
    TurboGaussianMixture(n_components=3).predict(X)
    check("predict before fit raises", False)
except RuntimeError:
    check("predict before fit raises", True)
a = TurboGaussianMixture(n_components=5, random_state=7).fit(X)
b = TurboGaussianMixture(n_components=5, random_state=7).fit(X)
dlb = abs(a.lower_bound_ - b.lower_bound_) / abs(b.lower_bound_)
check("reproducible to float32 tolerance for fixed random_state",
      np.array_equal(a.predict(X), b.predict(X)) and dlb < 1e-4,
      f"labels identical, lower_bound_ rel {dlb:.2e}")

Xb, _ = make_blobs(n_samples=60_000, centers=6, n_features=8, cluster_std=1.0,
                   random_state=0)

sub = TurboGaussianMixture(n_components=6, random_state=0, init_subsample=5_000).fit(Xb)
full = TurboGaussianMixture(n_components=6, random_state=0, init_subsample=None).fit(Xb)
check("subsampled init runs and converges", np.isfinite(sub.lower_bound_)
      and sub.converged_, f"lower_bound_={sub.lower_bound_:.6f}")
check("subsampled init loses no fit quality vs full-data k-means",
      sub.score(Xb) > full.score(Xb) - 1e-3,
      f"subsample {sub.score(Xb):.6f} vs full {full.score(Xb):.6f}")
check("init_subsample=None still fits", np.isfinite(full.lower_bound_))

big = TurboGaussianMixture(n_components=6, random_state=0,
                           init_subsample=10_000_000).fit(Xb)

_rel = abs(big.score(Xb) - full.score(Xb)) / abs(full.score(Xb))
check("init_subsample > N behaves as init_subsample=None", _rel < 1e-5,
      f"{big.score(Xb):.6f} vs {full.score(Xb):.6f} (rel {_rel:.2e})")
r1 = TurboGaussianMixture(n_components=6, random_state=3, init_subsample=5_000).fit(Xb)
r2 = TurboGaussianMixture(n_components=6, random_state=3, init_subsample=5_000).fit(Xb)
check("subsample draw is reproducible for fixed random_state",
      np.array_equal(r1.predict(Xb), r2.predict(Xb))
      and abs(r1.lower_bound_ - r2.lower_bound_) / abs(r2.lower_bound_) < 1e-4)

for bad, exc in [(2, ValueError), (1.5, TypeError), (True, TypeError)]:
    try:
        TurboGaussianMixture(n_components=6, init_subsample=bad)
        check(f"init_subsample={bad!r} raises", False, "did not raise")
    except exc:
        check(f"init_subsample={bad!r} raises {exc.__name__}", True)
    except Exception as e:
        check(f"init_subsample={bad!r} raises {exc.__name__}", False,
              f"raised {type(e).__name__}")

print()
print("=" * 88)
print(f"RESULT: {'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES'}")
if FAIL:
    for f in FAIL:
        print("   -", f)
print("=" * 88)
sys.exit(1 if FAIL else 0)
