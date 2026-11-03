"""A drop-in replacement for ``sklearn.mixture.GaussianMixture``.

Attribute names, shapes and semantics follow scikit-learn so that downstream code
(``bic``, ``aic``, ``score_samples``, ``sample``) behaves the same way.  One
deliberate difference: computation is float32 on the GPU, so fitted parameters agree
with scikit-learn to float32 tolerance rather than bitwise.  ``tests/`` quantifies the
agreement.
"""
import math

import numpy as np
import torch

from .kernels import em_step, lift_params

_SUPPORTED = ("diag", "spherical")


class TurboGaussianMixture:
    """Gaussian mixture fitted by materialization-free EM on the GPU.
    """

    def __init__(self, n_components=1, *, covariance_type="diag", tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params="kmeans",
                 init_subsample=500_000,
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False, verbose=0,
                 device="cuda", precision="tf32x3"):
        if covariance_type not in _SUPPORTED:
            raise ValueError(
                f"covariance_type={covariance_type!r} is not supported. Turbo-GMM "
                f"implements {_SUPPORTED}; the lift of Lemma 1 as written does not "
                f"cover 'full' or 'tied'."
            )
        if n_init != 1:
            raise NotImplementedError(
                "n_init > 1 is not implemented; run the estimator repeatedly with "
                "different random_state and keep the best lower_bound_."
            )
        if init_subsample is not None:
            if not isinstance(init_subsample, (int, np.integer)) or \
                    isinstance(init_subsample, bool):
                raise TypeError("init_subsample must be an int or None, got "
                                f"{type(init_subsample).__name__}")
            if init_subsample < n_components:
                raise ValueError(
                    f"init_subsample={init_subsample} must be >= n_components="
                    f"{n_components}, or None to initialise from every row."
                )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.init_subsample = init_subsample
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.precision = precision

    # ------------------------------------------------------------------ helpers
    def _check_fitted(self):
        if not hasattr(self, "means_"):
            raise RuntimeError("This TurboGaussianMixture is not fitted yet. "
                               "Call fit() before using this method.")

    def _as_device(self, X):
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return Xt.to(self.device, non_blocking=True)

    @staticmethod
    def _kmeans_device(Xs, K, gen, n_iter=25, tol=1e-4):
        """k-means++ seeding plus Lloyd, entirely on the device.
        """
        N = Xs.shape[0]
        # Greedy k-means++, matching scikit-learn: sample n_local trial centres from
        # the D^2 distribution per step and keep the one minimising resulting inertia,
        # rather than a single draw (which measurably loses fit quality).
        n_local = 2 + int(math.log(K)) if K > 1 else 1
        first = torch.randint(N, (1,), device=Xs.device, generator=gen)
        C = Xs[first].clone()
        d2 = ((Xs - C[0]) ** 2).sum(1)
        for _ in range(1, K):
            total = d2.sum()
            if not torch.isfinite(total) or total <= 0:
                cand = torch.randint(N, (n_local,), device=Xs.device, generator=gen)
            else:
                cand = torch.multinomial(d2 / total, n_local, replacement=True,
                                         generator=gen)
            # Looped rather than batched over candidates: a batched (n_local, N, d)
            # temporary would be 2 GB at N=4e6 with init_subsample=None, while
            # n_local is only 2-5.
            best_d2, best_c = None, None
            for j in range(cand.shape[0]):
                dj = ((Xs - Xs[cand[j]]) ** 2).sum(1)
                merged = torch.minimum(d2, dj)
                inertia = merged.sum()
                if best_d2 is None or inertia < best_inertia:
                    best_inertia, best_d2, best_c = inertia, merged, cand[j]
            C = torch.cat((C, Xs[best_c].unsqueeze(0)), 0)
            d2 = best_d2

        ones = torch.ones(N, device=Xs.device, dtype=Xs.dtype)
        for _ in range(n_iter):
            lbl = ((C * C).sum(1) - 2.0 * (Xs @ C.T)).argmin(1)
            tot = torch.zeros_like(C)
            cnt = torch.zeros(K, device=Xs.device, dtype=Xs.dtype)
            tot.index_add_(0, lbl, Xs)
            cnt.index_add_(0, lbl, ones)
            # An emptied cluster keeps its previous centre rather than becoming NaN.
            Cn = torch.where((cnt > 0)[:, None], tot / cnt.clamp_min(1.0)[:, None], C)
            shift = (Cn - C).abs().max()
            C = Cn
            if shift <= tol:
                break
        return C

    def _init(self, Xg, rng):
        """Return (weights, means, variances) as GPU tensors.

        Shapes follow what the kernels expect: means is (K, d) always; variances is
        (K, d) for covariance_type="diag" and (K,) for "spherical".
        """
        N, d = Xg.shape
        K = self.n_components
        if self.means_init is not None:
            means = torch.as_tensor(np.asarray(self.means_init),
                                    dtype=torch.float32, device=self.device)
            if means.shape != (K, d):
                raise ValueError(f"means_init must have shape {(K, d)}, got "
                                 f"{tuple(means.shape)}")
        elif self.init_params == "kmeans":
            # k-means for initialisation only, on the device .
            #
            sub = self.init_subsample
            if sub is not None and N > sub:
                sidx = torch.as_tensor(np.asarray(rng.choice(N, size=int(sub),
                                                             replace=False)),
                                       device=self.device)
                Xs = Xg[sidx]
            else:
                Xs = Xg
            gen = torch.Generator(device=Xg.device)
            gen.manual_seed(int(rng.integers(2 ** 63 - 1)))
            means = self._kmeans_device(Xs, K, gen).contiguous()
        elif self.init_params in ("random_from_data", "random"):
            idx = np.asarray(rng.choice(N, size=K, replace=False))
            means = Xg[torch.as_tensor(idx, device=self.device)].clone()
        else:
            raise ValueError(f"init_params={self.init_params!r} is not supported; "
                             f"use 'kmeans', 'random_from_data' or 'random'")

        if self.weights_init is not None:
            weights = torch.as_tensor(np.asarray(self.weights_init),
                                      dtype=torch.float32, device=self.device)
        else:
            weights = torch.full((K,), 1.0 / K, dtype=torch.float32,
                                 device=self.device)

        if self.precisions_init is not None:
            prec = np.asarray(self.precisions_init, dtype=np.float64)
            var_np = 1.0 / prec
            var = torch.as_tensor(var_np, dtype=torch.float32, device=self.device)
            if self.covariance_type == "diag" and var.ndim == 1:
                var = var.unsqueeze(1).expand(K, d)
            if self.covariance_type == "spherical" and var.ndim == 2:
                var = var.mean(1)
        else:
            # scikit-learn's default: the variance of the whole data set.
            v = Xg.var(0, unbiased=False).clamp_min(self.reg_covar)
            var = (v.mean().reshape(1).expand(K).clone()
                   if self.covariance_type == "spherical"
                   else v.unsqueeze(0).expand(K, d))
        return (weights.contiguous().float(), means.contiguous().float(),
                var.contiguous().float())

    # ---------------------------------------------------------------------- fit
    def fit(self, X, y=None):
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        Xg = self._as_device(X)
        if Xg.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {tuple(Xg.shape)}")
        N, d = Xg.shape
        if N < self.n_components:
            raise ValueError(f"n_samples={N} must be >= n_components="
                             f"{self.n_components}")
        rng = np.random.default_rng(self.random_state)

        if self.warm_start and hasattr(self, "means_"):
            w, mu, va = (self._w_dev, self._mu_dev, self._va_dev)
        else:
            w, mu, va = self._init(Xg, rng)

        lower_bound = -np.inf
        self.converged_ = False
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            prev = lower_bound
            # em_step returns the mean log-likelihood AT THE INPUT parameters, which
            # is scikit-learn's `lower_bound_` convention.
            lower_bound, w, mu, va = em_step(
                Xg, w, mu, va, ctype=self.covariance_type,
                reg_covar=self.reg_covar, prec=self.precision)
            self.n_iter_ = it
            if self.verbose:
                print(f"  iter {it:4d}  mean log-likelihood {lower_bound:.8f}")
            if it > 1 and abs(lower_bound - prev) < self.tol:
                self.converged_ = True
                break

        self._w_dev, self._mu_dev, self._va_dev = w, mu, va
        self.weights_ = w.detach().cpu().numpy().astype(np.float64)
        self.means_ = mu.detach().cpu().numpy().astype(np.float64)
        cov = va.detach().cpu().numpy().astype(np.float64)
        if self.covariance_type == "spherical":
            self.covariances_ = cov if cov.ndim == 1 else cov.mean(1)
            self.precisions_cholesky_ = 1.0 / np.sqrt(self.covariances_)
            self.precisions_ = 1.0 / self.covariances_
        else:
            self.covariances_ = cov
            self.precisions_cholesky_ = 1.0 / np.sqrt(cov)
            self.precisions_ = 1.0 / cov
        self.lower_bound_ = float(lower_bound)
        self.n_features_in_ = d
        return self.predict(X)

    # ------------------------------------------------------------- inference
    def _log_prob(self, X):
        """(N, K) log joint densities log p(x_i, k), computed via the lift."""
        self._check_fitted()
        Xg = self._as_device(X)
        w = torch.as_tensor(self.weights_, dtype=torch.float32, device=self.device)
        mu = torch.as_tensor(self.means_, dtype=torch.float32, device=self.device)
        if self.covariance_type == "spherical":
            va = torch.as_tensor(self.covariances_, dtype=torch.float32,
                                 device=self.device)
            va = va.unsqueeze(1).expand(-1, Xg.shape[1]).contiguous()
        else:
            va = torch.as_tensor(self.covariances_, dtype=torch.float32,
                                 device=self.device)
        A, T, b = lift_params(w, mu, va, "diag")
        return Xg @ A - 0.5 * (Xg * Xg) @ T + b[None, :]

    def score_samples(self, X):
        S = self._log_prob(X)
        return torch.logsumexp(S, dim=1).detach().cpu().numpy().astype(np.float64)

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())

    def predict(self, X):
        S = self._log_prob(X)
        return S.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    def predict_proba(self, X):
        S = self._log_prob(X)
        R = torch.softmax(S, dim=1)
        return R.detach().cpu().numpy().astype(np.float64)

    # ------------------------------------------------------- model selection
    def _n_parameters(self):
        d, K = self.n_features_in_, self.n_components
        cov_params = K * d if self.covariance_type == "diag" else K
        return int(cov_params + K * d + K - 1)

    def bic(self, X):
        n = np.asarray(X).shape[0]
        return float(-2 * self.score(X) * n + self._n_parameters() * np.log(n))

    def aic(self, X):
        n = np.asarray(X).shape[0]
        return float(-2 * self.score(X) * n + 2 * self._n_parameters())

    def sample(self, n_samples=1):
        """Draw from the fitted mixture. Returns (X, y) as scikit-learn does."""
        self._check_fitted()
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        rng = np.random.default_rng(self.random_state)
        counts = rng.multinomial(n_samples, self.weights_)
        d = self.n_features_in_
        var = (self.covariances_ if self.covariance_type == "diag"
               else np.repeat(self.covariances_.reshape(-1, 1), d, axis=1))
        Xs = np.vstack([
            rng.normal(self.means_[k], np.sqrt(var[k]), size=(c, d))
            for k, c in enumerate(counts) if c > 0])
        ys = np.concatenate([np.full(c, k, dtype=np.int64)
                             for k, c in enumerate(counts) if c > 0])
        return Xs, ys

    # scikit-learn compatibility surface
    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in
                ("n_components", "covariance_type", "tol", "reg_covar", "max_iter",
                 "n_init", "init_params", "init_subsample", "weights_init",
                 "means_init", "precisions_init", "random_state", "warm_start",
                 "verbose", "device", "precision")}

    def set_params(self, **kw):
        for k, v in kw.items():
            if k not in self.get_params():
                raise ValueError(f"Invalid parameter {k!r}")
            setattr(self, k, v)
        return self

    def __repr__(self):
        return (f"TurboGaussianMixture(n_components={self.n_components}, "
                f"covariance_type={self.covariance_type!r})")
