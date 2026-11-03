"""
Analytic sanity check on tiny matrices.

    analytic     the closed form, evaluated with mpmath at 60 decimal digits
    ours         turbo_gmm.kernels, both em_step_onepass (Alg. 2) and
                 em_step_twopass (Alg. 3), fp32, at prec="ieee" and "tf32x3"
    pytorch      Algorithm 1 (materialise the N x K responsibilities) in torch,
                 float64 and float32
    sklearn      GaussianMixture(covariance_type="diag", max_iter=1) with all
                 three inits pinned, float64

usage:  python tests/test_analytic_tiny.py
"""
import math
import os
import sys

import numpy as np
import torch
from mpmath import mp, mpf, exp as mexp, log as mlog, tanh as mtanh, pi as mpi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from turbo_gmm.kernels import em_step, em_step_onepass, em_step_twopass  # noqa: E402

mp.dps = 60
FAIL = []

# --------------------------------------------------------------------------- #
# the closed form, at 60 digits
# --------------------------------------------------------------------------- #


def analytic(X, pi, mu, tau, reg):
    """One EM step, evaluated exactly.  Pure python + mpmath, no array library."""
    N, d = len(X), len(X[0])
    K = len(mu)
    X = [[mpf(v) for v in row] for row in X]
    pi = [mpf(v) for v in pi]
    mu = [[mpf(v) for v in row] for row in mu]
    tau = [[mpf(v) for v in row] for row in tau]
    reg = mpf(reg)

    half_log_2pi_d = mpf(d) * mlog(2 * mpi) / 2
    S = []                                        # (N, K) scores
    for n in range(N):
        row = []
        for k in range(K):
            quad = sum((X[n][j] - mu[k][j]) ** 2 / tau[k][j] for j in range(d))
            logdet = sum(mlog(tau[k][j]) for j in range(d))
            row.append(mlog(pi[k]) - half_log_2pi_d - logdet / 2 - quad / 2)
        S.append(row)

    L, G = [], []                                 # normaliser and responsibilities
    for n in range(N):
        m = max(S[n])
        tot = sum(mexp(S[n][k] - m) for k in range(K))
        Ln = m + mlog(tot)
        L.append(Ln)
        G.append([mexp(S[n][k] - Ln) for k in range(K)])

    nk = [sum(G[n][k] for n in range(N)) for k in range(K)]
    sk = [[sum(G[n][k] * X[n][j] for n in range(N)) for j in range(d)] for k in range(K)]
    tk = [[sum(G[n][k] * X[n][j] ** 2 for n in range(N)) for j in range(d)]
          for k in range(K)]

    w2 = [nk[k] / N for k in range(K)]
    mu2 = [[sk[k][j] / nk[k] for j in range(d)] for k in range(K)]
    tau2 = [[tk[k][j] / nk[k] - mu2[k][j] ** 2 + reg for j in range(d)] for k in range(K)]
    ll = sum(L) / N
    return ll, w2, mu2, tau2


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #


def em_pytorch(X, pi, mu, tau, reg, dtype, lifted=False):
    """Algorithm 1 in PyTorch: materialise the N x K responsibility matrix.

    `lifted` selects how the score is evaluated:
      False  centered,  sum_j (x_j - mu_j)^2 / tau_j        -- what sklearn/PyTorch do
      True   lifted,    S = [x^2, x] @ [-1/(2 tau), mu/tau]^T + 1 b^T
    Both are algebraically identical (Lemma 1) and differ only in rounding. The
    lifted row is the control for our kernels, which evaluate the lift.
    """
    X = torch.tensor(X, dtype=dtype)
    pi = torch.tensor(pi, dtype=dtype)
    mu = torch.tensor(mu, dtype=dtype)
    tau = torch.tensor(tau, dtype=dtype)
    N, d = X.shape
    if lifted:
        Xt = torch.cat([X * X, X], 1)                          # (N, 2d)
        Th = torch.cat([-0.5 / tau, mu / tau], 1)               # (K, 2d)
        b = torch.log(pi) - 0.5 * (d * math.log(2 * math.pi)
                                   + torch.log(tau).sum(1) + (mu * mu / tau).sum(1))
        S = Xt @ Th.T + b[None, :]
    else:
        q = ((X[:, None, :] - mu[None, :, :]) ** 2 / tau[None, :, :]).sum(-1)
        S = torch.log(pi)[None, :] - 0.5 * (
            d * math.log(2 * math.pi) + torch.log(tau).sum(1)[None, :] + q)
    L = torch.logsumexp(S, dim=1)
    g = torch.exp(S - L[:, None])
    nk = g.sum(0)
    mu2 = (g.T @ X) / nk[:, None]
    tau2 = (g.T @ (X * X)) / nk[:, None] - mu2 ** 2 + reg
    return L.mean().item(), (nk / N), mu2, tau2


def em_sklearn(X, pi, mu, tau, reg):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=len(mu), covariance_type="diag",
                         max_iter=1, n_init=1, reg_covar=reg, tol=0.0,
                         init_params="random", random_state=0,
                         weights_init=np.asarray(pi, dtype=np.float64),
                         means_init=np.asarray(mu, dtype=np.float64),
                         precisions_init=1.0 / np.asarray(tau, dtype=np.float64))
    gm.fit(np.asarray(X, dtype=np.float64))
    return gm.lower_bound_, gm.weights_, gm.means_, gm.covariances_


def em_ours(X, pi, mu, tau, reg, fn, prec):
    Xc = torch.tensor(X, dtype=torch.float32, device="cuda")
    kw = dict(reg_covar=reg, prec=prec)
    out = fn(Xc,
             torch.tensor(pi, dtype=torch.float32, device="cuda"),
             torch.tensor(mu, dtype=torch.float32, device="cuda"),
             torch.tensor(tau, dtype=torch.float32, device="cuda"), **kw)
    return out[0], out[1], out[2], out[3]


# --------------------------------------------------------------------------- #
# error measurement
# --------------------------------------------------------------------------- #


def errs(got, ref):
    """(max abs err, max rel err) of `got` against the 60-digit `ref`.

    Relative error is taken only over reference entries with |ref| > 1e-9; for
    entries at or near zero a relative error is not defined and the absolute
    error is the honest measure.  Both are returned so neither can hide.
    """
    g = np.asarray(got.detach().cpu() if torch.is_tensor(got) else got,
                   dtype=np.float64).ravel()
    r = np.array([float(v) for v in np.asarray(ref, dtype=object).ravel()])
    assert g.shape == r.shape, (g.shape, r.shape)
    a = np.abs(g - r)
    big = np.abs(r) > 1e-9
    rel = (a[big] / np.abs(r[big])).max() if big.any() else 0.0
    return a.max(), rel


def flat(x):
    return np.asarray(x, dtype=object).ravel()


# --------------------------------------------------------------------------- #
# cases.  reg is chosen per case: 0 wherever the analytic covariance is strictly
# positive, and nonzero only where it is not (a zero covariance makes sklearn
# refuse to fit, and makes a relative error meaningless).
# --------------------------------------------------------------------------- #
CASES = [
    dict(name="A 2x2  K=2 symmetric",
         X=[[-1., -1.], [1., 1.]], pi=[.5, .5],
         mu=[[-1., -1.], [1., 1.]], tau=[[1., 1.], [1., 1.]], reg=0.0),
    dict(name="B 3x2  K=2 asymmetric",
         X=[[0., 0.], [1., 0.], [3., 4.]], pi=[.25, .75],
         mu=[[0., 0.], [2., 2.]], tau=[[1., 2.], [3., .5]], reg=0.0),
    dict(name="C 4x3  K=3",
         X=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
         pi=[1 / 3, 1 / 3, 1 / 3],
         mu=[[1., 0., 0.], [0., 1., 0.], [.5, .5, .5]],
         tau=[[1., 1., 1.], [2., .5, 1.], [.25, .25, .25]], reg=1e-3),
    dict(name="D 10x10 K=2",
         X=[[float((i * 7 + j * 3) % 5) - 2. for j in range(10)] for i in range(10)],
         pi=[.4, .6],
         mu=[[0.] * 10, [1.] * 10],
         tau=[[1.] * 10, [2.] * 10], reg=0.0),
    dict(name="E 10x4  K=3",
         X=[[math.sin(i + 1.) * (j + 1) / 3. for j in range(4)] for i in range(10)],
         pi=[.2, .3, .5],
         mu=[[-1., 0., 1., 0.], [0., 0., 0., 0.], [1., 1., -1., .5]],
         tau=[[.5, 1., 1.5, 2.], [1., 1., 1., 1.], [2., .5, .5, 1.]], reg=0.0),
    dict(name="F 5x3   K=1 (mean/var)",
         X=[[1., 2., 3.], [2., 4., 6.], [3., 6., 9.], [0., 1., 1.], [4., 4., 4.]],
         pi=[1.], mu=[[0., 0., 0.]], tau=[[1., 1., 1.]], reg=0.0),
    dict(name="G 1x2   K=2 (single row)",
         X=[[2., -1.]], pi=[.5, .5],
         mu=[[0., 0.], [1., 1.]], tau=[[1., 1.], [1., 1.]], reg=1e-2),
]


print("=" * 100)
print("CASE A -- closed form derived by hand, checked against the 60-digit reference")
print("=" * 100)
t2 = mtanh(2)
sym = dict(ll=-mlog(2 * mpi) - mlog(2) + mlog(1 + mexp(-4)),
           w=[mpf(1) / 2, mpf(1) / 2],
           mu=[[-t2, -t2], [t2, t2]],
           tau=[[1 - t2 ** 2] * 2] * 2)
a_ll, a_w, a_mu, a_tau = analytic(CASES[0]["X"], CASES[0]["pi"], CASES[0]["mu"],
                                 CASES[0]["tau"], CASES[0]["reg"])
d_sym = max([abs(sym["ll"] - a_ll)]
            + [abs(x - y) for x, y in zip(flat(sym["w"]), flat(a_w))]
            + [abs(x - y) for x, y in zip(flat(sym["mu"]), flat(a_mu))]
            + [abs(x - y) for x, y in zip(flat(sym["tau"]), flat(a_tau))])
print(f"  pi'   = 1/2                    = {float(sym['w'][0]):.17f}")
print(f"  mu'_2 = tanh(2)                = {float(t2):.17f}")
print(f"  tau'  = sech^2(2) = 1-tanh^2 2 = {float(1 - t2 ** 2):.17f}")
print(f"  meanLL= -log(2pi)-log2+log(1+e^-4) = {float(sym['ll']):.17f}")
print(f"  max |symbolic - 60-digit numeric| = {float(d_sym):.3e}")
if d_sym > mpf("1e-50"):
    FAIL.append("case A symbolic vs numeric reference")

# --------------------------------------------------------------------------- #
# every case, every implementation
# --------------------------------------------------------------------------- #
IMPLS = [
    ("ours  Alg2 ieee",   lambda c: em_ours(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], em_step_onepass, "ieee")),
    ("ours  Alg2 tf32x3", lambda c: em_ours(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], em_step_onepass, "tf32x3")),
    ("ours  Alg3 ieee",   lambda c: em_ours(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], em_step_twopass, "ieee")),
    ("ours  Alg3 tf32x3", lambda c: em_ours(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], em_step_twopass, "tf32x3")),
    ("ours  dispatch",    lambda c: em_ours(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], em_step, "tf32x3")),
    ("torch fp64 center",  lambda c: em_pytorch(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], torch.float64, False)),
    ("torch fp64 lifted",  lambda c: em_pytorch(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], torch.float64, True)),
    ("torch fp32 center",  lambda c: em_pytorch(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], torch.float32, False)),
    ("torch fp32 lifted",  lambda c: em_pytorch(c["X"], c["pi"], c["mu"], c["tau"], c["reg"], torch.float32, True)),
    ("sklearn fp64",       lambda c: em_sklearn(c["X"], c["pi"], c["mu"], c["tau"], c["reg"])),
]

TOL_FP32, TOL_FP64 = 2.0e-6, 1e-12


def tol_for(name, kappa):
    if "fp64" in name or "sklearn" in name:
        return TOL_FP64
    return TOL_FP32 * max(1.0, kappa)


results = {}
for c in CASES:
    ref = analytic(c["X"], c["pi"], c["mu"], c["tau"], c["reg"])
    kappa = max([2 * float(m) ** 2 / float(t)
                 for mk, tk in zip(ref[2], ref[3]) for m, t in zip(mk, tk)
                 if float(t) > 1e-12] or [1.0])
    c["kappa"] = kappa
    print()
    print("=" * 100)
    print(f"{c['name']}   N={len(c['X'])} d={len(c['X'][0])} K={len(c['mu'])} "
          f"reg_covar={c['reg']:g}   cancellation gain kappa=max 2mu'^2/tau' = "
          f"{kappa:.1f}x  ->  fp32 tol {TOL_FP32 * max(1.0, kappa):.1e}")
    print("=" * 100)
    print(f"  {'implementation':18s} {'meanLL abs':>12s} {'pi rel':>11s} "
          f"{'mu rel':>11s} {'tau rel':>11s} {'worst rel':>11s}  verdict")
    for name, fn in IMPLS:
        try:
            got = fn(c)
        except Exception as e:                                   # noqa: BLE001
            # sklearn refuses N < n_components on input validation. That is an API
            # precondition, not a numerical disagreement, so it is N/A and not a
            # failure. Anything else is a real failure.
            if "minimum of" in str(e) or "n_samples" in str(e):
                print(f"  {name:18s} {'N/A':>12s}  (declines this input: {e.__class__.__name__})")
            else:
                print(f"  {name:18s} {'RAISED':>12s}  {type(e).__name__}: {e}")
                FAIL.append(f"{c['name']} / {name}: {type(e).__name__}: {e}")
            continue
        ll_a = abs(float(got[0]) - float(ref[0]))
        ll_r = ll_a / abs(float(ref[0])) if abs(float(ref[0])) > 1e-9 else 0.0
        e_w, e_mu, e_t = (errs(got[1], ref[1]), errs(got[2], ref[2]),
                          errs(got[3], ref[3]))
        worst = max(ll_r, e_w[1], e_mu[1], e_t[1])
        worst_abs = max(ll_a, e_w[0], e_mu[0], e_t[0])
        tol = tol_for(name, kappa)
        ok = worst <= tol and worst_abs <= max(tol, 1e-9)
        if not ok:
            FAIL.append(f"{c['name']} / {name}: worst rel {worst:.2e} > {tol:.0e}")
        print(f"  {name:18s} {ll_a:12.3e} {e_w[1]:11.3e} {e_mu[1]:11.3e} "
              f"{e_t[1]:11.3e} {worst:11.3e}  {'PASS' if ok else 'FAIL'}")
        results[(c["name"], name)] = (got, worst, worst_abs)

    
    ours = results.get((c["name"], "ours  Alg2 ieee"))
    skl = results.get((c["name"], "torch fp64 center"))
    if ours and skl:
        dm = np.abs(np.asarray(ours[0][2].detach().cpu(), dtype=np.float64)
                    - np.asarray(skl[0][2], dtype=np.float64)).max()
        print(f"  {'ours vs fp64 ref':18s} max |dmu| = {dm:.3e}")
    
    lif = results.get((c["name"], "torch fp32 lifted"))
    if ours and lif:
        for q, lab in ((1, "pi"), (2, "mu"), (3, "tau")):
            g = np.asarray(ours[0][q].detach().cpu(), dtype=np.float64)
            h = np.asarray(lif[0][q].detach().cpu(), dtype=np.float64)
            print(f"  {'ours vs fp32 lift':18s} max |d{lab}| = "
                  f"{np.abs(g - h).max():.3e}")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURES")
    for f in FAIL:
        print("   -", f)
else:
    print("RESULT: ALL PASS -- every implementation matches the closed form within "
          "its arithmetic")
print("=" * 100)
sys.exit(1 if FAIL else 0)
