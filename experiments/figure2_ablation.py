"""
Mechanism ablation for Figure 2: decompose the end-to-end speedup of Algorithm 2
into the individual dataflow changes that produce it.

usage:
    python experiments/figure2_ablation.py --verify     # correctness of every rung
    python experiments/figure2_ablation.py --latency    # left panel data
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from turbo_gmm.kernels import (  # noqa: E402
    em_step_onepass, em_step_twopass, finalize, lift_params, autoconfig)

LOG2PI = math.log(2 * math.pi)


def _alg2_fits(K, d):
    """em_step's dispatch predicate, duplicated here only to label expected misses."""
    bk = max(16, triton.next_power_of_2(K))
    bd = max(16, triton.next_power_of_2(d))
    return bk <= 128 and bk * bd <= 4096


# --------------------------------------------------------------------------- #
# A2's kernel: the lifted score, written out to HBM in full.
# This is deliberately NOT fused -- it is the materializing dataflow of A0/A1
# expressed in Triton, so that it can serve as the language control.
# --------------------------------------------------------------------------- #
@triton.jit
def _score_lifted(X, A, T, b, S, N, d, K,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_D: tl.constexpr, PREC: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    nm = offs_n < N
    km = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        dm = offs_d < d
        x = tl.load(X + offs_n[:, None] * d + offs_d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0)
        a = tl.load(A + offs_d[:, None] * K + offs_k[None, :],
                    mask=dm[:, None] & km[None, :], other=0.0)
        t = tl.load(T + offs_d[:, None] * K + offs_k[None, :],
                    mask=dm[:, None] & km[None, :], other=0.0)
        acc += tl.dot(x, a, input_precision=PREC)
        acc -= 0.5 * tl.dot(x * x, t, input_precision=PREC)
    acc += tl.load(b + offs_k, mask=km, other=0.0)[None, :]
    tl.store(S + offs_n[:, None] * K + offs_k[None, :], acc,
             mask=nm[:, None] & km[None, :])


# --------------------------------------------------------------------------- #
# the five rungs.  Each returns (mean_ll, weights, means, cov).
# --------------------------------------------------------------------------- #
def _mstep_materialized(X, S, N, K, d, reg_covar):
    """Shared tail of A0/A1/A2: normalize S in HBM, then contract Gamma by GEMM.

    This is Kernels 2-5 of Algorithm 1: Gamma is a real N x K array in HBM and it is
    read three times, once per sufficient statistic.
    """
    L = torch.logsumexp(S, dim=1)
    G = torch.exp(S - L[:, None])              # the N x K materialization
    Nk = G.sum(0)
    Sk = G.T @ X
    Tk = G.T @ (X * X)
    return (L.sum().item() / N,) + finalize(Nk, Sk, Tk, N, "diag", reg_covar)


def a0_torch_pointwise(X, w, mu, va, reg_covar=1e-6, **kw):
    N, d = X.shape
    K = mu.shape[0]
    q = ((X[:, None, :] - mu[None, :, :]) ** 2 / va[None, :, :]).sum(-1)
    S = torch.log(w)[None, :] - 0.5 * (d * LOG2PI + torch.log(va).sum(1)[None, :] + q)
    return _mstep_materialized(X, S, N, K, d, reg_covar)


def a1_torch_lifted(X, w, mu, va, reg_covar=1e-6, **kw):
    N, d = X.shape
    K = mu.shape[0]
    A, T, b = lift_params(w, mu, va, "diag")       # A, T are (d, K)
    S = X @ A - 0.5 * (X * X) @ T + b[None, :]
    return _mstep_materialized(X, S, N, K, d, reg_covar)


SMEM_LIMIT = 166_912          # A100 max dynamic shared memory per block, in bytes


NUM_WARPS = 4
ACC_REGS_PER_THREAD = 32     # budget for the fp32 accumulator alone


def _tiles_for(BLOCK_D, K):
    """(BLOCK_N, BLOCK_K, num_stages) subject to BOTH shared memory and registers.

    Shared memory: the pipeline holds num_stages copies of the x tile (BLOCK_N x
    BLOCK_D) and the parameter tiles (BLOCK_D x BLOCK_K) at 4B each, bounded by
    SMEM_LIMIT. Registers: the BLOCK_N x BLOCK_K fp32 accumulator lives in registers
    at BLOCK_N*BLOCK_K/(32*NUM_WARPS) values per thread, so it is capped at
    32*NUM_WARPS*ACC_REGS_PER_THREAD -- without this a spilling A2 silently biases
    the ablation, making the A1 -> A2 rung look too small and A2 -> A3 too large.
    """
    acc_cap = 32 * NUM_WARPS * ACC_REGS_PER_THREAD
    for num_stages in (3, 2):
        for bn in (128, 64, 32):
            for bk in (64, 32, 16):
                if bk > max(16, triton.next_power_of_2(K)):
                    continue
                if bn * bk > acc_cap:
                    continue
                if num_stages * 4 * (bn + bk) * BLOCK_D <= SMEM_LIMIT:
                    return bn, bk, num_stages
    return 32, 16, 2


def a2_triton_lifted(X, w, mu, va, reg_covar=1e-6, prec="tf32x3", **kw):
    N, d = X.shape
    K = mu.shape[0]
    A, T, b = lift_params(w, mu, va, "diag")
    S = torch.empty((N, K), device=X.device, dtype=torch.float32)
    BLOCK_D = max(16, triton.next_power_of_2(d))
    BLOCK_N, BLOCK_K, num_stages = _tiles_for(BLOCK_D, K)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _score_lifted[grid](X, A, T, b, S, N, d, K,
                        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                        PREC=prec, num_warps=NUM_WARPS, num_stages=num_stages)
    return _mstep_materialized(X, S, N, K, d, reg_covar)


def a3_fused_twopass(X, w, mu, va, reg_covar=1e-6, prec="tf32x3", **kw):
    return em_step_twopass(X, w, mu, va, reg_covar=reg_covar, prec=prec)


def a4_fused_onepass(X, w, mu, va, reg_covar=1e-6, prec="tf32x3", **kw):
    return em_step_onepass(X, w, mu, va, reg_covar=reg_covar, prec=prec)


LADDER = [
    ("A1 torch lifted mat.",    a1_torch_lifted),
    ("A2 +fused score kernel",  a2_triton_lifted),
    ("A3 fused two-pass",       a3_fused_twopass),
    ("A4 fused one-pass",       a4_fused_onepass),
]
RUNGS = [("A0 torch pointwise mat.", a0_torch_pointwise)] + LADDER


# --------------------------------------------------------------------------- #
def mkparams(X, K, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, d = X.shape
    idx = torch.randperm(N, generator=g)[:K].to(X.device)
    return (torch.full((K,), 1.0 / K, device=X.device),
            X[idx].clone(),
            torch.ones(K, d, device=X.device))


def bench(fn, X, w, mu, va, reps=5):
    """min-of-reps wall clock, matching the estimator used for Figure 1."""
    try:
        fn(X, w, mu, va)
    except Exception as e:                                        # noqa: BLE001
        return None, f"{type(e).__name__}"
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(X, w, mu, va)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts), None


SHAPES = [(1_000_000, 128, 32), (1_000_000, 1024, 32), (4_000_000, 32, 32),
          (4_000_000, 64, 16), (4_000_000, 16, 128), (20_000_000, 32, 32)]


def run_one(rung_idx, N, K, d, reps=1, prec="tf32x3"):
    """`reps` EM iterations inside a profiler window. Used as an ncu target.

    `reps` exists to VALIDATE the window: if cudaProfilerStart/Stop really gates the
    region, total bytes must be proportional to `reps`. If they are not, ncu is
    counting the warm-up and the tensor allocation too, and every number is inflated.
    """
    torch.manual_seed(0)
    X = torch.randn(N, d, device="cuda", dtype=torch.float32)
    w, mu, va = mkparams(X, K)
    fn = LADDER[rung_idx][1]
    fn(X, w, mu, va)                      # warm up + JIT, outside the window
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(reps):
        fn(X, w, mu, va)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--run-one", nargs=5, type=int,
                    metavar=("RUNG","N","K","D","REPS"),
                    help="internal: profiled target, REPS EM iterations")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "ablation.jsonl"))
    args = ap.parse_args()

    if args.run_one:
        run_one(*args.run_one)
        return

    if args.verify:
        # Every rung must agree with A0 on all four returned quantities. A0 is the
        # Algorithm 1 archetype, so this checks the ablation is an ablation of
        # DATAFLOW and not of the mathematics.
        print("=" * 92)
        print("VERIFY: each rung against A0 (same EM iteration, different dataflow)")
        print("=" * 92)
        bad = 0
        for (N, K, d) in [(20_000, 32, 32), (100_000, 128, 32), (50_000, 16, 128),
                          (100_000, 512, 32)]:
            torch.manual_seed(0)
            X = torch.randn(N, d, device="cuda", dtype=torch.float32)
            w, mu, va = mkparams(X, K)
            ref = a0_torch_pointwise(X, w, mu, va)
            print(f"\n  N={N} K={K} d={d}")
            for name, fn in RUNGS:
                try:
                    got = fn(X, w, mu, va)
                except Exception as e:                            # noqa: BLE001
                    expected = (name.startswith("A4")
                                and not _alg2_fits(K, d)
                                and "shared memory" in str(e))
                    tag = "N/A (outside Alg. 2 envelope)" if expected else \
                          f"RAISED {type(e).__name__}: {e}"
                    print(f"    {name:26s} {tag}")
                    bad += (not expected)
                    continue
                dll = abs(got[0] - ref[0]) / max(abs(ref[0]), 1e-30)
                errs = [dll] + [
                    ((got[i] - ref[i]).abs().max()
                     / ref[i].abs().max().clamp_min(1e-30)).item() for i in (1, 2, 3)]
                ok = max(errs) < 2e-5
                bad += (not ok)
                print(f"    {name:26s} dLL {errs[0]:.2e}  dw {errs[1]:.2e}  "
                      f"dmu {errs[2]:.2e}  dcov {errs[3]:.2e}  "
                      f"{'PASS' if ok else 'FAIL'}")
            del X
            torch.cuda.empty_cache()
        print("\n" + ("ALL PASS" if bad == 0 else f"{bad} FAILURES"))
        sys.exit(1 if bad else 0)

    if args.latency:
        rows = []
        for (N, K, d) in SHAPES:
            torch.manual_seed(0)
            X = torch.randn(N, d, device="cuda", dtype=torch.float32)
            w, mu, va = mkparams(X, K)
            print(f"\n=== N={N:,} K={K} d={d} ===")
            for name, fn in LADDER:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ms, err = bench(fn, X, w, mu, va)
                peak = torch.cuda.max_memory_allocated() / 1e6
                rows.append(dict(N=N, K=K, d=d, rung=name,
                                 ms=ms, err=err, peak_mb=peak))
                print(f"  {name:26s} " + (f"{ms:9.3f} ms   peak {peak:9.1f} MB"
                                          if ms else f"{err}"))
            del X
            torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
