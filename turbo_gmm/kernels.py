"""
Triton kernels for one EM step of a diagonal (or spherical) Gaussian mixture.

"""
import torch
import triton
import triton.language as tl

# Finite sentinel, never -inf: -inf - (-inf) = NaN would poison the (m, lambda) merge.
NEG_SENTINEL: tl.constexpr = -3.0e38


# ----------------------------------------------------------------- one pass
@triton.jit
def _em_onepass(
    X_ptr, A_ptr, T_ptr, b_ptr,           # X:(N,d) rowmajor; A,T:(d,K); b:(K,)
    Nk_ptr, Sk_ptr, Tk_ptr, ll_ptr,       # Nk:(K,); Sk,Tk:(K,d); ll:(1,)
    N, d, K, n_tiles,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d
    mask_k = offs_k < K

    # Parameters are tiny (K*d); hoist them into registers once, outside the row loop.
    pmask = mask_d[:, None] & mask_k[None, :]
    A = tl.load(A_ptr + offs_d[:, None] * K + offs_k[None, :], mask=pmask, other=0.0)
    Tm = tl.load(T_ptr + offs_d[:, None] * K + offs_k[None, :], mask=pmask, other=0.0)
    b = tl.load(b_ptr + offs_k, mask=mask_k, other=0.0)

    acc_Nk = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_Sk = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    acc_Tk = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    acc_ll = tl.zeros([], dtype=tl.float32)

    for tile in range(pid, n_tiles, n_prog):
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x = tl.load(X_ptr + offs_n[:, None].to(tl.int64) * d + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x2 = x * x                                    # in REGISTERS: no Xsq in HBM

        S = (tl.dot(x, A, input_precision=PREC)
             - 0.5 * tl.dot(x2, Tm, input_precision=PREC) + b[None, :])
        S = tl.where(mask_k[None, :], S, NEG_SENTINEL)

        m = tl.max(S, axis=1)
        p = tl.exp(S - m[:, None])
        lam = tl.sum(p, axis=1)
        L = m + tl.log(lam)
        g = p / lam[:, None]                          # == exp(S - L), one div, no 2nd exp
        g = tl.where(mask_n[:, None] & mask_k[None, :], g, 0.0)

        acc_Nk += tl.sum(g, axis=0)
        gT = tl.trans(g)
        # mma-accumulator form: measured ~5% faster than `acc += tl.dot(...)`
        acc_Sk = tl.dot(gT, x, acc_Sk, input_precision=PREC)
        acc_Tk = tl.dot(gT, x2, acc_Tk, input_precision=PREC)
        acc_ll += tl.sum(tl.where(mask_n, L, 0.0))

    tl.atomic_add(Nk_ptr + offs_k, acc_Nk, mask=mask_k)
    smask = mask_k[:, None] & mask_d[None, :]
    tl.atomic_add(Sk_ptr + offs_k[:, None] * d + offs_d[None, :], acc_Sk, mask=smask)
    tl.atomic_add(Tk_ptr + offs_k[:, None] * d + offs_d[None, :], acc_Tk, mask=smask)
    tl.atomic_add(ll_ptr, acc_ll)


# ----------------------------------------------------------------- two pass
@triton.jit
def _em_pass1(
    X_ptr, A_ptr, T_ptr, b_ptr, L_ptr, ll_ptr,
    N, d, K, n_tiles, n_ktiles,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    """Pass 1: row normalizers only.  Rows outer, component tiles inner, so X is read
    once and the carried state is just (m, lambda) per row."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d
    acc_ll = tl.zeros([], dtype=tl.float32)

    for tile in range(pid, n_tiles, n_prog):
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x = tl.load(X_ptr + offs_n[:, None].to(tl.int64) * d + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x2 = x * x

        m = tl.full([BLOCK_N], NEG_SENTINEL, dtype=tl.float32)
        lam = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(n_ktiles):
            kk = t * BLOCK_K + offs_k
            mk = kk < K
            pmask = mask_d[:, None] & mk[None, :]
            A = tl.load(A_ptr + offs_d[:, None] * K + kk[None, :], mask=pmask, other=0.0)
            Tm = tl.load(T_ptr + offs_d[:, None] * K + kk[None, :], mask=pmask, other=0.0)
            b = tl.load(b_ptr + kk, mask=mk, other=0.0)
            S = (tl.dot(x, A, input_precision=PREC)
                 - 0.5 * tl.dot(x2, Tm, input_precision=PREC) + b[None, :])
            S = tl.where(mk[None, :], S, NEG_SENTINEL)
            # online-softmax merge; the sentinel makes exp(m - m_new) underflow to 0
            # on the first visit instead of producing NaN.
            mt = tl.max(S, axis=1)
            lt = tl.sum(tl.exp(S - mt[:, None]), axis=1)
            mn = tl.maximum(m, mt)
            lam = lam * tl.exp(m - mn) + lt * tl.exp(mt - mn)
            m = mn

        L = m + tl.log(lam)
        tl.store(L_ptr + offs_n, L, mask=mask_n)
        acc_ll += tl.sum(tl.where(mask_n, L, 0.0))

    tl.atomic_add(ll_ptr, acc_ll)


@triton.jit
def _em_pass2(
    X_ptr, A_ptr, T_ptr, b_ptr, L_ptr,
    Nk_ptr, Sk_ptr, Tk_ptr,
    N, d, K, n_tiles,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    """Pass 2: statistics.  COMPONENT TILE on grid dim 1, rows strided on dim 0, so the
    accumulator stays (BLOCK_K, BLOCK_D) at any K.  Costs one re-read of X per
    component tile."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    kt = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_d = offs_d < d
    mask_k = offs_k < K

    pmask = mask_d[:, None] & mask_k[None, :]
    A = tl.load(A_ptr + offs_d[:, None] * K + offs_k[None, :], mask=pmask, other=0.0)
    Tm = tl.load(T_ptr + offs_d[:, None] * K + offs_k[None, :], mask=pmask, other=0.0)
    b = tl.load(b_ptr + offs_k, mask=mask_k, other=0.0)

    acc_Nk = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_Sk = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    acc_Tk = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    for tile in range(pid, n_tiles, n_prog):
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        x = tl.load(X_ptr + offs_n[:, None].to(tl.int64) * d + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x2 = x * x
        L = tl.load(L_ptr + offs_n, mask=mask_n, other=0.0)

        S = (tl.dot(x, A, input_precision=PREC)
             - 0.5 * tl.dot(x2, Tm, input_precision=PREC) + b[None, :])
        g = tl.exp(S - L[:, None])                    # no divide; gamma in (0,1]
        g = tl.where(mask_n[:, None] & mask_k[None, :], g, 0.0)

        acc_Nk += tl.sum(g, axis=0)
        gT = tl.trans(g)
        acc_Sk = tl.dot(gT, x, acc_Sk, input_precision=PREC)
        acc_Tk = tl.dot(gT, x2, acc_Tk, input_precision=PREC)

    tl.atomic_add(Nk_ptr + offs_k, acc_Nk, mask=mask_k)
    smask = mask_k[:, None] & mask_d[None, :]
    tl.atomic_add(Sk_ptr + offs_k[:, None] * d + offs_d[None, :], acc_Sk, mask=smask)
    tl.atomic_add(Tk_ptr + offs_k[:, None] * d + offs_d[None, :], acc_Tk, mask=smask)


# ----------------------------------------------------------------- host side
def lift_params(weights, means, cov, ctype="diag"):
    """Build (A, T, b) of eq (6)-(7).  A, T are (d,K) K-minor; b is (K,)."""
    K, d = means.shape
    if ctype == "spherical":
        tau = (1.0 / cov).unsqueeze(1).expand(K, d).contiguous()
        logdet = 0.5 * d * torch.log(1.0 / cov)
    else:
        tau = 1.0 / cov
        logdet = 0.5 * torch.log(1.0 / cov).sum(1)
    A = (means * tau).T.contiguous()
    T = tau.T.contiguous()
    log2pi = torch.log(torch.tensor(2 * torch.pi, device=means.device, dtype=means.dtype))
    b = (torch.log(weights) - 0.5 * d * log2pi
         + logdet - 0.5 * (means * means * tau).sum(1)).contiguous()
    return A, T, b


def finalize(Nk, Sk, Tk, N, ctype="diag", reg_covar=1e-6):
    """The M-step of eq (5), matching sklearn's conventions (nk += 10*eps).
    """
    nk = Nk + 10 * torch.finfo(torch.float64).eps
    means = Sk / nk[:, None]
    cov = Tk / nk[:, None] - means * means + reg_covar
    if ctype == "spherical":
        cov = cov.mean(1)
    return nk / N, means, cov


def autoconfig(K, d):
    """(BLOCK_N, num_warps, num_stages) from measured A100 sweeps at N=4e6.
    """
    BLOCK_N = 64 if d <= 32 else 32
    num_warps = 4 if d <= 32 else 8
    num_stages = 1 if K >= 64 else 3
    return BLOCK_N, num_warps, num_stages


def _alloc(K, d, device):
    z = torch.zeros
    return (z(K, device=device), z(K, d, device=device), z(K, d, device=device),
            z(1, device=device))


def em_step_onepass(X, weights, means, cov, ctype="diag", reg_covar=1e-6,
                    BLOCK_N=None, BLOCK_K=None, num_programs=None,
                    num_warps=None, num_stages=None, prec="tf32x3"):
    """Algorithm 2. Requires BLOCK_K >= K. Returns (mean_ll, weights, means, cov)."""
    N, d = X.shape
    K = means.shape[0]
    _bn, _nw, _ns = autoconfig(K, d)
    BLOCK_N = BLOCK_N or _bn
    num_warps = num_warps or _nw
    num_stages = num_stages or _ns
    BLOCK_D = max(16, triton.next_power_of_2(d))
    BLOCK_K = BLOCK_K or max(16, triton.next_power_of_2(K))
    assert BLOCK_K >= K, "one-pass requires BLOCK_K >= K; use em_step_twopass"

    A, T, b = lift_params(weights, means, cov, ctype)
    Nk, Sk, Tk, ll = _alloc(K, d, X.device)
    n_tiles = triton.cdiv(N, BLOCK_N)
    if num_programs is None:
        n_sm = torch.cuda.get_device_properties(X.device).multi_processor_count
        num_programs = min(n_tiles, n_sm * 4)

    _em_onepass[(num_programs,)](
        X, A, T, b, Nk, Sk, Tk, ll, N, d, K, n_tiles,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, PREC=prec,
        num_warps=num_warps, num_stages=num_stages)

    return (ll.item() / N,) + finalize(Nk, Sk, Tk, N, ctype, reg_covar)


def em_step_twopass(X, weights, means, cov, ctype="diag", reg_covar=1e-6,
                    BLOCK_N=None, BLOCK_K=64, num_programs=None,
                    num_warps=None, num_stages=None, prec="tf32x3"):
    """Algorithm 3. Any K. Returns (mean_ll, weights, means, cov)."""
    N, d = X.shape
    K = means.shape[0]
    # BLOCK_K, not K, sets the accumulator shape here, so config on it instead.
    _bn, _nw, _ns = autoconfig(BLOCK_K, d)
    BLOCK_N = BLOCK_N or _bn
    num_warps = num_warps or _nw
    num_stages = num_stages or _ns
    BLOCK_D = max(16, triton.next_power_of_2(d))
    BLOCK_K = max(16, triton.next_power_of_2(BLOCK_K))

    A, T, b = lift_params(weights, means, cov, ctype)
    Nk, Sk, Tk, ll = _alloc(K, d, X.device)
    L = torch.empty(N, device=X.device, dtype=torch.float32)
    n_tiles = triton.cdiv(N, BLOCK_N)
    n_ktiles = triton.cdiv(K, BLOCK_K)
    if num_programs is None:
        n_sm = torch.cuda.get_device_properties(X.device).multi_processor_count
        num_programs = min(n_tiles, n_sm * 4)

    _em_pass1[(num_programs,)](
        X, A, T, b, L, ll, N, d, K, n_tiles, n_ktiles,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, PREC=prec,
        num_warps=num_warps, num_stages=num_stages)

    _em_pass2[(num_programs, n_ktiles)](
        X, A, T, b, L, Nk, Sk, Tk, N, d, K, n_tiles,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K, PREC=prec,
        num_warps=num_warps, num_stages=num_stages)

    return (ll.item() / N,) + finalize(Nk, Sk, Tk, N, ctype, reg_covar)


def em_step(X, weights, means, cov, **kw):
    K, d = means.shape
    BLOCK_D = max(16, triton.next_power_of_2(d))
    BLOCK_K = max(16, triton.next_power_of_2(K))
    if BLOCK_K <= 128 and BLOCK_K * BLOCK_D <= 4096:
        return em_step_onepass(X, weights, means, cov, **kw)
    return em_step_twopass(X, weights, means, cov, **kw)
