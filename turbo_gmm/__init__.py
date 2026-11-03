"""Turbo-GMM: materialization-free Gaussian mixture EM.

Public API
----------
em_step(X, weights, means, cov)          one fused EM iteration, auto-dispatched
em_step_onepass(...)                     Algorithm 2 (one sweep of the component axis)
em_step_twopass(...)                     Algorithm 3 (general K)
TurboGaussianMixture                     drop-in replacement for
                                         sklearn.mixture.GaussianMixture
"""
from .kernels import (                                              
    em_step,
    em_step_onepass,
    em_step_twopass,
    lift_params,
    finalize,
    autoconfig,
)

__all__ = [
    "em_step",
    "em_step_onepass",
    "em_step_twopass",
    "lift_params",
    "finalize",
    "autoconfig",
    "TurboGaussianMixture",
]

__version__ = "1.0.0"


def __getattr__(name):
    # TurboGaussianMixture is imported lazily so that `import turbo_gmm` does not
    # require scikit-learn when only the kernels are wanted.
    if name == "TurboGaussianMixture":
        from .sklearn_api import TurboGaussianMixture
        return TurboGaussianMixture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
