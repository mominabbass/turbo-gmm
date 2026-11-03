#!/usr/bin/env bash
# Environment for Turbo-GMM itself, for the PyTorch baseline, and for Flash-GMM.
#
# These three share one environment because they share one stack: torch 2.6.0+cu124
# with triton 3.2.0. Run this once, then `source .venv-core/bin/activate`.
#
#   bash env/setup-core.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HERE/.venv-core}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124"
"$VENV/bin/pip" install \
    "triton==3.2.0" \
    "numpy>=1.26,<3" \
    "scikit-learn==1.9.1" \
    "scipy>=1.11" \
    "mpmath>=1.3" \
    "sympy>=1.12" \
    "matplotlib>=3.8" \
    "pymupdf>=1.24"

echo
echo "Created $VENV"
echo "  source $VENV/bin/activate"
"$VENV/bin/python" - <<'PY'
import torch, triton
print(f"  torch {torch.__version__}  triton {triton.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY
