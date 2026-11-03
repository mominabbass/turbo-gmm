#!/usr/bin/env bash
# Shared environment for the three baselines that require a newer torch than
# Turbo-GMM does: scikit-learn, torchgmm and pomegranate.
#   bash baselines/env/setup-torch214.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HERE/.venv-torch214}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.14.0+cu130"
"$VENV/bin/pip" install \
    "scikit-learn==1.9.1" \
    "torchgmm==0.1.4" \
    "pomegranate==1.1.2" \
    "numpy>=1.26,<3" "scipy>=1.11"

echo "Created $VENV"
echo "  scikit-learn needs SCIPY_ARRAY_API=1 for its GPU path; the harness sets it."
