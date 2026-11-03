#!/usr/bin/env bash
# Environment for the gmmx baseline. Separate because gmmx is a JAX package and
# installing JAX alongside torch in one environment is not reliable.
#
#   bash baselines/gmmx/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$HERE/.venv-jax}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install "jax[cuda12]==0.10.2" "gmmx==0.7" "numpy>=1.26,<3"
echo "Created $VENV"
