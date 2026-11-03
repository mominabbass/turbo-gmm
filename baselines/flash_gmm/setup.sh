#!/usr/bin/env bash

#   bash baselines/flash_gmm/setup.sh
#
# Flash-GMM ships as source with no released version, so we pin the commit. It runs in
# the core environment (env/setup-core.sh); no separate interpreter is needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT="57c3621"
SRC="${FLASH_GMM_SRC:-$HERE/src}"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/IBM/Flash-GMM.git "$SRC"
fi
git -C "$SRC" fetch --all --tags
git -C "$SRC" checkout "$COMMIT"
echo "Flash-GMM checked out at $COMMIT in $SRC"
echo "Export FLASH_GMM_SRC=$SRC before running experiments/figure1_latency.py"
