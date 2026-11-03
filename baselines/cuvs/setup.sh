#!/usr/bin/env bash
# Environment for the cuVS baseline.
#
#   bash baselines/cuvs/setup.sh


set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEPS="${CUVS_DEPS:-$HERE/.deps}"
mkdir -p "$DEPS"
echo "Place the archived RAPIDS nightly wheels (libcuvs_cu12==26.12.0a14 and its"
echo "dependencies) in: $DEPS"
echo "Then: CUVS_DEPS=$DEPS bash $HERE/build.sh"
