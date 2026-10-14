#!/usr/bin/env bash
set -euo pipefail
CLIPPER_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CLIPPER_REPO_ROOT"
exec "${PYTHON:-python3}" -m clipper_gpu.deploy "$@"
