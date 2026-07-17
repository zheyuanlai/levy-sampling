#!/usr/bin/env bash
# Bounded JCP production launcher. Examples:
#   ./run_production.sh --gpus 4 --max-concurrent 1
#   ./run_production.sh --gpus 0,1 --max-concurrent 2
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python "$HERE/launch_production.py" "$@"
