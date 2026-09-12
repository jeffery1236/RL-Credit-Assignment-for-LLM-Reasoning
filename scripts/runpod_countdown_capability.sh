#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
RUN_NAME="${RUN_NAME:-countdown-capability-qwen35-4b}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-artifacts/phase2/capability/qwen3.5-4b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/runpod_experiment.sh"

case "$ACTION" in
  run|submit)
    exec "$HELPER" "$ACTION" "$RUN_NAME" "$LOCAL_OUTPUT" -- \
      python -m experiments.evaluate_countdown_capability \
      --model Qwen/Qwen3.5-4B \
      --output /workspace/results \
      --samples-per-problem 32 \
      --temperature 0.6 \
      --max-new-tokens 1024 \
      --generation-batch-size 8 \
      --device cuda
    ;;
  status|logs|wait|download|terminate)
    exec "$HELPER" "$ACTION" "$RUN_NAME"
    ;;
  -h|--help|help|"")
    cat >&2 <<EOF
Usage:
  $0 run
  $0 submit
  $0 status
  $0 logs
  $0 wait
  $0 download
  $0 terminate

Compatibility wrapper around scripts/runpod_experiment.sh.
Optional variables: RUN_NAME, LOCAL_OUTPUT, IMAGE, GPU_ID.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
