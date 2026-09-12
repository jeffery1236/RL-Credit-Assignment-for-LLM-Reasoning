#!/usr/bin/env bash
set -euo pipefail

RUNPOD_RESULTS_DIR="${RUNPOD_RESULTS_DIR:-/workspace/results}"
RUNPOD_SERVER_DIR="${RUNPOD_SERVER_DIR:-/workspace/result-server}"
RUNPOD_SERVE_RESULTS="${RUNPOD_SERVE_RESULTS:-1}"
export RUNPOD_RESULTS_DIR

if [[ -z "${RUNPOD_COMMAND_B64:-}" ]]; then
  echo "RUNPOD_COMMAND_B64 is required." >&2
  exit 2
fi

command="$(
  python -c \
    'import base64, os; print(base64.b64decode(os.environ["RUNPOD_COMMAND_B64"]).decode())'
)"
mkdir -p "$RUNPOD_RESULTS_DIR" "$RUNPOD_SERVER_DIR"

set +e
# Not a login shell: /etc/profile would reset PATH and hide the image venv.
# Output is teed into the results directory so a failed run downloads the
# traceback alongside its partial artifacts; a bare exit code is not
# diagnosable once the Pod is gone.
bash -c "$command" 2>&1 | tee "$RUNPOD_RESULTS_DIR/experiment.log"
experiment_exit_code="${PIPESTATUS[0]}"
set -e

RUNPOD_EXECUTED_COMMAND="$command" \
RUNPOD_EXIT_CODE="$experiment_exit_code" \
RUNPOD_SERVER_DIR="$RUNPOD_SERVER_DIR" \
  python -c '
import json
import os
from pathlib import Path

status = {
    "command": os.environ["RUNPOD_EXECUTED_COMMAND"],
    "exit_code": int(os.environ["RUNPOD_EXIT_CODE"]),
}
Path(os.environ["RUNPOD_SERVER_DIR"]).joinpath("status.json").write_text(
    json.dumps(status, sort_keys=True, indent=2) + "\n"
)
'
tar -czf "$RUNPOD_SERVER_DIR/experiment-results.tar.gz" \
  -C "$RUNPOD_RESULTS_DIR" .

echo "RESULT_SERVER_READY status=$experiment_exit_code"
if [[ "$RUNPOD_SERVE_RESULTS" == "0" ]]; then
  exit "$experiment_exit_code"
fi

exec python -m http.server 8000 \
  --bind 0.0.0.0 \
  --directory "$RUNPOD_SERVER_DIR"
