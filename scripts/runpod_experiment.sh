#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
if [[ -n "$ACTION" ]]; then
  shift
fi

RUNPOD_STATE_DIR="${RUNPOD_STATE_DIR:-.runpod}"
BUILD_TAG="${BUILD_TAG:-us-central1-docker.pkg.dev/gen-lang-client-0367367639/cloud-run-source-deploy/rl-research-runpod:latest}"
# `build` records the digest here and `submit` prefers it, because a mutable
# tag such as :latest can be served stale from a host or registry cache, which
# silently runs old code on a paid Pod.
PINNED_IMAGE_FILE="${PINNED_IMAGE_FILE:-$RUNPOD_STATE_DIR/image-digest.txt}"
IMAGE="${IMAGE:-}"
GPU_ID="${GPU_ID:-NVIDIA A40}"
GPU_COUNT="${GPU_COUNT:-1}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-50}"
RUNPOD_MAX_HOURS="${RUNPOD_MAX_HOURS:-6}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-21600}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/runpod_experiment.sh build
  scripts/runpod_experiment.sh run RUN_NAME LOCAL_OUTPUT -- COMMAND [ARG...]
  scripts/runpod_experiment.sh submit RUN_NAME LOCAL_OUTPUT -- COMMAND [ARG...]
  scripts/runpod_experiment.sh status RUN_NAME
  scripts/runpod_experiment.sh logs RUN_NAME
  scripts/runpod_experiment.sh wait RUN_NAME
  scripts/runpod_experiment.sh download RUN_NAME
  scripts/runpod_experiment.sh terminate RUN_NAME

The command must write artifacts beneath /workspace/results. A successful
`run` downloads and extracts that complete directory, then terminates the Pod.
A failed run downloads partial artifacts and leaves the Pod running.

Optional environment variables:
  IMAGE=registry/image:tag     # overrides the digest recorded by `build`
  BUILD_TAG=registry/image:tag # tag that `build` pushes to
  GPU_ID="NVIDIA A40"
  GPU_COUNT=1
  CLOUD_TYPE=SECURE
  CONTAINER_DISK_GB=50
  RUNPOD_MAX_HOURS=6
  WAIT_TIMEOUT_SECONDS=21600
  RUNPOD_STATE_DIR=.runpod
  NETWORK_VOLUME_ID=volume-id  # persists /workspace across named runs
  DATA_CENTER_IDS=EU-RO-1      # required data center when attaching a volume
EOF
}

validate_run_name() {
  local run_name="$1"
  if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run name '$run_name'; use letters, numbers, dots, underscores, or hyphens." >&2
    exit 2
  fi
}

state_path() {
  local run_name="$1"
  validate_run_name "$run_name"
  printf '%s/%s.json\n' "$RUNPOD_STATE_DIR" "$run_name"
}

state_field() {
  local run_name="$1"
  local field="$2"
  local path
  path="$(state_path "$run_name")"
  if [[ ! -f "$path" ]]; then
    echo "No tracked Pod for '$run_name'. Run '$0 submit ...' first." >&2
    exit 2
  fi
  python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$path" "$field"
}

pod_id_for() {
  state_field "$1" pod_id
}

# Resolves the image a Pod should run: an explicit IMAGE wins, then the digest
# recorded by the last build, then the mutable build tag.
resolve_image() {
  if [[ -n "$IMAGE" ]]; then
    printf '%s\n' "$IMAGE"
  elif [[ -s "$PINNED_IMAGE_FILE" ]]; then
    tr -d '[:space:]' <"$PINNED_IMAGE_FILE"
    printf '\n'
  else
    printf '%s\n' "$BUILD_TAG"
  fi
}

build_image() {
  gcloud builds submit \
    --config cloudbuild.runpod.yaml \
    --substitutions "_IMAGE=$BUILD_TAG" \
    .

  local digest
  digest="$(
    gcloud artifacts docker images describe "$BUILD_TAG" \
      --format 'value(image_summary.digest)'
  )"
  if [[ -z "$digest" ]]; then
    echo "Could not resolve a digest for '$BUILD_TAG'." >&2
    exit 1
  fi
  mkdir -p "$RUNPOD_STATE_DIR"
  printf '%s@%s\n' "${BUILD_TAG%:*}" "$digest" >"$PINNED_IMAGE_FILE"
  echo "Pinned image: $(cat "$PINNED_IMAGE_FILE")"
}

submit() {
  if [[ "$#" -lt 4 ]]; then
    usage
    exit 2
  fi
  local run_name="$1"
  local local_output="$2"
  shift 2
  validate_run_name "$run_name"
  if [[ "$1" != "--" ]]; then
    echo "Expected '--' before the experiment command." >&2
    exit 2
  fi
  shift
  if [[ "$#" -eq 0 ]]; then
    echo "Experiment command must not be empty." >&2
    exit 2
  fi

  local command command_b64 environment terminate_after response pod_id path
  local resolved_image
  resolved_image="$(resolve_image)"
  local -a volume_args=() placement_args=()
  command="$(
    python -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@"
  )"
  command_b64="$(
    RUNPOD_COMMAND="$command" python -c \
      'import base64, os; print(base64.b64encode(os.environ["RUNPOD_COMMAND"].encode()).decode())'
  )"
  environment="$(
    RUNPOD_COMMAND_B64="$command_b64" python -c '
import json
import os
print(json.dumps({
    "RUNPOD_COMMAND_B64": os.environ["RUNPOD_COMMAND_B64"],
    "RUNPOD_RESULTS_DIR": "/workspace/results",
}))
'
  )"
  terminate_after="$(
    RUNPOD_MAX_HOURS="$RUNPOD_MAX_HOURS" python -c '
import datetime
import os
hours = int(os.environ["RUNPOD_MAX_HOURS"])
deadline = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=hours)
print(deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))
'
  )"
  if [[ -n "${NETWORK_VOLUME_ID:-}" ]]; then
    volume_args=(
      --network-volume-id "$NETWORK_VOLUME_ID"
      --volume-mount-path /workspace
    )
  fi
  # A network volume only attaches to Pods in its own data center.
  if [[ -n "${DATA_CENTER_IDS:-}" ]]; then
    placement_args=(--data-center-ids "$DATA_CENTER_IDS")
  fi
  response="$(
    runpodctl pod create \
      --name "$run_name" \
      --image "$resolved_image" \
      --gpu-id "$GPU_ID" \
      --gpu-count "$GPU_COUNT" \
      --cloud-type "$CLOUD_TYPE" \
      --container-disk-in-gb "$CONTAINER_DISK_GB" \
      --ports 8000/http \
      --terminate-after "$terminate_after" \
      --env "$environment" \
      ${volume_args[@]+"${volume_args[@]}"} \
      ${placement_args[@]+"${placement_args[@]}"} \
      --output json
  )"
  pod_id="$(
    python -c 'import json, sys; print(json.load(sys.stdin)["id"])' <<<"$response"
  )"

  mkdir -p "$RUNPOD_STATE_DIR"
  path="$(state_path "$run_name")"
  RUN_NAME="$run_name" \
  POD_ID="$pod_id" \
  LOCAL_OUTPUT="$local_output" \
  RUNPOD_COMMAND="$command" \
  RUNPOD_IMAGE="$resolved_image" \
  STATE_PATH="$path" \
    python -c '
import json
import os
from pathlib import Path

state = {
    "command": os.environ["RUNPOD_COMMAND"],
    "image": os.environ["RUNPOD_IMAGE"],
    "local_output": os.environ["LOCAL_OUTPUT"],
    "pod_id": os.environ["POD_ID"],
    "run_name": os.environ["RUN_NAME"],
}
Path(os.environ["STATE_PATH"]).write_text(
    json.dumps(state, sort_keys=True, indent=2) + "\n"
)
'
  echo "Created Pod $pod_id for '$run_name' on $GPU_ID."
}

status() {
  local run_name="$1"
  runpodctl pod get "$(pod_id_for "$run_name")"
}

logs() {
  local run_name="$1"
  runpodctl pod logs "$(pod_id_for "$run_name")" --follow
}

download() {
  local run_name="$1"
  local pod_id local_output status_path archive_path result_url exit_code
  pod_id="$(pod_id_for "$run_name")"
  local_output="$(state_field "$run_name" local_output)"
  mkdir -p "$local_output" "$RUNPOD_STATE_DIR"
  status_path="$RUNPOD_STATE_DIR/$run_name.status.json"
  archive_path="$RUNPOD_STATE_DIR/$run_name.results.tar.gz"
  result_url="https://${pod_id}-8000.proxy.runpod.net"

  curl --fail --location --retry 30 --retry-delay 10 --retry-all-errors \
    "$result_url/status.json" --output "$status_path"
  curl --fail --location --retry 30 --retry-delay 10 --retry-all-errors \
    "$result_url/experiment-results.tar.gz" --output "$archive_path"
  tar -xzf "$archive_path" -C "$local_output"
  exit_code="$(
    python -c 'import json, sys; print(json.load(open(sys.argv[1]))["exit_code"])' \
      "$status_path"
  )"
  echo "Downloaded results for '$run_name' to $local_output (experiment exit $exit_code)."
  return "$exit_code"
}

terminate() {
  local run_name="$1"
  local pod_id path
  pod_id="$(pod_id_for "$run_name")"
  path="$(state_path "$run_name")"
  runpodctl pod delete "$pod_id"
  rm -f "$path" \
    "$RUNPOD_STATE_DIR/$run_name.status.json" \
    "$RUNPOD_STATE_DIR/$run_name.results.tar.gz"
  echo "Terminated Pod $pod_id for '$run_name'."
}

wait_for_results() {
  local run_name="$1"
  local pod_id deadline recent_logs
  pod_id="$(pod_id_for "$run_name")"
  deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  echo "Waiting for '$run_name'. Follow logs with: $0 logs $run_name"
  while (( SECONDS < deadline )); do
    recent_logs="$(runpodctl pod logs "$pod_id" --tail 100 2>/dev/null || true)"
    if [[ "$recent_logs" == *"RESULT_SERVER_READY status="* ]]; then
      if download "$run_name"; then
        terminate "$run_name"
        return 0
      fi
      echo "Experiment failed; partial results were downloaded and the Pod remains running." >&2
      return 1
    fi
    if [[ "$recent_logs" == *"CUDA out of memory"* ]] \
      || [[ "$recent_logs" == *"No space left on device"* ]] \
      || [[ "$recent_logs" == *"RUNPOD_COMMAND_B64 is required"* ]]; then
      echo "$recent_logs" >&2
      echo "Pod reported an infrastructure error; leaving it running." >&2
      return 1
    fi
    sleep 15
  done
  echo "Timed out waiting for '$run_name'; leaving the Pod running." >&2
  return 1
}

require_run_name() {
  if [[ "$#" -ne 1 ]]; then
    usage
    exit 2
  fi
  validate_run_name "$1"
}

case "$ACTION" in
  build)
    if [[ "$#" -ne 0 ]]; then usage; exit 2; fi
    build_image
    ;;
  submit)
    submit "$@"
    ;;
  run)
    if [[ "$#" -lt 4 ]]; then usage; exit 2; fi
    run_name="$1"
    submit "$@"
    wait_for_results "$run_name"
    ;;
  status)
    require_run_name "$@"
    status "$1"
    ;;
  logs)
    require_run_name "$@"
    logs "$1"
    ;;
  wait)
    require_run_name "$@"
    wait_for_results "$1"
    ;;
  download)
    require_run_name "$@"
    download "$1"
    ;;
  terminate)
    require_run_name "$@"
    terminate "$1"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
