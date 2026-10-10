#!/usr/bin/env bash
set -euo pipefail

# The generic helper untars whatever the Pod produced into the directory it is
# handed. Handing it the Benchmark A artifact root would let a partial or
# unexpected archive overwrite results that cannot be regenerated, so this
# launcher downloads into an isolated staging directory and copies only the
# stratified subtree into place, verifying the Benchmark A manifest on both
# sides of that copy.

ACTION="${1:-}"
if [[ -n "$ACTION" ]]; then
  shift
fi

RUN_NAME="${RUN_NAME:-phase3-credit-estimator-pilot}"
BENCHMARK_A_ROOT="${BENCHMARK_A_ROOT:-artifacts/phase3/credit-estimator-pilot}"
BENCHMARK_A_MANIFEST="${BENCHMARK_A_MANIFEST:-$BENCHMARK_A_ROOT/benchmark_a_sha256sums.txt}"
RAW_ARCHIVE_MANIFEST_NAME="raw_artifacts_manifest.json"
COMPACT_PILOT_SUMMARY_PATH="qwen3.5-4b/phase3_credit_estimator_pilot_summary.json"
COMPACT_PILOT_REPORT_PATH="qwen3.5-4b/phase3_credit_estimator_pilot_summary.md"
STRATIFIED_SUBTREE_NAME="stratified"
STRATIFIED_DESTINATION="${STRATIFIED_DESTINATION:-$BENCHMARK_A_ROOT/qwen3.5-4b/$STRATIFIED_SUBTREE_NAME}"
PROBLEM_COUNT="${PROBLEM_COUNT:-50}"
PROJECTED_HOURS="${PROJECTED_HOURS:-}"
HOURLY_RATE_USD="${HOURLY_RATE_USD:-}"
BUDGET_CEILING_USD=50
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/runpod_experiment.sh"
RUNPOD_MAX_HOURS="${RUNPOD_MAX_HOURS:-6}"
RUNPOD_STATE_DIR="${RUNPOD_STATE_DIR:-.runpod}"
STAGING_ROOT="${STAGING_ROOT:-$RUNPOD_STATE_DIR/phase3-staging}"
PINNED_IMAGE_FILE="${PINNED_IMAGE_FILE:-$RUNPOD_STATE_DIR/image-digest.txt}"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/runpod_phase3_credit_estimator_pilot.sh preflight \
    --problem-count 50 --projected-hours HOURS --hourly-rate-usd USD
  PROJECTED_HOURS=HOURS HOURLY_RATE_USD=USD \
    scripts/runpod_phase3_credit_estimator_pilot.sh run
  scripts/runpod_phase3_credit_estimator_pilot.sh verify-benchmark-a
  scripts/runpod_phase3_credit_estimator_pilot.sh promote STAGING_DIRECTORY

`preflight` only calculates the declared projection; it never contacts RunPod.
The 50-problem Phase 3 pilot is refused when its projected cost exceeds $50.
`run` performs the same check before delegating to runpod_experiment.sh. It
requires NETWORK_VOLUME_ID so resumable Benchmark B artifacts persist.

`run` never lets the helper write into the Benchmark A artifact root. It
downloads into STAGING_ROOT/RUN_NAME and then promotes only the stratified
subtree. `verify-benchmark-a` and `promote` expose those two steps so they can
be exercised without launching a Pod.
EOF
}

parse_projection_arguments() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --problem-count) PROBLEM_COUNT="$2"; shift 2 ;;
      --projected-hours) PROJECTED_HOURS="$2"; shift 2 ;;
      --hourly-rate-usd) HOURLY_RATE_USD="$2"; shift 2 ;;
      *) echo "Unknown preflight argument: $1" >&2; usage; exit 2 ;;
    esac
  done
  if [[ ! "$PROBLEM_COUNT" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "$PROJECTED_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || [[ ! "$HOURLY_RATE_USD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Problem count, projected hours, and hourly rate must be positive numbers." >&2
    exit 2
  fi
}

preflight() {
  parse_projection_arguments "$@"
  local projected_cost
  projected_cost="$(awk "BEGIN { printf \"%.2f\", $PROJECTED_HOURS * $HOURLY_RATE_USD }")"
  if awk "BEGIN { exit !($projected_cost > $BUDGET_CEILING_USD) }"; then
    echo "Refusing Phase 3 run: projected cost: \$$projected_cost exceeds \$$BUDGET_CEILING_USD." >&2
    exit 2
  fi
  echo "Phase 3 projected cost: \$$projected_cost for $PROBLEM_COUNT problem(s)."
  echo "No Pod was launched."
}

validate_immutable_image_reference() {
  local image_reference
  if [[ -n "${IMAGE:-}" ]]; then
    image_reference="$IMAGE"
  elif [[ -s "$PINNED_IMAGE_FILE" ]]; then
    image_reference="$(tr -d '[:space:]' <"$PINNED_IMAGE_FILE")"
  else
    echo "Refusing Phase 3 run: an immutable image digest is required." >&2
    exit 2
  fi
  if [[ ! "$image_reference" =~ ^[^[:space:]@]+@sha256:[[:xdigit:]]{64}$ ]]; then
    echo "Refusing Phase 3 run: an immutable image digest is required." >&2
    exit 2
  fi
}

validate_runpod_maximum_hours() {
  if [[ ! "$RUNPOD_MAX_HOURS" =~ ^[1-6]$ ]]; then
    echo "RUNPOD_MAX_HOURS must be a whole number between 1 and 6." >&2
    exit 2
  fi
}

# Exit code 3 is reserved for Benchmark A integrity failures so a caller can
# tell them apart from the argument and budget refusals above.
#
# Hashes alone only prove the listed files are unmodified; an archive that adds
# a file next to them would pass. The path set is therefore compared too. The
# stratified destination and repository metadata live under the Benchmark A
# root by design and are excluded from the historical raw-artifact path set.
verify_benchmark_a() {
  local stage="$1"
  if [[ ! -f "$BENCHMARK_A_MANIFEST" ]]; then
    echo "Refusing Phase 3 $stage: Benchmark A manifest $BENCHMARK_A_MANIFEST is missing." >&2
    exit 3
  fi
  local manifest_directory manifest_absolute verification_status
  manifest_directory="$(cd "$(dirname "$BENCHMARK_A_MANIFEST")" && pwd)"
  manifest_absolute="$manifest_directory/$(basename "$BENCHMARK_A_MANIFEST")"

  local expected_paths observed_paths unexpected_paths
  expected_paths="$(
    sed -n 's/^[[:xdigit:]]\{64\}  //p' "$manifest_absolute" | LC_ALL=C sort
  )"
  observed_paths="$(
    cd "$BENCHMARK_A_ROOT" \
      && find . -type d -name "$STRATIFIED_SUBTREE_NAME" -prune -o -type f -print \
      | sed 's|^\./||' \
      | {
          grep -v -x -F \
            -e "$(basename "$BENCHMARK_A_MANIFEST")" \
            -e "$RAW_ARCHIVE_MANIFEST_NAME" \
            -e "$COMPACT_PILOT_SUMMARY_PATH" \
            -e "$COMPACT_PILOT_REPORT_PATH" || true
        } \
      | LC_ALL=C sort
  )"
  unexpected_paths="$(
    comm -13 \
      <(printf '%s' "$expected_paths") \
      <(printf '%s' "$observed_paths")
  )"
  if [[ -n "$unexpected_paths" ]]; then
    echo "Refusing Phase 3 $stage: Benchmark A has files the manifest does not list:" >&2
    printf '%s\n' "$unexpected_paths" | sed 's/^/  /' >&2
    exit 3
  fi

  set +e
  if command -v shasum >/dev/null 2>&1; then
    (cd "$BENCHMARK_A_ROOT" && shasum -a 256 -c --status "$manifest_absolute")
    verification_status=$?
  elif command -v sha256sum >/dev/null 2>&1; then
    (cd "$BENCHMARK_A_ROOT" && sha256sum -c --status "$manifest_absolute")
    verification_status=$?
  else
    set -e
    echo "Refusing Phase 3 $stage: no SHA-256 checker is available." >&2
    exit 3
  fi
  set -e
  if [[ "$verification_status" -ne 0 ]]; then
    echo "Refusing Phase 3 $stage: Benchmark A manifest verification failed." >&2
    exit 3
  fi
  echo "Verified Benchmark A manifest ($stage)."
}

# Copying over the destination would leave files from an earlier run that this
# one no longer produces, so the staged subtree replaces the destination whole
# through a rename.
promote_stratified_subtree() {
  local staging_directory="$1"
  local source_subtree="$staging_directory/$STRATIFIED_SUBTREE_NAME"
  verify_benchmark_a "promotion"
  if [[ ! -d "$source_subtree" ]]; then
    echo "Refusing to promote: $source_subtree is missing, so the run produced no complete stratified subtree." >&2
    exit 3
  fi
  # A downloaded archive is untrusted input; a symlink inside it would let a
  # later write land outside the destination.
  if [[ -n "$(find "$source_subtree" -type l -print -quit)" ]]; then
    echo "Refusing to promote: $source_subtree contains a symbolic link." >&2
    exit 3
  fi

  local incoming replaced
  incoming="$STRATIFIED_DESTINATION.incoming.$$"
  replaced="$STRATIFIED_DESTINATION.replaced.$$"
  rm -rf "${incoming:?}" "${replaced:?}"
  mkdir -p "$(dirname "$STRATIFIED_DESTINATION")"
  cp -R "$source_subtree" "$incoming"
  if [[ -e "$STRATIFIED_DESTINATION" ]]; then
    mv "$STRATIFIED_DESTINATION" "$replaced"
  fi
  if ! mv "$incoming" "$STRATIFIED_DESTINATION"; then
    if [[ -e "$replaced" ]]; then
      mv "$replaced" "$STRATIFIED_DESTINATION"
    fi
    echo "Refusing to promote: could not move $incoming into $STRATIFIED_DESTINATION." >&2
    exit 3
  fi
  rm -rf "${replaced:?}"
  verify_benchmark_a "post-promotion"
  echo "Promoted $source_subtree to $STRATIFIED_DESTINATION."
}

run() {
  preflight
  if [[ -z "${NETWORK_VOLUME_ID:-}" ]]; then
    echo "NETWORK_VOLUME_ID is required to persist Phase 3 checkpoints." >&2
    exit 2
  fi
  validate_runpod_maximum_hours
  validate_immutable_image_reference
  verify_benchmark_a "launch"

  if [[ -z "$STAGING_ROOT" ]]; then
    echo "STAGING_ROOT must not be empty." >&2
    exit 2
  fi
  local staging_directory helper_status
  staging_directory="$STAGING_ROOT/$RUN_NAME"
  rm -rf "${staging_directory:?}"
  mkdir -p "$staging_directory"

  export PINNED_IMAGE_FILE RUNPOD_MAX_HOURS
  set +e
  "$HELPER" run "$RUN_NAME" "$staging_directory" -- \
    python -m experiments.benchmark_credit_estimator_fidelity \
    --validation-dataset data/v1/validation.jsonl \
    --benchmark stratified \
    --output /workspace/results/stratified \
    --problem-count "$PROBLEM_COUNT" \
    --model Qwen/Qwen3.5-4B \
    --device cuda \
    --batch-size 1
  helper_status=$?
  set -e
  if [[ "$helper_status" -ne 0 ]]; then
    verify_benchmark_a "failed-run"
    echo "Phase 3 run failed with status $helper_status; staged output remains in $staging_directory and Benchmark A was not modified." >&2
    exit "$helper_status"
  fi
  promote_stratified_subtree "$staging_directory"
}

case "$ACTION" in
  preflight)
    preflight "$@"
    ;;
  verify-benchmark-a)
    if [[ "$#" -ne 0 ]]; then usage; exit 2; fi
    verify_benchmark_a "verification"
    ;;
  promote)
    if [[ "$#" -ne 1 ]]; then usage; exit 2; fi
    promote_stratified_subtree "$1"
    ;;
  run)
    if [[ "$#" -ne 0 ]]; then usage; exit 2; fi
    run
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage
    exit 2
    ;;
esac
