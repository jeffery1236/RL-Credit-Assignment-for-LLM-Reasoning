#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "Set PROJECT_ID to a billed Google Cloud project." >&2
  exit 2
fi

REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-countdown-capability}"
JOB_NAME="${JOB_NAME:-countdown-capability-qwen35-4b}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-countdown-capability}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
RESULT_OBJECT="qwen3.5-4b/countdown_capability_results.json"
IMAGE_URI="${IMAGE_URI:-}"

deploy() {
  gcloud config set project "${PROJECT_ID}"
  gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com

  if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${BUCKET}" \
      --location="$REGION" \
      --uniform-bucket-level-access
  fi

  if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
      --display-name="Countdown capability Cloud Run job"
  fi

  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin"

  if [[ -n "$IMAGE_URI" ]]; then
    image_source=(--image="$IMAGE_URI")
  else
    image_source=(--source=.)
  fi

  gcloud run jobs deploy "$JOB_NAME" \
    "${image_source[@]}" \
    --region="$REGION" \
    --service-account="$SERVICE_ACCOUNT" \
    --cpu=8 \
    --memory=32Gi \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --no-gpu-zonal-redundancy \
    --task-timeout=3600s \
    --max-retries=3 \
    --tasks=1 \
    --parallelism=1 \
    --add-volume="name=results,type=cloud-storage,bucket=${BUCKET}" \
    --add-volume-mount="volume=results,mount-path=/results" \
    --args="--model=Qwen/Qwen3.5-4B,--output=/results/qwen3.5-4b,--samples-per-problem=32,--temperature=0.6,--max-new-tokens=1024,--generation-batch-size=8,--device=cuda"

  echo "Deployed $JOB_NAME in $REGION."
  echo "Estimated active cost: roughly \$1.42/hour for one L4, 8 vCPU, and 32 GiB."
}

execute() {
  gcloud run jobs execute "$JOB_NAME" \
    --region="$REGION" \
    --wait
}

status() {
  gcloud run jobs executions list \
    --job="$JOB_NAME" \
    --region="$REGION"
}

download() {
  mkdir -p artifacts/phase2/capability/qwen3.5-4b
  gcloud storage cp \
    "gs://${BUCKET}/${RESULT_OBJECT}" \
    artifacts/phase2/capability/qwen3.5-4b/countdown_capability_results.json
}

case "$ACTION" in
  deploy)
    deploy
    ;;
  execute)
    execute
    ;;
  status)
    status
    ;;
  download)
    download
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  PROJECT_ID=my-project scripts/cloud_run_countdown_capability.sh deploy
  PROJECT_ID=my-project scripts/cloud_run_countdown_capability.sh execute
  PROJECT_ID=my-project scripts/cloud_run_countdown_capability.sh status
  PROJECT_ID=my-project scripts/cloud_run_countdown_capability.sh download

Optional variables:
  REGION=us-central1
  BUCKET=my-results-bucket
  JOB_NAME=countdown-capability-qwen35-4b
  IMAGE_URI=us-central1-docker.pkg.dev/my-project/cloud-run-source-deploy/countdown-capability-qwen35-4b:latest
EOF
    exit 2
    ;;
esac
