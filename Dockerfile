FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY env ./env
COPY experiments ./experiments
COPY inference ./inference
COPY oracle ./oracle
COPY policies ./policies
COPY data/countdown-capability ./data/countdown-capability

ENTRYPOINT ["python", "-m", "experiments.evaluate_countdown_capability"]
CMD ["--help"]
