FROM python:3.12-slim

# uv binary, copied from Astral's official image (no pip install needed)
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1. Dependencies only: this layer is rebuilt only when the lock file changes
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. Application code
COPY app/ app/
COPY scripts/ scripts/
COPY eval/ eval/

# Models downloaded at build time: fast, reproducible startup
RUN python -m scripts.download_models

# Non-root user
RUN useradd -m appuser && mkdir -p /app/logs /app/data && chown -R appuser /app/logs /app/data /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# 1 worker: Prometheus counters stay consistent (single process)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
