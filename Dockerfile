# ---------------------------------------------------------------------------
# ARC Race - single container serving both the API (backend/) and the static
# frontend (frontend/).
#
# One uvicorn worker on purpose: race sessions live in process memory, so a
# second worker would not see another worker's races.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/arc-race

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
# The vendored ARC-AGI-1 public training set (Apache-2.0).
COPY backend/data backend/data
COPY frontend frontend

# Run unprivileged; nothing in the image needs to be writable.
RUN useradd --create-home --uid 10001 arcrace \
    && chown -R arcrace:arcrace /srv/arc-race
USER arcrace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
