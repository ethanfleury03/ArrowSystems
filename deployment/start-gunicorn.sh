#!/usr/bin/env bash
set -euo pipefail

WORKERS="${GUNICORN_WORKERS:-1}"
TIMEOUT="${GUNICORN_TIMEOUT:-600}"
PORT_BIND="${PORT:-8080}"

SERVICE="${K_SERVICE:-unknown}"
REVISION="${K_REVISION:-unknown}"
HOSTNAME_VAL="${HOSTNAME:-unknown}"
PID="$$"

if [[ "${TIMEOUT}" == "60" ]]; then
  echo "[GUNICORN_START] WARNING: timeout=60 detected; forcing timeout=600" >&2
  TIMEOUT="600"
fi

echo "[GUNICORN_START] pid=${PID} service=${SERVICE} revision=${REVISION} hostname=${HOSTNAME_VAL} workers=${WORKERS} timeout=${TIMEOUT} port=${PORT_BIND}"

exec gunicorn backend.api:app \
  --workers "${WORKERS}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT_BIND}" \
  --timeout "${TIMEOUT}" \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile - \
  --log-level info

