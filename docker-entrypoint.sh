#!/bin/bash
set -e

# Fix permissions for cache directory (always run as root, then switch to app user)
mkdir -p /app/.cache/huggingface/hub
chown -R app:app /app/.cache || true
chmod -R 755 /app/.cache || true

# Switch to app user and execute the main command
exec gosu app "$@"

