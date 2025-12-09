#!/bin/sh

set -e

echo "🛠 Running DB migrations (upgrade to head)..."
python -m backend.utils.migration_runner upgrade
echo "✅ DB migrations complete"

echo "🚀 Starting backend application..."
exec /bin/bash /app/run_backend.sh


