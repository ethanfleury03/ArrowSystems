# syntax=docker/dockerfile:1
# Lightweight Dockerfile for RAG App - Development & Production
# Models download at runtime, not during build

# Build argument to determine environment
ARG BUILD_ENV=production

# =============================================================================
# Base Stage - Minimal system dependencies
# =============================================================================
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal runtime dependencies only
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        libgomp1 \
        nodejs \
        npm \
        && rm -rf /var/lib/apt/lists/*

# Install Prisma CLIs required for generating the Python client
RUN npm install -g prisma prisma-client-py

# Create non-root user early
RUN useradd -m -u 1000 appuser

# =============================================================================
# Dependencies Stage - Install Python packages (with build tools)
# =============================================================================
FROM base AS dependencies

# Redeclare ARG for this stage
ARG BUILD_ENV=production

WORKDIR /app

# Install build-essential ONLY in this stage for compiling Python packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Conditionally install development dependencies
RUN if [ "$BUILD_ENV" = "development" ]; then \
        pip install watchdog python-dotenv; \
    fi

# =============================================================================
# Final Stage - Application setup (no build tools, no models)
# =============================================================================
FROM base AS final

# Redeclare ARG for this stage
ARG BUILD_ENV=production

# Set as ENV so it's available at runtime
ENV BUILD_ENV=${BUILD_ENV}

# Set HuggingFace cache directory (models download here at runtime)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Create necessary directories
# TODO: In production, mount GCS bucket for /data and use vector DB for latest_model
RUN mkdir -p /app/data /app/latest_model /app/logs /app/storage /app/.cache/huggingface

# Copy Python dependencies from dependencies stage (no build tools)
# Only copy installed packages, not build tools
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create startup script (updated for runtime model downloads)
RUN printf '#!/bin/bash\nset -e\n\n# Load .env file if it exists\nif [ -f ".env" ]; then\n    echo "📋 Loading environment variables from .env file..."\n    set -a\n    while IFS= read -r line || [ -n "$line" ]; do\n        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then\n            continue\n        fi\n        export "$(echo "$line" | tr -d '\r')"\n    done < .env\n    set +a\n    echo "✅ Environment variables loaded"\nfi\n\necho "=========================================="\nif [ "$BUILD_ENV" = "development" ]; then\n    echo "🔧 DuraFlex Technical Assistant (DEV)"\nelse\n    echo "🔧 DuraFlex Technical Assistant"\nfi\necho "=========================================="\necho ""\necho "📥 Models will download automatically on first use if not cached..."\necho ""\n\n# Check if index exists\nif [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then\n    echo "✅ RAG index found in latest_model/"\n    echo "   📊 Indexed chunks: $(python -c 'import json; print(len(json.load(open("latest_model/docstore.json"))["docstore/data"]))' 2>/dev/null || echo 'unknown')"\n    echo ""\nelse\n    echo "⚠️  RAG Index Not Found! Running ingestion..."\n    python ingest.py\n    echo "✅ Ingestion complete!"\nfi\n\necho "🚀 Starting FastAPI backend server..."\necho ""\necho "API will be available at: http://localhost:8000"\necho "API docs available at: http://localhost:8000/docs"\necho ""\necho "⚠️  Note: This container runs the backend API only."\necho "   Use docker-compose.yml to run both backend and frontend together."\necho ""\n\n# Start FastAPI with environment-specific settings\nif [ "$BUILD_ENV" = "development" ]; then\n    exec python api.py --host 0.0.0.0 --port 8000 --reload\nelse\n    exec python api.py --host 0.0.0.0 --port 8000\nfi\n' > /app/start.sh && chmod +x /app/start.sh

# Create health check script
RUN printf '#!/bin/bash\nif curl -f http://localhost:8000/health > /dev/null 2>&1; then\n    exit 0\nelse\n    exit 1\nfi\n' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

# Copy application code (this is the frequently changing part)
COPY . .

# Install Prisma CLI and generate Python client during build
RUN npm install -g prisma && \
    python -m prisma generate && \
    python - <<'PY' \
import pathlib, shutil, prisma, os
src = pathlib.Path('/app/prisma/generated')
dst = pathlib.Path(prisma.__file__).parent
for item in src.iterdir():
    target = dst / item.name
    if target.exists():
        if target.is_file() or target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
    if item.is_dir():
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)
PY

# Set ownership and permissions
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app/.cache

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD /app/healthcheck.sh || exit 1

CMD ["/bin/bash", "/app/start.sh"]
