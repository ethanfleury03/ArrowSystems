# syntax=docker/dockerfile:1
# Single Optimized Dockerfile for RAG App - Development & Production

# Build argument to determine environment
ARG BUILD_ENV=production

# =============================================================================
# Base Stage - Common dependencies and system setup
# =============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in a single layer with BuildKit cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user early
RUN useradd -m -u 1000 appuser

# =============================================================================
# Dependencies Stage - Install Python packages
# =============================================================================
FROM base AS dependencies

# Redeclare ARG for this stage (required - ARGs don't persist across FROM)
ARG BUILD_ENV=production

WORKDIR /app

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
# Preload Stage - Download and cache models during build
# =============================================================================
FROM dependencies AS preload

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

WORKDIR /app

# Create cache directory
RUN mkdir -p /app/.cache/huggingface && chmod -R 755 /app/.cache

# Copy preload script (better caching - only reruns if script changes)
COPY preload_models.py /tmp/preload_models.py

# Preload all HuggingFace models during build (cached in image layer)
# This happens once during build, making container startup instant
# Expect this to take 5-10 minutes on first build (downloads ~2-3GB of models)
RUN echo "🔄 Preloading HuggingFace models (this takes 5-10 minutes on first build)..." && \
    python /tmp/preload_models.py || (echo "❌ CRITICAL: Model preload FAILED! Check errors above." && exit 1) && \
    rm /tmp/preload_models.py

# =============================================================================
# Final Stage - Application setup
# =============================================================================
FROM preload AS final

# Redeclare ARG for this stage (required - ARGs don't persist across FROM)
ARG BUILD_ENV=production

# Set as ENV so it's available at runtime (for script conditionals)
ENV BUILD_ENV=${BUILD_ENV}

# Set HuggingFace cache directory to a location appuser can write to
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Copy application code (without chown yet - we'll preload first)
COPY . .

# Create necessary directories including cache directory for HuggingFace
RUN mkdir -p /app/data /app/latest_model /app/logs /app/storage /app/.cache/huggingface

# Preload the RAG index during build (if it exists) to warm up the cache
# This makes index loading instant on container startup
RUN echo "🔄 Preloading RAG index..." && \
    if [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then \
        python -c " \
        import os; \
        import sys; \
        sys.path.insert(0, '/app'); \
        os.chdir('/app'); \
        import torch; \
        device = 'cuda' if torch.cuda.is_available() else 'cpu'; \
        print(f'🖥️ Using device: {device}'); \
        print('📥 Loading embedding model for index warmup...'); \
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
        from llama_index.core import StorageContext, load_index_from_storage, Settings; \
        embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-large-en-v1.5', cache_folder='/app/.cache/huggingface/hub', trust_remote_code=True, device=device); \
        Settings.embed_model = embed_model; \
        print('🔄 Loading index from latest_model/...'); \
        storage_context = StorageContext.from_defaults(persist_dir='latest_model'); \
        index = load_index_from_storage(storage_context); \
        print('✅ Index preloaded successfully!'); \
        print(f'   Index type: {type(index).__name__}'); \
        if hasattr(index, 'as_retriever'): \
            retriever = index.as_retriever(similarity_top_k=1); \
            _ = retriever.retrieve('test query warmup'); \
        print('✅ Index warmed up and ready!'); \
        " 2>&1 || echo "⚠️ Index preload failed, continuing (will load on first run)"; \
    else \
        echo "⚠️ Index not found at build time (will load on first run)"; \
    fi

# Now chown everything to appuser and switch user
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app/.cache

# Switch to non-root user
USER appuser

# Create startup script that adapts to environment
# Using RUN heredoc instead of COPY heredoc for better compatibility
# Note: Single quotes around 'EOF' prevent variable expansion at build time
# BUILD_ENV will be evaluated at runtime via ENV variable
RUN printf '#!/bin/bash\nset -e\n\n# Load .env file if it exists\nif [ -f ".env" ]; then\n    echo "📋 Loading environment variables from .env file..."\n    set -a\n    # Use a while loop to load .env and strip carriage returns\n    while IFS= read -r line || [ -n "$line" ]; do\n        # Skip empty lines and comments\n        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then\n            continue\n        fi\n        # Strip carriage returns and export\n        export "$(echo "$line" | tr -d '\''\r'\'')"\n    done < .env\n    set +a\n    echo "✅ Environment variables loaded"\nfi\n\necho "=========================================="\nif [ "$BUILD_ENV" = "development" ]; then\n    echo "🔧 DuraFlex Technical Assistant (DEV)"\nelse\n    echo "🔧 DuraFlex Technical Assistant"\nfi\necho "=========================================="\necho ""\necho "✅ RAG models preloaded - ready to use instantly!"\necho ""\n\n# Check if index exists\nif [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then\n    echo "✅ RAG index found in latest_model/"\n    echo "   📊 Indexed chunks: $(python -c '\''import json; print(len(json.load(open("latest_model/docstore.json"))["docstore/data"]))'\'' 2>/dev/null || echo '\''unknown'\'')"\n    echo ""\nelse\n    echo "⚠️  RAG Index Not Found! Running ingestion..."\n    python ingest.py\n    echo "✅ Ingestion complete!"\nfi\n\necho "🔐 Login: admin/admin123 or tech1/tech123"\necho "🚀 Starting Streamlit server..."\necho ""\n\n# Start Streamlit with environment-specific settings\nif [ "$BUILD_ENV" = "development" ]; then\n    # Development: Enable hot reloading\n    exec python -m streamlit run app.py \\\n        --server.port=8501 \\\n        --server.address=0.0.0.0 \\\n        --server.headless=true \\\n        --server.runOnSave=true \\\n        --server.fileWatcherType=poll\nelse\n    # Production: Optimized settings\n    exec python -m streamlit run app.py \\\n        --server.port=8501 \\\n        --server.address=0.0.0.0 \\\n        --server.headless=true \\\n        --server.enableCORS=false \\\n        --server.enableXsrfProtection=false\nfi\n' > /app/start.sh && chmod +x /app/start.sh

# Create health check script
RUN printf '#!/bin/bash\nif curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then\n    exit 0\nelse\n    exit 1\nfi\n' > /app/healthcheck.sh && chmod +x /app/healthcheck.sh

EXPOSE 8501

# Health check (always enabled for production)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh || exit 1

CMD ["/bin/bash", "/app/start.sh"]
