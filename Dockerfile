# Production Dockerfile for RAG App - Google Cloud Ready
FROM python:3.11-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements-production.txt .

# Install Python dependencies with specific versions that work together
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/latest_model /app/logs /app/storage

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check script
COPY --chown=appuser:appuser <<EOF /app/healthcheck.sh
#!/bin/bash
# Health check for the application
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    exit 0
else
    exit 1
fi
EOF

RUN chmod +x /app/healthcheck.sh

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Startup script that handles everything
COPY --chown=appuser:appuser <<EOF /app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting RAG Application..."

# Check if we need to ingest data
if [ ! -f "/app/latest_model/index_store.json" ] || [ ! -f "/app/latest_model/docstore.json" ]; then
    echo "📚 No existing index found. Starting data ingestion..."
    python ingest.py
    echo "✅ Data ingestion completed!"
else
    echo "📚 Existing index found. Skipping ingestion."
fi

# Start the Streamlit app
echo "🌐 Starting Streamlit application..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
EOF

RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"]
