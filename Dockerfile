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
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with specific versions that work together
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy .env file if it exists
COPY .env* ./

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

# Environment variables will be set at runtime for security
# These are just defaults - override them when running the container
ENV ANTHROPIC_API_KEY=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_DEFAULT_REGION="us-east-1"

# Create startup script with proper Unix line endings
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Load .env file if it exists' >> /app/start.sh && \
    echo 'if [ -f ".env" ]; then' >> /app/start.sh && \
    echo '    echo "📋 Loading environment variables from .env file..."' >> /app/start.sh && \
    echo '    set -a' >> /app/start.sh && \
    echo '    source .env' >> /app/start.sh && \
    echo '    set +a' >> /app/start.sh && \
    echo '    echo "✅ Environment variables loaded"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'echo "=========================================="' >> /app/start.sh && \
    echo 'echo "🔧 DuraFlex Technical Assistant"' >> /app/start.sh && \
    echo 'echo "=========================================="' >> /app/start.sh && \
    echo 'echo ""' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Check if index exists' >> /app/start.sh && \
    echo 'if [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then' >> /app/start.sh && \
    echo '    echo "✅ RAG index found in latest_model/"' >> /app/start.sh && \
    echo '    echo "   📊 Indexed chunks: $(python -c "import json; print(len(json.load(open(\"latest_model/docstore.json\"))[\"docstore/data\"]))" 2>/dev/null || echo "unknown")"' >> /app/start.sh && \
    echo '    echo ""' >> /app/start.sh && \
    echo 'else' >> /app/start.sh && \
    echo '    echo "⚠️  RAG Index Not Found! Running ingestion..."' >> /app/start.sh && \
    echo '    python ingest.py' >> /app/start.sh && \
    echo '    echo "✅ Ingestion complete!"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'echo "🔐 Login: admin/admin123 or tech1/tech123"' >> /app/start.sh && \
    echo 'echo "🚀 Starting Streamlit server..."' >> /app/start.sh && \
    echo 'exec python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true' >> /app/start.sh

# Start the application
CMD ["/bin/bash", "/app/start.sh"]
