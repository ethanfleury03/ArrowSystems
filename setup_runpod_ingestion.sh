#!/bin/bash
# RunPod Ingestion Environment Setup - Production Configuration
# Based on your actual Cloud Run environment variables

# ============================================
# REQUIRED - Database
# IMPORTANT: Cloud Run uses Unix socket, RunPod needs TCP
# Option 1: Use Cloud SQL Proxy (recommended)
#   Start proxy first: cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres
#   Then use localhost connection:
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"

# Option 2: Direct connection (if Cloud SQL allows external IPs)
# export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app"

# ============================================
# REQUIRED - GCS Configuration
# ============================================
export DOCS_GCS_BUCKET="arrow-rag-support-prod-docs"
export DOCS_GCS_PREFIX="ROOT"
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcs-key.json"

# ============================================
# REQUIRED - RAG Index Configuration
# ============================================
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"
export RAG_INDEX_LOCAL_DIR="/workspace/latest_model"

# ============================================
# REQUIRED - Environment
# ============================================
export ENV="prod"

# ============================================
# RECOMMENDED - HuggingFace Cache
# ============================================
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="/workspace/.cache/huggingface"

# ============================================
# OPTIONAL - Anthropic (Claude Rewriting)
# ============================================
# export ANTHROPIC_API_KEY="sk-ant-api03-..."

# ============================================
# OPTIONAL - Ingestion Configuration
# ============================================
export ARROW_ALLOW_APP_INGESTION="true"
export DISABLE_METADATA_UPDATE="1"

# ============================================
# OPTIONAL - Performance Tuning
# ============================================
# export MAX_DOC_CHARS_FOR_SMART_CHUNK="250000"
# export MAX_DOC_CHARS="250000"
# export MAX_CHUNKS_PER_DOC="5000"
# export MAX_SECONDS_PER_DOC="90"
# export CHUNK_HEARTBEAT_EVERY="50"
# export CHUNK_DEBUG="0"

echo "✅ Environment variables configured for RunPod ingestion"
echo "   DATABASE_URL: ${DATABASE_URL%%@*}@***"
echo "   DOCS_GCS_BUCKET: $DOCS_GCS_BUCKET"
echo "   RAG_INDEX_GCS_BUCKET: $RAG_INDEX_GCS_BUCKET"
echo "   RAG_INDEX_LOCAL_DIR: $RAG_INDEX_LOCAL_DIR"
echo "   ENV: $ENV"
echo ""
echo "⚠️  IMPORTANT: If using Cloud SQL Proxy, start it first:"
echo "   cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres &"

