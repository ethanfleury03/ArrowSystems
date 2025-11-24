# Makefile for Arrow Systems RAG Application
# Provides convenient commands for development and deployment

.PHONY: help upload-index check-index

help:
	@echo "Available commands:"
	@echo "  make upload-index    - Upload RAG index to GCS bucket"
	@echo "  make check-index     - Verify local index directory exists"

upload-index:
	@echo "📦 Uploading RAG index to GCS..."
	python -m backend.scripts.upload_index_to_gcs --dir latest_model

check-index:
	@echo "🔍 Checking local RAG index..."
	@if [ -f "latest_model/docstore.json" ]; then \
		echo "✅ Index found at latest_model/"; \
		ls -lh latest_model/*.json | head -5; \
	else \
		echo "❌ Index not found at latest_model/"; \
		echo "   Expected: latest_model/docstore.json"; \
		exit 1; \
	fi

