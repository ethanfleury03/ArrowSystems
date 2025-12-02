# Makefile for Arrow Systems RAG Application
# Provides convenient commands for development and deployment

.PHONY: help upload-index check-index

help:
	@echo "Available commands:"
	@echo "  make upload-index    - Upload RAG index to GCS bucket"
	@echo "  make check-index     - Verify local index directory exists"

upload-index:
	@echo "[UPLOAD] Uploading RAG index to GCS..."
	@echo "         Using gsutil for reliable large file uploads..."
	@echo "         Uploading to bucket ROOT: gs://arrow-rag-support-prod-rag/"
	@echo "         Files will appear at /app/latest_model/ when bucket root mounts to /app/latest_model/"
	gsutil -m cp -r latest_model/* gs://arrow-rag-support-prod-rag/
	@echo "[SUCCESS] Index uploaded successfully!"
	@echo "         Verify with: gsutil ls gs://arrow-rag-support-prod-rag/"

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

