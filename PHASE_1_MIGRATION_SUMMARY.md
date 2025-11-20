# Phase 1 Migration Summary: Documents and Metadata Off Local Disk

## Overview

Phase 1 of the GCP migration successfully moves document files, metadata, and glossary data off local disk storage and into Google Cloud Storage + PostgreSQL. This phase does NOT touch chunk JSONs, vector indexes, or RAG vector stores (those are reserved for Phase 2).

## Changes Made

### 1. New Database Tables and Models

#### Document Table (`documents`)
- **Location**: `backend/utils/db.py` - `Document` class
- **Purpose**: Stores document metadata (replaces `data/document_metadata.json`)
- **Key Fields**:
  - `id` (Integer, primary key)
  - `file_name` (String, indexed) - Original filename
  - `gcs_path` (String) - Cloud Storage path (e.g., `gs://bucket/path`)
  - `display_name` (String) - Name to show in UI
  - `machine_model` (String) - Machine model(s), can be JSON string for multiple
  - `category`, `product_family` (String, optional)
  - `is_active` (Boolean, default True, indexed)
  - `requires_admin_review` (Boolean, default False)
  - `file_size_bytes` (Integer)
  - `last_ingestion_date` (DateTime)
  - `created_at`, `updated_at` (DateTime)

#### Glossary Terms Table (`glossary_terms`)
- **Location**: `backend/utils/db.py` - `GlossaryTerm` class
- **Purpose**: Stores glossary definitions (replaces `data/glossary.csv`)
- **Key Fields**:
  - `id` (Integer, primary key)
  - `term` (String, indexed)
  - `definition` (Text)
  - `aliases` (JSON) - List of alias strings
  - `created_at`, `updated_at` (DateTime)

#### Migration File
- **Location**: `backend/migrations/versions/004_documents_and_glossary.py`
- **Revision**: `004_documents_and_glossary`
- **Revises**: `003_ingestion_phase1`
- **Creates**: Both `documents` and `glossary_terms` tables

### 2. Refactored Document Metadata Management

#### File: `backend/utils/document_metadata.py`
- **Before**: Read/wrote `data/document_metadata.json` file
- **After**: All functions use SQLAlchemy to interact with `documents` table
- **Key Functions**:
  - `get_all_documents(session, active_only=True)` - Get all documents from DB
  - `get_document_by_id(session, doc_id)` - Get document by ID
  - `get_document_by_filename(session, filename)` - Get document by filename
  - `upsert_document(session, ...)` - Create or update document record
  - `get_document_metadata(filename)` - Maintains backward compatibility
  - `update_document_metadata(filename, updates)` - Maintains backward compatibility

### 3. Migration Scripts (One-Off, Not Runtime)

#### Document Metadata Migration Script
- **Location**: `backend/scripts/migrate_document_metadata_json_to_db.py`
- **Purpose**: Migrate data from `data/document_metadata.json` to `documents` table
- **Usage**: `python -m backend.scripts.migrate_document_metadata_json_to_db [--gcs-bucket BUCKET] [--dry-run]`
- **Features**:
  - Reads existing JSON file
  - Creates Document records in database
  - Optionally generates GCS paths if bucket name provided
  - Dry-run mode for testing

#### Glossary Migration Script
- **Location**: `backend/scripts/migrate_glossary_csv_to_db.py`
- **Purpose**: Migrate data from `data/glossary.csv` to `glossary_terms` table
- **Usage**: `python -m backend.scripts.migrate_glossary_csv_to_db [--csv-file PATH] [--dry-run]`
- **Features**:
  - Reads CSV file
  - Creates GlossaryTerm records in database
  - Parses pipe-separated aliases
  - Dry-run mode for testing

### 4. Updated API Endpoints

#### Document Listing Endpoint (`/documents`)
- **File**: `backend/api.py` - `get_user_documents()`
- **Before**: Scanned local `data/` directory with `os.listdir()`
- **After**: Queries `documents` table from database
- **Changes**:
  - Removed filesystem scanning
  - Uses `get_all_documents()` to query DB
  - Filters by `is_active=True`
  - Maintains same response format for frontend compatibility

#### Document Serving Endpoint (`/documents/{filename:path}`)
- **File**: `backend/api.py` - `serve_document()`
- **Before**: Tried multiple local paths (`data/`, `/app/data/`, etc.)
- **After**: Downloads from Cloud Storage using GCS client
- **Changes**:
  - Looks up document in database by filename
  - Uses `gcs_path` field to download from Cloud Storage
  - Falls back to default bucket if `gcs_path` not set
  - Returns document as streaming response with correct Content-Type

### 5. Google Cloud Storage Client

#### File: `backend/utils/gcs_client.py`
- **Purpose**: Helper utilities for accessing files from Cloud Storage
- **Key Functions**:
  - `get_gcs_client()` - Get or create GCS client (lazy initialization)
  - `parse_gcs_path(gcs_path)` - Parse GCS path into bucket and blob name
  - `download_blob(bucket_name, blob_name)` - Download blob from GCS
  - `download_document(gcs_path)` - Download document using its GCS path
  - `download_document_by_filename(filename, bucket_name)` - Download by filename from default bucket
  - `generate_signed_url(...)` - Generate signed URL for direct access (optional)

### 6. Updated Glossary Loader

#### File: `backend/glossary_loader.py`
- **Before**: Only loaded from CSV/PDF files
- **After**: Loads from database first, falls back to file if needed
- **Changes**:
  - `load_glossary_from_db(session)` - New function to load from database
  - `load_glossary_any(path)` - Updated to try database first, then file
  - Maintains backward compatibility with file-based loading

#### Updated Orchestrator
- **File**: `backend/orchestrator.py` - `_load_glossary_index()`
- **Changes**: Updated to work with new glossary loader (database-first)

### 7. Docker and Configuration Updates

#### Updated `.dockerignore`
- **Changes**: Excludes `data/*.pdf`, `data/*.docx`, `data/document_metadata.json`, `data/glossary.csv`
- **Note**: Phase 2 items (`latest_model/`, `data/chunks/`) are NOT excluded yet

#### Updated `backend/requirements.txt`
- **Added**: `google-cloud-storage>=2.10.0` - For accessing documents from GCS buckets

#### Updated `backend/api.py`
- **Removed**: All references to `document_metadata.json` file operations
- **Updated**: Upload endpoint now updates database instead of JSON file

### 8. Environment Variables

#### New Environment Variables Required:
- **`DOCS_BUCKET_NAME`** - Cloud Storage bucket name for documents (e.g., `rag-postgres-prod-docs`)
- **`DATABASE_URL`** - PostgreSQL connection string (already existed)

#### GCS Client Configuration:
- Uses default Google Cloud credentials (via Application Default Credentials)
- For local development: `gcloud auth application-default login`
- For Cloud Run: Uses service account automatically
- No additional environment variables needed for authentication

## Migration Steps for Deployment

### 1. Run Database Migration
```bash
# Run Alembic migration to create new tables
alembic upgrade head
```

### 2. Migrate Document Metadata
```bash
# Migrate document_metadata.json to database
python -m backend.scripts.migrate_document_metadata_json_to_db \
    --gcs-bucket rag-postgres-prod-docs
```

### 3. Migrate Glossary
```bash
# Migrate glossary.csv to database
python -m backend.scripts.migrate_glossary_csv_to_db
```

### 4. Upload Documents to Cloud Storage
```bash
# Upload PDFs and DOCX files to GCS bucket
gsutil -m cp data/*.pdf gs://rag-postgres-prod-docs/
gsutil -m cp data/*.docx gs://rag-postgres-prod-docs/
```

### 5. Set Environment Variables
```bash
export DOCS_BUCKET_NAME=rag-postgres-prod-docs
export DATABASE_URL=postgresql://user:pass@host:port/dbname
```

### 6. Deploy Application
- The application will now read documents from Cloud Storage
- Document metadata is read from PostgreSQL
- Glossary is loaded from PostgreSQL

## Files NOT Modified (Phase 2 Reserved)

The following remain unchanged for Phase 2:
- `data/chunks/*.json` - Chunk JSON files
- `data/chunks_test/` - Test chunks directory
- `latest_model/*.json` - Vector index files
- `backend/utils/chunking_runner.py` - Chunking persistence (still writes JSON)
- `backend/utils/embedding_runner.py` - Embedding persistence (still writes JSON)
- `backend/orchestrator.py` - Vector index loading (still uses `latest_model/`)

## Backward Compatibility

### Maintained for Smooth Migration:
- `get_document_metadata(filename)` - Still works, returns same dict structure
- `update_document_metadata(filename, updates)` - Still works, updates database
- Glossary loader falls back to file if database is empty
- Frontend API response format unchanged

## Testing Checklist

- [ ] Run Alembic migration successfully
- [ ] Migrate document metadata from JSON to DB
- [ ] Migrate glossary from CSV to DB
- [ ] Upload documents to Cloud Storage
- [ ] Test document listing endpoint (returns documents from DB)
- [ ] Test document serving endpoint (downloads from GCS)
- [ ] Test glossary loading (loads from DB)
- [ ] Verify no reads from `data/document_metadata.json` at runtime
- [ ] Verify no reads from `data/glossary.csv` at runtime
- [ ] Verify PDFs/DOCX are served from Cloud Storage, not local filesystem

## Summary Statistics

- **New Models**: 2 (Document, GlossaryTerm)
- **New Tables**: 2 (`documents`, `glossary_terms`)
- **New Migration**: 1 (`004_documents_and_glossary.py`)
- **New Scripts**: 2 (document metadata migration, glossary migration)
- **New Utility Module**: 1 (`backend/utils/gcs_client.py`)
- **Updated Endpoints**: 2 (`/documents`, `/documents/{filename}`)
- **Updated Utility Modules**: 2 (`document_metadata.py`, `glossary_loader.py`)
- **New Dependency**: 1 (`google-cloud-storage`)
- **Environment Variables**: 1 (`DOCS_BUCKET_NAME`)

## Next Steps (Phase 2)

Phase 2 will migrate:
- Chunk JSON files (`data/chunks/*.json`) → Database table or remove entirely
- Vector index files (`latest_model/*.json`) → Qdrant or pgvector
- Chunking/embedding runners to use database instead of JSON files

Phase 1 is complete and ready for deployment! 🚀



