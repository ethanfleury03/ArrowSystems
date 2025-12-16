# GCS Document Storage

This document describes how documents are stored in Google Cloud Storage (GCS) and how to configure and use the GCS-based document storage system.

## Overview

The Arrow RAG application uses Google Cloud Storage as the primary storage location for original PDF documents. This ensures:

- **Scalability**: Documents are stored in cloud storage, not on local disk
- **Durability**: GCS provides high availability and redundancy
- **Separation of concerns**: Documents are stored separately from the application runtime
- **RunPod compatibility**: GPU workers can access documents without needing local copies

## Architecture

### Document Flow

1. **Upload via Web App** (`/admin/documents/upload`):
   - User uploads PDF through admin interface
   - File is uploaded to GCS bucket: `gs://{DOCS_GCS_BUCKET}/{DOCS_GCS_PREFIX}{metadata_id}/{filename}`
   - `Document.gcs_path` is populated in database
   - `DocumentIngestionMetadata` record is created

2. **Ingestion on RunPod** (`python ingest.py`):
   - Script scans GCS bucket for all PDFs
   - Downloads each PDF to temporary file
   - Processes through chunking/embedding pipeline
   - Syncs document metadata to database
   - Cleans up temporary files

3. **Chunking/Embedding**:
   - `chunking_runner.py` prefers `gcs_path` over local `file_path`
   - Downloads from GCS to temporary file if needed
   - Processes document and cleans up temp file

## Configuration

### Required Environment Variables

#### `DOCS_GCS_BUCKET` (Required)
- **Description**: GCS bucket name where documents are stored
- **Example**: `arrow-documents-prod`
- **Required in**: Production, RunPod workers
- **Optional in**: Development (if using local storage fallback)

#### `DOCS_GCS_PREFIX` (Optional)
- **Description**: Prefix/path within bucket for document storage
- **Default**: `documents/`
- **Example**: `documents/` or `prod/documents/`
- **Note**: Trailing slash is automatically added if missing

#### `DOCS_LOCAL_SAVE_ENABLED` (Optional)
- **Description**: Whether to also save files locally (for dev/backward compatibility)
- **Default**: `false`
- **Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Use case**: Development environments where you want both GCS and local copies

#### `GOOGLE_APPLICATION_CREDENTIALS` (Required for GCS access)
- **Description**: Path to service account JSON key file
- **Example**: `/path/to/service-account-key.json`
- **Required in**: RunPod workers, any environment accessing GCS
- **Note**: Uses Application Default Credentials (ADC) - can also use `gcloud auth application-default login` for local dev

### Example Configuration

#### Production (Cloud Run)
```bash
DOCS_GCS_BUCKET=arrow-documents-prod
DOCS_GCS_PREFIX=documents/
DOCS_LOCAL_SAVE_ENABLED=false
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-service-account.json
```

#### RunPod Worker
```bash
DOCS_GCS_BUCKET=arrow-documents-prod
DOCS_GCS_PREFIX=documents/
DOCS_LOCAL_SAVE_ENABLED=false
GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-key.json
ARROW_ALLOW_APP_INGESTION=true
```

#### Development (Local)
```bash
DOCS_GCS_BUCKET=arrow-documents-dev
DOCS_GCS_PREFIX=documents/
DOCS_LOCAL_SAVE_ENABLED=true  # Also save locally for testing
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
# OR use: gcloud auth application-default login
```

## Object Naming Convention

Documents are stored in GCS using the following naming convention:

```
{DOCS_GCS_PREFIX}{metadata_id}/{sanitized_filename}
```

**Example**:
- Metadata ID: `abc-123-def-456`
- Filename: `User Manual.pdf`
- GCS Path: `gs://arrow-documents-prod/documents/abc-123-def-456/User_Manual.pdf`

**Benefits**:
- Unique paths per document (metadata_id ensures uniqueness)
- Easy to identify document from GCS path
- Supports multiple versions of same filename

## Migration

### Migrating Existing Local PDFs to GCS

If you have existing documents stored locally that need to be migrated to GCS:

```bash
# Dry run (see what would be migrated)
python backend/scripts/migrate_local_pdfs_to_gcs.py --dry-run

# Actual migration
python backend/scripts/migrate_local_pdfs_to_gcs.py
```

**What the script does**:
1. Queries database for documents with `file_path` but no `gcs_path`
2. Uploads each local PDF to GCS using the naming convention
3. Updates `Document.gcs_path` in database
4. Skips documents that already have `gcs_path` (safe to rerun)

**Requirements**:
- `DOCS_GCS_BUCKET` must be set
- `GOOGLE_APPLICATION_CREDENTIALS` must be configured
- Local files must exist at paths stored in `DocumentIngestionMetadata.file_path`

## Running Ingestion on RunPod

### Setup

1. **Configure environment variables**:
   ```bash
   export DOCS_GCS_BUCKET=arrow-documents-prod
   export DOCS_GCS_PREFIX=documents/
   export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-key.json
   export ARROW_ALLOW_APP_INGESTION=true
   ```

2. **Upload service account key**:
   - Download service account JSON key from GCP Console
   - Upload to RunPod workspace (e.g., `/workspace/gcs-key.json`)
   - Ensure file has read permissions: `chmod 600 /workspace/gcs-key.json`

3. **Run ingestion**:
   ```bash
   cd /workspace/ArrowSystems/backend
   python ingest.py
   ```

### What Happens

1. `ingest.py` scans GCS bucket for all PDFs under `{DOCS_GCS_PREFIX}`
2. For each PDF:
   - Downloads to temporary file
   - Processes through chunking/embedding pipeline
   - Creates/updates `Document` and `DocumentIngestionMetadata` records
   - Cleans up temporary file
3. Builds vector index from all processed documents
4. Saves index to storage directory

### Logging

The ingestion process logs:
- GCS objects being processed: `gs://bucket/key`
- Document IDs (if parsed from GCS path)
- Success/failure for each document
- Database sync status

## Database Schema

### Document Table

- `gcs_path` (String, nullable): Full GCS URI (e.g., `gs://bucket/path/to/file.pdf`)
- `file_name` (String): Original filename
- Other fields: `machine_model`, `is_active`, etc.

### DocumentIngestionMetadata Table

- `file_path` (String, nullable): Local file path (only set if `DOCS_LOCAL_SAVE_ENABLED=true`)
- `filename` (String): Original filename
- `status` (String): Ingestion status (`PENDING_INGESTION`, `COMPLETE`, etc.)

## Troubleshooting

### "DOCS_GCS_BUCKET not configured"

**Error**: `DOCS_GCS_BUCKET environment variable not set`

**Solution**: Set `DOCS_GCS_BUCKET` environment variable to your GCS bucket name.

### "Failed to upload file to GCS"

**Error**: Upload endpoint returns 500 error

**Possible causes**:
- GCS bucket doesn't exist
- Service account doesn't have write permissions
- `GOOGLE_APPLICATION_CREDENTIALS` not set or invalid

**Solution**:
1. Verify bucket exists: `gsutil ls gs://your-bucket-name`
2. Check service account permissions (needs `storage.objects.create`)
3. Verify credentials: `gcloud auth application-default print-access-token`

### "Failed to download from GCS" during ingestion

**Error**: `ingest.py` fails to download files

**Possible causes**:
- Service account doesn't have read permissions
- GCS object doesn't exist
- Network connectivity issues

**Solution**:
1. Check service account has `storage.objects.get` permission
2. Verify object exists: `gsutil ls gs://bucket/path/to/file.pdf`
3. Test download manually: `gsutil cp gs://bucket/path/to/file.pdf /tmp/test.pdf`

### Documents not appearing in ingestion

**Issue**: `ingest.py` doesn't find documents in GCS

**Check**:
1. Verify `DOCS_GCS_PREFIX` matches actual object paths
2. Check objects exist: `gsutil ls -r gs://bucket/documents/`
3. Verify objects are PDFs (case-insensitive `.pdf` extension)

## Best Practices

1. **Use separate buckets for dev/prod**: Prevents accidental data mixing
2. **Set up lifecycle policies**: Archive old documents to cheaper storage classes
3. **Monitor storage costs**: GCS charges for storage and operations
4. **Regular backups**: Consider versioning or object versioning for critical documents
5. **Access control**: Use IAM to restrict access to service accounts only

## Security

- **Service account keys**: Store securely, never commit to git
- **IAM roles**: Grant minimum required permissions (`Storage Object Admin` for full access, or custom role with specific permissions)
- **Bucket permissions**: Use bucket-level IAM policies to restrict access
- **Encryption**: GCS encrypts data at rest by default; ensure encryption in transit (HTTPS) is used

## Related Documentation

- [Ingestion Safety](./INGESTION_SAFETY.md): How ingestion is protected from web app
- [Deployment Guide](./DEPLOYMENT_GUIDE.md): General deployment information
- [Google Cloud Deployment](./GOOGLE_CLOUD_DEPLOYMENT.md): GCP-specific deployment details


