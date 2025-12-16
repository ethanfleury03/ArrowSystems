# Document Ingestion Safety

## Overview

The Arrow RAG application has **ingestion disabled by default** to protect the embedding index from automatic writes. This ensures that ingestion only occurs via controlled, external GPU pipelines (e.g., RunPod workers), not from the web application.

## Configuration

### Backend Flag

The backend uses the `ARROW_ALLOW_APP_INGESTION` environment variable to control whether the web app can trigger ingestion:

```bash
# Default (ingestion disabled from app)
ARROW_ALLOW_APP_INGESTION=false

# Only set to true in dedicated GPU ingestion environments
ARROW_ALLOW_APP_INGESTION=true
```

**⚠️ WARNING:** This flag should **ONLY** be set to `true` in dedicated GPU ingestion environments (e.g., RunPod workers, separate ingestion pods). **NEVER** set this to `true` in the main production frontend/backend deployment.

### Frontend Flag

The frontend uses the `NEXT_PUBLIC_ALLOW_APP_INGESTION` environment variable to control UI visibility:

```bash
# Default (ingestion UI hidden)
NEXT_PUBLIC_ALLOW_APP_INGESTION=false

# Only set to true in dedicated GPU ingestion environments
NEXT_PUBLIC_ALLOW_APP_INGESTION=true
```

## Protected Operations

When `ARROW_ALLOW_APP_INGESTION=false`, the following operations are blocked:

1. **Document Upload Ingestion** (`/admin/documents/upload`)
   - File upload and metadata creation still work
   - Chunking and embedding are blocked
   - Returns 403 with clear error message

2. **Document Deletion with Rebuild** (`/admin/documents/metadata/{metadata_id}`)
   - Metadata deletion still works
   - Index rebuild is blocked
   - Returns 403 with clear error message

3. **Chunk Summary Regeneration** (`/admin/chunks/{chunk_id}/regenerate-summary`)
   - Blocked for consistency (doesn't write embeddings, but gated anyway)

4. **Background Ingestion Functions**
   - `run_chunking()` - blocked
   - `run_embedding()` - blocked
   - `run_delete_and_reindex()` - blocked

## Allowed Operations

The following operations **always work**, regardless of the ingestion flag:

- Document upload (metadata only)
- Document listing and viewing
- Metadata editing (machine_model, category, product_family)
- Toggling document active/inactive status
- Document deletion (metadata only, no index rebuild)

## Recommended Workflow

1. **Upload documents via admin UI** (metadata only)
   - Documents are saved to `data/original_pdfs/`
   - Metadata is stored in the database
   - Status is set to `PENDING_INGESTION`

2. **Trigger ingestion manually via external GPU job**
   - Use a dedicated GPU worker (RunPod, separate pod, etc.)
   - Set `ARROW_ALLOW_APP_INGESTION=true` in that environment only
   - Run ingestion scripts that directly write to the index
   - Use the same config and database as the main app

3. **Re-upload or sync the index into the serving environment**
   - Copy the generated index to the serving environment
   - Or use shared storage (GCS, S3, etc.) that both environments can access

## Status Display

When ingestion is disabled:

- Documents with `ingestion_status=null` and `chunk_count > 0` show: **"Managed externally"**
- Documents with `ingestion_status=null` and `chunk_count=0` show: **"Index status unknown"**
- The UI does not show "Rebuilding index..." unless the status is explicitly set in the database

## Code Markers

All ingestion/index-write code paths are marked with:

```python
# INDEX-WRITE PATH: creates/updates embeddings
```

This makes it easy to identify all write paths during code reviews.

## Testing

To test that ingestion is properly blocked:

1. Set `ARROW_ALLOW_APP_INGESTION=false` in your environment
2. Attempt to upload a document via `/admin/documents/upload`
3. Verify you receive a 403 error with message: "Ingestion is disabled in this environment..."
4. Verify the document metadata was still created in the database
5. Verify no chunks or embeddings were generated

## Environment Variables

### Production (Main App)
```bash
ARROW_ALLOW_APP_INGESTION=false
NEXT_PUBLIC_ALLOW_APP_INGESTION=false
```

### GPU Ingestion Worker
```bash
ARROW_ALLOW_APP_INGESTION=true
NEXT_PUBLIC_ALLOW_APP_INGESTION=true
```

## Troubleshooting

**Q: Why do I see "Rebuilding index..." for documents?**
A: This status is read-only from the database. If you see this, it means the status was set by a previous ingestion operation. The app will not automatically change this status when ingestion is disabled.

**Q: Can I still upload documents when ingestion is disabled?**
A: Yes! Document upload works for metadata. Only the chunking/embedding step is blocked.

**Q: How do I manually trigger ingestion?**
A: Use external scripts or GPU workers that have `ARROW_ALLOW_APP_INGESTION=true` set. These should call the same ingestion functions (`run_chunking`, `run_embedding`) but from a controlled environment.

