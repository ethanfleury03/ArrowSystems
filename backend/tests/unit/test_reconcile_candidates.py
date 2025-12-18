from backend.scripts.reconcile_docs import DbRow


def test_reconcile_candidates_use_stored_gcs_uri_authoritatively():
    row = DbRow(
        metadata_id="meta-123",
        filename="Manual.pdf",
        status="PENDING_INGESTION",
        error_message=None,
        meta_file_path=None,
        document_id=1,
        doc_gcs_path="gs://bucket/Manual.pdf",
        doc_is_active=True,
    )
    cands = row.expected_object_candidates(configured_prefix="")
    assert cands[0] == "Manual.pdf"


def test_reconcile_candidates_root_includes_root_and_legacy_when_missing_stored_path():
    row = DbRow(
        metadata_id="meta-123",
        filename="Manual.pdf",
        status="PENDING_INGESTION",
        error_message=None,
        meta_file_path=None,
        document_id=None,
        doc_gcs_path=None,
        doc_is_active=None,
    )
    cands = row.expected_object_candidates(configured_prefix="")
    # new canonical root
    assert "Manual.pdf" in cands
    # legacy fallback
    assert "meta-123/Manual.pdf" in cands


