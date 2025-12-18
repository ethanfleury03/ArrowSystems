import os


def test_ingest_docs_prefix_env_preserves_empty(monkeypatch):
    """
    backend.ingest.main() reads docs prefix via os.environ.get(...) and normalize_gcs_prefix.
    This test asserts an explicitly empty env var stays empty (bucket root).
    """
    monkeypatch.setenv("GCS_DOCS_PREFIX", "")
    from backend.config.env import normalize_gcs_prefix
    raw = os.environ.get("GCS_DOCS_PREFIX")
    assert raw == ""
    assert normalize_gcs_prefix(raw) == ""


