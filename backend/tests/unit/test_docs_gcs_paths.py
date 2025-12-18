from backend.utils.docs_gcs_paths import choose_docs_upload_object_name


def test_upload_object_name_root_no_metadata_folder_when_no_collision():
    name = choose_docs_upload_object_name(
        docs_prefix="",
        sanitized_filename="Manual.pdf",
        metadata_id="meta-123",
        object_exists=lambda _: False,
    )
    assert name == "Manual.pdf"
    assert "meta-123" not in name
    assert "/" not in name


def test_upload_object_name_root_collision_uses_metadata_suffix():
    def exists(n: str) -> bool:
        return n in {"Manual.pdf"}  # base collides

    name = choose_docs_upload_object_name(
        docs_prefix="",
        sanitized_filename="Manual.pdf",
        metadata_id="meta-123",
        object_exists=exists,
    )
    assert name == "Manual__meta-123.pdf"
    assert "/" not in name


def test_upload_object_name_nonempty_prefix_collision_stays_in_prefix():
    def exists(n: str) -> bool:
        return n in {"docs/Manual.pdf"}  # base collides

    name = choose_docs_upload_object_name(
        docs_prefix="docs/",
        sanitized_filename="Manual.pdf",
        metadata_id="meta-123",
        object_exists=exists,
    )
    assert name == "docs/Manual__meta-123.pdf"


