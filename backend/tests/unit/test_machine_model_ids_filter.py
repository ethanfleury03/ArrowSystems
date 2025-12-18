import pytest


from backend.orchestrator import HybridRetriever


class DummyNode:
    def __init__(self, metadata):
        self.metadata = metadata


def test_machine_model_ids_overlap_matches_any():
    hr = HybridRetriever.__new__(HybridRetriever)  # avoid heavy init
    node = DummyNode(metadata={"machine_model_ids": [1, 2]})
    assert HybridRetriever._matches_filters(hr, node, {"machine_model_ids": [1]}) is True
    assert HybridRetriever._matches_filters(hr, node, {"machine_model_ids": [2]}) is True
    assert HybridRetriever._matches_filters(hr, node, {"machine_model_ids": [3]}) is False


def test_machine_model_ids_overlap_requires_non_empty_when_filter_present():
    hr = HybridRetriever.__new__(HybridRetriever)
    node_empty = DummyNode(metadata={"machine_model_ids": []})
    node_missing = DummyNode(metadata={})
    assert HybridRetriever._matches_filters(hr, node_empty, {"machine_model_ids": [1]}) is False
    assert HybridRetriever._matches_filters(hr, node_missing, {"machine_model_ids": [1]}) is False


def test_machine_model_ids_empty_filter_list_noop():
    hr = HybridRetriever.__new__(HybridRetriever)
    node = DummyNode(metadata={})
    # no filtering requested
    assert HybridRetriever._matches_filters(hr, node, {"machine_model_ids": []}) is True


