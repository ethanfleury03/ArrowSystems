"""
Unit tests for translation module.
"""

import pytest
from utils.translation import (
    detect_language,
    translate_to_english,
    process_query_for_retrieval,
    LangDetectResult,
    TranslationResult,
)


def test_detect_language_short_text():
    """Test that very short text returns English with low confidence."""
    result = detect_language("Hi")
    assert result.lang == "en"
    assert result.confidence < 0.3


def test_detect_language_symbols():
    """Test that mostly numeric/symbol text returns English with low confidence."""
    result = detect_language("12345 !@#$%")
    assert result.lang == "en"
    assert result.confidence < 0.3


def test_detect_language_english():
    """Test detection of English text."""
    result = detect_language("How do I troubleshoot print quality issues?")
    assert result.lang == "en"
    assert result.confidence > 0.5


def test_translate_already_english():
    """Test that English text is returned as-is."""
    result = translate_to_english("How do I fix this?", "en")
    assert result.translated_text == "How do I fix this?"
    assert result.provider == "none"


def test_process_query_english():
    """Test processing an English query (no translation needed)."""
    query = "How do I troubleshoot print quality?"
    query_retrieval, lang_result, translation_result = process_query_for_retrieval(query)
    
    assert query_retrieval == query  # Should be unchanged
    assert lang_result.lang == "en"
    assert translation_result is None or translation_result.provider == "none"


def test_technical_token_preservation():
    """Test that technical tokens are preserved during translation."""
    # This test would require mocking the LLM translation
    # For now, we just verify the function doesn't crash
    query = "Check error code E1234 at C:\\Program Files\\app.log"
    result = detect_language(query)
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


