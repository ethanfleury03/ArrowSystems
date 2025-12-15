"""
Multilingual query translation module.

Handles language detection and translation to English for RAG retrieval,
while preserving technical tokens (URLs, error codes, file paths, etc.).
"""

import os
import re
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Language detection - try fasttext first, fallback to langdetect
_langdetect_available = False
_langdetect_method = None

try:
    import fasttext
    # Try to load fasttext model (will download on first use)
    try:
        _fasttext_model = fasttext.load_model('lid.176.bin')
        _langdetect_available = True
        _langdetect_method = "fasttext"
        logger.info("Using fasttext for language detection")
    except Exception as e:
        logger.warning(f"Fasttext model not available: {e}, trying langdetect...")
        _fasttext_model = None
        try:
            from langdetect import detect_langs, LangDetectException
            _langdetect_available = True
            _langdetect_method = "langdetect"
            logger.info("Using langdetect for language detection")
        except ImportError:
            logger.warning("Neither fasttext nor langdetect available, language detection disabled")
            _langdetect_available = False
            _langdetect_method = None
except ImportError:
    _fasttext_model = None
    try:
        from langdetect import detect_langs, LangDetectException
        _langdetect_available = True
        _langdetect_method = "langdetect"
        logger.info("Using langdetect for language detection")
    except ImportError:
        logger.warning("No language detection library available, language detection disabled")
        _langdetect_available = False
        _langdetect_method = None


@dataclass
class LangDetectResult:
    """Language detection result."""
    lang: str  # ISO 639-1 language code (e.g., "en", "es", "fr")
    confidence: float  # 0.0 to 1.0


@dataclass
class TranslationResult:
    """Translation result."""
    translated_text: str
    provider: str  # "llm", "fasttext", etc.


# Technical token patterns to preserve during translation
TECH_TOKEN_PATTERNS = [
    # URLs
    (r'https?://[^\s]+', '__URL__'),
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '__EMAIL__'),
    # Windows file paths
    (r'[A-Za-z]:\\[^\s]+', '__WINPATH__'),
    # Unix file paths
    (r'/[^\s]+', '__UNIXPATH__'),
    # IP addresses
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '__IP__'),
    # Hex strings / long IDs / serials (e.g., 0xDEADBEEF, ABC123DEF456)
    (r'0x[0-9a-fA-F]+', '__HEX__'),
    (r'\b[A-Z0-9]{8,}\b', '__SERIAL__'),
    # Error codes (ALLCAPS with numbers/underscores, e.g., E1234, ERR_45, RESULT_123)
    (r'\b[A-Z][A-Z0-9_]*[0-9][A-Z0-9_]*\b', '__ERRORCODE__'),
    # Printer model names (mixed letters+digits like "DuraFlex123", "X2-9000")
    (r'\b[A-Z][A-Za-z0-9-]+[0-9][A-Za-z0-9-]*\b', '__MODEL__'),
    # Code blocks (backtick-delimited)
    (r'`[^`]+`', '__CODE__'),
]


def detect_language(text: str) -> LangDetectResult:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        LangDetectResult with detected language and confidence
    """
    if not text or len(text.strip()) < 6:
        # Very short text or mostly symbols - default to English with low confidence
        return LangDetectResult(lang="en", confidence=0.1)
    
    # Check if text is mostly numeric/symbols
    alphanumeric_ratio = sum(1 for c in text if c.isalnum()) / len(text) if text else 0
    if alphanumeric_ratio < 0.3:
        return LangDetectResult(lang="en", confidence=0.2)
    
    if not _langdetect_available:
        # Fallback: assume English if detection unavailable
        return LangDetectResult(lang="en", confidence=0.5)
    
    try:
        if _langdetect_method == "fasttext":
            # Fasttext detection
            predictions = _fasttext_model.predict(text.replace('\n', ' '), k=1)
            lang_code = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            # Normalize language code to ISO 639-1
            lang_code = lang_code[:2] if len(lang_code) > 2 else lang_code
            
            return LangDetectResult(lang=lang_code, confidence=confidence)
        
        elif _langdetect_method == "langdetect":
            # Langdetect detection
            from langdetect import detect_langs, LangDetectException
            try:
                detected = detect_langs(text)
                if detected:
                    top_lang = detected[0]
                    return LangDetectResult(
                        lang=top_lang.lang,
                        confidence=top_lang.prob
                    )
            except LangDetectException:
                pass
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
    
    # Fallback to English
    return LangDetectResult(lang="en", confidence=0.5)


def _extract_technical_tokens(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract technical tokens from text and replace with placeholders.
    
    Returns:
        Tuple of (text_with_placeholders, list of (placeholder, original_token))
    """
    tokens = []
    result_text = text
    placeholder_map = {}
    
    # Apply patterns in order
    for pattern, prefix in TECH_TOKEN_PATTERNS:
        matches = list(re.finditer(pattern, result_text))
        for i, match in enumerate(matches):
            placeholder = f"{prefix}{len(placeholder_map)}"
            original = match.group(0)
            placeholder_map[placeholder] = original
            result_text = result_text.replace(original, placeholder, 1)
    
    # Return text with placeholders and mapping
    return result_text, [(k, v) for k, v in placeholder_map.items()]


def _restore_technical_tokens(text: str, token_map: List[Tuple[str, str]]) -> str:
    """Restore technical tokens from placeholders."""
    result = text
    for placeholder, original in token_map:
        result = result.replace(placeholder, original)
    return result


@lru_cache(maxsize=5000)
def _translate_with_llm(text: str, source_lang: str) -> str:
    """
    Translate text to English using LLM (Claude).
    
    This is cached to avoid repeated API calls for the same input.
    """
    try:
        import anthropic
        
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Build translation prompt
        prompt = f"""Translate the following text from {source_lang} to English. 
Translate ONLY the text, preserving all placeholders exactly as they appear (they represent technical tokens).
Do not add any commentary, explanation, or formatting. Return only the translated text.

Text to translate:
{text}"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,  # Deterministic translation
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        translated = response.content[0].text.strip()
        return translated
        
    except Exception as e:
        logger.error(f"LLM translation failed: {e}")
        raise


def translate_to_english(text: str, source_lang: str) -> TranslationResult:
    """
    Translate text to English while preserving technical tokens.
    
    Args:
        text: Source text to translate
        source_lang: Source language code (ISO 639-1)
        
    Returns:
        TranslationResult with translated text and provider name
        
    Raises:
        Exception if translation fails
    """
    # If already English, return as-is
    if source_lang == "en":
        return TranslationResult(translated_text=text, provider="none")
    
    # Extract technical tokens
    text_with_placeholders, token_map = _extract_technical_tokens(text)
    
    # Translate using LLM (since we already use Claude for other operations)
    try:
        translated_with_placeholders = _translate_with_llm(text_with_placeholders, source_lang)
        
        # Restore technical tokens
        final_translation = _restore_technical_tokens(translated_with_placeholders, token_map)
        
        return TranslationResult(translated_text=final_translation, provider="llm")
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


def process_query_for_retrieval(query_original: str, confidence_threshold: float = 0.6) -> Tuple[str, LangDetectResult, Optional[TranslationResult]]:
    """
    Process a user query for retrieval: detect language and translate if needed.
    
    Args:
        query_original: Original user query
        confidence_threshold: Minimum confidence to attempt translation (default 0.6)
        
    Returns:
        Tuple of (query_for_retrieval, lang_detect_result, translation_result)
        - query_for_retrieval: The query to use for retrieval (English)
        - lang_detect_result: Language detection result
        - translation_result: Translation result if translation occurred, None otherwise
    """
    # Detect language
    lang_result = detect_language(query_original)
    
    # Determine if we should translate
    should_translate = (
        lang_result.lang != "en" and 
        lang_result.confidence >= confidence_threshold
    )
    
    translation_result = None
    query_for_retrieval = query_original
    
    if should_translate:
        try:
            translation_result = translate_to_english(query_original, lang_result.lang)
            query_for_retrieval = translation_result.translated_text
            logger.info(
                f"Translated query from {lang_result.lang} to English "
                f"(confidence: {lang_result.confidence:.2f})"
            )
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")
            # Fallback to original query
            query_for_retrieval = query_original
    
    return query_for_retrieval, lang_result, translation_result

