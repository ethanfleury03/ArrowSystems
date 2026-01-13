"""
Deterministic machine model matcher for ticket text.

Matches machine model names/aliases against ticket conversation text.
Uses word-boundary matching and scoring to avoid false positives.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .machine_models_loader import MachineModel


# Common English words to exclude from matching (short tokens that cause false positives)
COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let",
    "put", "say", "she", "too", "use", "any", "ask", "cut", "fix", "run", "set"
}


def normalize_text(text: str) -> str:
    """
    Normalize text for matching: lowercase, collapse whitespace, remove punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_tokens(text: str, min_length: int = 3) -> List[str]:
    """
    Extract tokens from text, filtering out common words and short tokens.
    
    Args:
        text: Input text
        min_length: Minimum token length
        
    Returns:
        List of tokens
    """
    normalized = normalize_text(text)
    tokens = normalized.split()
    
    # Filter: min length, not common word
    filtered = [
        t for t in tokens
        if len(t) >= min_length and t not in COMMON_WORDS
    ]
    
    return filtered


def find_matches(
    text: str,
    models: List[MachineModel],
    min_score: int = 50
) -> List[Dict[str, Any]]:
    """
    Find machine model matches in text.
    
    Args:
        text: Ticket conversation text to search
        models: List of MachineModel objects to match against
        min_score: Minimum score threshold (default: 50)
        
    Returns:
        List of match dicts, sorted by score (descending):
        {
            "model_id": int,
            "model_name": str,
            "match_source": "name" | "alias" | "token",
            "score": int,
            "evidence_snippet": str
        }
    """
    if not text or not models:
        return []
    
    normalized_text = normalize_text(text)
    matches = []
    
    for model in models:
        # Try exact full name match (highest score)
        name_normalized = normalize_text(model.name)
        if name_normalized in normalized_text:
            # Find snippet (50 chars before/after)
            idx = normalized_text.find(name_normalized)
            start = max(0, idx - 50)
            end = min(len(normalized_text), idx + len(name_normalized) + 50)
            snippet = text[start:end] if text else normalized_text[start:end]
            
            matches.append({
                "model_id": model.id,
                "model_name": model.name,
                "match_source": "name",
                "score": 100,
                "evidence_snippet": snippet
            })
            continue  # Don't check aliases if exact name match found
        
        # Try alias matches (medium score)
        best_alias_match = None
        best_alias_score = 0
        
        for alias in model.aliases:
            alias_normalized = normalize_text(alias)
            if alias_normalized != name_normalized and alias_normalized in normalized_text:
                # Use word-boundary matching for aliases (more strict)
                # Check if alias appears as whole word
                pattern = r'\b' + re.escape(alias_normalized) + r'\b'
                if re.search(pattern, normalized_text):
                    idx = normalized_text.find(alias_normalized)
                    start = max(0, idx - 50)
                    end = min(len(normalized_text), idx + len(alias_normalized) + 50)
                    snippet = text[start:end] if text else normalized_text[start:end]
                    
                    if not best_alias_match or len(alias) > len(best_alias_match["alias"]):
                        best_alias_match = {
                            "model_id": model.id,
                            "model_name": model.name,
                            "match_source": "alias",
                            "score": 80,
                            "evidence_snippet": snippet,
                            "alias": alias
                        }
        
        if best_alias_match:
            matches.append(best_alias_match)
            continue
        
        # Try partial token match (weak score, only if unique)
        # Extract meaningful tokens from model name
        model_tokens = extract_tokens(model.name, min_length=3)
        if model_tokens:
            # Check if any token appears as whole word
            for token in model_tokens:
                pattern = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern, normalized_text):
                    # Check if this token is unique to this model
                    is_unique = True
                    for other_model in models:
                        if other_model.id == model.id:
                            continue
                        other_tokens = extract_tokens(other_model.name, min_length=3)
                        if token in other_tokens:
                            is_unique = False
                            break
                    
                    if is_unique:
                        idx = normalized_text.find(token)
                        start = max(0, idx - 50)
                        end = min(len(normalized_text), idx + len(token) + 50)
                        snippet = text[start:end] if text else normalized_text[start:end]
                        
                        matches.append({
                            "model_id": model.id,
                            "model_name": model.name,
                            "match_source": "token",
                            "score": 50,
                            "evidence_snippet": snippet
                        })
                        break  # Only one token match per model
    
    # Filter by min_score and sort by score descending
    filtered = [m for m in matches if m["score"] >= min_score]
    filtered.sort(key=lambda x: x["score"], reverse=True)
    
    return filtered


def determine_assignment(
    matches: List[Dict[str, Any]],
    ambiguous_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Determine assignment status based on matches.
    
    Args:
        matches: List of match dicts (from find_matches)
        ambiguous_threshold: If top score - second score < threshold * top_score, mark as ambiguous
        
    Returns:
        Assignment dict:
        {
            "machine_model_ids": List[int],
            "status": "unassigned" | "assigned" | "ambiguous",
            "confidence": float,
            "method": str
        }
    """
    if not matches:
        return {
            "machine_model_ids": [],
            "status": "unassigned",
            "confidence": 0.0,
            "method": "regex_match_v1"
        }
    
    # Get unique model IDs (deduplicate by model_id)
    seen_ids = set()
    unique_matches = []
    for match in matches:
        if match["model_id"] not in seen_ids:
            seen_ids.add(match["model_id"])
            unique_matches.append(match)
    
    if len(unique_matches) == 0:
        return {
            "machine_model_ids": [],
            "status": "unassigned",
            "confidence": 0.0,
            "method": "regex_match_v1"
        }
    
    # Sort by score
    unique_matches.sort(key=lambda x: x["score"], reverse=True)
    
    top_score = unique_matches[0]["score"]
    model_ids = [m["model_id"] for m in unique_matches]
    
    # Determine status
    if len(unique_matches) == 1:
        # Single match: assigned
        status = "assigned"
        confidence = top_score / 100.0
    elif len(unique_matches) > 1:
        # Multiple matches: check if ambiguous
        second_score = unique_matches[1]["score"]
        score_diff = top_score - second_score
        threshold_score = top_score * ambiguous_threshold
        
        if score_diff < threshold_score:
            # Scores are too close: ambiguous
            status = "ambiguous"
            confidence = top_score / 100.0  # Use top score as confidence
        else:
            # Top score is clearly higher: assigned (but include all matches)
            status = "assigned"
            confidence = top_score / 100.0
    else:
        status = "unassigned"
        confidence = 0.0
    
    return {
        "machine_model_ids": model_ids,
        "status": status,
        "confidence": confidence,
        "method": "regex_match_v1"
    }


if __name__ == "__main__":
    # Self-test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.machine_models_loader import MachineModel
    
    # Create test models
    models = [
        MachineModel(1, "DuraFlex"),
        MachineModel(2, "EZCut 330"),
        MachineModel(3, "2800"),
    ]
    
    # Test cases
    test_cases = [
        ("I have a DuraFlex machine that's not working", ["DuraFlex"]),
        ("The Dura Flex printer is broken", ["DuraFlex"]),
        ("My EZCut 330 needs repair", ["EZCut 330"]),
        ("Model 2800 is having issues", ["2800"]),
        ("I need help with my printer", []),  # No match
        ("DuraFlex and EZCut 330 both mentioned", ["DuraFlex", "EZCut 330"]),  # Multiple
    ]
    
    print("Running self-test...")
    for text, expected_models in test_cases:
        matches = find_matches(text, models, min_score=50)
        matched_names = [m["model_name"] for m in matches]
        status = "✓" if set(matched_names) == set(expected_models) else "✗"
        print(f"{status} '{text}' → {matched_names} (expected: {expected_models})")
    
    print("\nSelf-test complete!")
