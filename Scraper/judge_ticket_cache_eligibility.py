#!/usr/bin/env python3
"""
Deterministic-first cache eligibility classification for Zendesk tickets.

Pipeline (Option 1):
1) Hard-block check: Strict word-boundary safe patterns for denied/onsite/replacement
2) Hard-allow check: Strict allow-list for tickets with explicit confirmation + steps + problem
3) Sonnet judge: Only for remaining tickets (not blocked, not allowed)

Configuration:
  Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL in Scraper/.env
  Default model: claude-sonnet-4-20250514
  Override model per-run with --model flag

Example usage:
  # Judge single ticket (dry-run)
  python judge_ticket_cache_eligibility.py --id 3599 --dry-run
  
  # Spot-check a ticket (print detailed output)
  python judge_ticket_cache_eligibility.py --id 4246 --print
  
  # Judge all solved tickets (limit 20)
  python judge_ticket_cache_eligibility.py --all --limit 20
  
  # Re-judge existing tickets
  python judge_ticket_cache_eligibility.py --all --force
  
  # Deterministic-only mode (no LLM calls)
  python judge_ticket_cache_eligibility.py --all --deterministic-only
  
  # Deterministic report (summary only, no DB writes)
  python judge_ticket_cache_eligibility.py --all --deterministic-report
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

import config
import db

# Prompt version for reproducibility
PROMPT_VERSION = "cache_elig_v2"

# Transcript limits
MAX_MESSAGE_CHARS = 2000
MAX_TOTAL_TRANSCRIPT_CHARS = 40000

# Hard-block phrases (case-insensitive)
# Short keywords (< 5 chars) should use word boundaries; multi-word phrases are safe as-is
# IMPORTANT: Only include phrases that indicate business outcomes, not technical log messages
HARD_BLOCK_PHRASES = {
    "denied": [
        "warranty claim is denied",
        "claim denied",
        "warranty denied",
        "rma denied",
        "claim rejected",
        "warranty claim rejected",
        "outside the terms",
        "outside warranty",
        "outside warranty terms",
        # Note: Generic "denied" and "rejected" require claim/warranty context check
        "rejected",  # Will check for claim context
    ],
    "needs_onsite": [
        "onsite",
        "on site",
        "on-site",
        "site visit",
        "field service",
        "dispatch",
        "dispatched",
        "technician visit",
        "send a tech",
        "send technician",
        "truck roll",
        "cannot be resolved remotely",
        "cannot resolve remotely",
        "not possible remotely",
    ],
    "needs_replacement": [
        "requires return",
        "send back",
        "rma",  # Short keyword - will use word boundaries
        "requires replacement",
        "needs replacement",
        "install a new one",
        "replace cradle",
        "replace printhead",
        "replace board",
        "return to memjet",
        "return to manufacturer",
    ],
    "no_fix_provided": [
        "unable to reproduce",
        "cannot reproduce",
        "customer handled offline",
        "no fix provided",
        "no resolution",
    ],
}

# Short keywords that need word boundary matching (to avoid false positives)
SHORT_KEYWORDS_NEED_BOUNDARIES = {"rma", "onsite", "dispatch"}

# Patterns that indicate log lines (should be ignored unless paired with explicit denial phrases)
LOG_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO timestamp
    r"MISC_REJECTED",
    r"DEVICE_MANAGER",
    r"QaLssMgr",
    r"Ngq",
    r"Operation not allowed",
    r"invalid certificate",
    r"system rejected",
    r"connection denied",
    r"access denied",
    r"row sync",
    r"sync rejected",
]

# Technical contexts that should NOT trigger denial hard-block
TECHNICAL_DENIAL_IGNORE = [
    "connection denied",
    "access denied",
    "permission denied",
    "system rejected",
    "sync rejected",
    "row sync",
    "device rejected",
    "certificate rejected",
]


def _regex_hit(pattern: str, text: str) -> Optional[re.Match]:
    """
    Check if pattern matches in text using regex with word boundaries.
    
    Args:
        pattern: Regex pattern (will be wrapped with word boundaries if needed)
        text: Text to search
        
    Returns:
        Match object if found, None otherwise
    """
    # Ensure word boundaries for the pattern
    if not pattern.startswith(r"\b"):
        pattern = r"\b" + re.escape(pattern) + r"\b"
    else:
        # Pattern already has boundaries, just escape special chars
        pattern = pattern
    
    try:
        return re.search(pattern, text, re.IGNORECASE)
    except re.error:
        # Fallback to simple substring if regex fails
        return None


def _context(text: str, match: re.Match, window: int = 30) -> str:
    """
    Extract context around a match for debugging.
    
    Args:
        text: Full text
        match: Regex match object
        window: Characters before/after to include
        
    Returns:
        Context string with match highlighted
    """
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    match_text = match.group()
    context_before = text[start:match.start()]
    context_after = text[match.end():end]
    return f"...{context_before}[{match_text}]{context_after}..."


def _check_phrase_match(phrase: str, text: str, use_word_boundaries: bool = False) -> Optional[Tuple[re.Match, str]]:
    """
    Check if phrase matches in text, optionally using word boundaries.
    
    Args:
        phrase: Phrase to search for
        text: Text to search
        use_word_boundaries: If True, use regex word boundaries
        
    Returns:
        Tuple of (match, context) if found, None otherwise
    """
    if use_word_boundaries:
        match = _regex_hit(phrase, text)
        if match:
            context = _context(text, match)
            return match, context
    else:
        # Multi-word phrases: simple substring search (safe)
        phrase_lower = phrase.lower()
        text_lower = text.lower()
        idx = text_lower.find(phrase_lower)
        if idx >= 0:
            # Create a pseudo-match for context extraction
            match_start = idx
            match_end = idx + len(phrase)
            # Extract context manually
            start = max(0, match_start - 30)
            end = min(len(text), match_end + 30)
            context_before = text[start:match_start]
            context_after = text[match_end:end]
            match_text = text[match_start:match_end]
            context = f"...{context_before}[{match_text}]{context_after}..."
            # Create a simple match-like object
            class SimpleMatch:
                def __init__(self, start, end, group):
                    self.start_val = start
                    self.end_val = end
                    self.group_val = group
                def start(self): return self.start_val
                def end(self): return self.end_val
                def group(self): return self.group_val
            return SimpleMatch(match_start, match_end, match_text), context
    
    return None


def _is_log_line(text: str) -> bool:
    """
    Check if text appears to be a log line (contains timestamps, module tags, etc.).
    
    Args:
        text: Text to check
        
    Returns:
        True if text looks like a log line
    """
    text_lower = text.lower()
    # Check for log patterns
    for pattern in LOG_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _is_technical_denial(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if a "denied" or "rejected" match is in a technical context (not warranty/claim).
    
    Args:
        text: Full text (lowercase)
        match_start: Start position of match
        match_end: End position of match
        
    Returns:
        True if match is in technical context (should be ignored)
    """
    # Extract context around match
    context_start = max(0, match_start - 100)
    context_end = min(len(text), match_end + 100)
    context = text[context_start:context_end]
    
    # Check for technical denial patterns
    for ignore_pattern in TECHNICAL_DENIAL_IGNORE:
        if ignore_pattern.lower() in context:
            return True
    
    # Check if it's in a log line
    line_start = text.rfind("\n", 0, match_start)
    line_end = text.find("\n", match_end)
    if line_start < 0:
        line_start = 0
    if line_end < 0:
        line_end = len(text)
    line_text = text[line_start:line_end]
    
    if _is_log_line(line_text):
        return True
    
    return False


def _has_claim_context(text: str, match_start: int, match_end: int) -> bool:
    """
    Check if a "rejected" or "denied" match has claim/warranty context nearby.
    
    Args:
        text: Full text (lowercase)
        match_start: Start position of match
        match_end: End position of match
        
    Returns:
        True if claim/warranty context found within 50 chars
    """
    # Look for claim/warranty keywords in nearby context
    context_start = max(0, match_start - 50)
    context_end = min(len(text), match_end + 50)
    context = text[context_start:context_end]
    
    claim_keywords = ["claim", "warranty", "rma", "return", "replacement"]
    return any(keyword in context for keyword in claim_keywords)


def hard_block(conversation: Dict[str, Any]) -> tuple[bool, List[str], Optional[str]]:
    """
    Check if ticket should be hard-blocked based on deterministic phrase matching.
    
    Uses word boundaries for short keywords to avoid false positives (e.g., "rma" in "information").
    Ignores technical/log contexts for "denied" and "rejected" unless paired with claim/warranty context.
    
    Args:
        conversation: Conversation JSON dict
        
    Returns:
        Tuple of (is_blocked: bool, blockers: List[str], outcome_category: str | None)
        outcome_category is one of: "denied", "needs_onsite", "needs_replacement", "no_fix_provided", "workaround_only"
    """
    # Get all text from conversation (preserve original case for context)
    text_parts_original = []
    text_parts_lower = []
    
    # Add subject
    request = conversation.get("request", {})
    subject = request.get("subject", "")
    if subject:
        text_parts_original.append(subject)
        text_parts_lower.append(subject.lower())
    
    # Add all message text (prioritize last messages)
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    
    # Sort by created_at
    messages.sort(key=lambda m: m.get("created_at", ""))
    
    # Get last 5 messages (most recent) for priority checking
    last_messages = messages[-5:] if len(messages) > 5 else messages
    
    for msg in messages:
        text = msg.get("text", "").strip()
        if text:
            text_parts_original.append(text)
            text_parts_lower.append(text.lower())
    
    # Combine all text (keep both original and lower for matching)
    full_text_original = " ".join(text_parts_original)
    full_text_lower = " ".join(text_parts_lower)
    
    # Check each category
    for outcome_type, phrases in HARD_BLOCK_PHRASES.items():
        for phrase in phrases:
            # Determine if we need word boundaries (short keywords)
            use_word_boundaries = phrase.lower() in SHORT_KEYWORDS_NEED_BOUNDARIES
            
            # Try matching
            match_result = _check_phrase_match(phrase, full_text_lower, use_word_boundaries=use_word_boundaries)
            
            if match_result:
                match_obj, context = match_result
                match_start = match_obj.start()
                match_end = match_obj.end()
                
                # Special handling for "rma": check for negative context
                if phrase.lower() == "rma":
                    # Check for negative indicators before/after the match
                    context_start = max(0, match_start - 20)
                    context_end = min(len(full_text_lower), match_end + 20)
                    context_text = full_text_lower[context_start:context_end]
                    # Negative indicators
                    negative_patterns = ["no rma", "not rma", "rma not", "rma is not", "without rma"]
                    if any(neg in context_text for neg in negative_patterns):
                        # Skip this match - it's negative context
                        continue
                
                # Special handling for "denied" category: check for technical contexts
                if outcome_type == "denied":
                    # For generic "denied" or "rejected", require claim/warranty context
                    if phrase.lower() in ["denied", "rejected"]:
                        if not _has_claim_context(full_text_lower, match_start, match_end):
                            # Skip - no claim/warranty context
                            continue
                    
                    # Check if this is a technical denial (connection denied, access denied, etc.)
                    if _is_technical_denial(full_text_lower, match_start, match_end):
                        # Skip - this is a technical denial, not a warranty/claim denial
                        continue
                
                # Special handling for replacement: check if there's confirmation after
                if outcome_type == "needs_replacement":
                    # If phrase appears in last messages, it's likely still pending
                    last_text_lower = " ".join([msg.get("text", "").lower() for msg in last_messages])
                    last_match = _check_phrase_match(phrase, last_text_lower, use_word_boundaries=use_word_boundaries)
                    
                    if last_match:
                        # Match in recent messages - likely still pending
                        pattern_desc = f"\\b{phrase}\\b" if use_word_boundaries else phrase
                        return True, [f"hard_blocked: {outcome_type} (pattern: '{pattern_desc}', match: '{match_obj.group()}', context: '{context}')"], outcome_type
                    
                    # If it's earlier, check for success indicators after
                    after_text = full_text_lower[match_start:match_start + 500]
                    success_indicators = ["resolved", "fixed", "working", "success", "installed", "replaced"]
                    if not any(indicator in after_text for indicator in success_indicators):
                        pattern_desc = f"\\b{phrase}\\b" if use_word_boundaries else phrase
                        return True, [f"hard_blocked: {outcome_type} (pattern: '{pattern_desc}', match: '{match_obj.group()}', context: '{context}' without confirmed success)"], outcome_type
                else:
                    # Other categories: block immediately
                    pattern_desc = f"\\b{phrase}\\b" if use_word_boundaries else phrase
                    return True, [f"hard_blocked: {outcome_type} (pattern: '{pattern_desc}', match: '{match_obj.group()}', context: '{context}')"], outcome_type
    
    return False, [], None


def hard_allow(
    conversation: Dict[str, Any],
    require_requester_confirmation: bool = True
) -> tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Check if ticket should be hard-allowed (deterministic cache_eligible=1).
    
    Hard-allow conditions (ALL required):
    1) Confirmation phrase from requester/customer in final part of ticket
    2) Actionable steps exist (>=2 steps OR >=1 step + concrete config/value)
    3) Clear problem statement exists (first requester message with error/symptom)
    4) Not hard-blocked
    
    Args:
        conversation: Conversation JSON dict
        require_requester_confirmation: If True, only accept requester confirmations (default: True)
        
    Returns:
        Tuple of (allowed: bool, evidence: dict | None, blockers: List[str])
    """
    blockers = []
    
    # Check hard-block first
    is_blocked, block_reasons, _ = hard_block(conversation)
    if is_blocked:
        blockers.extend(block_reasons)
        return False, None, blockers
    
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    
    # Sort by created_at
    messages.sort(key=lambda m: m.get("created_at", ""))
    
    if not messages:
        blockers.append("no messages found")
        return False, None, blockers
    
    # 1) Check for confirmation from requester/customer in final part
    # Use shared confirmation detection function
    requester_author_ids = get_requester_author_ids(conversation, messages)
    confirmation_evidence = find_requester_confirmation_evidence(
        messages, 
        require_requester_confirmation,
        conversation=conversation,
        requester_author_ids=requester_author_ids
    )
    
    confirmation_found = confirmation_evidence is not None
    confirmation_text = confirmation_evidence.get("quote", "") if confirmation_evidence else None
    confirmation_message_idx = confirmation_evidence.get("message_index") if confirmation_evidence else None
    
    if not confirmation_found:
        blockers.append("missing explicit requester confirmation in final messages")
    
    # 2) Extract actionable steps from agent messages
    steps = []
    step_message_indices = []
    
    # Imperative verb patterns
    imperative_verb_pattern = r"\b(check|replace|reseat|power off|swap|update|install|verify|run|ssh|execute|set|configure|reset|restart|clear|enable|disable|change|adjust|test|scan|reboot|power cycle)"
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "").lower()
        text = msg.get("text", "").strip()
        
        # Only look at agent messages for steps
        if role not in ["agent", "staff", "admin"]:
            continue
        
        # Look for bullet points
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for bullet points: "-", "*", "1.", "2.", etc.
            if re.match(r"^[-*•]\s+", line) or re.match(r"^\d+\.\s+", line):
                # Extract step text (remove bullet)
                step_text = re.sub(r"^[-*•]\s+", "", line)
                step_text = re.sub(r"^\d+\.\s+", "", step_text)
                if len(step_text) >= 10:  # Minimum step length
                    steps.append(step_text)
                    if i not in step_message_indices:
                        step_message_indices.append(i)
            
            # Check for imperative verb sentences
            elif re.search(imperative_verb_pattern, line, re.IGNORECASE):
                # Extract sentence
                sentences = re.split(r"[.!?]\s+", line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if re.search(imperative_verb_pattern, sentence, re.IGNORECASE):
                        if len(sentence) >= 10:
                            steps.append(sentence)
                            if i not in step_message_indices:
                                step_message_indices.append(i)
                            break
    
    # Check if we have enough steps
    # Require >=2 steps OR >=1 step + concrete config/value (part number, serial, command, speed value)
    has_concrete_value = False
    if steps:
        # Check for concrete values in steps
        concrete_patterns = [
            (r"\b\d{4,}\b", None),  # Part numbers, serials (4+ digits)
            (r"\b[A-Z]{2,}\d+\b", None),  # Alphanumeric codes
            (r"\b(speed|voltage|current|frequency)\s*[:=]\s*\d+", re.IGNORECASE),  # Config values
            (r"\b(command|run|execute)\s*[:=]", re.IGNORECASE),  # Commands
        ]
        for step in steps:
            for pattern_tuple in concrete_patterns:
                pattern, flags = pattern_tuple
                if flags:
                    if re.search(pattern, step, flags):
                        has_concrete_value = True
                        break
                else:
                    if re.search(pattern, step):
                        has_concrete_value = True
                        break
            if has_concrete_value:
                break
    
    if len(steps) < 2 and not (len(steps) >= 1 and has_concrete_value):
        blockers.append(f"insufficient actionable steps (found {len(steps)}, need >=2 or >=1 with concrete value)")
    
    # 3) Check for clear problem statement in first requester message
    problem_found = False
    problem_text = None
    problem_message_idx = None
    
    # Find first requester message
    for i, msg in enumerate(messages):
        role = msg.get("role", "").lower()
        text = msg.get("text", "").strip()
        
        if role in ["requester", "user", "customer"] and len(text) >= 30:
            # Check for technical tokens: ALL CAPS, underscore error tokens, digits
            technical_patterns = [
                r"\b[A-Z]{3,}_[A-Z_]+\b",  # ERROR_CODE, RESULT_NOT_READY
                r"\b[A-Z]{4,}\b",  # VC, RSYNC, PEP
                r"\b\d{3,}\b",  # Error codes with digits
            ]
            
            has_technical_token = any(re.search(pattern, text) for pattern in technical_patterns)
            
            # Also check for explicit error/symptom keywords
            error_keywords = ["error", "fault", "fail", "issue", "problem", "not working", "broken"]
            has_error_keyword = any(keyword in text.lower() for keyword in error_keywords)
            
            if has_technical_token or has_error_keyword:
                problem_found = True
                problem_text = text[:200]  # Trim to first 200 chars
                problem_message_idx = i
                break
    
    if not problem_found:
        blockers.append("no clear problem statement in first requester message (need error code/symptom + component)")
    
    # All checks passed?
    # Require: confirmation_found AND (steps >= 2 OR (steps >= 1 AND has_concrete_value)) AND problem_found
    if confirmation_found and (len(steps) >= 2 or (len(steps) >= 1 and has_concrete_value)) and problem_found:
        evidence = {
            "confirmation": {
                "text": confirmation_text,
                "message_idx": confirmation_message_idx
            },
            "steps": steps,
            "step_message_indices": step_message_indices,
            "problem": {
                "text": problem_text,
                "message_idx": problem_message_idx
            }
        }
        return True, evidence, []
    else:
        return False, None, blockers


def get_requester_author_ids(conversation: Dict[str, Any], messages: List[Dict[str, Any]]) -> set:
    """
    Identify requester author IDs from ticket conversation.
    
    Uses multiple strategies:
    1. Extract requester_id from conversation.request if available
    2. Find first message with role in ["requester", "user", "customer"] and use its author_id
    3. Fallback: find non-staff author that appears most in early messages
    
    Args:
        conversation: Full conversation dict with 'request' metadata
        messages: List of message dicts
        
    Returns:
        Set of author_id strings that belong to the requester
    """
    requester_ids = set()
    
    # Strategy 1: Get requester_id from request metadata
    request = conversation.get("request", {})
    if isinstance(request, dict):
        requester_id = request.get("requester_id")
        if requester_id:
            requester_ids.add(str(requester_id))
    
    # Strategy 2: Find first message with requester role
    sorted_messages = sorted(messages, key=lambda m: m.get("created_at", ""))
    for msg in sorted_messages:
        role = msg.get("role", "").lower()
        author_id = msg.get("author_id", "")
        if role in ["requester", "user", "customer"] and author_id:
            requester_ids.add(str(author_id))
            break  # Use first requester message's author_id
    
    # Strategy 3: Fallback - find most common non-staff author in early messages
    if not requester_ids:
        # Look at first 5 messages to find requester
        early_messages = sorted_messages[:5] if len(sorted_messages) > 5 else sorted_messages
        author_counts = {}
        for msg in early_messages:
            role = msg.get("role", "").lower()
            author_id = msg.get("author_id", "")
            # Skip staff/agent roles
            if role not in ["agent", "staff", "admin"] and author_id:
                author_counts[author_id] = author_counts.get(author_id, 0) + 1
        
        if author_counts:
            # Use most common non-staff author
            most_common = max(author_counts.items(), key=lambda x: x[1])
            requester_ids.add(str(most_common[0]))
    
    return requester_ids


def find_requester_confirmation_evidence(
    messages: List[Dict[str, Any]],
    require_requester_confirmation: bool = True,
    conversation: Optional[Dict[str, Any]] = None,
    requester_author_ids: Optional[set] = None
) -> Optional[Dict[str, Any]]:
    """
    Find requester confirmation evidence in messages using deterministic patterns.
    
    This is the single source of truth for confirmation detection, used by both
    hard_allow() and backfill operations.
    
    Args:
        messages: List of message dicts with 'role', 'text', 'created_at', 'author_id'
        require_requester_confirmation: If True, only accept requester messages (default: True)
        conversation: Optional conversation dict for requester ID detection
        requester_author_ids: Optional pre-computed set of requester author IDs
        
    Returns:
        Evidence dict with:
            - message_index: int (index in chronological messages list)
            - created_at: str (message timestamp)
            - role: str (requester/user/customer vs agent/staff)
            - author_id: str (optional)
            - quote: str (short snippet, max 200 chars, containing matched text)
            - matched_pattern: str (pattern string that matched)
        Or None if no valid confirmation found
    """
    if not messages:
        return None
    
    # Sort by created_at to ensure chronological order
    sorted_messages = sorted(messages, key=lambda m: m.get("created_at", ""))
    
    # Identify requester author IDs if needed
    if require_requester_confirmation and requester_author_ids is None:
        if conversation:
            requester_author_ids = get_requester_author_ids(conversation, sorted_messages)
        else:
            # Fallback: use role-based detection
            requester_author_ids = set()
            for msg in sorted_messages[:5]:  # Check first 5 messages
                role = msg.get("role", "").lower()
                author_id = msg.get("author_id", "")
                if role in ["requester", "user", "customer"] and author_id:
                    requester_author_ids.add(str(author_id))
    
    # Examine last N messages (N=max(3, len(messages)//4)) - same as hard_allow
    last_n = max(3, len(sorted_messages) // 4)
    last_messages = sorted_messages[-last_n:] if len(sorted_messages) > last_n else sorted_messages
    
    # Improved confirmation regex patterns - catch more variations
    confirmation_patterns = [
        # Strong positive signals
        (re.compile(r"\b(issue|problem|it|this|that) (is )?(resolved|fixed|working|solved|gone|fixed now)\b", re.IGNORECASE), "resolved/fixed/working"),
        (re.compile(r"\bnow (it )?(is )?(working|fixed|resolved|solved)\b", re.IGNORECASE), "now working/fixed"),
        (re.compile(r"\b(it|this|that) (is )?working\b", re.IGNORECASE), "is working"),
        (re.compile(r"\b.* (is|are) working\b", re.IGNORECASE), "is working"),
        (re.compile(r"\bsolved\b", re.IGNORECASE), "solved"),
        (re.compile(r"\bfixed\b", re.IGNORECASE), "fixed"),
        (re.compile(r"\b(resolved|fixed|solved) (the )?(issue|problem)\b", re.IGNORECASE), "resolved issue"),
        # Removed/cleared patterns (e.g., "it removed streak")
        (re.compile(r"\b(it|this|that) (removed|cleared|fixed|resolved)\b", re.IGNORECASE), "removed/cleared"),
        (re.compile(r"\b(removed|cleared|fixed|resolved) (the )?(streak|issue|problem|error)\b", re.IGNORECASE), "removed issue"),
        # Close ticket patterns (only if paired with positive signal)
        (re.compile(r"\b(close|closing) (the )?(case|ticket).*(working|fixed|resolved|solved)\b", re.IGNORECASE), "close ticket + working"),
        (re.compile(r"\b.* (working|fixed|resolved|solved).*(close|closing) (the )?(case|ticket)\b", re.IGNORECASE), "working + close ticket"),
        (re.compile(r"\b(we|i) (can|will) close\b", re.IGNORECASE), "can close"),
        (re.compile(r"\bclose (the )?(case|ticket)\b", re.IGNORECASE), "close ticket"),
        # Thank you with resolution
        (re.compile(r"\bthank you.*(resolved|fixed|working|solved)\b", re.IGNORECASE), "thank you + resolved"),
        (re.compile(r"\b(resolved|fixed|working|solved).*thank you\b", re.IGNORECASE), "resolved + thank you"),
        # Initialization/startup success
        (re.compile(r"\b(it|this|that) (is )?(initializing|initialized)\b", re.IGNORECASE), "initializing"),
    ]
    
    # Negative patterns (should NOT match) - block uncertain/conditional statements
    negative_patterns = [
        re.compile(r"can you confirm", re.IGNORECASE),
        re.compile(r"please confirm", re.IGNORECASE),
        re.compile(r"resolved\?", re.IGNORECASE),
        re.compile(r"\bshould (work|be|fix)", re.IGNORECASE),
        re.compile(r"\b(hope|hoping) (it|this|that)", re.IGNORECASE),
        re.compile(r"\bwill try\b", re.IGNORECASE),
        re.compile(r"\bi think\b", re.IGNORECASE),
        re.compile(r"\bnot sure (yet|if)", re.IGNORECASE),
        re.compile(r"\bstill (happening|occurring|getting)", re.IGNORECASE),
        re.compile(r"^thank you\.?$", re.IGNORECASE),  # "Thank you" alone is NOT confirmation
    ]
    
    # Search from end (most recent) backwards to find latest confirmation
    for i in range(len(last_messages) - 1, -1, -1):
        msg = last_messages[i]
        role = msg.get("role", "").lower()
        text = msg.get("text", "").strip()
        created_at = msg.get("created_at", "")
        author_id = msg.get("author_id", "")
        
        if not text:
            continue
        
        # Check for negative patterns first
        if any(neg_pattern.search(text) for neg_pattern in negative_patterns):
            continue
        
        # Check if this is a requester message (by author_id or role)
        is_requester = False
        if require_requester_confirmation:
            if requester_author_ids and author_id:
                is_requester = str(author_id) in requester_author_ids
            else:
                # Fallback to role-based check
                is_requester = role in ["requester", "user", "customer"]
        else:
            is_requester = True  # Allow any role
        
        # Check for confirmation patterns
        for pattern, pattern_str in confirmation_patterns:
            match = pattern.search(text)
            if match:
                if require_requester_confirmation and not is_requester:
                    # Check if agent is quoting requester
                    if '"' in text or "'" in text:
                        # Might be quoting, but be strict - require explicit quote markers
                        match_start = match.start()
                        match_end = match.end()
                        quote_start = max(0, match_start - 50)
                        quote_end = min(len(text), match_end + 50)
                        quote = text[quote_start:quote_end].strip()
                        
                        # Find original message index in full sorted list
                        original_idx = sorted_messages.index(msg)
                        
                        return {
                            "message_index": original_idx,
                            "created_at": created_at,
                            "role": "requester",  # Quoted from requester
                            "author_id": author_id,
                            "quote": quote[:200],  # Max 200 chars
                            "matched_pattern": pattern_str
                        }
                    continue
                
                # Found confirmation from requester
                match_start = match.start()
                match_end = match.end()
                quote_start = max(0, match_start - 50)
                quote_end = min(len(text), match_end + 50)
                quote = text[quote_start:quote_end].strip()
                
                # Find original message index in full sorted list
                original_idx = sorted_messages.index(msg)
                
                return {
                    "message_index": original_idx,
                    "created_at": created_at,
                    "role": role if is_requester else "requester",
                    "author_id": author_id,
                    "quote": quote[:200],  # Max 200 chars
                    "matched_pattern": pattern_str
                }
    
    return None


def outcome_requires_steps(outcome: str) -> bool:
    """
    Determine if an outcome requires resolution steps to determine cache eligibility.
    
    Args:
        outcome: Outcome string (e.g., "resolved_remotely_actionable", "needs_replacement", etc.)
        
    Returns:
        True only when outcome == "resolved_remotely_actionable"
    """
    return outcome == "resolved_remotely_actionable"


def _self_test_hard_allow() -> None:
    """
    Self-test cases for hard-allow detection.
    Tests that hard-allow only triggers on tickets with all required conditions.
    """
    test_cases = [
        # (description, conversation_dict, should_allow, expected_blockers)
        ("Perfect ticket: confirmation + steps + problem", {
            "request": {"subject": "VC consumables error"},
            "messages": [
                {"role": "requester", "text": "Getting VC consumables error RESULT_NOT_READY", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "agent", "text": "Try these steps:\n- Check consumables level\n- Replace if needed\n- Run command: verify_consumables", "created_at": "2024-01-01T10:05:00Z"},
                {"role": "requester", "text": "Issue is resolved, thank you!", "created_at": "2024-01-01T10:10:00Z"}
            ]
        }, True, []),
        
        ("Missing confirmation", {
            "request": {"subject": "VC error"},
            "messages": [
                {"role": "requester", "text": "VC consumables error RESULT_NOT_READY", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "agent", "text": "Try:\n- Check consumables\n- Replace if needed", "created_at": "2024-01-01T10:05:00Z"}
            ]
        }, False, ["missing explicit requester confirmation"]),
        
        ("Missing steps", {
            "request": {"subject": "VC error"},
            "messages": [
                {"role": "requester", "text": "VC consumables error RESULT_NOT_READY", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "agent", "text": "We will review this issue", "created_at": "2024-01-01T10:05:00Z"},
                {"role": "requester", "text": "Issue is resolved", "created_at": "2024-01-01T10:10:00Z"}
            ]
        }, False, ["insufficient actionable steps"]),
        
        ("Missing problem statement", {
            "request": {"subject": "Question"},
            "messages": [
                {"role": "requester", "text": "How do I configure this?", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "agent", "text": "Try:\n- Step 1\n- Step 2", "created_at": "2024-01-01T10:05:00Z"},
                {"role": "requester", "text": "Working now", "created_at": "2024-01-01T10:10:00Z"}
            ]
        }, False, ["no clear problem statement"]),
        
        ("Hard-blocked ticket (RMA)", {
            "request": {"subject": "RMA needed"},
            "messages": [
                {"role": "requester", "text": "Device error RSYNC_OVERRUN", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "agent", "text": "We need to process an RMA", "created_at": "2024-01-01T10:05:00Z"}
            ]
        }, False, ["hard_blocked"]),
    ]
    
    print("Running hard-allow self-tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for desc, conversation, should_allow, expected_blockers in test_cases:
        allowed, evidence, blockers = hard_allow(conversation, require_requester_confirmation=True)
        
        # Check if allowing matches expectation
        allow_match = (allowed == should_allow)
        
        # Check blockers if not allowed
        blockers_match = True
        if not allowed and expected_blockers:
            # Check if any expected blocker is present
            blocker_text = " ".join(blockers).lower()
            blockers_match = any(expected.lower() in blocker_text for expected in expected_blockers)
        
        if allow_match and blockers_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Expected: allow={should_allow}, blockers={expected_blockers}")
        print(f"  Got: allow={allowed}, blockers={blockers}")
        if not allow_match:
            print(f"  ERROR: Allow mismatch!")
        if not blockers_match:
            print(f"  ERROR: Blockers mismatch!")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_hard_block() -> None:
    """
    Self-test cases for hard-block detection.
    Tests word boundary matching to avoid false positives.
    """
    test_cases = [
        # (description, conversation_dict, should_block, expected_outcome_type)
        ("RMA required", {
            "request": {"subject": "RMA required"},
            "messages": [{"text": "We need to process an RMA for this unit", "created_at": "2024-01-01"}]
        }, True, "needs_replacement"),
        
        ("infoRMAtion (should NOT block)", {
            "request": {"subject": "Information request"},
            "messages": [{"text": "Please provide more information about the issue", "created_at": "2024-01-01"}]
        }, False, None),
        
        ("rma. (with punctuation)", {
            "request": {"subject": "Return request"},
            "messages": [{"text": "Please initiate rma. Thank you.", "created_at": "2024-01-01"}]
        }, True, "needs_replacement"),
        
        ("no rma required (negative context)", {
            "request": {"subject": "No return needed"},
            "messages": [{"text": "No RMA required for this issue", "created_at": "2024-01-01"}]
        }, False, None),  # Conservative: don't block negative cases
        
        ("warranty claim is denied", {
            "request": {"subject": "Warranty claim"},
            "messages": [{"text": "The warranty claim is denied", "created_at": "2024-01-01"}]
        }, True, "denied"),
        
        ("onsite visit", {
            "request": {"subject": "Onsite needed"},
            "messages": [{"text": "We need to schedule an onsite visit", "created_at": "2024-01-01"}]
        }, True, "needs_onsite"),
        
        # False positive tests - should NOT block
        ("system rejected in log", {
            "request": {"subject": "System issue"},
            "messages": [{"text": "2024-09-10T10:00:00 system rejected on 2 additional row syncs", "created_at": "2024-01-01"}]
        }, False, None),
        
        ("MISC_REJECTED in log", {
            "request": {"subject": "Device issue"},
            "messages": [{"text": "MISC_REJECTED Operation not allowed for device with an invalid certificate", "created_at": "2024-01-01"}]
        }, False, None),
        
        ("connection denied", {
            "request": {"subject": "Network issue"},
            "messages": [{"text": "Connection denied to server", "created_at": "2024-01-01"}]
        }, False, None),
        
        ("access denied", {
            "request": {"subject": "Access issue"},
            "messages": [{"text": "Access denied to the system", "created_at": "2024-01-01"}]
        }, False, None),
        
        # True positive tests - should block
        ("warranty claim is denied", {
            "request": {"subject": "Warranty claim"},
            "messages": [{"text": "The warranty claim is denied", "created_at": "2024-01-01"}]
        }, True, "denied"),
        
        ("claim rejected with context", {
            "request": {"subject": "Claim"},
            "messages": [{"text": "Your claim has been rejected due to warranty terms", "created_at": "2024-01-01"}]
        }, True, "denied"),
    ]
    
    print("Running hard-block self-tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for desc, conversation, should_block, expected_outcome in test_cases:
        is_blocked, blockers, outcome_category = hard_block(conversation)
        reason = blockers[0] if blockers else None
        
        # Check if blocking matches expectation
        block_match = (is_blocked == should_block)
        
        # Check outcome type if blocked
        outcome_match = True
        if is_blocked and expected_outcome:
            outcome_match = expected_outcome in reason.lower() if reason else False
        
        if block_match and outcome_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Expected: block={should_block}, outcome={expected_outcome}")
        print(f"  Got: block={is_blocked}, reason={reason}")
        if not block_match:
            print(f"  ERROR: Blocking mismatch!")
        if not outcome_match:
            print(f"  ERROR: Outcome type mismatch!")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_review_status() -> None:
    """
    Self-test cases for review_status determination using new asymmetric policy.
    Tests that review_status is set correctly based on confidence thresholds.
    """
    test_cases = [
        # (description, outcome, confirmed, steps_count, confidence, stage, cache_eligible, approve_min, reject_min, unclear_reject_min, expected_status, expected_cache_eligible)
        ("denied -> rejected", "denied", False, 0, 0.50, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("needs_replacement -> rejected", "needs_replacement", False, 0, 0.70, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("resolved_remotely_actionable @0.90 confirmed True -> approved", "resolved_remotely_actionable", True, 2, 0.92, "sonnet", 1, 0.90, 0.60, 0.80, "approved", 1),
        ("resolved_remotely_actionable @0.85 confirmed True -> needs_review", "resolved_remotely_actionable", True, 2, 0.85, "sonnet", 0, 0.90, 0.60, 0.80, "needs_review", 0),
        ("unclear @0.30 -> needs_review", "unclear", False, 0, 0.30, "sonnet", 0, 0.90, 0.60, 0.80, "needs_review", 0),
        ("cache_eligible==0 conf=0.70 outcome=denied -> rejected", "denied", False, 0, 0.70, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("hard_block stage -> rejected", "needs_replacement", False, 0, 0.50, "hard_block", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("hard_allow stage -> approved", "resolved_remotely_actionable", True, 2, 1.0, "hard_allow", 1, 0.90, 0.60, 0.80, "approved", 1),
        ("unclear @0.85 -> rejected", "unclear", False, 0, 0.85, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("needs_onsite -> rejected", "needs_onsite", False, 0, 0.50, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("no_fix_provided -> rejected", "no_fix_provided", False, 0, 0.50, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
        ("workaround_only -> rejected", "workaround_only", False, 0, 0.50, "sonnet", 0, 0.90, 0.60, 0.80, "rejected", 0),
    ]
    
    print("Running review_status self-tests (new asymmetric policy)...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for desc, outcome, confirmed, steps_count, confidence, stage, cache_eligible, approve_min, reject_min, unclear_reject_min, expected_status, expected_cache_eligible in test_cases:
        blockers = []
        if outcome != "resolved_remotely_actionable":
            blockers.append(f"outcome is '{outcome}'")
        if not confirmed and outcome == "resolved_remotely_actionable":
            blockers.append("No confirmation of issue resolution")
        if steps_count == 0 and outcome == "resolved_remotely_actionable":
            blockers.append("No actionable resolution steps provided")
        
        review_status, review_reason, review_reasons = determine_review_status(
            outcome=outcome,
            confirmation_confirmed=confirmed,
            resolution_steps_count=steps_count,
            model_confidence=confidence,
            blockers=blockers,
            stage=stage,
            cache_eligible=cache_eligible,
            approve_min_confidence=approve_min,
            reject_min_confidence=reject_min,
            unclear_reject_min_confidence=unclear_reject_min,
            require_requester_confirmation=True
        )
        
        # Determine cache_eligible based on review_status
        new_cache_eligible = 1 if review_status == "approved" else 0
        
        status_match = (review_status == expected_status)
        cache_match = (new_cache_eligible == expected_cache_eligible)
        
        if status_match and cache_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Outcome: {outcome}, confirmed={confirmed}, steps={steps_count}, conf={confidence}, stage={stage}")
        print(f"  Expected: review_status={expected_status}, cache_eligible={expected_cache_eligible}")
        print(f"  Got: review_status={review_status}, cache_eligible={new_cache_eligible}, reason={review_reason}")
        if not status_match:
            print(f"  ERROR: Review status mismatch!")
        if not cache_match:
            print(f"  ERROR: Cache eligible mismatch!")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_confirmation_evidence() -> None:
    """
    Self-test cases for confirmation evidence validation.
    Tests that confirmed=True without evidence is forced to confirmed=False.
    """
    print("Running confirmation evidence validation self-tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test case: confirmed=True but evidence quote missing
    test_cases = [
        {
            "desc": "confirmed=True with missing evidence quote -> should force confirmed=False",
            "confirmed": True,
            "evidence": {},  # Missing quote
            "transcript": "Ticket #123\n[0] <2024-01-01> <requester> author=user\nIssue resolved, thank you!",
            "require_requester_confirmation": True,
            "expected_confirmed": False,
            "expected_blocker": "confirmation_evidence_not_grounded"
        },
        {
            "desc": "confirmed=True with quote not in transcript -> should force confirmed=False",
            "confirmed": True,
            "evidence": {"quote": "this quote does not exist", "author_role": "requester"},
            "transcript": "Ticket #123\n[0] <2024-01-01> <requester> author=user\nDifferent text here",
            "require_requester_confirmation": True,
            "expected_confirmed": False,
            "expected_blocker": "confirmation_evidence_not_grounded"
        },
        {
            "desc": "confirmed=True with valid quote -> should keep confirmed=True",
            "confirmed": True,
            "evidence": {"quote": "Issue resolved", "author_role": "requester"},
            "transcript": "Ticket #123\n[0] <2024-01-01> <requester> author=user\nIssue resolved, thank you!",
            "require_requester_confirmation": True,
            "expected_confirmed": True,
            "expected_blocker": None
        },
        {
            "desc": "confirmed=True with agent role when requester required -> should force confirmed=False",
            "confirmed": True,
            "evidence": {"quote": "Issue resolved", "author_role": "agent"},
            "transcript": "Ticket #123\n[0] <2024-01-01> <agent> author=agent\nIssue resolved",
            "require_requester_confirmation": True,
            "expected_confirmed": False,
            "expected_blocker": "confirmation_evidence_not_grounded"
        }
    ]
    
    for test_case in test_cases:
        desc = test_case["desc"]
        confirmed = test_case["confirmed"]
        evidence = test_case["evidence"]
        transcript = test_case["transcript"]
        require_requester_confirmation = test_case["require_requester_confirmation"]
        expected_confirmed = test_case["expected_confirmed"]
        expected_blocker = test_case.get("expected_blocker")
        
        # Simulate validation logic
        blockers = []
        if confirmed:
            evidence_valid = False
            evidence_quote = evidence.get("quote", "")
            evidence_role = evidence.get("author_role", "")
            
            if evidence_quote:
                if evidence_quote.lower() in transcript.lower():
                    evidence_valid = True
                    # Check role if required
                    if require_requester_confirmation and evidence_role:
                        if evidence_role.lower() != "requester":
                            evidence_valid = False
            
            if not evidence_valid:
                confirmed = False
                blockers.append("confirmation_evidence_not_grounded")
        
        # Check results
        confirmed_match = (confirmed == expected_confirmed)
        blocker_match = (expected_blocker is None) or (expected_blocker in blockers)
        
        if confirmed_match and blocker_match:
            passed += 1
            print(f"  PASS: {desc}")
        else:
            failed += 1
            print(f"  FAIL: {desc}")
            if not confirmed_match:
                print(f"    Expected confirmed={expected_confirmed}, got {confirmed}")
            if not blocker_match:
                print(f"    Expected blocker={expected_blocker}, got {blockers}")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_tail_preserving_transcript() -> None:
    """
    Self-test for tail-preserving transcript truncation.
    Tests that last messages are preserved even when truncating.
    """
    print("Running tail-preserving transcript self-tests...")
    print("=" * 60)
    
    # Create a conversation with many messages
    messages = []
    for i in range(100):
        messages.append({
            "message_id": str(i),
            "role": "requester" if i == 0 else ("requester" if i % 2 == 0 else "agent"),
            "created_at": f"2024-01-01T{i:02d}:00:00Z",
            "text": f"Message {i}: " + "x" * 100  # Each message ~110 chars
        })
    
    # Add confirmation in last message
    messages[-1]["text"] = "Issue resolved, thank you!"
    
    conversation = {
        "ticket_id": "12345",
        "request": {
            "subject": "Test ticket",
            "status": "solved",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T23:00:00Z"
        },
        "messages": messages
    }
    
    transcript = build_transcript(conversation)
    
    # Check that last message (with confirmation) is in transcript
    if "Issue resolved" in transcript:
        print("  PASS: Last message with confirmation preserved in transcript")
        print("  PASSED: Tail-preserving truncation works")
    else:
        print("  FAIL: Last message with confirmation NOT found in transcript")
        print("  FAILED: Tail-preserving truncation failed")
        sys.exit(1)
    
    print("=" * 60)


def _self_test_hard_block_outcome() -> None:
    """
    Self-test that hard_block returns explicit outcome category.
    """
    print("Running hard_block explicit outcome self-tests...")
    print("=" * 60)
    
    test_cases = [
        {
            "desc": "denied phrase -> outcome='denied'",
            "conversation": {
                "request": {"subject": "Warranty claim"},
                "messages": [{"text": "The warranty claim is denied", "created_at": "2024-01-01"}]
            },
            "expected_blocked": True,
            "expected_outcome": "denied"
        },
        {
            "desc": "onsite phrase -> outcome='needs_onsite'",
            "conversation": {
                "request": {"subject": "Site visit"},
                "messages": [{"text": "We need to dispatch a technician onsite", "created_at": "2024-01-01"}]
            },
            "expected_blocked": True,
            "expected_outcome": "needs_onsite"
        },
        {
            "desc": "replacement phrase -> outcome='needs_replacement'",
            "conversation": {
                "request": {"subject": "RMA"},
                "messages": [{"text": "We need to process an RMA for this unit", "created_at": "2024-01-01"}]
            },
            "expected_blocked": True,
            "expected_outcome": "needs_replacement"
        },
        {
            "desc": "no blocking phrase -> not blocked",
            "conversation": {
                "request": {"subject": "General question"},
                "messages": [{"text": "How do I configure the system?", "created_at": "2024-01-01"}]
            },
            "expected_blocked": False,
            "expected_outcome": None
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        desc = test_case["desc"]
        conversation = test_case["conversation"]
        expected_blocked = test_case["expected_blocked"]
        expected_outcome = test_case["expected_outcome"]
        
        is_blocked, blockers, outcome_category = hard_block(conversation)
        
        blocked_match = (is_blocked == expected_blocked)
        outcome_match = (outcome_category == expected_outcome)
        
        if blocked_match and outcome_match:
            passed += 1
            print(f"  PASS: {desc}")
        else:
            failed += 1
            print(f"  FAIL: {desc}")
            if not blocked_match:
                print(f"    Expected blocked={expected_blocked}, got {is_blocked}")
            if not outcome_match:
                print(f"    Expected outcome={expected_outcome}, got {outcome_category}")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_find_confirmation_evidence() -> None:
    """
    Self-test cases for find_requester_confirmation_evidence function.
    Tests that evidence is found correctly with proper index/quote/role.
    """
    print("Running find_requester_confirmation_evidence self-tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test case 1: requester confirmation present -> evidence found
    conversation1 = {"request": {"requester_id": "user1"}, "messages": []}
    messages1 = [
        {"role": "requester", "text": "I have an issue with my printer", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Try these steps:\n1. Check cables\n2. Restart printer", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "Thank you! The issue is resolved now.", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence1 = find_requester_confirmation_evidence(messages1, require_requester_confirmation=True, conversation=conversation1)
    if evidence1 and evidence1.get("quote") and "resolved" in evidence1["quote"].lower():
        passed += 1
        print(f"  PASS: Requester confirmation found with correct quote")
    else:
        failed += 1
        print(f"  FAIL: Expected requester confirmation evidence, got {evidence1}")
    
    # Test case 2: "now it is working" -> evidence found
    conversation2 = {"request": {"requester_id": "user1"}, "messages": []}
    messages2 = [
        {"role": "requester", "text": "I have an issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Try this fix", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "After changing switch pump two error showed up -> replaced circulation pump now it is working. Thank you.", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence2 = find_requester_confirmation_evidence(messages2, require_requester_confirmation=True, conversation=conversation2)
    if evidence2 and "working" in evidence2.get("quote", "").lower():
        passed += 1
        print(f"  PASS: 'now it is working' pattern detected")
    else:
        failed += 1
        print(f"  FAIL: Expected evidence for 'now it is working', got {evidence2}")
    
    # Test case 3: "dongle is working... close ticket" -> evidence found
    conversation3 = {"request": {"requester_id": "user1"}, "messages": []}
    messages3 = [
        {"role": "requester", "text": "Dongle issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Update license", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "The dongle is working after updating the license. I will close the ticket. Thanks", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence3 = find_requester_confirmation_evidence(messages3, require_requester_confirmation=True, conversation=conversation3)
    if evidence3 and ("working" in evidence3.get("quote", "").lower() or "close" in evidence3.get("quote", "").lower()):
        passed += 1
        print(f"  PASS: 'dongle is working... close ticket' pattern detected")
    else:
        failed += 1
        print(f"  FAIL: Expected evidence for 'dongle is working... close ticket', got {evidence3}")
    
    # Test case 4: "it removed streak" -> evidence found
    conversation4 = {"request": {"requester_id": "user1"}, "messages": []}
    messages4 = [
        {"role": "requester", "text": "Bad streak issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Check tubing", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "After I changed the tubing, it removed steak. Thank you for the help.", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence4 = find_requester_confirmation_evidence(messages4, require_requester_confirmation=True, conversation=conversation4)
    if evidence4 and "removed" in evidence4.get("quote", "").lower():
        passed += 1
        print(f"  PASS: 'it removed streak' pattern detected")
    else:
        failed += 1
        print(f"  FAIL: Expected evidence for 'it removed streak', got {evidence4}")
    
    # Test case 5: agent confirmation only -> evidence not found (when requester required)
    conversation5 = {"request": {"requester_id": "user1"}, "messages": []}
    messages5 = [
        {"role": "requester", "text": "I have an issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "The issue is resolved", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"}
    ]
    
    evidence5 = find_requester_confirmation_evidence(messages5, require_requester_confirmation=True, conversation=conversation5)
    if evidence5 is None:
        passed += 1
        print(f"  PASS: Agent-only confirmation correctly rejected when requester required")
    else:
        failed += 1
        print(f"  FAIL: Expected no evidence for agent-only confirmation, got {evidence5}")
    
    # Test case 6: "Thank you" alone -> not confirmed
    conversation6 = {"request": {"requester_id": "user1"}, "messages": []}
    messages6 = [
        {"role": "requester", "text": "I have an issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Try this", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "Thank you.", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence6 = find_requester_confirmation_evidence(messages6, require_requester_confirmation=True, conversation=conversation6)
    if evidence6 is None:
        passed += 1
        print(f"  PASS: 'Thank you' alone correctly rejected")
    else:
        failed += 1
        print(f"  FAIL: Expected no evidence for 'Thank you' alone, got {evidence6}")
    
    # Test case 7: "should work" -> not confirmed (negative pattern)
    conversation7 = {"request": {"requester_id": "user1"}, "messages": []}
    messages7 = [
        {"role": "requester", "text": "I have an issue", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Try this", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "It should work now", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence7 = find_requester_confirmation_evidence(messages7, require_requester_confirmation=True, conversation=conversation7)
    if evidence7 is None:
        passed += 1
        print(f"  PASS: 'should work' correctly rejected (uncertain)")
    else:
        failed += 1
        print(f"  FAIL: Expected no evidence for 'should work', got {evidence7}")
    
    # Test case 8: valid confirmation with proper message_index
    conversation8 = {"request": {"requester_id": "user1"}, "messages": []}
    messages8 = [
        {"role": "requester", "text": "Problem report", "created_at": "2024-01-01T10:00:00Z", "author_id": "user1"},
        {"role": "agent", "text": "Solution steps", "created_at": "2024-01-01T11:00:00Z", "author_id": "agent1"},
        {"role": "requester", "text": "The problem is fixed, thank you!", "created_at": "2024-01-01T12:00:00Z", "author_id": "user1"}
    ]
    
    evidence8 = find_requester_confirmation_evidence(messages8, require_requester_confirmation=True, conversation=conversation8)
    if evidence8 and evidence8.get("message_index") == 2:
        passed += 1
        print(f"  PASS: Correct message_index returned (expected 2, got {evidence8['message_index']})")
    else:
        failed += 1
        print(f"  FAIL: Expected message_index=2, got {evidence8.get('message_index') if evidence8 else None}")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def _self_test_blocker_generation() -> None:
    """
    Self-test cases for blocker generation logic.
    Tests that blockers are only added when outcome requires steps.
    """
    test_cases = [
        # (description, outcome, steps_count, confirmed, should_have_steps_blocker, should_have_confirmation_blocker)
        ("resolved_remotely_actionable with steps=0", "resolved_remotely_actionable", 0, True, True, False),
        ("resolved_remotely_actionable with confirmed=False", "resolved_remotely_actionable", 2, False, False, True),
        ("resolved_remotely_actionable with both missing", "resolved_remotely_actionable", 0, False, True, True),
        ("resolved_remotely_actionable perfect", "resolved_remotely_actionable", 2, True, False, False),
        ("needs_replacement with steps=0", "needs_replacement", 0, False, False, False),
        ("needs_onsite with steps=0", "needs_onsite", 0, False, False, False),
        ("denied with steps=0", "denied", 0, False, False, False),
        ("unclear with steps=0", "unclear", 0, False, False, False),
    ]
    
    print("Running blocker generation self-tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for desc, outcome, steps_count, confirmed, should_have_steps_blocker, should_have_confirmation_blocker in test_cases:
        blockers = []
        requires_steps = outcome_requires_steps(outcome)
        
        # Simulate blocker generation logic
        if outcome == "resolved_remotely_actionable":
            if confirmed and steps_count >= 1:
                # Would be cache_eligible=1, no blockers needed
                pass
            else:
                if not confirmed:
                    blockers.append("No confirmation of issue resolution")
                if steps_count < 1:
                    blockers.append("No actionable resolution steps provided")
        else:
            blockers.append(f"outcome is '{outcome}' (not 'resolved_remotely_actionable')")
        
        # Check blockers
        has_steps_blocker = "No actionable resolution steps provided" in blockers
        has_confirmation_blocker = "No confirmation of issue resolution" in blockers
        
        steps_match = (has_steps_blocker == should_have_steps_blocker)
        confirmation_match = (has_confirmation_blocker == should_have_confirmation_blocker)
        
        if steps_match and confirmation_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Outcome: {outcome}, steps={steps_count}, confirmed={confirmed}")
        print(f"  Requires steps: {requires_steps}")
        print(f"  Expected: steps_blocker={should_have_steps_blocker}, confirmation_blocker={should_have_confirmation_blocker}")
        print(f"  Got: steps_blocker={has_steps_blocker}, confirmation_blocker={has_confirmation_blocker}")
        print(f"  Blockers: {blockers}")
        if not steps_match:
            print(f"  ERROR: Steps blocker mismatch!")
        if not confirmation_match:
            print(f"  ERROR: Confirmation blocker mismatch!")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("FAILED: Some tests did not pass")
        sys.exit(1)
    else:
        print("PASSED: All tests passed")


def build_transcript(conversation: Dict[str, Any]) -> str:
    """
    Build a compact transcript string from conversation JSON.
    
    Uses tail-preserving truncation: allocates 60% of chars to tail (latest messages)
    and 40% to head (earlier messages) to avoid missing confirmations in final messages.
    
    Args:
        conversation: Conversation JSON dict
        
    Returns:
        Formatted transcript string
    """
    ticket_id = conversation.get("ticket_id", "unknown")
    request = conversation.get("request", {})
    subject = request.get("subject", "N/A")
    status = request.get("status", "N/A")
    created_at = request.get("created_at", "N/A")
    updated_at = request.get("updated_at", "N/A")
    
    # Header
    header_lines = [
        f"Ticket #{ticket_id}, subject: {subject}, status: {status}",
        f"created_at: {created_at}, updated_at: {updated_at}",
        ""
    ]
    header_chars = sum(len(line) for line in header_lines)
    
    # Messages
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    
    # Sort by created_at
    messages.sort(key=lambda m: m.get("created_at", ""))
    
    if not messages:
        return "\n".join(header_lines)
    
    # Tail-preserving truncation:
    # - Allocate 60% of available chars to tail (last messages)
    # - Allocate 40% to head (earlier messages)
    # - This ensures confirmations in final messages are not lost
    available_chars = MAX_TOTAL_TRANSCRIPT_CHARS - header_chars
    tail_chars = int(available_chars * 0.6)  # 60% for tail
    head_chars = available_chars - tail_chars  # 40% for head
    
    # Format messages into (index, formatted_lines, char_count) tuples
    formatted_messages = []
    for i, msg in enumerate(messages):
        created_at = msg.get("created_at", "N/A")
        role = msg.get("role", "unknown")
        author_id = msg.get("author_id", "N/A")
        text = msg.get("text", "").strip()
        attachments = msg.get("attachments", [])
        
        # Truncate per-message text
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "...[TRUNCATED]"
        
        # Format message
        msg_lines = [
            f"[{i}] <{created_at}> <{role}> author={author_id}",
            text
        ]
        
        if attachments:
            msg_lines.append(f"attachments: {len(attachments)}")
        
        msg_lines.append("")
        
        msg_text = "\n".join(msg_lines)
        formatted_messages.append((i, msg_lines, len(msg_text)))
    
    # Build transcript: first add tail (last messages), then head (earlier messages)
    lines = header_lines.copy()
    total_chars = header_chars
    
    # Add tail messages (from end, working backwards)
    tail_lines = []
    tail_total = 0
    for i, msg_lines, msg_chars in reversed(formatted_messages):
        if tail_total + msg_chars > tail_chars:
            break
        tail_lines.insert(0, (i, msg_lines, msg_chars))
        tail_total += msg_chars
    
    # Add head messages (from start, working forwards, excluding those already in tail)
    tail_indices = {i for i, _, _ in tail_lines}
    head_lines = []
    head_total = 0
    for i, msg_lines, msg_chars in formatted_messages:
        if i in tail_indices:
            continue
        if head_total + msg_chars > head_chars:
            break
        head_lines.append((i, msg_lines, msg_chars))
        head_total += msg_chars
    
    # Combine: head first, then tail
    for i, msg_lines, _ in head_lines:
        lines.extend(msg_lines)
        total_chars += sum(len(line) for line in msg_lines)
    
    if head_lines and tail_lines:
        lines.append("...[MIDDLE TRUNCATED]")
    
    for i, msg_lines, _ in tail_lines:
        lines.extend(msg_lines)
        total_chars += sum(len(line) for line in msg_lines)
    
    return "\n".join(lines)


def get_extractor_prompt(transcript: str) -> tuple[str, str]:
    """
    Get system and user prompts for PASS A (Extractor).
    
    Args:
        transcript: Formatted transcript string
        
    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = """You are a strict JSON-only output classifier. Output ONLY valid JSON, no prose or explanations."""

    user_message = f"""Analyze this support ticket conversation and determine the outcome and cache eligibility.

CRITICAL RULES (MUST follow):
- If outcome != "resolved_remotely_actionable" => cache_eligible MUST be 0
- If confirmation.confirmed is false => cache_eligible MUST be 0
- If resolution.steps is empty => cache_eligible MUST be 0
- Denial language (warranty denied, claim denied, outside warranty) => outcome="denied"
- Replacement discussed without explicit success after replacement => outcome="needs_replacement" and cache_eligible=0
- Onsite visit required => outcome="needs_onsite" and cache_eligible=0
- No fix provided / unable to reproduce => outcome="no_fix_provided" and cache_eligible=0
- Workaround only (no actual fix) => outcome="workaround_only" and cache_eligible=0
- Unclear resolution => outcome="unclear" and cache_eligible=0

A ticket is cache_eligible=1 ONLY if:
1. outcome == "resolved_remotely_actionable"
2. There is a clear problem statement from the requester (grounded in their text)
3. There is at least one actionable resolution step from agent/tech guidance (not generic like "we will review")
4. confirmation.confirmed is true ONLY if there is an explicit statement that the issue is resolved/working/fixed OR for admin/info requests, an explicit acknowledgement of success (e.g., "received the file", "that answered it").
   - confirmed can ONLY be true if there is explicit confirmation text
   - You MUST provide the exact quote in confirmation.evidence.quote and the message_index it came from
   - For resolved_remotely_actionable outcomes, confirmation MUST come from requester/customer (author_role="requester")
   - Generic "thanks" or "ok" without resolution statement does NOT count as confirmed
5. You can quote verbatim evidence from the transcript for problem, resolution, and confirmation

Be conservative: if ANY doubt, set cache_eligible=0 and explain in blockers.

Output JSON with this exact schema:
{{
  "outcome": "resolved_remotely_actionable" | "needs_onsite" | "needs_replacement" | "denied" | "unclear" | "no_fix_provided" | "workaround_only",
  "problem": {{
    "summary": "clear problem statement or empty if unclear",
    "evidence_quote_ids": [0, 1]
  }},
  "resolution": {{
    "steps": ["step1", "step2", ...],
    "evidence_quote_ids": [2, 3]
  }},
  "confirmation": {{
    "confirmed": true|false,
    "evidence": {{
      "message_index": int | null,
      "author_role": "requester"|"agent"|"unknown"|null,
      "quote": str | null
    }},
    "evidence_quote_ids": [4]
  }},
  "cache_eligible": 0|1,
  "confidence": 0.0-1.0,
  "rationale": "brief explanation",
  "blockers": ["reason1", "reason2"]
}}

Transcript:
{transcript}"""

    return system_message, user_message


def get_verifier_prompt(transcript: str, extractor_json: Dict[str, Any]) -> tuple[str, str]:
    """
    Get system and user prompts for PASS B (Verifier).
    
    Args:
        transcript: Formatted transcript string
        extractor_json: JSON output from PASS A
        
    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = """You are a strict JSON-only output verifier. Output ONLY valid JSON, no prose or explanations."""

    extractor_json_str = json.dumps(extractor_json, indent=2, ensure_ascii=False)
    
    user_message = f"""Verify the extractor's analysis against the transcript.

CRITICAL RULES (MUST follow):
- If outcome != "resolved_remotely_actionable" => cache_eligible MUST be 0
- If confirmation.confirmed is false => cache_eligible MUST be 0
- If resolution.steps is empty => cache_eligible MUST be 0
- Denial language => outcome="denied" and cache_eligible=0
- Replacement without confirmed success => outcome="needs_replacement" and cache_eligible=0
- Onsite required => outcome="needs_onsite" and cache_eligible=0

Your task:
1. Check that evidence quote IDs reference valid message indices
2. Ensure resolution.steps are actionable (imperative or clear actions), not generic
3. Ensure confirmation.confirmed is true only if there's explicit confirmation
4. Verify outcome matches the actual conversation result
5. If ANY doubt or missing evidence => set cache_eligible=0

Hard rules:
- If evidence quote IDs don't match transcript => cache_eligible=0
- If resolution.steps are generic ("we will review") => cache_eligible=0
- If confirmation.confirmed is false => cache_eligible=0
- If ending indicates denial/onsite/unresolved/admin closure => cache_eligible=0
- If any required field is missing or weak => cache_eligible=0

Output final JSON with the SAME schema (overwrite fields if needed):
{{
  "outcome": "resolved_remotely_actionable" | "needs_onsite" | "needs_replacement" | "denied" | "unclear" | "no_fix_provided" | "workaround_only",
  "problem": {{
    "summary": "...",
    "evidence_quote_ids": [...]
  }},
  "resolution": {{
    "steps": ["...", ...],
    "evidence_quote_ids": [...]
  }},
  "confirmation": {{
    "confirmed": true|false,
    "evidence": {{
      "message_index": int | null,
      "author_role": "requester"|"agent"|"unknown"|null,
      "quote": str | null
    }},
    "evidence_quote_ids": [...]
  }},
  "cache_eligible": 0|1,
  "confidence": 0.0-1.0,
  "rationale": "...",
  "blockers": ["...", ...]
}}

Transcript:
{transcript}

Extractor Analysis:
{extractor_json_str}"""

    return system_message, user_message


def call_anthropic_api(
    client: Anthropic,
    model: str,
    system_message: str,
    user_message: str,
    max_tokens: int,
    retries: int = 3
) -> str:
    """
    Call Anthropic API with retry logic.
    
    Args:
        client: Anthropic client
        model: Model name
        system_message: System message
        user_message: User message
        max_tokens: Max output tokens
        retries: Number of retries
        
    Returns:
        Response text
    """
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            return response.content[0].text
        except Exception as e:
            if attempt < retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from response text, handling code blocks.
    
    Args:
        text: Response text
        
    Returns:
        Parsed JSON dict or None if invalid
    """
    # Try to extract JSON from code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def determine_review_status(
    outcome: str,
    confirmation_confirmed: bool,
    resolution_steps_count: int,
    model_confidence: float,
    blockers: List[str],
    stage: str = "sonnet",
    cache_eligible: int = 0,
    approve_min_confidence: float = 0.90,
    reject_min_confidence: float = 0.60,
    unclear_reject_min_confidence: float = 0.80,
    require_requester_confirmation: bool = True
) -> Tuple[str, str, List[str]]:
    """
    Determine review_status using asymmetric decision policy.
    
    APPROVE: Strict - only high-confidence eligible tickets
    REJECT: Automatic for clearly non-cacheable outcomes
    NEEDS_REVIEW: Borderline cases only
    
    Args:
        outcome: Outcome string
        confirmation_confirmed: Whether confirmation was confirmed
        resolution_steps_count: Number of resolution steps
        model_confidence: Model confidence score (0.0-1.0)
        blockers: List of blocker reasons
        stage: Stage that produced this judgment ("hard_block", "hard_allow", "sonnet")
        cache_eligible: Current cache_eligible value (0 or 1)
        approve_min_confidence: Minimum confidence to auto-approve (default: 0.90)
        reject_min_confidence: Minimum confidence to auto-reject (default: 0.60)
        unclear_reject_min_confidence: Minimum confidence to reject unclear outcomes (default: 0.80)
        require_requester_confirmation: If True, require confirmation for approval
        
    Returns:
        Tuple of (review_status, review_reason, review_reasons)
    """
    review_reasons = []
    
    # Hard-block outcomes that should always be rejected
    auto_reject_outcomes = {"denied", "needs_replacement", "needs_onsite", 
                           "no_fix_provided", "workaround_only"}
    
    # APPROVE: Strict criteria - ALL must be met
    if (stage != "hard_block" and
        outcome == "resolved_remotely_actionable" and
        model_confidence >= approve_min_confidence):
        
        # Check confirmation requirement
        if require_requester_confirmation and not confirmation_confirmed:
            return ("needs_review", "missing_confirmation", ["missing_confirmation"])
        
        # Check steps requirement (only if outcome requires steps)
        if outcome_requires_steps(outcome) and resolution_steps_count < 1:
            return ("needs_review", "missing_resolution_steps", ["missing_resolution_steps"])
        
        # All criteria met - approve
        return ("approved", "meets_all_criteria", [])
    
    # REJECT: Automatic for clearly non-cacheable outcomes
    # 1) Hard-block stage
    if stage == "hard_block":
        return ("rejected", f"hard_block_{outcome}", [outcome])
    
    # 2) Auto-reject outcomes (always reject regardless of confidence)
    if outcome in auto_reject_outcomes:
        return ("rejected", f"auto_reject_outcome_{outcome}", [outcome])
    
    # 3) Unclear outcome with high confidence -> reject
    if outcome == "unclear" and model_confidence >= unclear_reject_min_confidence:
        return ("rejected", f"unclear_high_confidence_{model_confidence:.2f}", 
               [f"unclear_outcome_confidence_{model_confidence:.2f}"])
    
    # 4) Not eligible + high confidence -> reject
    if (cache_eligible == 0 and 
        model_confidence >= reject_min_confidence and
        outcome != "resolved_remotely_actionable"):
        return ("rejected", f"not_eligible_high_confidence_{model_confidence:.2f}", 
               [f"outcome_{outcome}_confidence_{model_confidence:.2f}"])
    
    # NEEDS_REVIEW: Everything else (borderline cases)
    if outcome == "resolved_remotely_actionable":
        if model_confidence < approve_min_confidence:
            review_reasons.append(f"confidence_{model_confidence:.2f}_below_{approve_min_confidence}")
        if require_requester_confirmation and not confirmation_confirmed:
            review_reasons.append("missing_confirmation")
        if resolution_steps_count < 1:
            review_reasons.append("missing_resolution_steps")
        
        reason = "borderline_eligible" if review_reasons else f"low_confidence_{model_confidence:.2f}"
        return ("needs_review", reason, review_reasons)
    
    # Unclear or other outcomes with low/moderate confidence
    if outcome == "unclear":
        return ("needs_review", f"unclear_low_confidence_{model_confidence:.2f}", 
               [f"unclear_outcome_confidence_{model_confidence:.2f}"])
    
    # Other non-eligible outcomes with low confidence
    return ("needs_review", f"borderline_{outcome}_{model_confidence:.2f}", 
           [f"outcome_{outcome}_confidence_{model_confidence:.2f}"])


def judge_ticket(
    client: Anthropic,
    ticket_id: str,
    conversation: Dict[str, Any],
    model: Optional[str] = None,
    min_confidence: float = 0.90,
    approve_min_confidence: float = 0.90,
    reject_min_confidence: float = 0.60,
    unclear_reject_min_confidence: float = 0.80,
    require_requester_confirmation: bool = True
) -> Dict[str, Any]:
    """
    Judge a single ticket using two-pass LLM approach.
    
    Args:
        client: Anthropic client
        ticket_id: Ticket ID
        conversation: Conversation JSON dict
        model: Model name (defaults to config.get_anthropic_model() if None)
        min_confidence: Minimum confidence threshold (default: 0.90) - legacy, kept for compatibility
        approve_min_confidence: Minimum confidence to auto-approve (default: 0.90)
        reject_min_confidence: Minimum confidence to auto-reject (default: 0.60)
        unclear_reject_min_confidence: Minimum confidence to reject unclear outcomes (default: 0.80)
        require_requester_confirmation: If True, require confirmation for approval
        
    Returns:
        Judgement dict with all required fields
    """
    # Use configured model if not provided
    if model is None:
        model = config.get_anthropic_model()
    
    # Hard-block check is done in main pipeline, not here
    # This function is only called for tickets that passed hard-block and hard-allow checks
    
    # Build transcript
    transcript = build_transcript(conversation)
    
    # PASS A: Extractor
    system_msg_a, user_msg_a = get_extractor_prompt(transcript)
    extractor_response = call_anthropic_api(
        client, model, system_msg_a, user_msg_a, max_tokens=800
    )
    extractor_json = parse_json_response(extractor_response)
    
    if not extractor_json:
        # Retry with JSON repair prompt
        repair_prompt = f"""The previous response was invalid JSON. Please output ONLY valid JSON matching this schema:
{{
  "outcome": "resolved_remotely_actionable" | "needs_onsite" | "needs_replacement" | "denied" | "unclear" | "no_fix_provided" | "workaround_only",
  "problem": {{"summary": "...", "evidence_quote_ids": []}},
  "resolution": {{"steps": ["..."], "evidence_quote_ids": []}},
  "confirmation": {{"confirmed": true|false, "evidence_quote_ids": []}},
  "cache_eligible": 0|1,
  "confidence": 0.0-1.0,
  "rationale": "...",
  "blockers": ["..."]
}}

Previous response:
{extractor_response}"""
        extractor_response = call_anthropic_api(
            client, model, system_msg_a, repair_prompt, max_tokens=800
        )
        extractor_json = parse_json_response(extractor_response)
    
    if not extractor_json:
        # Fallback: invalid JSON - send to review
        return {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "problem": None,
            "resolution_steps_json": json.dumps([], ensure_ascii=False),
            "confirmation": None,
            "evidence_json": json.dumps({}, ensure_ascii=False),
            "blockers_json": json.dumps(["invalid_json"], ensure_ascii=False),
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "raw_response_json": json.dumps({"error": "invalid_json", "extractor_response": extractor_response}, ensure_ascii=False),
            "review_status": "needs_review",
            "review_reason": "llm_schema_or_confidence_missing",
            "review_reasons_json": json.dumps(["invalid_json"], ensure_ascii=False)
        }
    
    # PASS B: Verifier
    system_msg_b, user_msg_b = get_verifier_prompt(transcript, extractor_json)
    verifier_response = call_anthropic_api(
        client, model, system_msg_b, user_msg_b, max_tokens=600
    )
    verifier_json = parse_json_response(verifier_response)
    
    if not verifier_json:
        # Use extractor result if verifier fails
        verifier_json = extractor_json
    
    # Parse new schema (with backward compatibility)
    outcome = verifier_json.get("outcome", "unclear")
    
    # Handle old schema format (backward compatibility)
    if "problem" in verifier_json and isinstance(verifier_json["problem"], str):
        problem_summary = verifier_json.get("problem", "")
        problem_quote_ids = []
    else:
        problem_obj = verifier_json.get("problem", {})
        if isinstance(problem_obj, dict):
            problem_summary = problem_obj.get("summary", "")
            problem_quote_ids = problem_obj.get("evidence_quote_ids", [])
        else:
            problem_summary = str(problem_obj) if problem_obj else ""
            problem_quote_ids = []
    
    # Handle resolution (new schema)
    resolution_obj = verifier_json.get("resolution", {})
    if isinstance(resolution_obj, dict):
        resolution_steps = resolution_obj.get("steps", [])
        resolution_quote_ids = resolution_obj.get("evidence_quote_ids", [])
    else:
        # Backward compatibility: old format had resolution_steps directly
        resolution_steps = verifier_json.get("resolution_steps", [])
        resolution_quote_ids = []
    
    if not isinstance(resolution_steps, list):
        resolution_steps = []
    
    # Handle confirmation (new schema with evidence grounding)
    confirmation_obj = verifier_json.get("confirmation", {})
    confirmation_confirmed = False
    confirmation_quote_ids = []
    confirmation_text = ""
    confirmation_evidence = None
    
    if isinstance(confirmation_obj, dict):
        confirmation_confirmed = confirmation_obj.get("confirmed", False)
        confirmation_quote_ids = confirmation_obj.get("evidence_quote_ids", [])
        confirmation_text = confirmation_obj.get("text", "")
        confirmation_evidence = confirmation_obj.get("evidence", {})
    else:
        # Backward compatibility: old format had confirmation as string
        confirmation_text = str(confirmation_obj) if confirmation_obj else ""
        confirmation_confirmed = bool(confirmation_text)
        confirmation_quote_ids = []
    
    # Validate confirmation evidence (deterministic grounding check)
    if confirmation_confirmed:
        evidence_valid = False
        evidence_quote = None
        evidence_role = None
        
        if confirmation_evidence and isinstance(confirmation_evidence, dict):
            evidence_quote = confirmation_evidence.get("quote", "")
            evidence_role = confirmation_evidence.get("author_role", "")
            message_index = confirmation_evidence.get("message_index")
            
            # Check if quote exists in transcript/messages
            if evidence_quote:
                # Check if quote appears in transcript (case-insensitive, allow substring match)
                if evidence_quote.lower() in transcript.lower():
                    evidence_valid = True
                # Also check in messages directly
                elif conversation.get("messages"):
                    for msg in conversation.get("messages", []):
                        msg_text = msg.get("text", "").lower()
                        if evidence_quote.lower() in msg_text:
                            evidence_valid = True
                            # Verify role matches if provided
                            if evidence_role:
                                msg_role = msg.get("role", "").lower()
                                if evidence_role.lower() == "requester" and msg_role not in ["requester", "user", "customer"]:
                                    evidence_valid = False
                                elif evidence_role.lower() == "agent" and msg_role not in ["agent", "staff", "admin"]:
                                    evidence_valid = False
                            break
        
        # If require_requester_confirmation and outcome requires confirmation, enforce requester role
        if (require_requester_confirmation and 
            outcome == "resolved_remotely_actionable" and 
            evidence_role and 
            evidence_role.lower() != "requester"):
            evidence_valid = False
        
        # If evidence is invalid, force confirmed=False and add blocker
        if not evidence_valid:
            confirmation_confirmed = False
            blockers.append("confirmation_evidence_not_grounded")
            if evidence_quote:
                blockers.append(f"confirmation_quote_not_found_in_transcript: '{evidence_quote[:50]}...'")
            elif not evidence_quote:
                blockers.append("confirmation_evidence_missing_quote")
    
    # Handle evidence (backward compatibility)
    evidence = verifier_json.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    
    blockers = verifier_json.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    
    rationale = verifier_json.get("rationale", "")
    raw_confidence = float(verifier_json.get("confidence", 0.0))
    model_cache_eligible = int(verifier_json.get("cache_eligible", 0))
    
    # Apply strict decision logic:
    # cache_eligible = 1 ONLY if:
    #   - outcome == "resolved_remotely_actionable"
    #   - confirmation.confirmed is True
    #   - len(resolution.steps) >= 1
    #   - not hard_blocked (already checked above)
    #   - confidence >= MIN_CONFIDENCE (optional final gate)
    
    final_cache_eligible = 0
    requires_steps = outcome_requires_steps(outcome)
    
    if outcome == "resolved_remotely_actionable":
        if confirmation_confirmed and len(resolution_steps) >= 1:
            if raw_confidence >= min_confidence:
                final_cache_eligible = 1
            else:
                blockers.append(f"confidence {raw_confidence:.6f} < threshold {min_confidence:.6f}")
        else:
            # Only add steps/confirmation blockers when outcome requires steps
            if not confirmation_confirmed:
                blockers.append("No confirmation of issue resolution")
            if len(resolution_steps) < 1:
                blockers.append("No actionable resolution steps provided")
    else:
        blockers.append(f"outcome is '{outcome}' (not 'resolved_remotely_actionable')")
        # Do NOT add steps/confirmation blockers for non-actionable outcomes
    
    # Use function parameters (already passed in, don't access args)
    # approve_min_confidence, reject_min_confidence, unclear_reject_min_confidence, require_requester_confirmation
    # are already function parameters
    
    # Determine review_status using new asymmetric policy
    review_status, review_reason, review_reasons = determine_review_status(
        outcome=outcome,
        confirmation_confirmed=confirmation_confirmed,
        resolution_steps_count=len(resolution_steps),
        model_confidence=raw_confidence,
        blockers=blockers,
        stage="sonnet",
        cache_eligible=final_cache_eligible,
        approve_min_confidence=approve_min_confidence,
        reject_min_confidence=reject_min_confidence,
        unclear_reject_min_confidence=unclear_reject_min_confidence,
        require_requester_confirmation=require_requester_confirmation
    )
    
    # Ensure cache_eligible=0 if needs_review
    if review_status == "needs_review":
        final_cache_eligible = 0
    
    # Override model decision if it conflicts with our logic
    if model_cache_eligible != final_cache_eligible:
        blockers.append(f"model decision ({model_cache_eligible}) overridden by strict logic ({final_cache_eligible})")
    
    return {
        "ticket_id": ticket_id,
        "cache_eligible": final_cache_eligible,
        "confidence": raw_confidence,  # Store raw float, no rounding
        "problem": problem_summary or None,
        "resolution_steps_json": json.dumps(resolution_steps, ensure_ascii=False),
        "confirmation": confirmation_text or None,
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
        "blockers_json": json.dumps(blockers, ensure_ascii=False),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "raw_response_json": json.dumps(verifier_json, ensure_ascii=False),
        "review_status": review_status,
        "review_reason": review_reason,
        "review_reasons_json": json.dumps(review_reasons, ensure_ascii=False),
        # Metadata for logging
        "_outcome": outcome,
        "_hard_blocked": False,
        "_hard_block_reason": None,
        "_confirmation_confirmed": confirmation_confirmed,
        "_resolution_steps_count": len(resolution_steps),
        "_min_confidence": min_confidence,
        "_rationale": rationale
    }


def process_ticket_deterministic_first(
    client: Anthropic,
    ticket_id: str,
    conversation: Dict[str, Any],
    conn,
    args,
    default_model: str,
    require_requester_confirmation: bool = True
) -> Dict[str, Any]:
    """
    Process a ticket through deterministic-first pipeline:
    1) Hard-block check
    2) Hard-allow check
    3) Sonnet judge (only if not blocked and not allowed)
    
    Returns:
        Dict with final cache_eligible and metadata
    """
    # Stage 1: Hard-block check
    hard_blocked, blockers, outcome_category = hard_block(conversation)
    
    if hard_blocked:
        # Hard-blocked: set final cache_eligible=0, no LLM calls
        # Use explicit outcome_category from hard_block() instead of guessing from blocker text
        outcome = outcome_category if outcome_category else "unclear"
        
        judgement = {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "problem": None,
            "resolution_steps_json": json.dumps([], ensure_ascii=False),
            "confirmation": None,
            "evidence_json": json.dumps({}, ensure_ascii=False),
            "blockers_json": json.dumps(blockers, ensure_ascii=False),
            "model": "hard_block",
            "prompt_version": "hard_block_v1",
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "raw_response_json": json.dumps({"hard_blocked": True, "blockers": blockers, "outcome": outcome}, ensure_ascii=False),
            "review_status": "rejected",
            "review_reason": f"hard_block_{outcome}",
            "review_reasons_json": json.dumps([outcome], ensure_ascii=False)
        }
        
        if not args.dry_run and not args.deterministic_report:
            db.upsert_ticket_judgement(conn, judgement)
        
        return {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "stage": "hard_block",
            "blocked": True,
            "allowed": False,
            "steps": 0,
            "confirmed": False,
            "blockers": blockers,
            "outcome": outcome,
            "review_status": "rejected",
            "review_reason": f"hard_block_{outcome}"
        }
    
    # Stage 2: Hard-allow check
    hard_allowed, evidence, allow_blockers = hard_allow(conversation, require_requester_confirmation)
    
    if hard_allowed and evidence:
        # Hard-allowed: set final cache_eligible=1, confidence=1.0, no LLM calls
        steps_list = evidence.get("steps", [])
        confirmation_obj = evidence.get("confirmation", {})
        problem_obj = evidence.get("problem", {})
        
        judgement = {
            "ticket_id": ticket_id,
            "cache_eligible": 1,
            "confidence": 1.0,
            "problem": problem_obj.get("text", ""),
            "resolution_steps_json": json.dumps(steps_list, ensure_ascii=False),
            "confirmation": confirmation_obj.get("text", ""),
            "evidence_json": json.dumps({
                "problem_quotes": [problem_obj.get("text", "")],
                "resolution_quotes": steps_list,
                "confirmation_quotes": [confirmation_obj.get("text", "")]
            }, ensure_ascii=False),
            "blockers_json": json.dumps([], ensure_ascii=False),
            "model": "hard_allow",
            "prompt_version": "hard_allow_v1",
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "raw_response_json": json.dumps({
                "hard_allowed": True,
                "evidence": evidence,
                "outcome": "resolved_remotely_actionable"
            }, ensure_ascii=False),
            "review_status": "approved",
            "review_reason": "hard_allow_deterministic",
            "review_reasons_json": json.dumps([], ensure_ascii=False)
        }
        
        if not args.dry_run and not args.deterministic_report:
            db.upsert_ticket_judgement(conn, judgement)
        
        return {
            "ticket_id": ticket_id,
            "cache_eligible": 1,
            "confidence": 1.0,
            "stage": "hard_allow",
            "blocked": False,
            "allowed": True,
            "steps": len(steps_list),
            "confirmed": True,
            "blockers": [],
            "outcome": "resolved_remotely_actionable",
            "review_status": "approved",
            "review_reason": "hard_allow_deterministic"
        }
    
    # Stage 3: Sonnet judge (only if not blocked and not allowed)
    if args.deterministic_only or args.deterministic_report:
        # Deterministic-only mode: don't call Sonnet
        judgement = {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "problem": None,
            "resolution_steps_json": json.dumps([], ensure_ascii=False),
            "confirmation": None,
            "evidence_json": json.dumps({}, ensure_ascii=False),
            "blockers_json": json.dumps(allow_blockers, ensure_ascii=False),
            "model": "deterministic_only",
            "prompt_version": "deterministic_only_v1",
            "judged_at": datetime.now(timezone.utc).isoformat(),
            "raw_response_json": json.dumps({"deterministic_only": True, "blockers": allow_blockers}, ensure_ascii=False)
        }
        
        if not args.dry_run and not args.deterministic_report:
            db.upsert_ticket_judgement(conn, judgement)
        
        stage_name = "deterministic_report" if args.deterministic_report else "deterministic_only"
        return {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "stage": stage_name,
            "blocked": False,
            "allowed": False,
            "steps": 0,
            "confirmed": False,
            "blockers": allow_blockers,
            "outcome": "unclear"  # Deterministic-only didn't reach Sonnet, so outcome is unclear
        }
    
    # Call Sonnet judge (skip if deterministic_only or deterministic_report)
    if args.deterministic_only or args.deterministic_report:
        # Should not reach here, but handle gracefully
        return {
            "ticket_id": ticket_id,
            "cache_eligible": 0,
            "confidence": 0.0,
            "stage": "deterministic_only" if args.deterministic_only else "deterministic_report",
            "blocked": False,
            "allowed": False,
            "steps": 0,
            "confirmed": False,
            "blockers": allow_blockers,
            "outcome": "unclear"  # Deterministic-only didn't reach Sonnet, so outcome is unclear
        }
    
    sonnet_model = args.model or default_model
    judgement = judge_ticket(
        client, ticket_id, conversation, 
        model=sonnet_model, 
        min_confidence=args.min_confidence,
        approve_min_confidence=getattr(args, 'approve_min_confidence', 0.90),
        reject_min_confidence=getattr(args, 'reject_min_confidence', 0.60),
        unclear_reject_min_confidence=getattr(args, 'unclear_reject_min_confidence', 0.80),
        require_requester_confirmation=args.require_requester_confirmation
    )
    
    # Extract metadata for logging
    outcome = judgement.get("_outcome", "unknown")
    confirmation_confirmed = judgement.get("_confirmation_confirmed", False)
    resolution_steps_count = judgement.get("_resolution_steps_count", 0)
    cache_eligible = judgement["cache_eligible"]
    
    if not args.dry_run and not args.deterministic_report:
        db_judgement = {k: v for k, v in judgement.items() if not k.startswith("_")}
        db.upsert_ticket_judgement(conn, db_judgement)
    
    return {
        "ticket_id": ticket_id,
        "cache_eligible": cache_eligible,
        "confidence": judgement.get("confidence", 0.0),
        "stage": "sonnet",
        "blocked": False,
        "allowed": False,
        "steps": resolution_steps_count,
        "confirmed": confirmation_confirmed,
        "blockers": json.loads(judgement.get("blockers_json", "[]")),
        "outcome": outcome,
        "review_status": judgement.get("review_status", "needs_review"),
        "review_reason": judgement.get("review_reason", "unknown")
    }


def export_review_queue(conn, output_path: str) -> None:
    """
    Export tickets needing review to CSV file with full context.
    
    Includes: ticket_id, outcome, confidence, review_reason, confirmation status,
    evidence quotes, transcript excerpts, resolution steps, and metadata.
    
    Args:
        conn: Database connection
        output_path: Output CSV file path
    """
    import csv
    import json
    
    cursor = conn.cursor()
    
    # Get all needs_review tickets with raw_response_json
    cursor.execute("""
        SELECT 
            ticket_id, 
            raw_response_json,
            confidence, 
            review_reason, 
            review_reasons_json,
            blockers_json,
            judged_at, 
            model
        FROM ticket_judgements
        WHERE review_status = 'needs_review'
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No tickets needing review found.")
        return
    
    # Process and enrich each row
    enriched_rows = []
    for row in rows:
        row_dict = dict(row)  # Convert Row to dict
        ticket_id = row_dict['ticket_id']
        raw_json_str = row_dict.get('raw_response_json', '{}')
        
        try:
            raw_json = json.loads(raw_json_str) if raw_json_str else {}
        except (json.JSONDecodeError, TypeError):
            raw_json = {}
        
        # Extract fields from raw_json
        outcome = raw_json.get('outcome', 'unclear')
        confirmation_obj = raw_json.get('confirmation', {})
        confirmation_confirmed = confirmation_obj.get('confirmed', False) if isinstance(confirmation_obj, dict) else bool(confirmation_obj)
        confirmation_evidence = confirmation_obj.get('evidence', {}) if isinstance(confirmation_obj, dict) else {}
        confirmation_quote = confirmation_evidence.get('quote', '') if confirmation_evidence else ''
        
        resolution_obj = raw_json.get('resolution', {})
        resolution_steps = resolution_obj.get('steps', []) if isinstance(resolution_obj, dict) else []
        resolution_steps_str = ' | '.join(resolution_steps) if resolution_steps else ''
        
        # Get conversation for transcript excerpt
        conversation = db.get_ticket_detail_json(conn, ticket_id)
        transcript_excerpt = ''
        if conversation:
            messages = conversation.get('messages', [])
            if isinstance(messages, list):
                # Sort by created_at and take last 10
                sorted_messages = sorted(messages, key=lambda m: m.get('created_at', ''))
                last_messages = sorted_messages[-10:] if len(sorted_messages) > 10 else sorted_messages
                
                # Build excerpt with author_id/role
                excerpt_parts = []
                for msg in last_messages:
                    role = msg.get('role', 'unknown')
                    author_id = msg.get('author_id', '')
                    text = msg.get('text', '')[:200]  # Truncate long messages
                    excerpt_parts.append(f"[{role}|{author_id}]: {text}")
                transcript_excerpt = ' || '.join(excerpt_parts)
        
        # Parse blockers
        blockers_str = ''
        try:
            blockers = json.loads(row_dict.get('blockers_json', '[]') or '[]')
            blockers_str = ' | '.join(blockers) if blockers else ''
        except (json.JSONDecodeError, TypeError):
            blockers_str = row_dict.get('blockers_json', '')
        
        # Parse review_reasons
        review_reasons_str = ''
        try:
            review_reasons = json.loads(row_dict.get('review_reasons_json', '[]') or '[]')
            review_reasons_str = ' | '.join(review_reasons) if review_reasons else ''
        except (json.JSONDecodeError, TypeError):
            review_reasons_str = row_dict.get('review_reasons_json', '')
        
        enriched_rows.append({
            'ticket_id': ticket_id,
            'outcome': outcome,
            'confidence': row_dict.get('confidence', 0.0),
            'review_reason': row_dict.get('review_reason', ''),
            'review_reasons': review_reasons_str,
            'confirmation_confirmed': confirmation_confirmed,
            'confirmation_quote': confirmation_quote[:500],  # Limit quote length
            'resolution_steps': resolution_steps_str[:1000],  # Limit steps length
            'blockers': blockers_str[:500],  # Limit blockers length
            'transcript_excerpt': transcript_excerpt[:2000],  # Limit transcript length
            'model': row_dict.get('model', ''),
            'judged_at': row_dict.get('judged_at', '')
        })
    
    # Sort: missing_confirmation first, then borderline_eligible by confidence descending
    def sort_key(r):
        reason = r['review_reason']
        conf = r['confidence']
        if reason == 'missing_confirmation':
            return (0, -conf)  # First priority, higher confidence first
        elif reason == 'borderline_eligible':
            return (1, -conf)  # Second priority, higher confidence first
        else:
            return (2, -conf)  # Other reasons last
    
    enriched_rows.sort(key=sort_key)
    
    print(f"Exporting {len(enriched_rows)} tickets needing review to {output_path}...")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ticket_id', 'outcome', 'confidence', 'review_reason', 'review_reasons',
            'confirmation_confirmed', 'confirmation_quote', 'resolution_steps',
            'blockers', 'transcript_excerpt', 'model', 'judged_at'
        ])
        
        for row in enriched_rows:
            writer.writerow([
                row['ticket_id'],
                row['outcome'],
                row['confidence'],
                row['review_reason'],
                row['review_reasons'],
                row['confirmation_confirmed'],
                row['confirmation_quote'],
                row['resolution_steps'],
                row['blockers'],
                row['transcript_excerpt'],
                row['model'],
                row['judged_at']
            ])
    
    print(f"Exported {len(enriched_rows)} tickets to {output_path}")
    print(f"  - missing_confirmation: {sum(1 for r in enriched_rows if r['review_reason'] == 'missing_confirmation')}")
    print(f"  - borderline_eligible: {sum(1 for r in enriched_rows if r['review_reason'] == 'borderline_eligible')}")
    print(f"  - other: {sum(1 for r in enriched_rows if r['review_reason'] not in ['missing_confirmation', 'borderline_eligible'])}")


def backfill_confirmation_evidence(conn, args) -> None:
    """
    Deterministic backfill of confirmation evidence for existing ticket judgments.
    
    Reads tickets with confirmed=True but missing evidence, finds confirmation in
    conversation messages using same logic as runtime, and updates raw_response_json.
    
    NO API CALLS - purely deterministic.
    
    Args:
        conn: Database connection
        args: Parsed arguments (may include --only-ids, --dry-run, --db)
    """
    cursor = conn.cursor()
    
    # Print diagnostics: DB path
    db_path = getattr(args, 'db', None) or db.DEFAULT_DB_PATH
    abs_db_path = os.path.abspath(db_path)
    print("=" * 70)
    print("Backfill Confirmation Evidence Diagnostics")
    print("=" * 70)
    print(f"Database path: {abs_db_path}")
    print(f"Database exists: {os.path.exists(abs_db_path)}")
    print()
    
    # Print diagnostic counts BEFORE candidate selection
    print("Pre-selection diagnostics:")
    cursor.execute("SELECT COUNT(*) as count FROM ticket_judgements")
    total_rows = cursor.fetchone()["count"]
    print(f"  Total rows in ticket_judgements: {total_rows}")
    
    cursor.execute("SELECT COUNT(*) as count FROM ticket_judgements WHERE review_reason = 'missing_confirmation'")
    missing_confirmation_count = cursor.fetchone()["count"]
    print(f"  Rows with review_reason='missing_confirmation': {missing_confirmation_count}")
    
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM ticket_judgements 
        WHERE json_extract(raw_response_json, '$.outcome') = 'resolved_remotely_actionable'
    """)
    resolved_count = cursor.fetchone()["count"]
    print(f"  Rows with outcome='resolved_remotely_actionable': {resolved_count}")
    
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM ticket_judgements 
        WHERE json_extract(raw_response_json, '$.outcome') = 'resolved_remotely_actionable'
        AND (
            COALESCE(json_extract(raw_response_json, '$.confirmation.confirmed'), 0) = 0
            OR json_extract(raw_response_json, '$.confirmation.confirmed') IS NULL
        )
        AND model != 'hard_block'
    """)
    resolved_no_confirmation_count = cursor.fetchone()["count"]
    print(f"  Rows with outcome='resolved_remotely_actionable' AND confirmation.confirmed=0/NULL (non-hard_block): {resolved_no_confirmation_count}")
    print()
    
    # Build query - filter by ticket IDs if --only-ids specified
    if hasattr(args, 'only_ids') and args.only_ids:
        ticket_ids = [tid.strip() for tid in args.only_ids.split(',')]
        placeholders = ','.join(['?' for _ in ticket_ids])
        query = f"""
            SELECT 
                ticket_id,
                raw_response_json,
                review_status,
                review_reason,
                model
            FROM ticket_judgements
            WHERE ticket_id IN ({placeholders})
            ORDER BY ticket_id
        """
        cursor.execute(query, ticket_ids)
        print(f"Filtering to specific ticket IDs: {args.only_ids}")
    else:
        # Find tickets that need evidence backfill:
        # A) review_reason == 'missing_confirmation' (primary)
        # B) outcome == 'resolved_remotely_actionable' AND confirmation.confirmed is 0/NULL AND model != 'hard_block'
        # AND confirmation.evidence is missing/empty
        query = """
            SELECT 
                ticket_id,
                raw_response_json,
                review_status,
                review_reason,
                model
            FROM ticket_judgements
            WHERE (
                -- Primary: tickets with missing_confirmation review_reason
                review_reason = 'missing_confirmation'
                OR
                -- Secondary: resolved_remotely_actionable with no confirmation
                (
                    json_extract(raw_response_json, '$.outcome') = 'resolved_remotely_actionable'
                    AND (
                        COALESCE(json_extract(raw_response_json, '$.confirmation.confirmed'), 0) = 0
                        OR json_extract(raw_response_json, '$.confirmation.confirmed') IS NULL
                    )
                    AND model != 'hard_block'
                )
            )
            -- Ensure evidence is missing or empty
            AND (
                json_extract(raw_response_json, '$.confirmation.evidence') IS NULL
                OR json_extract(raw_response_json, '$.confirmation.evidence.quote') IS NULL
                OR json_extract(raw_response_json, '$.confirmation.evidence.quote') = ''
            )
            ORDER BY ticket_id
        """
        print("Candidate selection SQL:")
        print(query)
        print()
        cursor.execute(query)
    
    rows = cursor.fetchall()
    
    print(f"Candidates selected: {len(rows)}")
    if len(rows) > 0:
        print(f"Sample candidate ticket IDs: {', '.join([r['ticket_id'] for r in rows[:10]])}")
    print()
    
    if not rows:
        print("No tickets found for backfill.")
        return
    
    print(f"Scanning {len(rows)} candidates for confirmation evidence backfill...")
    print(f"Require requester confirmation: {args.require_requester_confirmation}")
    if args.dry_run:
        print("[DRY RUN MODE] No changes will be made to the database.")
    print()
    
    candidates = 0
    updated_with_evidence = 0
    updated_set_false = 0
    skipped_invalid_json = 0
    skipped_no_conversation = 0
    skipped_hard_block = 0
    errors = 0
    
    # Track samples for dry-run output
    evidence_samples = []
    no_evidence_samples = []
    
    for row in rows:
        ticket_id = row["ticket_id"]
        raw_response_json = dict(row).get("raw_response_json", "{}")
        review_status = dict(row).get("review_status", "")
        review_reason = dict(row).get("review_reason", "")
        model = dict(row).get("model", "")
        
        # Skip hard_block rows (shouldn't happen with SQL filter, but safety check)
        if model == "hard_block":
            skipped_hard_block += 1
            continue
        
        try:
            raw_json = json.loads(raw_response_json)
        except (json.JSONDecodeError, TypeError):
            skipped_invalid_json += 1
            continue
        
        # All rows from SQL query are already candidates (filtered by SQL)
        candidates += 1
        
        # Load conversation from tickets_detail
        conversation = db.get_ticket_detail_json(conn, ticket_id)
        if not conversation:
            skipped_no_conversation += 1
            continue
        
        messages = conversation.get("messages", [])
        if not isinstance(messages, list):
            skipped_no_conversation += 1
            continue
        
        # Identify requester author IDs
        requester_author_ids = get_requester_author_ids(conversation, messages)
        
        # Find confirmation evidence using shared function
        try:
            evidence = find_requester_confirmation_evidence(
                messages,
                require_requester_confirmation=args.require_requester_confirmation,
                conversation=conversation,
                requester_author_ids=requester_author_ids
            )
            
            # Update raw_response_json
            if not raw_json.get("confirmation"):
                raw_json["confirmation"] = {}
            
            if evidence:
                # Found evidence - update confirmation object
                raw_json["confirmation"]["confirmed"] = True
                raw_json["confirmation"]["evidence"] = {
                    "message_index": evidence["message_index"],
                    "author_role": evidence["role"],
                    "quote": evidence["quote"]
                }
                # Add author_id if available (for requester validation)
                if "author_id" in evidence:
                    raw_json["confirmation"]["evidence"]["author_id"] = evidence["author_id"]
                # Add created_at if available (optional but helpful)
                if "created_at" in evidence:
                    raw_json["confirmation"]["evidence"]["created_at"] = evidence["created_at"]
                # Preserve existing evidence_quote_ids if present
                if "evidence_quote_ids" not in raw_json["confirmation"]:
                    raw_json["confirmation"]["evidence_quote_ids"] = [evidence["message_index"]]
                
                updated_with_evidence += 1
                
                # Collect samples for dry-run
                if args.dry_run and len(evidence_samples) < 5:
                    evidence_samples.append({
                        "ticket_id": ticket_id,
                        "quote": evidence['quote'][:100]
                    })
                
                if args.print or args.dry_run:
                    print(f"  [{ticket_id}] Found evidence: '{evidence['quote'][:50]}...'")
            else:
                # No evidence found - leave confirmed as-is (don't flip to False unless it was incorrectly True)
                # Only update if it was incorrectly set to True
                confirmation_obj = raw_json.get("confirmation", {})
                if not isinstance(confirmation_obj, dict):
                    confirmation_obj = {}
                
                current_confirmed = confirmation_obj.get("confirmed", False)
                
                # Only set to False if it was incorrectly True
                if current_confirmed:
                    raw_json["confirmation"]["confirmed"] = False
                    raw_json["confirmation"]["evidence"] = None
                    raw_json["confirmation"]["reason"] = "no_requester_confirmation_found"
                    
                    # Add blocker if not already present
                    blockers = raw_json.get("blockers", [])
                    if not isinstance(blockers, list):
                        blockers = []
                    if "missing_grounded_requester_confirmation" not in blockers:
                        blockers.append("missing_grounded_requester_confirmation")
                    raw_json["blockers"] = blockers
                    
                    updated_set_false += 1
                    
                    # Collect samples for dry-run
                    if args.dry_run and len(no_evidence_samples) < 5:
                        no_evidence_samples.append({
                            "ticket_id": ticket_id,
                            "current_confirmed": current_confirmed,
                            "review_reason": review_reason
                        })
                    
                    if args.print or args.dry_run:
                        print(f"  [{ticket_id}] No evidence found - set confirmed=False (was incorrectly True)")
                else:
                    # Already False or missing - no change needed
                    if args.print or args.dry_run:
                        print(f"  [{ticket_id}] No evidence found - confirmed already False, no change")
            
            # Write updated raw_response_json back to DB
            if not args.dry_run:
                cursor.execute("""
                    UPDATE ticket_judgements
                    SET raw_response_json = ?
                    WHERE ticket_id = ?
                """, (
                    json.dumps(raw_json, ensure_ascii=False),
                    ticket_id
                ))
                conn.commit()
        
        except Exception as e:
            errors += 1
            print(f"  [{ticket_id}] ERROR: {e}", file=sys.stderr)
            continue
    
    print()
    print("=" * 60)
    print("Backfill Summary:")
    print("=" * 60)
    print(f"Total tickets scanned: {len(rows)}")
    print(f"Candidates for backfill: {candidates}")
    print(f"Would update with evidence: {updated_with_evidence}")
    print(f"Would update (set confirmed=False): {updated_set_false}")
    print(f"Skipped (invalid JSON): {skipped_invalid_json}")
    print(f"Skipped (no conversation): {skipped_no_conversation}")
    print(f"Skipped (hard_block): {skipped_hard_block}")
    print(f"Errors: {errors}")
    
    # Show samples in dry-run mode
    if args.dry_run:
        print("\n" + "=" * 60)
        print("Sample Evidence Found (first 5):")
        print("=" * 60)
        if evidence_samples:
            for sample in evidence_samples:
                print(f"  [{sample['ticket_id']}] {sample['quote']}...")
        else:
            print("  (none)")
        
        print("\n" + "=" * 60)
        print("Sample No Evidence (first 5):")
        print("=" * 60)
        if no_evidence_samples:
            for sample in no_evidence_samples:
                print(f"  [{sample['ticket_id']}] Currently confirmed={sample['current_confirmed']}")
        else:
            print("  (none)")
        
        print("\n[DRY RUN] No changes were made to the database.")
    else:
        print(f"\nUpdated {updated_with_evidence + updated_set_false} tickets in database.")
        print("Run --relabel-from-db to recompute review_status/cache_eligible.")


def relabel_from_db(conn, args) -> None:
    """
    Relabel tickets from existing database judgments without API calls.
    
    Loads stored judgments and recomputes review_status + cache_eligible using new policy.
    
    Args:
        conn: Database connection
        args: Parsed arguments
    """
    cursor = conn.cursor()
    
    # Get all tickets with judgments (default to --all if neither specified)
    if args.id:
        cursor.execute("""
            SELECT ticket_id, raw_response_json, confidence, cache_eligible, model, prompt_version
            FROM ticket_judgements
            WHERE ticket_id = ?
        """, (args.id,))
    else:
        # Default to all tickets if --all not explicitly set
        cursor.execute("""
            SELECT ticket_id, raw_response_json, confidence, cache_eligible, model, prompt_version
            FROM ticket_judgements
            ORDER BY ticket_id
        """)
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No tickets with judgments found.")
        return
    
    print(f"Relabeling {len(rows)} tickets from existing judgments...")
    print(f"Approve min confidence: {args.approve_min_confidence}")
    print(f"Reject min confidence: {args.reject_min_confidence}")
    print(f"Unclear reject min confidence: {args.unclear_reject_min_confidence}")
    print()
    
    updated = 0
    skipped = 0
    stats = {"approved": 0, "rejected": 0, "needs_review": 0}
    
    for row in rows:
        ticket_id = row["ticket_id"]
        # sqlite3.Row doesn't have .get(), use dict() conversion or try/except
        raw_response_json = dict(row).get("raw_response_json", "{}")
        confidence = dict(row).get("confidence", 0.0)
        current_cache_eligible = dict(row).get("cache_eligible", 0)
        
        # Parse outcome from raw_response_json
        try:
            raw_json = json.loads(raw_response_json)
            outcome = raw_json.get("outcome", "unclear")
            
            # Initialize blockers early so we can append to it
            blockers = raw_json.get("blockers", [])
            if not isinstance(blockers, list):
                blockers = []
            
            # Extract confirmation and steps from raw JSON
            confirmation_obj = raw_json.get("confirmation", {})
            if isinstance(confirmation_obj, dict):
                confirmation_confirmed = confirmation_obj.get("confirmed", False)
                # Validate confirmation evidence if present (for relabel, we trust stored value but note if evidence missing)
                confirmation_evidence = confirmation_obj.get("evidence", {})
                if confirmation_confirmed and confirmation_evidence:
                    # Check if evidence quote exists (best-effort validation)
                    evidence_quote = confirmation_evidence.get("quote", "")
                    if not evidence_quote:
                        # Evidence missing - this would fail validation in live judging
                        # For relabel, we keep confirmed=True but note it in blockers
                        if "confirmation_evidence_missing_quote" not in blockers:
                            blockers.append("confirmation_evidence_missing_quote")
            else:
                confirmation_confirmed = bool(confirmation_obj)
            
            resolution_obj = raw_json.get("resolution", {})
            if isinstance(resolution_obj, dict):
                resolution_steps = resolution_obj.get("steps", [])
            else:
                resolution_steps = raw_json.get("resolution_steps", [])
            
            if not isinstance(resolution_steps, list):
                resolution_steps = []
            
        except (json.JSONDecodeError, TypeError, AttributeError):
            print(f"  [{ticket_id}] SKIP: Invalid raw_response_json", file=sys.stderr)
            skipped += 1
            continue
        
        # Determine stage from model field
        model = dict(row).get("model", "")
        if model == "hard_block":
            stage = "hard_block"
        elif model == "hard_allow":
            stage = "hard_allow"
        else:
            stage = "sonnet"
        
        # Derive fresh "base eligibility signal" from stored raw_response_json, not stale cache_eligible
        # This ensures relabel reflects CURRENT policy applied to stored facts
        base_cache_signal = 0
        stored_cache_eligible = raw_json.get("cache_eligible", 0)
        if stored_cache_eligible == 1:
            # Only trust stored cache_eligible if outcome and conditions match
            if (outcome == "resolved_remotely_actionable" and 
                confirmation_confirmed and 
                len(resolution_steps) >= 1):
                base_cache_signal = 1
        
        # Recompute review_status using new policy
        # Pass base_cache_signal instead of current_cache_eligible to avoid stale data dependency
        review_status, review_reason, review_reasons = determine_review_status(
            outcome=outcome,
            confirmation_confirmed=confirmation_confirmed,
            resolution_steps_count=len(resolution_steps),
            model_confidence=confidence,
            blockers=blockers,
            stage=stage,
            cache_eligible=base_cache_signal,  # Use fresh signal, not stale DB value
            approve_min_confidence=args.approve_min_confidence,
            reject_min_confidence=args.reject_min_confidence,
            unclear_reject_min_confidence=args.unclear_reject_min_confidence,
            require_requester_confirmation=args.require_requester_confirmation
        )
        
        # Determine cache_eligible based on review_status
        new_cache_eligible = 1 if review_status == "approved" else 0
        
        # Update database
        if not args.dry_run:
            cursor.execute("""
                UPDATE ticket_judgements
                SET review_status = ?,
                    cache_eligible = ?,
                    review_reason = ?,
                    review_reasons_json = ?
                WHERE ticket_id = ?
            """, (
                review_status,
                new_cache_eligible,
                review_reason,
                json.dumps(review_reasons, ensure_ascii=False),
                ticket_id
            ))
            conn.commit()
        
        stats[review_status] = stats.get(review_status, 0) + 1
        updated += 1
        
        # Log changes
        if args.print or args.dry_run:
            print(f"  [{ticket_id}] {outcome} conf={confidence:.2f} => {review_status} (cache_eligible={new_cache_eligible})")
    
    print()
    print("=" * 60)
    print("Relabel Summary:")
    print("=" * 60)
    print(f"Total processed: {len(rows)}")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Needs review: {stats['needs_review']}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes were made to the database.")


def main():
    parser = argparse.ArgumentParser(
        description="Judge ticket cache eligibility using LLM",
        epilog="""
Examples:
  # Judge single ticket (dry-run)
  python judge_ticket_cache_eligibility.py --id 3599 --dry-run
  
  # Spot-check a ticket (print detailed output)
  python judge_ticket_cache_eligibility.py --id 4246 --print
  
  # Judge all solved tickets (limit 20)
  python judge_ticket_cache_eligibility.py --all --limit 20
  
  # Re-judge existing tickets
  python judge_ticket_cache_eligibility.py --all --force
        """
    )
    parser.add_argument("--id", type=str, help="Judge only this ticket_id")
    parser.add_argument("--all", action="store_true", help="Judge all solved tickets")
    parser.add_argument("--limit", type=int, help="Limit number of tickets (when --all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB; just print JSON")
    parser.add_argument("--force", action="store_true", help="Re-judge even if already judged")
    parser.add_argument("--db", type=str, help="Database path override")
    parser.add_argument("--model", type=str, default=None, help=f"Model name override (default: from ANTHROPIC_MODEL env or {config.DEFAULT_MODEL})")
    parser.add_argument("--min-confidence", type=float, default=0.90, help="Minimum confidence threshold (default: 0.90, uses >= comparison) - legacy")
    parser.add_argument("--approve-min-confidence", type=float, default=0.90, help="Minimum confidence to auto-approve (default: 0.90)")
    parser.add_argument("--reject-min-confidence", type=float, default=0.60, help="Minimum confidence to auto-reject (default: 0.60)")
    parser.add_argument("--unclear-reject-min-confidence", type=float, default=0.80, help="Minimum confidence to reject unclear outcomes (default: 0.80)")
    parser.add_argument("--print", action="store_true", help="Print detailed judgement output (for spot-check)")
    parser.add_argument("--export-review-queue", type=str, help="Export tickets needing review to CSV file")
    parser.add_argument("--review-only", action="store_true", help="Only process tickets not already reviewed/approved/rejected")
    parser.add_argument("--relabel-from-db", action="store_true", help="Recompute review_status from existing judgments (no API calls)")
    parser.add_argument("--backfill-confirmation-evidence", action="store_true", help="Backfill confirmation evidence for existing judgments (no API calls)")
    parser.add_argument("--only-ids", type=str, help="Comma-separated list of ticket IDs to process (for backfill or relabel)")
    parser.add_argument("--self-test", action="store_true", help="Run all self-test cases (hard-block, hard-allow, blocker generation, review_status)")
    parser.add_argument("--self-test-hard-block", action="store_true", help="Run self-test cases for hard-block detection only")
    parser.add_argument("--self-test-hard-allow", action="store_true", help="Run self-test cases for hard-allow detection only")
    parser.add_argument("--self-test-blockers", action="store_true", help="Run self-test cases for blocker generation only")
    parser.add_argument("--self-test-review-status", action="store_true", help="Run self-test cases for review_status determination only")
    parser.add_argument("--self-test-confirmation-evidence", action="store_true", help="Run self-test cases for confirmation evidence validation")
    parser.add_argument("--self-test-transcript", action="store_true", help="Run self-test cases for tail-preserving transcript truncation")
    parser.add_argument("--self-test-hard-block-outcome", action="store_true", help="Run self-test cases for hard_block explicit outcome")
    parser.add_argument("--self-test-find-confirmation-evidence", action="store_true", help="Run self-test cases for find_requester_confirmation_evidence function")
    
    # Deterministic pipeline flags
    parser.add_argument("--deterministic-only", action="store_true", help="Run only hard_block + hard_allow; never call LLM; write results")
    parser.add_argument("--deterministic-report", action="store_true", help="No DB writes; print summary counts for hard_blocked, hard_allowed, needs_llm")
    parser.add_argument("--require-requester-confirmation", dest="require_requester_confirmation", action="store_true", default=True, help="Require requester confirmation for hard-allow (default: True)")
    parser.add_argument("--no-require-requester-confirmation", dest="require_requester_confirmation", action="store_false", help="Allow Sonnet to decide confirmation (less safe)")
    
    args = parser.parse_args()
    
    # Self-test mode
    if (args.self_test or args.self_test_hard_block or args.self_test_hard_allow or 
        args.self_test_blockers or args.self_test_review_status or args.self_test_confirmation_evidence or
        args.self_test_transcript or args.self_test_hard_block_outcome or args.self_test_find_confirmation_evidence):
        if args.self_test or args.self_test_hard_block:
            _self_test_hard_block()
        if args.self_test or args.self_test_hard_allow:
            _self_test_hard_allow()
        if args.self_test or args.self_test_blockers:
            _self_test_blocker_generation()
        if args.self_test or args.self_test_review_status:
            _self_test_review_status()
        if args.self_test or args.self_test_confirmation_evidence:
            _self_test_confirmation_evidence()
        if args.self_test or args.self_test_transcript:
            _self_test_tail_preserving_transcript()
        if args.self_test or args.self_test_hard_block_outcome:
            _self_test_hard_block_outcome()
        if args.self_test or args.self_test_find_confirmation_evidence:
            _self_test_find_confirmation_evidence()
        sys.exit(0)
    
    # Validate args (allow --relabel-from-db, --backfill-confirmation-evidence, and --export-review-queue without --id or --all)
    if not args.id and not args.all and not args.relabel_from_db and not args.backfill_confirmation_evidence and not args.export_review_queue:
        print("Error: Must specify either --id <ticket_id> or --all or --relabel-from-db or --backfill-confirmation-evidence or --export-review-queue", file=sys.stderr)
        sys.exit(1)
    
    # Get API key and model from config
    try:
        api_key = config.get_anthropic_api_key()
        default_model = config.get_anthropic_model()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Log configuration
    config.log_config()
    
    # Initialize DB
    db_path = args.db or db.DEFAULT_DB_PATH
    db.init_db(db_path)
    
    conn = db.get_connection(db_path)
    
    # Handle export-review-queue mode
    if args.export_review_queue:
        export_review_queue(conn, args.export_review_queue)
        conn.close()
        return
    
    # Handle backfill-confirmation-evidence mode (no API calls needed)
    if args.backfill_confirmation_evidence:
        # Pass db_path to args for diagnostics
        args.db = db_path
        backfill_confirmation_evidence(conn, args)
        conn.close()
        return
    
    # Handle relabel-from-db mode (no API calls needed)
    if args.relabel_from_db:
        relabel_from_db(conn, args)
        conn.close()
        return
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    try:
        if args.id:
            # Single ticket mode
            conversation = db.get_ticket_detail_json(conn, args.id)
            if not conversation:
                print(f"Error: Ticket {args.id} not found in tickets_detail", file=sys.stderr)
                sys.exit(1)
            
            # Check if already judged (unless force)
            if not args.force:
                cursor = conn.cursor()
                cursor.execute("SELECT ticket_id FROM ticket_judgements WHERE ticket_id = ?", (args.id,))
                if cursor.fetchone():
                    print(f"Ticket {args.id} already judged. Use --force to re-judge.", file=sys.stderr)
                    sys.exit(1)
            
            # Deterministic-first pipeline
            result = process_ticket_deterministic_first(
                client, args.id, conversation, conn, args, default_model, args.require_requester_confirmation
            )
            
            # Log decision details
            outcome = result.get('outcome', 'unknown')
            steps_count = result['steps']
            requires_steps = outcome_requires_steps(outcome)
            review_status = result.get('review_status', 'unknown')
            review_reason = result.get('review_reason', '')
            confidence = result.get('confidence', 0.0)
            
            log_parts = [
                f"[{args.id}]",
                f"stage={result['stage']}",
                f"outcome={outcome}",
                f"conf={confidence:.2f}",
                f"review_status={review_status}",
                f"cache_eligible={result['cache_eligible']}",
            ]
            
            # Format steps based on whether outcome requires them
            if requires_steps:
                log_parts.append(f"steps={steps_count}")
            else:
                if steps_count == 0:
                    log_parts.append(f"steps=0 (not required)")
                else:
                    log_parts.append(f"steps={steps_count} (not required)")
            
            log_parts.append(f"confirmed={result['confirmed']}")
            
            # Add reason (review_reason or first blocker)
            reason = review_reason or (result["blockers"][0][:50] if result.get("blockers") else "")
            if reason:
                log_parts.append(f"(reason: {reason})")
            
            print(" ".join(log_parts), file=sys.stderr)
            
            if args.dry_run or args.print:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif not args.print:
                print(f"[OK] Processed ticket {args.id}: cache_eligible={result['cache_eligible']}, stage={result['stage']}")
        
        elif args.all:
            # Bulk mode
            if args.force:
                ticket_ids = db.get_all_solved_ticket_ids(conn)
            else:
                ticket_ids = db.get_ticket_ids_needing_judgement(conn, only_solved=True)
            
            if not ticket_ids:
                print("No tickets to process.")
                return
            
            if args.limit:
                ticket_ids = ticket_ids[:args.limit]
            
            print(f"Processing {len(ticket_ids)} tickets with deterministic-first pipeline...")
            if args.deterministic_only:
                print("DETERMINISTIC ONLY MODE: Will not call Sonnet")
            elif args.deterministic_report:
                print("DETERMINISTIC REPORT MODE: No DB writes, summary only")
            else:
                print(f"Sonnet model: {args.model or default_model}")
                print(f"Min confidence threshold: {args.min_confidence} (>= comparison)")
            print(f"Require requester confirmation: {args.require_requester_confirmation}")
            print()
            
            processed = 0
            cache_eligible_count = 0
            errors = 0
            
            # Statistics
            hard_blocked_count = 0
            hard_allowed_count = 0
            sonnet_called_count = 0
            review_status_counts = {"approved": 0, "rejected": 0, "needs_review": 0}
            
            for i, ticket_id in enumerate(ticket_ids, 1):
                try:
                    conversation = db.get_ticket_detail_json(conn, ticket_id)
                    if not conversation:
                        print(f"  [{i}/{len(ticket_ids)}] {ticket_id} SKIP: No conversation detail")
                        continue
                    
                    # Deterministic-first pipeline
                    result = process_ticket_deterministic_first(
                        client, ticket_id, conversation, conn, args, default_model, args.require_requester_confirmation
                    )
                    
                    # Update statistics
                    if result["blocked"]:
                        hard_blocked_count += 1
                    elif result["allowed"]:
                        hard_allowed_count += 1
                    elif result["stage"] == "sonnet":
                        sonnet_called_count += 1
                    
                    if result["cache_eligible"] == 1:
                        cache_eligible_count += 1
                    
                    # Update review_status counts
                    review_status = result.get("review_status", "unknown")
                    if review_status in review_status_counts:
                        review_status_counts[review_status] += 1
                    
                    # Log decision details
                    outcome = result.get('outcome', 'unknown')
                    steps_count = result['steps']
                    requires_steps = outcome_requires_steps(outcome)
                    review_status = result.get('review_status', 'unknown')
                    review_reason = result.get('review_reason', '')
                    confidence = result.get('confidence', 0.0)
                    
                    log_parts = [
                        f"  [{i}/{len(ticket_ids)}] {ticket_id}",
                        f"stage={result['stage']}",
                        f"outcome={outcome}",
                        f"conf={confidence:.2f}",
                        f"review_status={review_status}",
                        f"cache_eligible={result['cache_eligible']}",
                    ]
                    
                    # Format steps based on whether outcome requires them
                    if requires_steps:
                        log_parts.append(f"steps={steps_count}")
                    else:
                        if steps_count == 0:
                            log_parts.append(f"steps=0 (not required)")
                        else:
                            log_parts.append(f"steps={steps_count} (not required)")
                    
                    log_parts.append(f"confirmed={result['confirmed']}")
                    
                    # Add reason (review_reason or first blocker)
                    reason = review_reason or (result["blockers"][0][:50] if result.get("blockers") else "")
                    if reason:
                        log_parts.append(f"(reason: {reason})")
                    
                    print(" ".join(log_parts))
                    
                    if args.dry_run or args.print:
                        print(f"\n[{i}/{len(ticket_ids)}] {ticket_id}:")
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    
                    processed += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"  [{i}/{len(ticket_ids)}] {ticket_id} ERROR: {e}", file=sys.stderr)
            
            print()
            print("=" * 60)
            print("Summary:")
            print("=" * 60)
            print(f"Total tickets: {len(ticket_ids)}")
            print(f"Processed: {processed}")
            print(f"Hard-blocked: {hard_blocked_count}")
            print(f"Hard-allowed: {hard_allowed_count}")
            print(f"Sonnet called: {sonnet_called_count}")
            print(f"Cache eligible: {cache_eligible_count}")
            print(f"\nReview Status:")
            print(f"  Approved: {review_status_counts['approved']}")
            print(f"  Rejected: {review_status_counts['rejected']}")
            print(f"  Needs review: {review_status_counts['needs_review']}")
            print(f"\nErrors: {errors}")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

