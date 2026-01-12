#!/usr/bin/env python3
"""
Cheap model triage stage for cache eligibility classification.

This module uses Claude Haiku to triage tickets into:
- "deny": Definitely not cache-eligible, skip Sonnet stage
- "candidate": Potentially cache-eligible, run Sonnet stage
- "uncertain": Ambiguous, default to deny unless flag enabled

IMPORTANT: This module NEVER sets cache_eligible=1. Only Sonnet can do that.
"""

import json
import random
import sys
import time
from typing import Any, Dict, Optional

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

import config

# Prompt version for reproducibility
TRIAGE_PROMPT_VERSION = "triage_v1"

# Transcript limits (same as judge script)
MAX_MESSAGE_CHARS = 2000
MAX_TOTAL_TRANSCRIPT_CHARS = 40000


def build_transcript(conversation: Dict[str, Any]) -> str:
    """
    Build a compact transcript string from conversation JSON.
    Same logic as judge script for consistency.
    
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
    lines = [
        f"Ticket #{ticket_id}, subject: {subject}, status: {status}",
        f"created_at: {created_at}, updated_at: {updated_at}",
        ""
    ]
    
    # Messages
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    
    # Sort by created_at
    messages.sort(key=lambda m: m.get("created_at", ""))
    
    total_chars = sum(len(line) for line in lines)
    
    for i, msg in enumerate(messages):
        created_at = msg.get("created_at", "N/A")
        role = msg.get("role", "unknown")
        author_id = msg.get("author_id", "N/A")
        text = msg.get("text", "").strip()
        attachments = msg.get("attachments", [])
        
        # Truncate per-message text
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "...[TRUNCATED]"
        
        msg_lines = [
            f"[{i}] <{created_at}> <{role}> author={author_id}",
            text
        ]
        
        if attachments:
            msg_lines.append(f"attachments: {len(attachments)}")
        
        msg_lines.append("")
        
        msg_text = "\n".join(msg_lines)
        if total_chars + len(msg_text) > MAX_TOTAL_TRANSCRIPT_CHARS:
            lines.append("...[TRUNCATED]")
            break
        
        lines.extend(msg_lines)
        total_chars += len(msg_text)
    
    return "\n".join(lines)


def get_triage_prompt(transcript: str) -> tuple[str, str]:
    """
    Get system and user prompts for triage stage.
    
    Args:
        transcript: Formatted transcript string
        
    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = """You are a conservative triage classifier. Output ONLY valid JSON, no prose or explanations.

Your task: Classify tickets into exactly one of three categories:
- "deny": Definitely NOT cache-eligible (missing problem/steps/confirmation, or has denial/onsite/replacement)
- "candidate": Potentially cache-eligible (has clear problem, actionable steps, explicit confirmation)
- "uncertain": Ambiguous or unclear (default to this if unsure)

CRITICAL RULES:
- Default to "deny" unless there is EXPLICIT evidence of ALL three:
  1. A specific problem clearly stated
  2. A specific actionable set of steps attempted
  3. Explicit confirmation the issue is resolved (e.g., "resolved", "fixed", "working now", requester confirms)
- If ANY of these are missing or unclear => return "deny" or "uncertain" (prefer "deny" if clearly not eligible)
- If you see denial/onsite/replacement language => return "deny"
- You CANNOT set cache_eligible=1 - only the final judge can do that
- Be conservative: when in doubt, choose "deny" or "uncertain"
"""

    user_message = f"""Analyze this support ticket conversation and classify it.

Output JSON with this exact schema:
{{
  "triage_label": "deny" | "candidate" | "uncertain",
  "confidence": 0.0-1.0,
  "reason": "brief explanation of classification",
  "signals": {{
    "has_clear_problem": true|false,
    "has_actionable_steps": true|false,
    "has_explicit_confirmation": true|false,
    "mentions_replacement_or_rma": true|false,
    "mentions_onsite": true|false,
    "mentions_denied": true|false
  }}
}}

Transcript:
{transcript}"""

    return system_message, user_message


def call_anthropic_api(
    client: Anthropic,
    model: str,
    system_message: str,
    user_message: str,
    max_tokens: int = 400,
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


def triage_ticket(
    client: Anthropic,
    ticket_id: str,
    conversation: Dict[str, Any],
    model: str = "claude-3-haiku-20240307"
) -> Dict[str, Any]:
    """
    Triage a single ticket using cheap model (Haiku).
    
    IMPORTANT: This function NEVER sets cache_eligible=1. Only Sonnet can do that.
    
    Args:
        client: Anthropic client
        ticket_id: Ticket ID
        conversation: Conversation JSON dict
        model: Model name (default: Claude 3 Haiku)
        
    Returns:
        Triage dict with triage_label, confidence, reason, signals, etc.
    """
    # Build transcript
    transcript = build_transcript(conversation)
    
    # Get prompts
    system_msg, user_msg = get_triage_prompt(transcript)
    
    # Call API
    response_text = call_anthropic_api(
        client, model, system_msg, user_msg, max_tokens=400
    )
    
    # Parse JSON
    triage_json = parse_json_response(response_text)
    
    if not triage_json:
        # Fallback: invalid JSON -> uncertain
        return {
            "ticket_id": ticket_id,
            "triage_label": "uncertain",
            "triage_confidence": 0.0,
            "triage_reason": "invalid_json_response",
            "triage_model": model,
            "triage_prompt_version": TRIAGE_PROMPT_VERSION,
            "triage_raw_response_json": json.dumps({"error": "invalid_json", "response": response_text}, ensure_ascii=False),
            "signals": {
                "has_clear_problem": False,
                "has_actionable_steps": False,
                "has_explicit_confirmation": False,
                "mentions_replacement_or_rma": False,
                "mentions_onsite": False,
                "mentions_denied": False
            }
        }
    
    # Validate and normalize triage_label
    triage_label = triage_json.get("triage_label", "uncertain").lower()
    if triage_label not in ["deny", "candidate", "uncertain"]:
        triage_label = "uncertain"
    
    # Clamp confidence to [0, 1]
    confidence = float(triage_json.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    
    # Extract signals
    signals = triage_json.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    
    return {
        "ticket_id": ticket_id,
        "triage_label": triage_label,
        "triage_confidence": confidence,
        "triage_reason": triage_json.get("reason", ""),
        "triage_model": model,
        "triage_prompt_version": TRIAGE_PROMPT_VERSION,
        "triage_raw_response_json": json.dumps(triage_json, ensure_ascii=False),
        "signals": signals
    }

