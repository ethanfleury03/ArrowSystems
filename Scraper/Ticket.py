import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class Ticket:
    """
    A ticket-derived cache layer for RAG systems.
    
    This class represents a finished/resolved ticket conversation as a clean,
    comparable "problem + solution" record for semantic matching and LLM context.
    
    The class is designed to be stable and deterministic:
    - `__init__` only stores raw data and computes cheap deterministic fields
    - LLM-derived fields are populated via `derive()` method
    - No LLM work is performed in `__init__`
    
    Fields:
    - ticket_id: Unique identifier for the ticket
    - ticket_content: Raw JSON of the ticket conversation
    - message_count: Number of messages in the conversation (deterministic)
    - error_codes: List of error codes extracted from conversation text (deterministic)
    - ticket_title: LLM-generated one-sentence summary (populated via derive())
    - ticket_description: LLM-generated description (populated via derive())
    - is_resolved: Whether the ticket was resolved (populated via derive())
    - solution_description: LLM-generated solution description (populated via derive())
    """
    
    def __init__(self, ticket_id: str, ticket_content: Any):
        """
        Initialize a Ticket with raw data.
        
        Args:
            ticket_id: Unique identifier for the ticket
            ticket_content: Raw JSON (dict, list, or any structure) of the ticket conversation
        """
        self.ticket_id = ticket_id
        self.ticket_content = ticket_content
        
        # Deterministic fields computed immediately
        messages = self.extract_messages(ticket_content)
        self.message_count = len(messages)
        self.error_codes = self._extract_error_codes()
        
        # LLM-derived fields (None until derive() is called)
        self.ticket_title: Optional[str] = None
        self.ticket_description: Optional[str] = None
        self.is_resolved: Optional[bool] = None
        self.solution_description: Optional[str] = None
    
    def extract_messages(self, ticket_content: Any) -> List[Dict[str, Any]]:
        """
        Extract and normalize messages from ticket_content.
        
        Returns normalized messages with structure:
        {
            "id": str (or index),
            "role": str (e.g., "user", "agent", "system", or "unknown"),
            "text": str
        }
        
        Supports common shapes:
        - ticket_content["messages"] list
        - ticket_content["conversation"] list
        - ticket_content["thread"] list
        - if ticket_content is already a list -> treat as messages
        
        Within each message dict, text may be under keys like:
        "text", "body", "comment", "message", "content"
        
        Args:
            ticket_content: Raw ticket JSON
            
        Returns:
            List of normalized message dictionaries
        """
        raw_messages = []
        
        # Handle different input shapes
        if isinstance(ticket_content, list):
            raw_messages = ticket_content
        elif isinstance(ticket_content, dict):
            # Try common keys
            for key in ['messages', 'conversation', 'thread', 'comments', 'replies']:
                if key in ticket_content and isinstance(ticket_content[key], list):
                    raw_messages = ticket_content[key]
                    break
        
        # Normalize messages
        normalized = []
        for idx, msg in enumerate(raw_messages):
            if isinstance(msg, str):
                # Simple string message
                normalized.append({
                    "id": str(idx),
                    "role": "unknown",
                    "text": msg
                })
            elif isinstance(msg, dict):
                # Extract text from common fields
                text = None
                for field in ['text', 'body', 'comment', 'message', 'content']:
                    if field in msg and isinstance(msg[field], str):
                        text = msg[field]
                        break
                
                if text is None:
                    # Try to stringify the whole dict
                    text = str(msg)
                
                # Extract role if available
                role = msg.get('role', msg.get('author', msg.get('sender', 'unknown')))
                if not isinstance(role, str):
                    role = 'unknown'
                
                # Extract id if available
                msg_id = msg.get('id', msg.get('message_id', str(idx)))
                if not isinstance(msg_id, str):
                    msg_id = str(msg_id)
                
                normalized.append({
                    "id": msg_id,
                    "role": role.lower() if isinstance(role, str) else "unknown",
                    "text": text
                })
        
        return normalized
    
    def _extract_error_codes(self) -> List[str]:
        """
        Extract error codes from the ticket conversation text using regex.
        
        Captures patterns like:
        - "Error 52", "Error: 52" -> normalized to "ERROR_52"
        - "E52", "ERR52", "E-52"
        - Codes like "DF-102" or "AB1234" (2-5 letters + optional dash + 2-5 digits)
        
        Returns:
            List of unique error codes found (sorted, normalized)
        """
        text = self.combined_text
        
        error_codes = set()
        
        # Pattern 1: "Error" + number (Error 52, Error: 52)
        error_number_pattern = r'Error\s*:?\s*(\d+)'
        matches = re.findall(error_number_pattern, text, re.IGNORECASE)
        for match in matches:
            error_codes.add(f"ERROR_{match}")
        
        # Pattern 2: E + digits (E52, E-52, E123)
        e_pattern = r'\bE-?\d+\b'
        matches = re.findall(e_pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = match.replace('-', '').upper()
            error_codes.add(normalized)
        
        # Pattern 3: ERR + alphanumeric (ERR52, ERR123, ERRCODE123)
        err_pattern = r'\bERR[A-Z0-9]*\d+\b'
        matches = re.findall(err_pattern, text, re.IGNORECASE)
        for match in matches:
            error_codes.add(match.upper())
        
        # Pattern 4: Letter codes with optional dash + digits (DF-102, AB1234)
        # 2-5 letters, optional dash, 2-5 digits
        letter_code_pattern = r'\b[A-Z]{2,5}-?\d{2,5}\b'
        matches = re.findall(letter_code_pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize: remove dash, uppercase
            normalized = match.replace('-', '').upper()
            # Only add if it looks like an error code (has both letters and digits)
            if re.search(r'[A-Z]', normalized) and re.search(r'\d', normalized):
                error_codes.add(normalized)
        
        return sorted(list(error_codes))
    
    @property
    def combined_text(self) -> str:
        """
        Get a clean concatenation of all message text from the conversation.
        
        Messages are joined with blank lines for readability.
        This is deterministic and used for embeddings/semantic matching.
        Does not call the LLM.
        
        Returns:
            Concatenated text from all messages (separated by blank lines)
        """
        messages = self.extract_messages(self.ticket_content)
        text_parts = [msg.get("text", "") for msg in messages if msg.get("text")]
        return "\n\n".join(text_parts)
    
    def derive(
        self,
        *,
        force: bool = False,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 1200,
        env_path: Optional[str] = None
    ) -> None:
        """
        Populate LLM-derived fields using Anthropic API with tool-use for structured output.
        
        If fields are already present and force=False, does nothing.
        
        Args:
            force: If True, regenerate fields even if they already exist
            model: Anthropic model to use (default: claude-3-5-sonnet-20241022)
            max_tokens: Maximum tokens for response
            env_path: Optional path to .env file (default: Scraper/.env relative to this file)
        
        Raises:
            ValueError: If ANTHROPIC_API_KEY is not found
            RuntimeError: If LLM response doesn't contain expected tool-use block
        """
        if not force:
            if all([
                self.ticket_title is not None,
                self.ticket_description is not None,
                self.is_resolved is not None,
                self.solution_description is not None
            ]):
                return
        
        # Load environment variables (only when needed, not at import time)
        if env_path is None:
            # Default to Scraper/.env relative to this file
            ticket_file = Path(__file__).parent
            env_path = ticket_file / ".env"
        else:
            env_path = Path(env_path)
        
        # Load .env file if it exists
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        
        # Get API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                f"ANTHROPIC_API_KEY not found in environment. "
                f"Please set it in {env_path} or as an environment variable."
            )
        
        # Import anthropic only when needed
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        
        # Create client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare input text (truncate if extremely long)
        combined = self.combined_text
        if len(combined) > 100000:  # Very long conversations
            # Keep first 50k chars and last 10k chars
            combined = combined[:50000] + "\n\n[... truncated ...]\n\n" + combined[-10000:]
        
        # Build user message
        user_content = f"""Ticket ID: {self.ticket_id}

Error Codes Found: {', '.join(self.error_codes) if self.error_codes else 'None'}

Ticket Conversation:
{combined}

Please extract the ticket information using the tool."""
        
        # Define tool schema
        tool_schema = {
            "type": "object",
            "properties": {
                "ticket_title": {
                    "type": "string",
                    "description": "A concise one-sentence summary of the ticket"
                },
                "ticket_description": {
                    "type": "string",
                    "description": "A detailed description of the problem or issue"
                },
                "is_resolved": {
                    "type": "boolean",
                    "description": "Whether the ticket was resolved/solved"
                },
                "solution_description": {
                    "type": "string",
                    "description": "If resolved, describe the solution. If not resolved, describe why or what was attempted."
                }
            },
            "required": ["ticket_title", "ticket_description", "is_resolved", "solution_description"],
            "additionalProperties": False
        }
        
        # Make API call
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system="You are a ticket analysis system. Return the result ONLY by calling the extract_ticket_record tool. Do not output extra text or explanations.",
                messages=[
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                tools=[
                    {
                        "name": "extract_ticket_record",
                        "description": "Extract structured information from a ticket conversation",
                        "input_schema": tool_schema
                    }
                ],
                tool_choice={"type": "tool", "name": "extract_ticket_record"}
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")
        
        # Parse response - find tool_use block
        tool_use_block = None
        for content_block in message.content:
            if content_block.type == "tool_use" and content_block.name == "extract_ticket_record":
                tool_use_block = content_block
                break
        
        if not tool_use_block:
            # Log the raw response for debugging
            raw_content = "\n".join(str(block) for block in message.content)
            raise RuntimeError(
                f"Expected tool_use block 'extract_ticket_record' not found in response. "
                f"Raw response content:\n{raw_content}"
            )
        
        # Extract fields from tool input
        try:
            extracted = tool_use_block.input
            self.ticket_title = extracted.get("ticket_title", "")
            self.ticket_description = extracted.get("ticket_description", "")
            self.is_resolved = extracted.get("is_resolved", False)
            self.solution_description = extracted.get("solution_description", "")
        except Exception as e:
            raise RuntimeError(f"Failed to parse tool_use input: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ticket to a dictionary representation.
        
        Returns:
            Dictionary with all ticket fields
        """
        return {
            "ticket_id": self.ticket_id,
            "message_count": self.message_count,
            "error_codes": self.error_codes,
            "ticket_title": self.ticket_title,
            "ticket_description": self.ticket_description,
            "is_resolved": self.is_resolved,
            "solution_description": self.solution_description,
        }
    
    def save_derived(self, path: Optional[str] = None) -> str:
        """
        Save derived ticket data to a JSON file.
        
        Args:
            path: Optional file path. If None, uses Scraper/out/derived_{ticket_id}.json
            
        Returns:
            Path to saved file
        """
        if path is None:
            ticket_file = Path(__file__).parent
            out_dir = ticket_file / "out"
            out_dir.mkdir(exist_ok=True)
            path = str(out_dir / f"derived_{self.ticket_id}.json")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(path)
