import re
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
    
    def __init__(self, ticket_id: str, ticket_content: Dict[str, Any]):
        """
        Initialize a Ticket with raw data.
        
        Args:
            ticket_id: Unique identifier for the ticket
            ticket_content: Raw JSON dictionary of the ticket conversation
        """
        self.ticket_id = ticket_id
        self.ticket_content = ticket_content
        
        # Deterministic fields computed immediately
        self.message_count = self._get_ticket_length(ticket_content)
        self.error_codes = self._extract_error_codes(ticket_content)
        
        # LLM-derived fields (None until derive() is called)
        self.ticket_title: Optional[str] = None
        self.ticket_description: Optional[str] = None
        self.is_resolved: Optional[bool] = None
        self.solution_description: Optional[str] = None
    
    def _extract_messages(self, ticket_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract a list of messages from ticket_content.
        
        Tries common keys like 'messages', 'conversation', 'thread'.
        If ticket_content is already a list, returns it.
        Otherwise returns empty list.
        
        Args:
            ticket_content: Raw ticket JSON
            
        Returns:
            List of message dictionaries
        """
        if isinstance(ticket_content, list):
            return ticket_content
        
        if isinstance(ticket_content, dict):
            # Try common keys
            for key in ['messages', 'conversation', 'thread', 'comments', 'replies']:
                if key in ticket_content and isinstance(ticket_content[key], list):
                    return ticket_content[key]
        
        return []
    
    def _get_ticket_length(self, ticket_content: Dict[str, Any]) -> int:
        """
        Get the number of messages in the ticket conversation.
        
        Args:
            ticket_content: Raw ticket JSON
            
        Returns:
            Number of messages (0 if unable to determine)
        """
        messages = self._extract_messages(ticket_content)
        return len(messages)
    
    def _extract_error_codes(self, ticket_content: Dict[str, Any]) -> List[str]:
        """
        Extract error codes from the ticket conversation text using regex.
        
        Captures patterns like:
        - E123, ERR123 (alphanumeric codes)
        - Error 52, Error: 52 (Error + number)
        
        Args:
            ticket_content: Raw ticket JSON
            
        Returns:
            List of unique error codes found (sorted)
        """
        # Get combined text from messages
        messages = self._extract_messages(ticket_content)
        text_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                for field in ['body', 'text', 'content', 'message', 'comment']:
                    if field in msg and isinstance(msg[field], str):
                        text_parts.append(msg[field])
                        break
            elif isinstance(msg, str):
                text_parts.append(msg)
        text = ' '.join(text_parts)
        
        # Patterns to match:
        # - Alphanumeric codes: E123, ERR123, ERROR123, etc.
        # - "Error" + number: Error 52, Error: 52, etc.
        patterns = [
            r'\bE[A-Z0-9]+\d+\b',  # E123, ERR123, ERROR123
            r'\bERR[A-Z0-9]*\d+\b',  # ERR123, ERRCODE123
            r'Error\s*:?\s*(\d+)',  # Error 52, Error: 52
        ]
        
        error_codes = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # For groups, take the first non-empty group
                    match = next((m for m in match if m), '')
                if match:
                    error_codes.add(str(match).upper())
        
        return sorted(list(error_codes))
    
    @property
    def combined_text(self) -> str:
        """
        Get a clean concatenation of all message text from the conversation.
        
        This is deterministic and used for embeddings/semantic matching.
        Does not call the LLM.
        
        Returns:
            Concatenated text from all messages
        """
        messages = self._extract_messages(self.ticket_content)
        
        text_parts = []
        for msg in messages:
            # Try common text fields
            if isinstance(msg, dict):
                for field in ['body', 'text', 'content', 'message', 'comment']:
                    if field in msg and isinstance(msg[field], str):
                        text_parts.append(msg[field])
                        break
            elif isinstance(msg, str):
                text_parts.append(msg)
        
        return ' '.join(text_parts)
    
    def derive(self, llm_client: Any, *, force: bool = False) -> None:
        """
        Populate LLM-derived fields using the provided LLM client.
        
        If fields are already present and force=False, does nothing.
        
        Args:
            llm_client: LLM client instance (provider-agnostic interface)
            force: If True, regenerate fields even if they already exist
        """
        if not force:
            if all([
                self.ticket_title is not None,
                self.ticket_description is not None,
                self.is_resolved is not None,
                self.solution_description is not None
            ]):
                return
        
        # Generate LLM-derived fields
        self.ticket_title = self._generate_ticket_title(llm_client, self.ticket_content)
        self.ticket_description = self._generate_ticket_description(llm_client, self.ticket_content)
        self.is_resolved = self._generate_is_resolved(llm_client, self.ticket_content)
        
        if self.is_resolved:
            self.solution_description = self._generate_solution_description(llm_client, self.ticket_content)
        else:
            self.solution_description = "The ticket was not resolved."
    
    def _generate_ticket_title(self, llm_client: Any, ticket_content: Dict[str, Any]) -> str:
        """
        Use LLM to generate a one-sentence summary of the entire ticket.
        
        Args:
            llm_client: LLM client instance
            ticket_content: Raw ticket JSON
            
        Returns:
            One-sentence summary string
        """
        # Placeholder: implement with your LLM client
        # Example: return llm_client.generate_title(ticket_content)
        return ""
    
    def _generate_ticket_description(self, llm_client: Any, ticket_content: Dict[str, Any]) -> str:
        """
        Use LLM to generate a description of the ticket.
        
        Args:
            llm_client: LLM client instance
            ticket_content: Raw ticket JSON
            
        Returns:
            Ticket description string
        """
        # Placeholder: implement with your LLM client
        # Example: return llm_client.generate_description(ticket_content)
        return ""
    
    def _generate_is_resolved(self, llm_client: Any, ticket_content: Dict[str, Any]) -> bool:
        """
        Use LLM to determine if the ticket was resolved.
        
        Args:
            llm_client: LLM client instance
            ticket_content: Raw ticket JSON
            
        Returns:
            True if resolved, False otherwise
        """
        # Placeholder: implement with your LLM client
        # Example: return llm_client.is_resolved(ticket_content)
        return False
    
    def _generate_solution_description(self, llm_client: Any, ticket_content: Dict[str, Any]) -> str:
        """
        Use LLM to generate a description of the solution to the ticket.
        
        Only called if is_resolved is True.
        
        Args:
            llm_client: LLM client instance
            ticket_content: Raw ticket JSON
            
        Returns:
            Solution description string
        """
        # Placeholder: implement with your LLM client
        # Example: return llm_client.generate_solution_description(ticket_content)
        return ""
