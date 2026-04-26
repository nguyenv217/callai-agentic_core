import json

from typing import Protocol

from agentic_core.constants import TRUNCATE_DEFAULT_TEXT_THRESHOLD, TRUNCATE_DEFAULT_TOOL_THRESHOLD

class TruncationStrategy(Protocol):
    """Protocol for pluggable memory truncation logic."""
    def truncate(self, messages: list[dict], max_chars: int) -> list[dict]:
        """
        Processes a list of messages to fit within constraints.
        Returns a modified list of messages.
        """
        ...

class NoTruncationStrategy(TruncationStrategy):
    """Strategy that prevents any message truncation. Oftentimes financially better for API providers supporing context caching."""
    
    def truncate(self, messages: list[dict], max_chars: int) -> list[dict]:
        """Returns the entire list of messages, bypassing truncation logic."""
        return messages

class DefaultTruncationStrategy(TruncationStrategy):
    """
    Standard strategy that prioritizes truncating tool outputs and 
    long plaintext messages over deleting history.
    """
    def __init__(self, tool_threshold: int = TRUNCATE_DEFAULT_TOOL_THRESHOLD, text_threshold: int = TRUNCATE_DEFAULT_TEXT_THRESHOLD):
        self.tool_threshold = tool_threshold
        self.text_threshold = text_threshold

    def truncate(self, messages: list[dict], max_chars: int) -> list[dict]:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars <= max_chars:
            return messages

        # Iterate backwards to preserve recent context
        # Create a copy to avoid mutating the original list if it's passed by reference 
        # although in this context the MemoryManager owns the list.
        messages_copy = [m.copy() for m in messages]

        for i in range(len(messages_copy) - 1, -1, -1):
            msg = messages_copy[i]
            role = msg.get("role")
            content = str(msg.get("content", ""))

            if role == "tool":
                # Truncate if it exceeds the default threshold OR the total limit
                msg["content"] = self._truncate_tool(content, min(self.tool_threshold, max_chars))
            
            elif role in ["user", "assistant"]:
                # Truncate if it exceeds the default threshold OR the total limit
                msg["content"] = self._truncate_text(content, min(self.text_threshold, max_chars))
            
            # elif role == "image": # Reserved for future
            #     msg["content"] = self._truncate_image(content)

            # Re-calculate and break if we are under limit
            new_total = sum(len(str(m.get("content", ""))) for m in messages_copy)
            if new_total <= max_chars:
                break
                
        return messages_copy

    def _truncate_tool(self, content: str, threshold: int) -> str:
        if len(content) <= threshold:
            return content
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return json.dumps(data[:3]) + "\n... [ARRAY TRUNCATED] ..."
            return content[:threshold] + "\n... [JSON TRUNCATED] ..."
        except json.JSONDecodeError:
            return content[:threshold] + "\n... [TEXT TRUNCATED] ..."

    def _truncate_text(self, content: str, threshold: int) -> str:
        if len(content) <= threshold:
            return content
        return content[:threshold] + "\n... [LONG TEXT TRUNCATED] ..."

    def _truncate_image(self, content: str) -> str:
        # Placeholder: could replace base64 data with a URL or descriptive text
        return "[IMAGE DATA REMOVED TO SAVE SPACE]"
