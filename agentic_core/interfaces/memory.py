from typing import Protocol, List, Dict

class TruncationStrategy(Protocol):
    """Protocol for pluggable memory truncation logic."""
    def truncate(self, messages: List[Dict], max_chars: int) -> List[Dict]:
        """
        Processes a list of messages to fit within constraints.
        Returns a modified list of messages.
        """
        ...
