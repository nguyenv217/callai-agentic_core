import hashlib
import json

from agentic_core.constants import MEMORY_MANAGER_MAX_CHARS

from .strategies import NoTruncationStrategy, TruncationStrategy, DefaultTruncationStrategy

class MemoryManager:
    def __init__(self, max_chars: int = MEMORY_MANAGER_MAX_CHARS, strategy: TruncationStrategy = None):
        """
        Initialize the MemoryManager instance.

        Args:
            max_chars : int, optional
                The maximum number of characters in the conversation history. Defaults to 80000.
            strategy : TruncationStrategy, optional
                A strategy for content-level truncation. Defaults to DefaultTruncationStrategy.
        """
        self.messages: list[dict] = []
        self.system_prompt: dict = None
        self.max_chars = max_chars
        self.strategy = strategy or NoTruncationStrategy()
        self._hash_obj = hashlib.sha256()
        self._current_hash = None

    def system_prompt_exists(self):
        return self.system_prompt is not None

    def set_system_prompt(self, content: str):
        self.system_prompt = {"role": "system", "content": content}
        self._update_hash()

    def add_message(self, message: dict):
        """Adds standard messages (user, assistant)."""
        self.messages.append(message)
        self._update_hash()

    def add_tool_result(self, name: str, tool_call_id: str, content: str):
        """Adds a tool result message."""
        msg = {"role": "tool", "tool_name": name, "tool_call_id": tool_call_id, "content": content}
        self.messages.append(msg)
        self._update_hash()

    def get_history(self) -> list[dict]:
        history = []
        if self.system_prompt:
            history.append(self.system_prompt)
        history.extend(self.messages)
        return history

    def get_hash(self) -> str:
        if self._current_hash is None:
            self._update_hash()
        return self._current_hash

    def enforce_context_limits(self):
        """Delegates complexity to the pluggable strategy."""
        self.messages = self.strategy.truncate(self.messages, self.max_chars)
        self._update_hash()

    def _update_hash(self):
        self._hash_obj = hashlib.sha256()
        if self.system_prompt:
            self._hash_obj.update(json.dumps(self.system_prompt, sort_keys=True).encode())
        for msg in self.messages:
            self._hash_obj.update(json.dumps(msg, sort_keys=True).encode())
        self._current_hash = self._hash_obj.hexdigest()

    def clear(self):
        self.messages = []
        self._update_hash()
    
    def is_new_session(self):
        return len(self.messages) == 0

    def export_state(self) -> dict:
        """Exports the current state of memory for snapshotting."""
        return {
            "system_prompt": self.system_prompt,
            "messages": list(self.messages),
            "max_chars": self.max_chars
        }

    def import_state(self, state: dict):
        """Restores the memory state from a provided state dictionary."""
        self.system_prompt = state.get("system_prompt")
        self.messages = list(state.get("messages", []))
        self.max_chars = state.get("max_chars", self.max_chars)
        self._update_hash()
