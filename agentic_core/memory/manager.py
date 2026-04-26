import hashlib
import json

from agentic_core.constants import MEMORY_MANAGER_MAX_CHARS

from .strategies import TruncationStrategy, DefaultTruncationStrategy

class MemoryManager:
    def __init__(self, max_messages: int | None = None, max_chars: int = MEMORY_MANAGER_MAX_CHARS, strategy: TruncationStrategy = None):
        """
        Initialize the MemoryManager instance.

        Args:
            max_messages : int, optional
                (Deprecated/Not recommended) The maximum number of messages to store. `max_messages` structurally pops older messages. May invalidate your message cache for many provders. Defaults to no limit.  
            max_chars : int, optional
                The maximum number of characters in the conversation history. Defaults to 80000.
            strategy : TruncationStrategy, optional
                A strategy for content-level truncation. Defaults to DefaultTruncationStrategy.
        """
        self.messages: list[dict] = []
        self.system_prompt: dict = None
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.strategy = strategy or DefaultTruncationStrategy()
        self._hash_obj = hashlib.sha256()
        self._current_hash = None
        # self.truncate_by_pop = False

    def system_prompt_exists(self):
        return self.system_prompt is not None

    def set_system_prompt(self, content: str):
        self.system_prompt = {"role": "system", "content": content}
        self._update_hash()

    def inject_dynamic_system_prompt(self, content: str):
        if self.system_prompt:
            self.system_prompt = {"role": "system", "content": self.system_prompt['content'] + content}
        else:
            self.system_prompt = {"role": "system", "content": content}

    def add_message(self, message: dict):
        """Adds standard messages (user, assistant)."""
        self.messages.append(message)
        self._update_hash()
        self._enforce_message_limit()

    def add_tool_result(self, name: str, tool_call_id: str, content: str):
        """Adds a tool result message."""
        msg = {"role": "tool", "name": name, "tool_call_id": tool_call_id, "content": content}
        self.messages.append(msg)
        self._update_hash()
        self._enforce_message_limit()

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
        # First handle the structural pruning (message count) -> 
        self._enforce_message_limit()
        
        # Use the strategy for content-level truncation
        self.messages = self.strategy.truncate(self.messages, self.max_chars)
        
        self._update_hash()

    # THIS IS DEPRECATED BUT KEPT FOR BACKWARD COMPABILITY/or anyone who actually uses it anyway 
    def _enforce_message_limit(self):
        """
        Prunes messages in valid structural pairs to maintain LLM API context validity.
        IMPORTANT: Deprecated `pop()` truncation and will defaults to replacing with placeholder instead to preserve structural integrity.
        """
        if not self.max_messages:
            return

        will_change = False
        if len(self.messages) > self.max_messages:
            will_change = True

        while len(self.messages) > self.max_messages:
            first_msg = self.messages[0]
            first_role = first_msg.get("role")
            
            if first_role == "user":
                assistant_idx = next((i for i in range(1, len(self.messages)) if self.messages[i].get("role") == "assistant"), None)
                if assistant_idx is not None:
                    self.messages.pop(0)
                else:
                    self.messages.pop(0)
                    
            elif first_role == "assistant":
                tool_calls = first_msg.get("tool_calls", [])
                num_tool_results_needed = len(tool_calls) if tool_calls else 0
                
                if num_tool_results_needed > 0:
                    tool_indices = [i for i in range(1, len(self.messages)) if self.messages[i].get("role") == "tool"]
                    if len(tool_indices) >= num_tool_results_needed:
                        for idx in reversed(tool_indices[:num_tool_results_needed]):
                            self.messages.pop(idx)
                        self.messages.pop(0)
                    else:
                        self.messages.pop(0)
                else:
                    if len(self.messages) > 1 and self.messages[1].get("role") == "tool":
                        self.messages.pop(1)
                    else:
                        self.messages.pop(0)
            else:
                self.messages.pop(0)
        
        if will_change:
            self._update_hash()
    
    def _update_hash(self):
        self._hash_obj = hashlib.sha256()
        if self.system_prompt:
            self._hash_obj.update(json.dumps(self.system_prompt, sort_keys=True).encode())
        for msg in self.messages:
            # We strip out non-hashable/volatile internal data if needed here
            self._hash_obj.update(json.dumps(msg, sort_keys=True).encode())
        self._current_hash = self._hash_obj.hexdigest()

    def clear(self):
        self.messages = []
        self._update_hash()
    
    def is_new_session(self):
        return len(self.messages) == 0