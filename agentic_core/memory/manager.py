import hashlib
import json
from typing import List, Dict

class MemoryManager:
    def __init__(self, max_messages: int = 50, max_chars: int = 80000):
        self.messages: List[Dict] = []
        self.system_prompt: Dict = None
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._hash_obj = hashlib.sha256()
        self._current_hash = None

    def set_system_prompt(self, content: str):
        self.system_prompt = {"role": "system", "content": content}
        self._update_hash()

    def inject_dynamic_system_prompt(self, content: str):
        if self.system_prompt:
            self.system_prompt = {"role": "system", "content": self.system_prompt['content'] + content}
        else:
            self.system_prompt = {"role": "system", "content": content}

    def add_message(self, message: Dict):
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

    def get_history(self) -> List[Dict]:
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
        """
        Prevents context window overflow. Uses smart truncation for tool outputs 
        to preserve JSON schema integrity whenever possible.
        """
        total_chars = sum(len(str(m.get("content", ""))) for m in self.messages)
        
        # If we exceed the limit, prune starting from the oldest non-system messages
        # If we exceed the limit, prune starting from the oldest non-system messages
        if total_chars > getattr(self, 'max_chars', 80000) and (len(self.messages) > 0):
            running_total = 0
            
            # Iterate backwards (newest to oldest)
            # Iterate backwards (newest to oldest)
            for i in range(len(self.messages) - 1, -1, -1): 
                msg = self.messages[i]
                content_str = str(msg.get("content", ""))
                running_total += len(content_str)
                
                # Only heavily truncate tool outputs (where massive payloads usually occur)
                if running_total > self.max_chars and msg.get("role") == "tool":
                    # Use a dynamic threshold: 25% of max_chars, capped at 3000
                    threshold = min(3000, self.max_chars // 4)
                    if len(content_str) > threshold:
                        try:
                            # Smart Truncation: Try to preserve JSON structure
                            data = json.loads(content_str)
                            if isinstance(data, list):
                                # Keep only the first 3 items to maintain the schema example
                                msg["content"] = json.dumps(data[:3]) + "\n\n... [ARRAY TRUNCATED: Exceeded context limits] ..."
                            elif isinstance(data, dict):
                                # Truncate dict string representation
                                msg["content"] = content_str[:threshold] + "\n\n... [JSON TRUNCATED: Exceeded context limits] ..."
                        except json.JSONDecodeError:
                            # Fallback for plain text
                            msg["content"] = content_str[:threshold] + "\n\n... [TEXT TRUNCATED: Exceeded context limits] ..."
                        
                        # Adjust running total after truncation
                        running_total -= (len(content_str) - len(msg["content"]))
        
        self._update_hash()

    def _enforce_message_limit(self):
        """Prunes messages in valid structural pairs to maintain LLM API context validity."""
        while len(self.messages) > self.max_messages:
            if not self.messages:
                break
                
            first_msg = self.messages[0]
            first_role = first_msg.get("role")
            
            if first_role == "user":
                assistant_idx = next((i for i in range(1, len(self.messages)) if self.messages[i].get("role") == "assistant"), None)
                if assistant_idx is not None:
                    self.messages.pop(0) # pop user
                    self.messages.pop(0) # pop assistant
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