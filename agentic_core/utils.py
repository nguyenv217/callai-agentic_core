import re

def clean_context_for_downstream(raw_response: str) -> str:
    """Removes <think> blocks to prevent context pollution."""
    cleaned = re.sub(r'<think>.*?</think>\n?', '', raw_response, flags=re.DOTALL)
    return cleaned.strip()