import re

def clean_context_for_downstream(raw_response: str) -> str:
    """Removes <think> blocks to prevent context pollution."""
    cleaned = re.sub(r'<think>.*?</think>\n?', '', raw_response, flags=re.DOTALL)
    return cleaned.strip()

def convert_exception_to_message(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"