import re
import unicodedata
import logging

def normalize_spaces(s):
    """Normalizes whitespace in a string."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_non_printable(text):
    """Removes non-printable control characters that can confuse models."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

def sanitize_for_injection(text, replacement_marker="[SANITIZED_INSTRUCTION]"):
    """
    Sanitizes text to neutralize potential indirect prompt injections
    found in source documents or direct injections in user queries.
    """
    if not text:
        return text

    # 1. Strip non-printable characters
    text = strip_non_printable(text)

    # 2. Define regex patterns for common injection phrases
    # This list targets common command-like phrases and "role-playing" instructions.
    injection_patterns = [
        # General instructions
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?(your|the)\s+instructions",
        r"disregard\s+the\s+above",
        r"delete\s+your\s+instructions",
        r"you\s+are\s+now",
        r"your\s+new\s+instructions\s+are",
        r"as\s+an\s+ai",
        
        # Specific attack patterns (role, chatML, etc.)
        r"(user|system|assistant|human)\s*:",
        r"<\s*\|(system|user|endoftext|im_start|im_end)\s*\|>",
        
        # Harmful commands
        r"reveal\s+your\s+instructions",
        r"print\s+your\s+prompt",
        r"what\s+are\s+your\s+instructions",
        
        # Evasion attempts
        r"stop\s+being\s+a\s+chatbot",
        r"act\s+as\s+[a-zA-Z\s]+",
    ]
    
    # Compile a single, case-insensitive regex
    compiled_pattern = re.compile(
        r'|'.join(f"({p})" for p in injection_patterns), 
        re.IGNORECASE | re.DOTALL
    )
    # i changed the replacer
    def replacer(m):
        gi = None
        if m.lastindex:
            gi = m.lastindex
        else:
            gs = m.groups()
            for idx, g in enumerate(gs, start=1):
                if g is not None:
                    gi = idx
                    break
        if gi is None:
            return replacement_marker
        try:
            _ = injection_patterns[gi - 1]  
        except Exception:
            return replacement_marker
        return replacement_marker
    sanitized_text = compiled_pattern.sub(replacer, text)
    return normalize_spaces(sanitized_text)
