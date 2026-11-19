import re
import logfire
from groq import Groq

from app.config import settings

# 1. Regex patterns for common enterprise PII & sensitive secrets
PATTERNS = {
    "API_KEY": r"(?:gsk_|sk-|AIzaSy)[A-Za-z0-9_\-]{20,}",
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}\b",
    "BEARER_TOKEN": r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",
}

# 2. Heuristic patterns for explicit jailbreaks and destructive commands
JAILBREAK_PATTERNS = [
    r"(?i)\b(?:developer mode|dan mode|jailbreak|unrestricted assistant)\b",
    r"(?i)\b(?:bypass|override)\b.*\b(?:safety|rules|filters|instructions)\b",
    r"(?i)\b(?:rm -rf|drop table|delete collection|format c:)\b",
]

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def sanitize_pii(text: str) -> tuple[str, bool]:
    """
    Scans text and redacts sensitive PII and API keys in sub-1ms.
    Returns (sanitized_text, was_pii_found).
    """
    sanitized = text
    pii_detected = False
    for label, pattern in PATTERNS.items():
        if re.search(pattern, sanitized):
            pii_detected = True
            sanitized = re.sub(pattern, f"[REDACTED_{label}]", sanitized)
    if pii_detected:
        logfire.warning("PII or sensitive secret detected and redacted from input.")
    return sanitized, pii_detected


def check_prompt_injection(query: str, threshold: float = 0.85) -> tuple[bool, float]:
    """
    Defense-in-depth evaluation:
    1. Heuristic regex scan for explicit jailbreak/command execution (< 0.1ms).
    2. Groq's Llama-Prompt-Guard-2-86m model for semantic injection (< 120ms).
    Returns (is_attack, attack_probability).
    """
    # 1. Fast Heuristic Scan
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, query):
            logfire.error("🚨 Heuristic Security Rail: Jailbreak pattern detected!")
            return True, 1.0

    # 2. AI Prompt Guard Model
    client = _get_groq_client()
    try:
        with logfire.span("Security Guardrail: Prompt Guard Check"):
            response = client.chat.completions.create(
                model="meta-llama/llama-prompt-guard-2-86m",
                messages=[{"role": "user", "content": query}],
                max_tokens=10,
            )
            score_str = response.choices[0].message.content.strip()
            score = float(score_str)
            is_attack = score >= threshold
            if is_attack:
                logfire.error(
                    f" Prompt Injection BLOCKED! Score: {score:.4f} (Threshold: {threshold})"
                )
            else:
                logfire.info(f"Prompt Guard PASSED. Score: {score:.4f}")
            return is_attack, score
    except Exception as e:
        logfire.warning(f"Prompt Guard check bypassed due to error: {e}")
        # Fail-Open or Fail-Closed based on enterprise policy (here fail-open with logging)
        return False, 0.0


def sanitize_output(text: str) -> str:
    """
    Scans final generated answer to ensure no credentials or secrets leaked out.
    """
    sanitized, _ = sanitize_pii(text)
    return sanitized
