"""Production logs: one schema for every LLM call site, ingestion with validation and
de-duplication, and PII redaction before anything is stored."""
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field


@dataclass
class LogEntry:
    id: str
    prompt: str
    response: str
    feature: str = ""                # which product feature made the call
    model: str = ""
    system_prompt: str = ""
    confidence: float = None         # the model's own confidence, if it gives one
    latency_ms: float = None
    tokens: dict = field(default_factory=dict)
    feedback: str = ""               # thumbs_up, thumbs_down, edited, retried
    time: float = None


REQUIRED = ("id", "prompt", "response")
RULES = {   # redaction rules: (pattern, replacement)
    "email": (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    "card": (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    "iban": (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), "[IBAN]"),
    # a phone number has a leading + or separators between its digit groups; a bare digit run is left alone
    "phone": (re.compile(r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]?\d{3,4}(?!\w)"), "[PHONE]"),
}


def luhn(digits):
    total, flip = 0, False
    for d in reversed(digits):
        n = int(d) * (2 if flip else 1)
        total, flip = total + (n - 9 if n > 9 else n), not flip
    return total % 10 == 0


def redact(text):
    """(redacted text, {rule: count}). Card numbers must pass the Luhn check, so order numbers survive."""
    counts = {}
    for name, (pattern, token) in RULES.items():
        def swap(m):
            if name == "card" and not luhn(re.sub(r"\D", "", m.group())):
                return m.group()
            counts[name] = counts.get(name, 0) + 1
            return token
        text = pattern.sub(swap, text)
    return text, counts


def ingest(lines):
    """JSON lines in, valid redacted entries out, exact duplicates (same prompt and response) dropped.
    Returns (entries, report)."""
    out, seen, report = [], set(), {"read": 0, "invalid": 0, "duplicates": 0, "redacted": {}}
    for line in lines:
        report["read"] += 1
        try:
            raw = json.loads(line)
            if any(not raw.get(k) for k in REQUIRED):
                raise ValueError("missing field")
            entry = LogEntry(**{k: v for k, v in raw.items() if k in LogEntry.__dataclass_fields__})
        except (ValueError, TypeError):
            report["invalid"] += 1
            continue
        key = hashlib.sha256(f"{entry.prompt}\x00{entry.response}".encode()).hexdigest()
        if key in seen:
            report["duplicates"] += 1
            continue
        seen.add(key)
        for f in ("prompt", "response", "system_prompt"):
            text, counts = redact(getattr(entry, f))
            setattr(entry, f, text)
            for k, v in counts.items():
                report["redacted"][k] = report["redacted"].get(k, 0) + v
        out.append(entry)
    return out, report


def dump(entries):
    return "\n".join(json.dumps(asdict(e)) for e in entries)
