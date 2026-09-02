"""Pre-response privacy guard for langchain_agent.py's cloud tier.

A direct port of scan_text()/_redact() from ThesisVault's
LanguageDeprivationAgent/agent/privacy_scan.py -- same patterns, same
high-severity redaction, no new dependency (pure re, stdlib only). The rest
of that file (the vault-sweep CLI, its .ragconfig.yaml-driven skip dirs) is
Mac-tooling-specific and intentionally not ported; only the scan primitive
langchain_agent.py needs is here.

Gated behind BRAIN_PRIVACY_SCAN=1 (default off) so it never runs on retrieved
context until it has been checked against real vault content on-device.
"""

import re

PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "api_key_sk": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "api_key_generic": re.compile(r"\b(?:api[_-]?key|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}"),
    "aws_access": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b"),
}

HIGH_SEVERITY = {"ssn", "credit_card", "api_key_sk", "api_key_generic",
                  "aws_access", "jwt"}

PUBLICATION_KEYWORD_RE = re.compile(
    r"\b(doi|isbn|issn|orcid|pmid|arxiv|ciencia|cienciavitae|sagepub|jstor|crossref)\b",
    re.IGNORECASE,
)


def _is_publication_id(match, context):
    bare = match.replace(" ", "").replace("-", "")
    if bare.startswith(("978", "979")):
        return True
    if bare.startswith("0000") and len(bare) == 16:
        return True
    if PUBLICATION_KEYWORD_RE.search(context):
        return True
    return False


def _redact(s):
    if len(s) <= 8:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def scan_text(text):
    """Return {pattern_name: [matched strings, up to 3]} for whatever PII/
    secret patterns hit. High-severity hits (ssn/credit_card/api keys/aws/
    jwt) come back redacted; ssn/credit_card matches that look like a
    publication identifier (DOI/ISBN/etc.) in their surrounding context are
    dropped as false positives, same as the Mac scanner."""
    hits = {}
    for name, rx in PATTERNS.items():
        kept = []
        for m in rx.finditer(text):
            match_str = m.group(0)
            if name in ("credit_card", "ssn"):
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 50)
                if _is_publication_id(match_str, text[start:end]):
                    continue
            kept.append(match_str)
            if len(kept) >= 3:
                break
        if kept:
            if name in HIGH_SEVERITY:
                kept = [_redact(s) for s in kept]
            hits[name] = kept
    return hits
