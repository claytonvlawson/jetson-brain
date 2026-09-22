#!/usr/bin/env python3
"""research_fetch.py - verified full-text retrieval for the research agent.

Adapted from gallaudet-rag's papers.py plus its hard-won access rules
(docs/paper-access-verification.md). The headline lesson encoded here:
an HTTP 200 with a big HTML payload is NOT full text, and an OpenAlex
"open access" label is NOT a retrievable file. Access is only believed
when body content was actually extracted.

The route table:
  - Known-dead publisher domains (ScienceDirect, Sage PDF routes, Muse,
    LWW, OUP, ASHA without its full-text container divs) are not fetched
    at all: candidates go straight to abstract-only.
  - PMC HTML/pdf URLs are a trap twice over (reCAPTCHA with HTTP 200, HTML
    landing page instead of PDF); the ONLY machine route is NCBI efetch
    JATS XML, reached via the idconv DOI-to-PMCID API.
  - Everything else is fetched with curl_cffi Chrome impersonation (plain
    TLS clients get 403 from half the publishers), PDF via pypdf, HTML via
    trafilatura when installed with a BeautifulSoup fallback.
  - Any failure falls back DOI-to-PMC, then to the open-access indexes in
    research_oa.py (Unpaywall, OpenAlex, Europe PMC, Semantic Scholar, CORE,
    arXiv), then to abstract-only, which is a legitimate outcome (the note
    becomes a metadata stub with the abstract).

verify_fulltext() is the gate: PDF magic bytes or content-type for PDFs;
for HTML, a minimum extracted length AND no bot-challenge markers.

Never raises into the caller. Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import io
import os
import re
import json
import time
import urllib.parse
import urllib.request

MAILTO = os.environ.get("OA_MAILTO", "claytonvlawson@gmail.com")
USER_AGENT = "jetson-brain-research-agent/1.0 (mailto:%s)" % MAILTO
FETCH_TIMEOUT = 60
MIN_BODY_CHARS = 500          # papers.py threshold: less than this is not a body
FULLTEXT_CAP = 20000          # cap stored full text so notes stay indexable
NCBI_PAUSE_S = 0.4            # max 3 req/s without an API key

# Domains where no automated client gets full text (verified 2026-07-02 in
# gallaudet-rag). Fetching them wastes time and can look like success (big
# HTML payloads that are abstract-only or JS challenge shells). The entry
# value documents the reason, printed in --dry-run listings.
DEAD_DOMAINS = {
    "sciencedirect.com": "403 JS challenge for all clients, even hybrid OA",
    "linkinghub.elsevier.com": "redirects into the ScienceDirect challenge",
    "journals.sagepub.com": "doi/pdf routes 403 even with impersonation",
    "muse.jhu.edu": "2 KB stub pages, treat abstract-only",
    "journals.lww.com": "body not retrievable",
    "academic.oup.com": "403 via curl_cffi; use PMC when deposited",
}

# ASHA is special-cased: pages load (200, large) but are usually abstract-only.
# Full text exists only when one of the Atypon container markers is present.
ASHA_DOMAIN = "pubs.asha.org"
ASHA_FULLTEXT_MARKERS = ("article__body", "hlFld-Fulltext", "sectionInfo")

BOT_CHALLENGE_MARKERS = ("recaptcha", "cf-challenge", "captcha-delivery",
                         "challenge-platform")

_PMC_URL_RE = re.compile(r"pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)", re.I)


def _domain(url):
    try:
        host = urllib.parse.urlparse(str(url)).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def dead_domain_reason(url):
    """The documented reason a URL's domain is unfetchable, or None."""
    d = _domain(url)
    for dead, reason in DEAD_DOMAINS.items():
        if d == dead or d.endswith("." + dead):
            return reason
    return None


# ---------------------------------------------------------------------------
# PMC route (stdlib only; usable before curl_cffi is installed).
# ---------------------------------------------------------------------------
def _http_get(url, params=None, timeout=FETCH_TIMEOUT, accept="*/*"):
    """Plain-urllib GET returning (status, bytes) or (None, b'')."""
    try:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT, "Accept": accept})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except Exception:
        return None, b""


def pmcid_from_doi(doi):
    """DOI to PMCID via the NCBI id converter. None when not deposited."""
    if not doi:
        return None
    time.sleep(NCBI_PAUSE_S)
    status, raw = _http_get(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        {"ids": doi, "format": "json"}, timeout=30)
    try:
        recs = json.loads(raw.decode("utf-8", "replace")).get("records", [])
        pmcid = recs[0].get("pmcid") if recs else None
        return pmcid or None
    except Exception:
        return None


def _et_text(elem):
    """All text under an ElementTree node, whitespace-collapsed."""
    return re.sub(r"\s+", " ", " ".join(elem.itertext())).strip()


def fetch_pmc_body(pmcid):
    """Full text via NCBI efetch JATS XML. PMC's own HTML pages serve a
    reCAPTCHA challenge (with HTTP 200) to non-browser TLS clients, so the
    E-utilities API is the only reliable machine route. Parsed with stdlib
    ElementTree (JATS body/sec/p carry no namespace), so this route has zero
    third-party dependencies. Returns '' on failure."""
    if not pmcid:
        return ""
    time.sleep(NCBI_PAUSE_S)
    status, raw = _http_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pmc", "id": pmcid, "rettype": "xml"})
    if status != 200 or not raw:
        return ""
    return jats_to_text(raw)


def jats_to_text(raw):
    """JATS XML bytes (NCBI efetch or Europe PMC fullTextXML) to markdown-ish
    text: '## Abstract', then each body section heading and its paragraphs.
    '' when the deposit has no <body> long enough to be an article."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw.decode("utf-8", "replace"))
        body_parts = []
        body = root.find(".//body")
        if body is not None:
            secs = body.findall("./sec")
            for sec in secs:
                title = sec.find("./title")
                if title is not None and _et_text(title):
                    body_parts.append("## " + _et_text(title))
                for p in sec.iter("p"):
                    text = _et_text(p)
                    if len(text) > 40:
                        body_parts.append(text)
            if not secs:
                for p in body.iter("p"):
                    text = _et_text(p)
                    if len(text) > 40:
                        body_parts.append(text)
        # Publisher-restricted deposits efetch WITHOUT a <body> (only an
        # abstract plus a notice). Gate on body-derived text alone, so an
        # abstract can never masquerade as verified full text downstream (the
        # abstract already reaches the note via the candidate metadata).
        if len("\n\n".join(body_parts)) < MIN_BODY_CHARS:
            return ""
        parts = []
        abstract = root.find(".//abstract")
        if abstract is not None:
            text = _et_text(abstract)
            if text:
                parts.append("## Abstract\n\n" + text)
        parts.extend(body_parts)
        return "\n\n".join(parts)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Publisher route (curl_cffi + pypdf + trafilatura/bs4; all optional).
# ---------------------------------------------------------------------------
def _curl_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _pdf_to_text(data):
    """PDF bytes to plain text via pypdf. '' when pypdf is missing/fails."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(pages).replace("-\n", "")   # undo hyphenation breaks
        paras = [re.sub(r"\s+", " ", p).strip()
                 for p in re.split(r"\n\s*\n", text)]
        return "\n\n".join(p for p in paras if len(p) > 40)
    except Exception:
        return ""


def _html_to_text(html, url):
    """Readable article text from HTML: trafilatura when installed, else a
    BeautifulSoup strip-the-chrome fallback. '' on failure."""
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url, include_comments=False)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header",
                         "footer", "aside", "form"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup
        text = main.get_text(separator="\n")
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        return "\n\n".join(ln for ln in lines if len(ln) > 40)
    except Exception:
        return ""


def verify_fulltext(url, body, raw_html=None):
    """The believe-nothing gate. Returns (ok, reason). ok=True only when the
    extracted body is long enough to be an article AND, when the raw page is
    available, it carries no bot-challenge markers; ASHA pages additionally
    need one of the Atypon full-text container divs."""
    if not body or len(body) < MIN_BODY_CHARS:
        return False, "extracted %d chars (< %d)" % (len(body or ""), MIN_BODY_CHARS)
    low = (raw_html or "").lower()
    if low:
        for marker in BOT_CHALLENGE_MARKERS:
            if marker in low:
                return False, "bot challenge marker: " + marker
        if _domain(url) == ASHA_DOMAIN and not any(
                m.lower() in low for m in ASHA_FULLTEXT_MARKERS):
            return False, "ASHA page without full-text container divs"
    return True, "ok"


def fetch_fulltext(candidate):
    """Try to retrieve verified full text for one candidate record.

    Returns {"body": str, "via": str, "note": str}. body='' means abstract-only
    (with `note` saying why); body non-empty means it PASSED verify_fulltext.
    Order: dead-domain skip, PMC-URL and DOI-to-PMC efetch route, then a
    curl_cffi impersonated fetch of the claimed fulltext_url. Never raises."""
    url = str(candidate.get("fulltext_url") or "").strip()
    doi = str(candidate.get("doi") or "").strip()

    if url:
        reason = dead_domain_reason(url)
        if reason:
            # Do not fetch; try the PMC mirror by DOI instead.
            body = fetch_pmc_body(pmcid_from_doi(doi)) if doi else ""
            if body and len(body) >= MIN_BODY_CHARS:
                return {"body": body[:FULLTEXT_CAP], "via": "pmc-efetch",
                        "note": "publisher skipped (%s), recovered via PMC" % reason}
            return {"body": "", "via": "", "note": "dead domain: " + reason}

        m = _PMC_URL_RE.search(url)
        if m:
            body = fetch_pmc_body(m.group(1))
            ok, why = verify_fulltext(url, body)
            if ok:
                return {"body": body[:FULLTEXT_CAP], "via": "pmc-efetch",
                        "note": ""}
            return {"body": "", "via": "", "note": "PMC efetch: " + why}

        session = _curl_session()
        if session is None:
            return {"body": "", "via": "",
                    "note": "curl_cffi not installed; queued abstract-only"}
        try:
            resp = session.get(url, timeout=FETCH_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                ctype = (resp.headers.get("content-type") or "").lower()
                if "pdf" in ctype or resp.content[:5] == b"%PDF-":
                    body = _pdf_to_text(resp.content)
                    ok, why = verify_fulltext(url, body)
                    if ok:
                        return {"body": body[:FULLTEXT_CAP], "via": "pdf",
                                "note": ""}
                else:
                    body = _html_to_text(resp.text, url)
                    ok, why = verify_fulltext(url, body, raw_html=resp.text)
                    if ok:
                        return {"body": body[:FULLTEXT_CAP], "via": "html",
                                "note": ""}
        except Exception:
            pass

    # Last resort for anything with a DOI: the PMC mirror.
    if doi:
        body = fetch_pmc_body(pmcid_from_doi(doi))
        if body and len(body) >= MIN_BODY_CHARS:
            return {"body": body[:FULLTEXT_CAP], "via": "pmc-efetch",
                    "note": "recovered via DOI-to-PMC"}

    # Open-access indexes (Unpaywall, OpenAlex, Europe PMC, Semantic Scholar,
    # CORE, arXiv). Lawful deposits only; same verify gate.
    if doi or candidate.get("title"):
        try:
            import research_oa
            oa = research_oa.fetch_open_access(candidate)
            if oa["body"]:
                return {"body": oa["body"], "via": oa["via"],
                        "note": "recovered via open-access index"}
        except Exception:
            pass

    return {"body": "", "via": "",
            "note": "no verified full text; abstract-only stub"}
