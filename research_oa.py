#!/usr/bin/env python3
"""research_oa.py - open-access full-text resolver for the research agent.

Given a DOI (or a title when no DOI is known), asks the legitimate
open-access indexes where a free, publisher- or author-deposited copy lives,
fetches each candidate URL, and returns the first body that passes
research_fetch.verify_fulltext(). Every source here is a lawful OA index;
nothing is fetched from shadow libraries.

Source chain, stop at the first verified body:
  1. Unpaywall        best_oa_location + every oa_locations[] pdf, then landing
  2. OpenAlex         best_oa_location + every is_oa location (pdf, then landing)
  3. Europe PMC       REST search by DOI -> fullTextXML (JATS) when OA / has PMCID
  4. Semantic Scholar openAccessPdf.url; also surfaces the arXiv id
  5. CORE             search by DOI -> fullText (already extracted) or downloadUrl
  6. arXiv            https://arxiv.org/pdf/<id> when an arXiv id is known

Usage as a script:
    python3 research_oa.py --doi 10.1044/jshd.3903.330
    python3 research_oa.py --title "Signed English: A Manual Approach" --out paper.md
    python3 research_oa.py --doi ... --json

Usage from the pipeline: research_fetch.fetch_fulltext() calls
fetch_open_access(candidate) after its publisher/PMC routes fail and before
it settles on an abstract-only stub. Same {"body","via","note"} contract.

Config (env, optional): OA_MAILTO (contact email sent to Unpaywall/OpenAlex),
CORE_API_KEY, S2_API_KEY. See oa.env.example.

Never raises into the caller. Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import research_fetch as rf

logging.getLogger("pypdf").setLevel(logging.ERROR)   # font-encoding chatter

OA_MAILTO = os.environ.get("OA_MAILTO") or rf.MAILTO
CORE_API_KEY = os.environ.get("CORE_API_KEY", "").strip()
S2_API_KEY = os.environ.get("S2_API_KEY", "").strip()

API_PAUSE_S = 0.5          # between index calls; S2 unauthenticated is ~1 req/s
API_TIMEOUT = 25
MAX_URL_FETCHES = 6        # per candidate, across all sources
MAX_SECONDS = 60           # per candidate wall clock

# PMC article pages (either host, with or without the PMC prefix) serve a
# challenge to non-browser clients; the only machine route is NCBI efetch.
_PMC_ANY_RE = re.compile(
    r"ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/(?:PMC)?(\d+)", re.I)
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7})", re.I)
_DOI_CLEAN_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.I)

# Per-run memo so repeat candidates across feedback rows do not refetch.
_CACHE = {}


def clean_doi(doi):
    doi = str(doi or "").strip()
    doi = _DOI_CLEAN_RE.sub("", doi)
    return doi.strip().rstrip(".")


def arxiv_id_from(*texts):
    for t in texts:
        m = _ARXIV_ID_RE.search(str(t or ""))
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Index clients: each returns a list of (url, source_tag) or, for sources that
# hand back text directly, (None, source_tag, text).
# ---------------------------------------------------------------------------
def _get_json(url, params=None, headers=None, retry_429=True):
    """GET JSON via urllib; one 5 s retry on 429. None on any failure."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in (0, 1):
        try:
            hdrs ={"User-Agent": rf.USER_AGENT, "Accept": "application/json"}
            hdrs.update(headers or {})
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:                       # noqa: BLE001
            code = getattr(exc, "code", None)
            if code == 429 and retry_429 and attempt == 0:
                time.sleep(5)
                continue
            return None
    return None


def unpaywall_urls(doi):
    data = _get_json("https://api.unpaywall.org/v2/" + urllib.parse.quote(doi, safe="/"),
                     {"email": OA_MAILTO})
    out = []
    if not data or not data.get("is_oa"):
        return out
    best = data.get("best_oa_location") or {}
    locs = [best] + [l for l in (data.get("oa_locations") or []) if l is not best]
    for loc in locs:
        pdf = loc.get("url_for_pdf")
        if pdf:
            out.append((pdf, "unpaywall-pdf"))
    for loc in locs:
        landing = loc.get("url_for_landing_page") or loc.get("url")
        if landing:
            out.append((landing, "unpaywall-html"))
    return out


def openalex_urls(doi):
    data = _get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/"),
                     {"mailto": OA_MAILTO})
    out = []
    if not data:
        return out
    best = data.get("best_oa_location") or {}
    locs = [best] + [l for l in (data.get("locations") or [])
                     if l is not best and l.get("is_oa")]
    for loc in locs:
        if loc.get("pdf_url"):
            out.append((loc["pdf_url"], "openalex-pdf"))
    for loc in locs:
        if loc.get("landing_page_url"):
            out.append((loc["landing_page_url"], "openalex-html"))
    return out


def europepmc_text(doi):
    """Full text straight from Europe PMC's JATS endpoint, or ''."""
    data = _get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     {"query": 'DOI:"%s"' % doi, "format": "json", "pageSize": 3})
    hits = ((data or {}).get("resultList") or {}).get("result") or []
    for hit in hits:
        pmcid = hit.get("pmcid")
        if not pmcid:
            continue
        if hit.get("isOpenAccess") != "Y" and hit.get("inEPMC") != "Y":
            continue
        status, raw = rf._http_get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML" % pmcid,
            timeout=API_TIMEOUT, accept="application/xml")
        if status == 200 and raw:
            text = rf.jats_to_text(raw)
            if text:
                return text
        # Author manuscripts are indexed but not served as XML by Europe PMC;
        # NCBI efetch still has the JATS deposit.
        text = rf.fetch_pmc_body(pmcid)
        if text:
            return text
    return ""


def crossref_doi_for_title(title):
    """Best-match DOI from Crossref for a title, or None. Requires the top
    hit's title to share most of the query's words so a near-miss is rejected."""
    if not title:
        return None
    data = _get_json("https://api.crossref.org/works",
                     {"query.bibliographic": title, "rows": 1, "mailto": OA_MAILTO})
    items = ((data or {}).get("message") or {}).get("items") or []
    if not items:
        return None
    hit = items[0]
    hit_title = " ".join(hit.get("title") or []).lower()
    want = {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 3}
    got = set(re.findall(r"[a-z0-9]+", hit_title))
    if want and len(want & got) / len(want) < 0.7:
        return None
    return hit.get("DOI")


def semanticscholar(doi=None, title=None):
    """(pdf_url or None, arxiv_id or None)."""
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    fields = "openAccessPdf,externalIds,title"
    if doi:
        data = _get_json("https://api.semanticscholar.org/graph/v1/paper/DOI:"
                         + urllib.parse.quote(doi, safe="/"),
                         {"fields": fields}, headers)
    elif title:
        res = _get_json("https://api.semanticscholar.org/graph/v1/paper/search",
                        {"query": title, "limit": 1, "fields": fields}, headers)
        data = ((res or {}).get("data") or [None])[0]
    else:
        data = None
    if not data:
        return None, None
    pdf = (data.get("openAccessPdf") or {}).get("url")
    arx = (data.get("externalIds") or {}).get("ArXiv")
    return pdf or None, arx or None


def core_lookup(doi):
    """(fulltext_str, download_url) from CORE; either may be empty."""
    headers = {"Authorization": "Bearer " + CORE_API_KEY} if CORE_API_KEY else {}
    data = _get_json("https://api.core.ac.uk/v3/search/works",
                     {"q": 'doi:"%s"' % doi, "limit": 3}, headers)
    for hit in (data or {}).get("results") or []:
        text = hit.get("fullText") or ""
        url = hit.get("downloadUrl") or ""
        if text or url:
            return text, url
    return "", ""


# ---------------------------------------------------------------------------
# Fetch + verify one URL (shared by every source above).
# ---------------------------------------------------------------------------
def _fetch_and_verify(url, source):
    """(body, reason). body='' when the URL did not yield verified text."""
    reason = rf.dead_domain_reason(url)
    if reason:
        return "", "dead domain: " + reason
    m = _PMC_ANY_RE.search(url)
    if m:
        body = rf.fetch_pmc_body("PMC" + m.group(1))
        ok, why = rf.verify_fulltext(url, body)
        return (body if ok else ""), "pmc-efetch: " + why
    session = rf._curl_session()
    try:
        if session is not None:
            resp = session.get(url, timeout=rf.FETCH_TIMEOUT, allow_redirects=True)
            status, content = resp.status_code, resp.content
            ctype = (resp.headers.get("content-type") or "").lower()
            text = resp.text
        else:
            status, content = rf._http_get(url)
            ctype = ""
            text = content.decode("utf-8", "replace")
    except Exception as exc:                            # noqa: BLE001
        return "", "fetch error: %s" % exc.__class__.__name__
    if status != 200:
        return "", "HTTP %s" % status
    if "pdf" in ctype or content[:5] == b"%PDF-":
        body = rf._pdf_to_text(content)
        ok, why = rf.verify_fulltext(url, body)
    else:
        body = rf._html_to_text(text, url)
        ok, why = rf.verify_fulltext(url, body, raw_html=text)
    return (body if ok else ""), why


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------
def resolve(doi=None, title=None, hints=(), log=None):
    """Walk the source chain. Returns {"body","via","note","tried"} where
    tried is a list of (source, url_or_None, reason) for the CLI report."""
    doi = clean_doi(doi)
    key = doi or (title or "").strip().lower()
    if key in _CACHE:
        return _CACHE[key]
    _log = log or (lambda *a: None)
    if not doi and title:
        doi = clean_doi(crossref_doi_for_title(title))
        _log("crossref title -> %s" % (doi or "no confident DOI match"))
    tried = []
    start = time.time()
    fetches = 0

    def done(body, via, note):
        res = {"body": body[:rf.FULLTEXT_CAP], "via": via, "note": note,
               "tried": tried}
        _CACHE[key] = res
        return res

    def budget_left():
        return fetches < MAX_URL_FETCHES and (time.time() - start) < MAX_SECONDS

    def try_urls(pairs):
        nonlocal fetches
        seen = set()
        for url, source in pairs:
            if not url or url in seen or not budget_left():
                continue
            seen.add(url)
            fetches += 1
            body, why = _fetch_and_verify(url, source)
            if why.startswith("pmc-efetch"):
                source = source.rsplit("-", 1)[0] + "-pmc"
            tried.append((source, url, why))
            _log("  %-16s %s -> %s" % (source, url[:90], why))
            if body:
                return body, source
        return "", ""

    arxiv_id = arxiv_id_from(*hints)

    if doi:
        _log("unpaywall")
        body, via = try_urls(unpaywall_urls(doi))
        if body:
            return done(body, via, "")
        time.sleep(API_PAUSE_S)

        _log("openalex")
        body, via = try_urls(openalex_urls(doi))
        if body:
            return done(body, via, "")
        time.sleep(API_PAUSE_S)

        _log("europepmc")
        body = europepmc_text(doi)
        tried.append(("europepmc-xml", None, "ok" if body else "no OA full text"))
        _log("  %-16s -> %s" % ("europepmc-xml", "ok" if body else "no OA full text"))
        if body:
            return done(body, "europepmc-xml", "")
        time.sleep(API_PAUSE_S)

    _log("semanticscholar")
    s2_pdf, s2_arxiv = semanticscholar(doi=doi or None, title=title)
    arxiv_id = arxiv_id or s2_arxiv
    if s2_pdf:
        body, via = try_urls([(s2_pdf, "s2-pdf")])
        if body:
            return done(body, via, "")
    else:
        tried.append(("s2-pdf", None, "no openAccessPdf"))
        _log("  %-16s -> no openAccessPdf" % "s2-pdf")
    time.sleep(API_PAUSE_S)

    if doi:
        _log("core")
        core_text, core_url = core_lookup(doi)
        if core_text:
            ok, why = rf.verify_fulltext(core_url or "core", core_text)
            tried.append(("core-text", core_url or None, why))
            _log("  %-16s -> %s" % ("core-text", why))
            if ok:
                return done(core_text, "core-text", "")
        if core_url:
            body, via = try_urls([(core_url, "core-pdf")])
            if body:
                return done(body, via, "")
        if not core_text and not core_url:
            tried.append(("core", None, "no hit"))
            _log("  %-16s -> no hit" % "core")
        time.sleep(API_PAUSE_S)

    if arxiv_id:
        _log("arxiv")
        body, via = try_urls([("https://arxiv.org/pdf/" + arxiv_id, "arxiv-pdf")])
        if body:
            return done(body, via, "")

    return done("", "", "no verified full text; abstract-only stub")


def fetch_open_access(candidate):
    """Pipeline entry: same contract as research_fetch.fetch_fulltext()."""
    try:
        res = resolve(
            doi=candidate.get("doi"),
            title=candidate.get("title"),
            hints=(candidate.get("fulltext_url"), candidate.get("landing_url")))
        return {"body": res["body"], "via": res["via"], "note": res["note"]}
    except Exception as exc:                            # noqa: BLE001
        return {"body": "", "via": "",
                "note": "OA resolver error: %s" % exc.__class__.__name__}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--doi", help="DOI, bare or as a doi.org URL")
    ap.add_argument("--title", help="title, used only when no DOI is known")
    ap.add_argument("--out", help="write the full text (markdown) here")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--quiet", action="store_true", help="no per-source trace")
    args = ap.parse_args(argv)
    if not args.doi and not args.title:
        ap.error("give --doi or --title")

    log = (lambda *a: None) if args.quiet or args.json else (
        lambda msg: print(msg, file=sys.stderr))
    res = resolve(doi=args.doi, title=args.title, log=log)

    if args.out and res["body"]:
        head = "# %s\n\nvia: %s\n\n" % (args.doi or args.title, res["via"])
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(head + res["body"] + "\n")

    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "tried"}
                         | {"tried": [list(t) for t in res["tried"]]}, indent=2))
    elif res["body"]:
        print("via: %s   chars: %d%s" % (
            res["via"], len(res["body"]),
            ("   written: " + args.out) if args.out else ""))
        print()
        print(res["body"][:300].rstrip() + (" ..." if len(res["body"]) > 300 else ""))
    else:
        print(res["note"])
    return 0 if res["body"] else 1


if __name__ == "__main__":
    sys.exit(main())
