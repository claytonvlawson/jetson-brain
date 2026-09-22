#!/usr/bin/env python3
"""research_sources.py - discovery clients for the deaf-education research agent.

Finds candidate literature on language support for deaf children from free,
keyless scholarly APIs, plus the Clerc Center's practitioner pages:

  1. OpenAlex (api.openalex.org): topic sweeps over language-deprivation and
     ASL-acquisition queries, an INSTITUTION sweep for everything published
     from Gallaudet University (OpenAlex institution I66108857, verified), and
     author sweeps for the focus scholars (Wyatte Hall, Naomi Caselli, ...).
  2. ERIC (api.ies.ed.gov/eric): education-focused, carries an explicit
     peerreviewed flag, so it anchors the peer-review gate.
  3. Clerc Center listing pages (clerccenter.gallaudet.edu): grey literature
     (Odyssey magazine, resources) that never reaches the scholarly indexes.
     Cloudflare 403s plain clients, so this uses curl_cffi Chrome
     impersonation (the proven gallaudet-rag route); when curl_cffi is not
     installed the Clerc sweep is skipped and everything else still runs.

Every client returns candidates in ONE record shape (the gallaudet-rag
papers.py manifest schema extended with provenance):

    {authors: [..], year, title, venue, doi, landing_url, fulltext_url,
     fulltext_type, access, relevance_note, abstract, oa_status,
     peer_reviewed, openalex_id, eric_id, discovery_source, discovered_at}

Never-raise contract per client: one API being down loses that sweep, not the
run. Standard library only for the JSON APIs (urllib), so discovery works on
the Jetson before any pip install. Python 3.9 compatible. No em dashes.
"""
from __future__ import annotations

import re
import json
import time
import datetime
import urllib.parse
import urllib.request

USER_AGENT = "jetson-brain-research-agent/1.0 (mailto:claytonvlawson@gmail.com)"
# OpenAlex polite pool: including a mailto gets a faster, more reliable tier.
OPENALEX_MAILTO = "claytonvlawson@gmail.com"
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_AUTHORS = "https://api.openalex.org/authors"
ERIC_API = "https://api.ies.ed.gov/eric/"

# Gallaudet University in OpenAlex (ROR 02b9aym09), verified live 2026-08-11.
GALLAUDET_INSTITUTION_ID = "I66108857"

# Topic sweeps: the thesis focus, phrased for title_and_abstract.search.
# OpenAlex search supports | for OR within one filter value.
OPENALEX_TOPIC_QUERIES = (
    '"language deprivation" deaf',
    '"sign language" acquisition deaf children',
    'deaf children "early intervention" language',
    '"bimodal bilingual" OR "bilingual-bimodal" deaf education',
    'deaf "language outcomes" children',
)

# Focus scholars (papers.py FOCUS_AUTHORS plus the two the vault tracks).
FOCUS_AUTHOR_NAMES = (
    "Wyatte C. Hall",
    "Naomi Caselli",
    "Maartje De Meulder",
    "Carey Ballard",
)

# ERIC sweeps. peerreviewed:"T" anchors the peer-review gate at the source.
ERIC_QUERIES = (
    '(deaf AND "language deprivation") AND peerreviewed:T',
    '("American Sign Language" AND acquisition) AND peerreviewed:T',
    '(deaf AND "early intervention" AND language) AND peerreviewed:T',
    '("deaf education" AND bilingual) AND peerreviewed:T',
)
ERIC_FIELDS = ("id,title,author,description,source,publicationdateyear,"
               "peerreviewed,publicationtype,url")

# Clerc Center grey-literature listing pages. A fixed allowlist, not a crawl:
# these index pages are diffed by link URL against previously seen state.
CLERC_LISTING_PAGES = (
    "https://clerccenter.gallaudet.edu/national-resources/odyssey-magazine/",
    "https://clerccenter.gallaudet.edu/national-resources/info-to-go/",
)
CLERC_IMPERSONATE = "chrome"   # curl_cffi target; plain clients get 403

REQUEST_TIMEOUT = 30
PER_PAGE = 50
MAX_PAGES_PER_SWEEP = 4        # 200 records per sweep per run is plenty
API_PAUSE_S = 0.25             # politeness gap between API pages


def _now_iso():
    try:
        return datetime.datetime.now().isoformat(timespec="seconds")
    except Exception:
        return ""


def _get_json(base_url, params, timeout=REQUEST_TIMEOUT):
    """GET a JSON API with urllib. Returns the parsed object or None."""
    try:
        url = base_url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def normalize_doi(doi):
    """Bare lowercase DOI ('10.x/...') from any of the shapes APIs emit
    (https://doi.org/10.x, doi:10.x, 10.x). '' when there is none."""
    s = str(doi or "").strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s if s.startswith("10.") else ""


def reconstruct_abstract(inv_idx, cap=4000):
    """OpenAlex ships abstracts as an inverted index {word: [positions]}.
    Rebuild the plain text. '' on any failure or when absent."""
    try:
        if not isinstance(inv_idx, dict) or not inv_idx:
            return ""
        pos = {}
        for word, positions in inv_idx.items():
            for p in positions:
                pos[int(p)] = word
        return " ".join(pos[i] for i in sorted(pos))[:cap]
    except Exception:
        return ""


def _blank_candidate():
    return {
        "authors": [], "year": None, "title": "", "venue": "", "doi": "",
        "landing_url": "", "fulltext_url": "", "fulltext_type": None,
        "access": "abstract-only", "relevance_note": "", "abstract": "",
        "oa_status": "", "peer_reviewed": False, "openalex_id": "",
        "eric_id": "", "discovery_source": "", "discovered_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def _openalex_candidate(work, discovery_source):
    """Map one OpenAlex work object to the shared candidate record."""
    c = _blank_candidate()
    c["discovery_source"] = discovery_source
    c["openalex_id"] = str(work.get("id") or "").rsplit("/", 1)[-1]
    c["title"] = str(work.get("display_name") or work.get("title") or "").strip()
    c["year"] = work.get("publication_year")
    c["doi"] = normalize_doi(work.get("doi"))
    c["landing_url"] = str(work.get("doi") or "") or str(
        (work.get("primary_location") or {}).get("landing_page_url") or "")
    try:
        c["authors"] = [
            str((a.get("author") or {}).get("display_name") or "").strip()
            for a in (work.get("authorships") or []) if a][:12]
        c["authors"] = [a for a in c["authors"] if a]
    except Exception:
        c["authors"] = []
    try:
        src = ((work.get("primary_location") or {}).get("source") or {})
        c["venue"] = str(src.get("display_name") or "").strip()
        # Journal-published article = the peer-review proxy OpenAlex offers.
        c["peer_reviewed"] = (work.get("type") == "article"
                              and src.get("type") == "journal")
    except Exception:
        pass
    try:
        oa = work.get("open_access") or {}
        c["oa_status"] = str(oa.get("oa_status") or "")
        oa_url = oa.get("oa_url") or (work.get("best_oa_location") or {}).get("pdf_url")
        if oa.get("is_oa") and oa_url:
            # A CLAIM, not a fact: research_fetch verifies by actual retrieval
            # (OpenAlex OA labels have confirmed false positives).
            c["fulltext_url"] = str(oa_url)
            c["access"] = "open"
    except Exception:
        pass
    c["abstract"] = reconstruct_abstract(work.get("abstract_inverted_index"))
    return c


def _openalex_sweep(filter_expr, discovery_source, max_pages=MAX_PAGES_PER_SWEEP):
    """One cursor-paginated OpenAlex works sweep. Returns candidates ([] on
    failure)."""
    out = []
    cursor = "*"
    for _page in range(max_pages):
        data = _get_json(OPENALEX_WORKS, {
            "filter": filter_expr, "per-page": PER_PAGE, "cursor": cursor,
            "mailto": OPENALEX_MAILTO,
        })
        if not data or not isinstance(data.get("results"), list):
            break
        for work in data["results"]:
            try:
                cand = _openalex_candidate(work, discovery_source)
                if cand["title"]:
                    out.append(cand)
            except Exception:
                continue
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or not data["results"]:
            break
        time.sleep(API_PAUSE_S)
    return out


def resolve_author_ids(names=FOCUS_AUTHOR_NAMES):
    """Resolve author display names to OpenAlex author ids (A...). Returns
    {name: id} for the names that resolved; callers cache this in state.json
    so it runs once, not weekly."""
    ids = {}
    for name in names:
        data = _get_json(OPENALEX_AUTHORS, {
            "search": name, "per-page": 1, "mailto": OPENALEX_MAILTO})
        try:
            results = (data or {}).get("results") or []
            if results:
                aid = str(results[0].get("id") or "").rsplit("/", 1)[-1]
                if aid.startswith("A"):
                    ids[name] = aid
        except Exception:
            continue
        time.sleep(API_PAUSE_S)
    return ids


def openalex_candidates(since_date, author_ids=None):
    """All three OpenAlex sweeps (topics, Gallaudet institution, focus
    authors) since `since_date` (YYYY-MM-DD). Returns a list of candidates;
    a failed sweep contributes nothing."""
    out = []
    datef = ",from_publication_date:" + since_date if since_date else ""
    for q in OPENALEX_TOPIC_QUERIES:
        expr = ('title_and_abstract.search:' + q
                + ',type:article' + datef)
        out.extend(_openalex_sweep(expr, "openalex-topic"))
    out.extend(_openalex_sweep(
        "authorships.institutions.id:" + GALLAUDET_INSTITUTION_ID + datef,
        "openalex-gallaudet"))
    for name, aid in (author_ids or {}).items():
        out.extend(_openalex_sweep(
            "author.id:" + aid + datef, "openalex-author:" + name))
    return out


def openalex_candidates_for_query(query, since_date=None, max_pages=1,
                                  discovery_source="openalex-feedback"):
    """One free-text OpenAlex sweep for a query that is NOT on the fixed
    topic list (feedback_research.py turns a parent's complaint into these).
    Same candidate shape as openalex_candidates(); [] on failure or empty
    query. One page (50 works) by default: a feedback run wants the top hits,
    not a backfill."""
    q = str(query or "").strip()
    if not q:
        return []
    # A stray comma would be read as a filter separator by OpenAlex.
    q = q.replace(",", " ")
    datef = ",from_publication_date:" + since_date if since_date else ""
    expr = "title_and_abstract.search:" + q + ",type:article" + datef
    return _openalex_sweep(expr, discovery_source, max_pages=max_pages)


# ---------------------------------------------------------------------------
# ERIC
# ---------------------------------------------------------------------------
def _eric_candidate(rec):
    c = _blank_candidate()
    c["discovery_source"] = "eric"
    c["eric_id"] = str(rec.get("id") or "")
    c["title"] = str(rec.get("title") or "").strip()
    try:
        c["year"] = int(rec.get("publicationdateyear"))
    except Exception:
        c["year"] = None
    authors = rec.get("author")
    if isinstance(authors, list):
        c["authors"] = [str(a).strip() for a in authors if str(a).strip()][:12]
    elif authors:
        c["authors"] = [str(authors).strip()]
    c["venue"] = str(rec.get("source") or "").strip()
    c["abstract"] = str(rec.get("description") or "").strip()[:4000]
    c["peer_reviewed"] = str(rec.get("peerreviewed") or "").upper() == "T"
    url = str(rec.get("url") or "").strip()
    if url:
        c["landing_url"] = url
    elif c["eric_id"]:
        c["landing_url"] = "https://eric.ed.gov/?id=" + c["eric_id"]
    return c


def eric_candidates(since_year):
    """All ERIC sweeps. `since_year` bounds publicationdateyear (int or None).
    Returns candidates; a failed sweep contributes nothing."""
    out = []
    for q in ERIC_QUERIES:
        query = q
        if since_year:
            query += " AND publicationdateyear:[%d TO 3000]" % int(since_year)
        start = 0
        for _page in range(MAX_PAGES_PER_SWEEP):
            data = _get_json(ERIC_API, {
                "search": query, "format": "json", "rows": PER_PAGE,
                "start": start, "fields": ERIC_FIELDS})
            try:
                docs = ((data or {}).get("response") or {}).get("docs") or []
            except Exception:
                docs = []
            if not docs:
                break
            for rec in docs:
                try:
                    cand = _eric_candidate(rec)
                    if cand["title"]:
                        out.append(cand)
                except Exception:
                    continue
            if len(docs) < PER_PAGE:
                break
            start += PER_PAGE
            time.sleep(API_PAUSE_S)
    return out


# ---------------------------------------------------------------------------
# Clerc Center grey literature
# ---------------------------------------------------------------------------
_CLERC_LINK_RE = re.compile(
    r'<a[^>]+href="(https://clerccenter\.gallaudet\.edu/[^"#?]+)"[^>]*>(.*?)</a>',
    re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def clerc_candidates(seen_urls=None):
    """Diff the fixed Clerc Center listing pages for new article links.
    Grey literature: peer_reviewed stays False and access 'grey-lit' so the
    review CLI displays these distinctly. Needs curl_cffi (Cloudflare); when
    unavailable, returns [] and the caller logs the skip."""
    try:
        from curl_cffi import requests as curl_requests
    except Exception:
        return None   # None = the sweep could not run (vs [] = ran, nothing new)
    seen = set(seen_urls or [])
    out = []
    try:
        session = curl_requests.Session(impersonate=CLERC_IMPERSONATE)
    except Exception:
        return None
    for page_url in CLERC_LISTING_PAGES:
        try:
            resp = session.get(page_url, timeout=REQUEST_TIMEOUT,
                               allow_redirects=True)
            if resp.status_code != 200:
                continue
            for href, label in _CLERC_LINK_RE.findall(resp.text):
                href = href.rstrip("/")
                if href in seen or href.rstrip("/") == page_url.rstrip("/"):
                    continue
                title = _TAG_RE.sub(" ", label)
                title = re.sub(r"\s+", " ", title).strip()
                if len(title) < 12:     # nav/button links, not articles
                    continue
                seen.add(href)
                c = _blank_candidate()
                c["discovery_source"] = "clerc-listing"
                c["title"] = title[:200]
                c["venue"] = "Clerc Center (Gallaudet University)"
                c["landing_url"] = href
                c["access"] = "grey-lit"
                out.append(c)
        except Exception:
            continue
        time.sleep(API_PAUSE_S)
    return out


def candidate_keys(cand):
    """Every dedup key this candidate answers to (normalized doi, openalex id,
    eric id, landing URL for grey-lit). Used against state.json's seen and
    rejected sets, the queue, and the vault scan."""
    keys = []
    doi = normalize_doi(cand.get("doi"))
    if doi:
        keys.append("doi:" + doi)
    if cand.get("openalex_id"):
        keys.append("openalex:" + cand["openalex_id"])
    if cand.get("eric_id"):
        keys.append("eric:" + cand["eric_id"])
    if not keys and cand.get("landing_url"):
        keys.append("url:" + cand["landing_url"].rstrip("/").lower())
    return keys
