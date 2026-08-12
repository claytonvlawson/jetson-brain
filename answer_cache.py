#!/usr/bin/env python3
"""answer_cache.py - compounding, self-optimizing answer cache for the brain.

Stores answered questions as small JSON files under a cache directory so repeat
or near-repeat questions are served instantly with no retrieval or model call.
Lookups try an exact normalized-question match first, then a content-word
Jaccard fuzzy match (grammatical filler stripped, so rephrasings collide).

This is v3. It keeps every v2 behavior (exact + fuzzy lookup, defensive
never-raise contract, /cache commands) and adds a "compound-interest knowledge
bank" that unifies two goals:

  (A) OPTIMIZE STORAGE ON EVERY STORE. Answer bodies are gzip+base64 compressed
      on disk (the largest field), and after each store a budget pass evicts the
      LEAST VALUABLE entries until the cache is back under a byte/entry budget.
      Storage stays bounded no matter how long the brain is used.

  (B) COMPOUND THE KNOWLEDGE BANK ON EVERY INTERACTION. Each entry earns
      compound interest on reuse:

          V = P * (1 + r) ** hits * 0.5 ** (age_days / HALF_LIFE_DAYS)

      P (principal) is the one-time worth of the answer (longer, source-grounded
      answers are worth more). r is the per-reuse interest rate: every cache hit
      compounds the entry's value by (1 + r). A recency half-life decays entries
      that stop being used. The BANK BALANCE is the sum of all entry values; it
      rises every interaction -- a brand-new question deposits fresh principal, a
      reused answer compounds an existing balance -- exactly like an
      interest-bearing account. Eviction spends the cheapest (lowest-value)
      knowledge first, so the bank keeps the compounding, high-yield knowledge.

Compression and value are transparent to callers: store()/lookup() take and
return plain answer text; the value math only drives eviction and /cache stats.

Runs under the Jetson's Python 3.10 with the standard library only.
"""

import re
import os
import json
import gzip
import math
import atexit
import base64
import hashlib
import datetime


# -- Fuzzy-match thresholds ---------------------------------------------------
# The fuzzy matcher scores CONTENT words (grammatical filler stripped) so that
# rephrasings of the same question collide: "what is X", "explain X", and "tell
# me about X" all reduce to {x} and score 1.0. Filler no longer dilutes the
# sets, so this sits below the old whole-token 0.85 while staying safe. It is
# kept above 0.667 on purpose: that is the score of a 2-of-3 superset
# ("what prevents X" vs stored "what is X"), a different question we must miss.
FUZZY_THRESHOLD = 0.70   # minimum content-word Jaccard to accept a fuzzy hit
FUZZY_MIN_TOKENS = 1     # both questions need at least this many content words
SLUG_TOKEN_COUNT = 6     # number of normalized tokens used to build the slug
SLUG_MAX_LEN = 48        # hard cap on slug length (characters)

# Grammatical filler removed before the fuzzy set match. Question words, verbs
# of asking (tell/explain/...), articles, prepositions, auxiliaries, pronouns,
# and the single-letter fragments a tokenizer leaves behind when it splits
# contractions ("what's" -> what, s). NEGATIONS ARE DELIBERATELY ABSENT so that
# "is X safe" and "is X not safe" never collapse together.
STOPWORDS = frozenset({
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    # conjunctions / prepositions
    "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "about",
    "from", "into", "by", "as", "than", "then", "over", "under", "between",
    # auxiliaries / copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "done", "have", "has", "had", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must",
    # pronouns / politeness
    "i", "you", "we", "he", "she", "it", "they", "me", "us", "him", "her",
    "them", "my", "your", "our", "its", "their", "please", "kindly",
    # question words / verbs of asking
    "what", "whats", "why", "how", "when", "where", "who", "which", "whose",
    "whom", "tell", "explain", "describe", "define", "give", "show", "list",
    "summarize", "summarise",
    # filler
    "so", "if", "just", "really", "actually", "more", "much", "many",
    # contraction fragments left by the tokenizer
    "s", "t", "m", "re", "ve", "ll", "d",
})

# Whole-word negations. Two questions whose negation words differ are treated as
# different intents and never fuzzy-match, so "is X safe" cannot be served from a
# cached "is X not safe" (the old 0.85 threshold blocked this for free; the lower
# content threshold would not, so we guard it explicitly). Contraction-split
# forms like "don't" -> don, t are unrecoverable and out of scope.
NEGATIONS = frozenset({
    "no", "not", "never", "without", "none", "cannot", "nor", "neither",
})

# Question words, guarded on the same principle as NEGATIONS and for the same
# reason: they are STOPWORDS, so they are stripped before the fuzzy match and
# two questions that differ only in what they ask for score 1.0 against each
# other. "Where is Gallaudet University?" reduced to {gallaudet, university} and
# matched a primed entry literally titled "Gallaudet University" at Jaccard
# 1.000, so a request for a city returned a 250-word institutional essay and
# never reached retrieval at all. Raising FUZZY_THRESHOLD cannot fix this: the
# content sets really are identical. The intent has to be compared separately.
#
# Grouped by the answer they demand, so genuine paraphrases still collide:
# "where is X" / "what is the location of X" are the same question, and
# "what/which" stay together because "what school" and "which school" are too.
_WH_GROUPS = {
    "where": "place", "when": "time", "who": "person", "whom": "person",
    "whose": "person", "why": "reason", "how": "manner",
    "what": "thing", "which": "thing",
}

# -- Compound-interest "knowledge bank" parameters ----------------------------
# V = principal  *  (1 + INTEREST_RATE) ** hits  *  recency decay.
# The decay clock is measured from an entry's LAST USE, so a reuse both
# compounds the value (x (1+r)) and resets the clock to full (decay = 1.0).
INTEREST_RATE = 0.15         # r: each reuse compounds an entry's value by +15%
HALF_LIFE_DAYS = 30.0        # value halves after this many days with no reuse
PRINCIPAL_REF_WORDS = 120.0  # W0: word count of a "complete" personal answer
PRINCIPAL_SOURCE_BONUS = 0.4 # grounded answers (with sources) are worth more
PRINCIPAL_MAX = 3.0          # cap so a giant answer cannot dominate the bank
VALUE_FLOOR = 1e-9           # never let a value hit exactly zero

# -- Storage budget (the eviction pass enforces these on every store) ---------
MAX_ENTRIES = 2000                 # hard cap on cached answers
MAX_BYTES = 25 * 1024 * 1024       # ~25 MB soft cap on total cache size
EVICT_MARGIN = 0.90                # evict down to 90% of budget to avoid churn


def _tokens(text):
    """Lowercase word tokens. Same character class as the brain tokenizer."""
    try:
        return re.findall(r"[a-z0-9]+", str(text).lower())
    except Exception:
        return []


def _normalized(text):
    """Normalize a question to a canonical whitespace-joined token string."""
    return " ".join(_tokens(text))


def _match_set(tokens):
    """The set the fuzzy matcher compares on: content words only (grammatical
    filler and contraction fragments removed), so phrasing ('what is',
    'explain', 'tell me about') never dilutes the score. Questions that reduce
    to no content words simply do not fuzzy-match. Never raises."""
    try:
        return {t for t in tokens if t not in STOPWORDS}
    except Exception:
        return set(tokens) if tokens else set()


def _negations(tokens):
    """Whole-word negations present in a token list. Used to keep opposite
    intents ('is X safe' vs 'is X not safe') from fuzzy-matching. Never raises."""
    try:
        return {t for t in tokens if t in NEGATIONS}
    except Exception:
        return set()


def _wh_intent(tokens):
    """The answer-type group a question asks for ('place', 'time', 'person',
    ...), or "" when it uses no question word. Read from the FULL token list,
    before _match_set strips the question word as filler. Only the FIRST
    question word counts: it governs the sentence, while a later one usually
    sits in a subordinate clause ('where is the school that teaches what?').
    Never raises."""
    try:
        for t in tokens:
            group = _WH_GROUPS.get(t)
            if group:
                return group
        return ""
    except Exception:
        return ""


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _jaccard(set_a, set_b):
    """Token-set Jaccard similarity. 0.0 when either side is empty.

    The union size is derived arithmetically (|A u B| = |A| + |B| - |A n B|)
    instead of materializing set_a | set_b, so each call builds only the
    intersection. This is an exact integer identity, so the returned ratio is
    bit-for-bit identical to the old two-set form; on the fuzzy scan it removes
    one full set construction per candidate entry.
    """
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    if inter == 0:
        return 0.0
    union = len(set_a) + len(set_b) - inter
    return inter / float(union)


def _now():
    return datetime.datetime.now()


def _parse_ts(value):
    """Parse an ISO timestamp string back to datetime, or None."""
    try:
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        return None


# -- answer body compression --------------------------------------------------
def _compress_answer(text):
    """gzip+base64 an answer string into an ASCII-safe blob. '' on failure."""
    try:
        raw = (text or "").encode("utf-8")
        packed = gzip.compress(raw, compresslevel=6)
        return base64.b64encode(packed).decode("ascii")
    except Exception:
        return ""


def _decompress_answer(blob):
    """Inverse of _compress_answer. Returns '' on any failure."""
    try:
        packed = base64.b64decode(blob.encode("ascii"))
        return gzip.decompress(packed).decode("utf-8")
    except Exception:
        return ""


def _entry_answer(entry):
    """Read the answer text from an entry, compressed (v3) or plain (v2)."""
    if not isinstance(entry, dict):
        return ""
    if entry.get("answer_gz"):
        return _decompress_answer(entry["answer_gz"])
    return entry.get("answer", "") or ""


def _principal(answer_text, sources):
    """One-time worth (deposit) of an answer, length-normalized and capped.

    P = 1 + ln(1 + W / W0), plus a bonus when the answer is source-grounded,
    clamped to PRINCIPAL_MAX so a huge answer cannot dominate the bank. A
    120-word grounded answer is worth ~2.1; a one-liner ~1.1; a 5k-word dump is
    capped at PRINCIPAL_MAX.
    """
    try:
        n_words = len(_tokens(answer_text))
        worth = 1.0 + math.log1p(n_words / PRINCIPAL_REF_WORDS)
        if sources:
            worth += PRINCIPAL_SOURCE_BONUS
        return max(VALUE_FLOOR, min(worth, PRINCIPAL_MAX))
    except Exception:
        return 1.0


def _entry_value(entry, now=None):
    """Compound-interest value of one entry:

        V = principal * (1 + r) ** hits * 0.5 ** (age_days / HALF_LIFE_DAYS)

    where age is measured from the last time the entry was used (or created).
    Never raises; returns VALUE_FLOOR on any trouble so eviction still works.
    """
    try:
        now = now or _now()
        principal = float(entry.get("principal") or 1.0)
        hits = int(entry.get("hits") or 0)
        compounded = principal * ((1.0 + INTEREST_RATE) ** hits)

        stamp = _parse_ts(entry.get("last_used") or entry.get("created"))
        if stamp is not None:
            age_days = max(0.0, (now - stamp).total_seconds() / 86400.0)
            decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
        else:
            decay = 1.0
        return max(VALUE_FLOOR, compounded * decay)
    except Exception:
        return VALUE_FLOOR


class AnswerCache:
    """File-backed, compounding, self-pruning cache of answered questions.

    Layout: cache_dir/<slug>-<hash8>.json, one entry per file. Entries load into
    memory on construction and stay in sync on store/lookup/clear. Answer bodies
    are compressed on disk; every store triggers a value-based eviction pass that
    keeps the cache under MAX_ENTRIES / MAX_BYTES.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        # Map of file path -> entry dict.
        self.entries = {}
        # path -> on-disk byte size, kept in sync so the eviction pass never has
        # to stat every file (was an O(n^2) getsize storm on each store).
        self._sizes = {}
        # normalized-question string -> path, for O(1) exact-match lookups
        # instead of re-hashing every entry on every question.
        self._norm_index = {}
        # -- fuzzy-scan acceleration (all in-memory only, NEVER serialized) --
        # A stored entry's content match-set and negation-set are pure functions
        # of its (immutable) tokens, so we compute them ONCE at index time and
        # reuse them on every fuzzy lookup instead of rebuilding two sets per
        # entry per query. path -> frozenset(...).
        self._match_cache = {}   # path -> frozenset of content words
        self._neg_cache = {}     # path -> frozenset of negation words
        self._wh_cache = {}      # path -> answer-type group (see _WH_GROUPS)
        # Inverted index: content word -> set of paths whose entry contains it.
        # A fuzzy query only needs to score entries that share >=1 content word
        # with it (Jaccard >= FUZZY_THRESHOLD forces intersection >= 1), so this
        # turns the fuzzy scan from O(all entries) into O(postings of the query's
        # content words) -- exact narrowing, not approximate.
        self._inverted = {}      # content word -> set(paths)
        # Paths whose in-memory hits/last_used have been bumped but not yet
        # persisted. _register_hit marks dirty instead of rewriting the full
        # entry JSON on every hit; flushes happen on store/clear/atexit.
        self._dirty = set()
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            # Directory could not be created; operate as an empty cache.
            return
        self._load_all()
        # Persist any deferred hit bumps on a clean interpreter exit. Guarded so
        # a registration failure can never break construction.
        try:
            atexit.register(self._flush_dirty)
        except Exception:
            pass

    # -- loading -----------------------------------------------------------
    def _load_all(self):
        """Load every *.json entry in the cache directory into memory."""
        try:
            names = os.listdir(self.cache_dir)
        except Exception:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.cache_dir, name)
            entry = self._read_entry(path)
            if entry is not None:
                self.entries[path] = entry
                # Seed the real on-disk size (getsize, not a recomputed length:
                # _read_entry backfills fields in memory without rewriting).
                try:
                    self._sizes[path] = os.path.getsize(path)
                except Exception:
                    self._sizes[path] = 1
                norm = entry.get("normalized")
                if norm:
                    self._norm_index[norm] = path
                # Precompute this entry's fuzzy sets + inverted-index postings.
                self._index_entry(path, entry)

    def _index_entry(self, path, entry):
        """Cache an entry's fuzzy match-set / negation-set and add its inverted-
        index postings. Pure function of the entry's tokens, so it is safe to
        memoize; idempotent per path. Never raises."""
        try:
            toks = entry.get("tokens") or (entry.get("normalized") or "").split()
            match = frozenset(_match_set(toks))
            self._match_cache[path] = match
            self._neg_cache[path] = frozenset(_negations(toks))
            self._wh_cache[path] = _wh_intent(toks)
            for word in match:
                self._inverted.setdefault(word, set()).add(path)
        except Exception:
            return

    def _deindex_entry(self, path):
        """Drop an entry's fuzzy-set caches and inverted-index postings, pruning
        emptied word buckets. Inverse of _index_entry. Never raises."""
        try:
            for word in self._match_cache.pop(path, ()):  # frozenset is iterable
                bucket = self._inverted.get(word)
                if bucket is not None:
                    bucket.discard(path)
                    if not bucket:
                        del self._inverted[word]
            self._neg_cache.pop(path, None)
            self._wh_cache.pop(path, None)
        except Exception:
            return

    def _flush_dirty(self):
        """Persist entries whose hits/last_used were bumped by a cache hit but
        not yet written. Skips paths no longer present (evicted/cleared) so a
        flush can never resurrect a deleted file. Never raises."""
        try:
            for path in list(self._dirty):
                entry = self.entries.get(path)
                if entry is not None:
                    self._write_entry(path, entry)
            self._dirty.clear()
        except Exception:
            return

    def _read_entry(self, path):
        """Read and lightly validate a single entry file. None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except Exception:
            return None
        if not isinstance(entry, dict):
            return None
        # Backfill derived fields so older or hand-edited entries still work.
        try:
            if not entry.get("normalized"):
                entry["normalized"] = _normalized(entry.get("question", ""))
            if not entry.get("tokens"):
                entry["tokens"] = entry["normalized"].split()
            if "hits" not in entry:
                entry["hits"] = 0
            # v2 entries stored the answer in plaintext and had no principal.
            if "principal" not in entry:
                entry["principal"] = _principal(
                    _entry_answer(entry), entry.get("sources"))
            if not entry.get("last_used"):
                entry["last_used"] = entry.get("created") or _now().isoformat()
        except Exception:
            return None
        return entry

    def _write_entry(self, path, entry):
        """Write an entry atomically (temp file + rename) and record its size.

        Writing to a sibling .tmp then os.replace() means a crash mid-write can
        never truncate or corrupt an existing entry -- readers see either the
        old file or the complete new one. json.dumps with these exact params
        yields the same bytes json.dump would write, so the recorded size equals
        the true on-disk size. Returns True on success, False otherwise."""
        tmp = path + ".tmp"
        try:
            data = json.dumps(entry, ensure_ascii=False, indent=2)
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, path)
            self._sizes[path] = len(data.encode("utf-8"))
            return True
        except Exception:
            # Best-effort cleanup so a failed write leaves no stray .tmp behind.
            try:
                os.remove(tmp)
            except Exception:
                pass
            return False

    # -- lookup ------------------------------------------------------------
    def lookup(self, question):
        """Return a cached entry for question, or None.

        Tries an exact match on the normalized-question sha1 first, then a
        content-word Jaccard fuzzy match (>= FUZZY_THRESHOLD, both sides >=
        FUZZY_MIN_TOKENS content tokens). On a hit the entry compounds: its
        hit counter
        increments, its last_used stamp refreshes, and its file is rewritten.
        The returned entry always carries a decompressed 'answer' field so
        callers read it exactly as before. Never raises.
        """
        try:
            normalized = _normalized(question)
            if not normalized:
                return None

            # Exact match: O(1) via the normalized-question index.
            hit_path = self._norm_index.get(normalized)
            if hit_path is not None:
                entry = self.entries.get(hit_path)
                if entry is not None:
                    return self._register_hit(hit_path, entry)

            # Fuzzy match on content words. Requires enough signal on the query.
            q_toks = normalized.split()
            q_match = _match_set(q_toks)
            if len(q_match) < FUZZY_MIN_TOKENS:
                return None
            q_neg = _negations(q_toks)
            q_wh = _wh_intent(q_toks)
            q_size = len(q_match)

            # Candidate narrowing: only entries that share at least one content
            # word with the query can reach FUZZY_THRESHOLD, because Jaccard >=
            # 0.70 forces a non-empty intersection. So we score just the union of
            # the inverted-index postings for the query's content words rather
            # than every entry. This is EXACT (zero-overlap entries score 0.0 and
            # were always discarded), and drops a full-cache scan to a handful of
            # candidates for a typical short query.
            candidates = set()
            for word in q_match:
                postings = self._inverted.get(word)
                if postings:
                    candidates |= postings

            best_path = None
            best_entry = None
            best_score = 0.0
            for path in candidates:
                entry = self.entries.get(path)
                if entry is None:
                    continue
                try:
                    # Cached content match-set (memoized in _index_entry). Lazy
                    # backfill covers any entry not yet indexed.
                    entry_match = self._match_cache.get(path)
                    if entry_match is None:
                        self._index_entry(path, entry)
                        entry_match = self._match_cache.get(path, frozenset())
                    e_size = len(entry_match)
                    if e_size < FUZZY_MIN_TOKENS:
                        continue
                    # Size-ratio upper bound: the largest Jaccard two content
                    # sets of sizes (lo, hi) can reach is lo/hi (smaller set
                    # fully contained in the larger). Using the SAME float
                    # division and comparison _jaccard uses guarantees we prune
                    # an entry only when its best-possible score would itself be
                    # rejected, so no would-pass hit is ever dropped.
                    lo, hi = (q_size, e_size) if q_size <= e_size else (e_size, q_size)
                    if lo / float(hi) < FUZZY_THRESHOLD:
                        continue
                    # Opposite intents (differing negations) never match.
                    if self._neg_cache.get(path, frozenset()) != q_neg:
                        continue
                    # Different answer-types never match either (see _WH_GROUPS).
                    # An entry with NO question word is a bare topic label
                    # ("Gallaudet University"), which answers no specific
                    # question, so it may only serve a query that also asks
                    # none. A query with no question word still matches
                    # anything, since "Gallaudet University" as a QUERY is an
                    # open request that any stored answer can satisfy.
                    e_wh = self._wh_cache.get(path, "")
                    if q_wh and e_wh != q_wh:
                        continue
                    score = _jaccard(q_match, entry_match)
                    if score > best_score:
                        best_score = score
                        best_path = path
                        best_entry = entry
                except Exception:
                    continue

            if best_entry is not None and best_score >= FUZZY_THRESHOLD:
                return self._register_hit(best_path, best_entry)
            return None
        except Exception:
            return None

    def _register_hit(self, path, entry):
        """Compound a reused entry: bump hits, refresh recency, and return a
        shallow copy carrying a decompressed 'answer'. Never raises.

        The bump is applied to the in-memory entry (which self.entries holds by
        reference, so eviction/stats/value math see it immediately) and the path
        is marked dirty; the full-entry JSON rewrite is DEFERRED off this hot
        read path to the next store()/clear()/atexit flush. The only thing lost
        on a hard kill before a flush is un-persisted hits/last_used bumps, which
        feed only eviction ranking and /cache stats -- never answer content."""
        try:
            entry["hits"] = int(entry.get("hits", 0)) + 1
        except Exception:
            entry["hits"] = 1
        entry["last_used"] = _now().isoformat()
        self._dirty.add(path)
        # Hand the caller a view with plaintext answer, leaving stored form gz.
        view = dict(entry)
        view["answer"] = _entry_answer(entry)
        return view

    # -- store -------------------------------------------------------------
    def store(self, question, answer, sources, model):
        """Persist an answered question, then optimize storage. Returns the file
        path, or None.

        Skips (returns None) when the answer is empty or is a control signal
        (an error, a SEARCH request, or a FETCH request). The answer body is
        compressed on disk and the entry is seeded with a principal deposit;
        afterwards a value-based eviction pass keeps the cache under budget.
        Never raises.
        """
        try:
            answer_text = answer if answer is not None else ""
            stripped = answer_text.strip()
            if not stripped:
                return None
            if (
                stripped.startswith("[error]")
                or stripped.startswith("SEARCH:")
                or stripped.startswith("FETCH:")
            ):
                return None

            normalized = _normalized(question)
            if not normalized:
                return None
            tokens = normalized.split()

            slug = self._make_slug(tokens)
            hash8 = _sha1(normalized)[:8]
            filename = slug + "-" + hash8 + ".json" if slug else hash8 + ".json"
            path = os.path.join(self.cache_dir, filename)

            # Normalize sources to a plain list.
            if sources is None:
                sources_list = []
            elif isinstance(sources, (list, tuple)):
                sources_list = list(sources)
            else:
                sources_list = [sources]

            now_iso = _now().isoformat()
            entry = {
                "question": question,
                "answer_gz": _compress_answer(answer_text),
                "sources": sources_list,
                "model": model,
                "created": now_iso,
                "last_used": now_iso,
                "hits": 0,
                "principal": _principal(answer_text, sources_list),
                "normalized": normalized,
                "tokens": tokens,
            }
            # If compression somehow failed, fall back to a plaintext body so we
            # never silently drop the answer.
            if not entry["answer_gz"]:
                del entry["answer_gz"]
                entry["answer"] = answer_text

            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except Exception:
                pass

            if not self._write_entry(path, entry):
                return None
            self.entries[path] = entry
            self._norm_index[normalized] = path
            # Refresh this path's fuzzy caches + inverted postings (a re-store
            # under the same key may have changed its tokens).
            self._deindex_entry(path)
            self._index_entry(path, entry)

            # A write is already happening here, so flush any deferred hit bumps
            # first -- unconditionally, so it is not gated by _enforce_budget's
            # under-budget early return -- keeping _sizes fresh before ranking.
            self._flush_dirty()

            # Optimize storage space on every store: prune the least valuable
            # knowledge until the cache is back under budget.
            self._enforce_budget()
            return path
        except Exception:
            return None

    def _make_slug(self, tokens):
        """Kebab-case slug from the first few normalized tokens, length capped."""
        try:
            head = tokens[:SLUG_TOKEN_COUNT]
            slug = "-".join(head)
            if len(slug) > SLUG_MAX_LEN:
                slug = slug[:SLUG_MAX_LEN].rstrip("-")
            return slug
        except Exception:
            return ""

    # -- storage optimization ---------------------------------------------
    def _total_bytes(self):
        """Sum of on-disk entry sizes from the in-memory size map (no syscalls)."""
        return sum(self._sizes.values())

    def _value_density(self, path, entry, now):
        """Compound-interest value per on-disk byte. Eviction key: dropping the
        lowest-density entries first is the greedy-optimal way to keep the most
        knowledge value under a byte budget (a fractional-knapsack argument)."""
        size = self._sizes.get(path, 1)
        return _entry_value(entry, now) / float(max(size, 1))

    def _enforce_budget(self):
        """Evict the lowest value-density entries until the cache is under
        budget. We spend the cheapest-per-byte knowledge first so compounding,
        frequently-reused answers survive. Runs after each store. Never raises.
        """
        try:
            over_count = len(self.entries) > MAX_ENTRIES
            over_bytes = self._total_bytes() > MAX_BYTES
            if not (over_count or over_bytes):
                return

            now = _now()
            # Ascending by value density: cheapest knowledge-per-byte first.
            ranked = sorted(
                self.entries.items(),
                key=lambda kv: self._value_density(kv[0], kv[1], now),
            )

            target_entries = int(MAX_ENTRIES * EVICT_MARGIN)
            target_bytes = int(MAX_BYTES * EVICT_MARGIN)

            for path, _entry in ranked:
                if len(self.entries) <= target_entries and \
                        self._total_bytes() <= target_bytes:
                    break
                try:
                    os.remove(path)
                except Exception:
                    pass
                self.entries.pop(path, None)
                self._sizes.pop(path, None)
                # Drop the fuzzy caches/inverted postings and any pending dirty
                # flag so nothing outlives the evicted entry or resurrects it.
                self._deindex_entry(path)
                self._dirty.discard(path)
                # Drop the norm-index entry only if it still points at this path
                # (a re-store under the same normalized key may have moved it).
                norm = _entry.get("normalized")
                if norm is not None and self._norm_index.get(norm) == path:
                    self._norm_index.pop(norm, None)
        except Exception:
            return

    # -- stats / clear -----------------------------------------------------
    def bank_balance(self):
        """Total compound-interest value across the cache: the knowledge bank
        balance. Rises with every interaction (new deposits + reuse compounding).
        """
        try:
            now = _now()
            return sum(_entry_value(e, now) for e in self.entries.values())
        except Exception:
            return 0.0

    def stats(self):
        """Return a summary of the cache and the knowledge bank. Never raises."""
        try:
            total_hits = 0
            for e in self.entries.values():
                try:
                    total_hits += int(e.get("hits") or 0)
                except Exception:
                    continue
            return {
                "count": len(self.entries),
                "dir": self.cache_dir,
                "bytes": self._total_bytes(),
                "hits": total_hits,
                "bank": self.bank_balance(),
            }
        except Exception:
            return {"count": 0, "dir": self.cache_dir, "bytes": 0,
                    "hits": 0, "bank": 0.0}

    def clear(self):
        """Delete all *.json in the cache dir, reset memory. Returns count."""
        removed = 0
        try:
            names = os.listdir(self.cache_dir)
        except Exception:
            names = []
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.cache_dir, name)
            try:
                os.remove(path)
                removed += 1
            except Exception:
                continue
        self.entries = {}
        self._sizes = {}
        self._norm_index = {}
        self._match_cache = {}
        self._neg_cache = {}
        self._wh_cache = {}
        self._inverted = {}
        self._dirty = set()
        return removed
