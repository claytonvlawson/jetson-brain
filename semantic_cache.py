#!/usr/bin/env python3
"""semantic_cache.py - an embedding concept-router that widens cache coverage.

The keyword AnswerCache only fires on lexical overlap (content-word Jaccard above a
high threshold), which long natural-sentence questions almost never reach: a real
question like "my pediatrician says my implanted child won't need to sign, is that
true" shares too few content words with a short seeded phrase to match, even though
it is squarely about one concept. Measured reachability of held-out questions against
the keyword cache was 0 out of 14, before and after adding 252 phrasings.

This layer bridges that gap semantically. It embeds each concept once (a rich
descriptor built from the concept's canonical question plus representative phrasings)
with a local embedding model, embeds the incoming question, and serves the concept's
vetted answer WHEN two conditions hold:

  1. the top concept's cosine similarity is at least `threshold`, and
  2. the top concept beats the runner-up by at least `margin`.

The margin gate is the safety mechanism. On this subject a confidently wrong answer is
worse than none, so an ambiguous question (two concepts close together) is NOT served a
guessed answer: match() returns None and the caller falls through to live RAG. Adjacent
concepts (language deprivation vs Deaf gain) are exactly the low-margin case the gate
rejects. Everything is stdlib plus one local Ollama embeddings call. Never raises.
"""

import os
import json
import math
import hashlib
import urllib.request


def _embed(text, host, model, timeout):
    """One Ollama embeddings call. Returns a vector, or None on any failure."""
    try:
        data = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            host + "/api/embeddings", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            vec = json.loads(resp.read().decode("utf-8")).get("embedding")
        return vec if isinstance(vec, list) and vec else None
    except Exception:
        return None


def _normalize(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class SemanticCache:
    """Concept router over the primed KNOWLEDGE. Build once, then match() queries.

    concepts: list of dicts with id, canonical_question, answer, sources,
    keyword_variants. Concept vectors are cached on disk keyed by a hash of the
    model plus the descriptors, so a rebuild with unchanged content is instant.
    Degrades to a no-op (match always returns None) if embeddings are unavailable,
    so the brain simply falls through to RAG and never breaks.
    """

    def __init__(self, concepts, host="http://127.0.0.1:11434",
                 model="nomic-embed-text", threshold=0.70, margin=0.06,
                 top_variants=8, timeout=30, cache_path=None):
        self.host = host
        self.model = model
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.timeout = timeout
        self.ready = False
        self.ids = []
        self.answers = {}
        self.sources = {}
        self.vecs = {}
        self._build(concepts or [], top_variants, cache_path)

    def _descriptor(self, c, top_variants):
        variants = list(c.get("keyword_variants") or [])[:top_variants]
        parts = [c.get("canonical_question") or ""]
        if variants:
            parts.append("; ".join(variants))
        return ". ".join(p for p in parts if p).strip()

    def _build(self, concepts, top_variants, cache_path):
        descriptors = {}
        for c in concepts:
            cid = c.get("id")
            if not cid or not (c.get("answer") or "").strip():
                continue
            self.ids.append(cid)
            self.answers[cid] = c.get("answer", "")
            self.sources[cid] = list(c.get("sources") or [])
            descriptors[cid] = self._descriptor(c, top_variants)
        if not self.ids:
            return

        key = hashlib.sha1(
            (self.model + "\x00" + json.dumps(descriptors, sort_keys=True))
            .encode("utf-8")).hexdigest()[:16]
        disk = None
        if cache_path:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        disk = json.load(fh)
                    if disk.get("key") != key:
                        disk = None
            except Exception:
                disk = None

        if disk and isinstance(disk.get("vecs"), dict):
            for cid in self.ids:
                v = disk["vecs"].get(cid)
                if v:
                    self.vecs[cid] = v
        else:
            for cid in self.ids:
                vec = _embed(descriptors[cid], self.host, self.model, self.timeout)
                if vec:
                    self.vecs[cid] = _normalize(vec)
            if cache_path and self.vecs:
                try:
                    with open(cache_path, "w", encoding="utf-8") as fh:
                        json.dump({"key": key, "vecs": self.vecs}, fh)
                except Exception:
                    pass

        self.ready = len(self.vecs) >= 2

    def match(self, question):
        """Return {answer, sources, concept, score, margin} if a confident,
        unambiguous concept match exists, else None. Never raises."""
        if not self.ready or not question:
            return None
        try:
            qvec = _embed(question, self.host, self.model, self.timeout)
            if not qvec:
                return None
            qvec = _normalize(qvec)
            scored = sorted(
                ((_dot(qvec, v), cid) for cid, v in self.vecs.items()),
                reverse=True)
            if not scored:
                return None
            top_score, top_id = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if top_score >= self.threshold and (top_score - second) >= self.margin:
                return {
                    "answer": self.answers.get(top_id, ""),
                    "sources": self.sources.get(top_id, []),
                    "concept": top_id,
                    "score": round(top_score, 4),
                    "margin": round(top_score - second, 4),
                }
            return None
        except Exception:
            return None
