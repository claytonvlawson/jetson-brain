#!/usr/bin/env python3
"""brain.py - interactive local RAG agent for the ThesisVault on a Jetson Orin Nano.

Loads a BM25 index built by build_index.py, routes each question through the
local vault first, and optionally falls out to a web search when the model
signals it needs external or current information. Answers stream to the
terminal. Everything runs locally against Ollama except the optional web fetch.

v2 adds three interlocking features on top of the v1 behavior, all additive:
  1. Wikilink graph expansion: notes linked from the retrieved notes get a
     representative chunk pulled in as extra "linked" context.
  2. Reference URLs: external links found in the retrieved notes are offered to
     the model, which may answer with a single 'FETCH: <url>' line to pull one.
  3. Answer cache: exact and fuzzy question caching for instant repeat answers.
None of the v1 behavior (model picker, streaming, SEARCH web fallback, rolling
history, defensive REST fallbacks, existing commands) is removed.
"""

import os
import re
import sys
import time
import atexit
import pickle
import unicodedata

# Make sure the sibling answer_cache module is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from answer_cache import AnswerCache
except Exception:
    AnswerCache = None
try:
    from web_cache import persist_web_fetch
except Exception:
    persist_web_fetch = None
try:
    import web_paths
except Exception:
    web_paths = None
try:
    from brain_ui import UI
except Exception:
    UI = None
try:
    import brain_input
except Exception:
    brain_input = None
try:
    import brain_transcript
except Exception:
    brain_transcript = None

# Paths (fixed install layout on the Jetson). BRAIN_INDEX overrides the pickle
# path so two index variants can be A/B'd side by side without touching the
# live index (eval runs point BRAIN_INDEX at a candidate build).
# BRAIN_ROOT honours the env var, matching update_index.py:52 and web_paths.py.
# It used to be a bare literal here while those two read the environment, so an
# overridden BRAIN_ROOT (tests, a side-by-side eval tree) moved the queue and the
# web paths but left CACHE_DIR pointing at the production cache. refresh_web's
# retire then purged a different answer cache than this file writes to and
# reported success having purged nothing. One source of truth closes that.
BRAIN_ROOT = os.environ.get("BRAIN_ROOT", "/mnt/nvme/brain")
INDEX_PATH = os.environ.get("BRAIN_INDEX") or os.path.join(
    BRAIN_ROOT, "index", "bm25.pkl")
CACHE_DIR = os.path.join(BRAIN_ROOT, "cache")
VAULT_NAME = "ThesisVault"   # Obsidian vault name for clickable source links

# Web-fetch persistence (web_cache.py). Every DuckDuckGo fetch is saved twice:
# a raw JSON record under WEB_CACHE_DIR for fast dedup/reuse (keyed by url +
# content hash), and a human-readable note under the notes dir.
#
# These paths are no longer defined here: web_paths.py is the single source of
# truth shared with web_cache.py, refresh_web.py and web_review.py, because two
# independent hardcoded copies of the same _WebCache path is how retire and
# refresh silently drift onto different directories. The old brain.py names are
# kept as aliases so nothing downstream has to change (web_paths spells the
# vault notes dir WEB_NOTES_DIR; brain.py has always called it WEB_VAULT_DIR).
#
# WEB_VAULT_DIR lives under the build_index.py vault root and is therefore
# indexable; WEB_QUEUE_DIR does NOT, and is where unreviewed fetches now land
# (see _persist_web). WEB_CACHE_DIR is raw-content caching only and is kept in
# its own cache/web subdir, separate from the AnswerCache Q/A cache under
# CACHE_DIR. The literals below are a fallback for the case where web_paths is
# missing; they must stay byte-identical to web_paths' definitions.
if web_paths is not None:
    WEB_CACHE_DIR  = web_paths.WEB_CACHE_DIR
    WEB_VAULT_DIR  = web_paths.WEB_NOTES_DIR
    WEB_QUEUE_DIR  = web_paths.WEB_QUEUE_DIR
    STORE_CHAR_CAP = web_paths.STORE_CHAR_CAP
else:
    WEB_CACHE_DIR  = os.path.join(CACHE_DIR, "web")
    WEB_VAULT_DIR  = os.path.join(BRAIN_ROOT, "vault", "_WebCache")
    WEB_QUEUE_DIR  = os.path.join(BRAIN_ROOT, "queue", "_WebCache")
    STORE_CHAR_CAP = 10000

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://127.0.0.1:11434"

# Two-tier models. FAST is the default: small, resident, quick draft answers.
# DEEP is escalated per-question via "/deep <q>": larger, slower, more grounded.
# DEFAULT_MODEL (above) is the single source of truth for the fast/default tier;
# FAST_MODEL is an alias of it, so swapping the small model means editing
# DEFAULT_MODEL only and both stay in lockstep.
FAST_MODEL = DEFAULT_MODEL
DEEP_MODEL = "qwen2.5:7b"

# Generation options threaded into every Ollama chat call: bounded output +
# context window and a low temperature for grounded answers.
GEN_OPTIONS = {
    "num_predict":    640,   # was 200, then 512 — room for fuller, fact-complete answers
    "num_ctx":       8192,   # was 3072, 4096, then 6144 — see the measurement below
    "temperature":    0.2,   # tighter citation adherence; BRAIN_TEMPERATURE overrides
    "repeat_penalty": 1.1,   # prevent looping on long outputs
    "num_gpu":        999,   # all layers on GPU, no CPU offload
}

# num_ctx was raised 6144 -> 8192 on measurement, not on intuition. Probing all
# 40 retrieval_gold questions through the real path (retrieve_context ->
# expand_with_links -> fit_context) showed the budget was binding hard:
#
#   questions that lost a chunk to fit_context : 17 of 40 (42.5%)
#   chunks dropped                             : 26 (up to 4 on one question)
#   estimated prompt tokens min/mean/max       : 2909 / 4662 / 5371
#   budget at num_ctx=6144                     : 5376
#
# The max estimate landed 5 tokens under the budget, so retrieval was routinely
# handing the model chunks that fit_context then threw away. Production sends
# ~11 chunks, not TOP_K_FAST=8: expand_with_links adds roughly three more after
# retrieval, and the old comment on this line still assumed a 6-chunk window.
#
# At 8192 the budget becomes 7424, which clears the observed 5371 maximum with
# room to spare and takes the drop count to zero. The cost is KV cache: llama3.2
# 3B is 28 layers x 8 KV heads x 128 head_dim, so 2 * 8 * 128 * 28 = 57344
# values per token, and Ollama on the Jetson runs OLLAMA_KV_CACHE_TYPE=q8_0
# (~1 byte/value), i.e. ~56 KB/token -- about 336 MB at 6144 and 448 MB at 8192.
# The extra ~112 MB fits the ~1.0 GB the Orin had free with the 3B resident.
# Do not raise this further without re-checking `free -m` on the device: the
# same arithmetic puts 12288 at ~672 MB, which does not fit.
try:
    GEN_OPTIONS["num_ctx"] = int(os.environ["BRAIN_NUM_CTX"])
except (KeyError, ValueError):
    pass

# Eval reproducibility: BRAIN_TEMPERATURE=0 makes generation deterministic so
# an A/B between two configs measures the change, not sampling noise. Unset
# (production) keeps the 0.2 default above.
try:
    GEN_OPTIONS["temperature"] = float(os.environ["BRAIN_TEMPERATURE"])
except (KeyError, ValueError):
    pass

# BRAIN_SEED pins the sampler. Temperature alone does NOT make a run
# reproducible at the default 0.2, so two arms of an A/B differed by the change
# under test AND by sampling, and a 1-2 question swing could not be told from
# noise. With a seed, an unchanged arm re-run must reproduce exactly -- which is
# also the cheapest way to prove the harness itself is stable. Unset in
# production: pinning the seed there would freeze every retry into the same
# wording, including a bad one.
try:
    GEN_OPTIONS["seed"] = int(os.environ["BRAIN_SEED"])
except (KeyError, ValueError):
    pass

# keep_alive per tier (8GB cannot hold both large models resident): the fast
# model stays pinned (-1) for warm repeats; the deep 7B loads on demand and is
# released a couple minutes after use so it never fights the fast model for VRAM.
KEEP_ALIVE_FAST = -1
KEEP_ALIVE_DEEP = "2m"


def _keep_alive(model):
    """keep_alive for a chat call: the deep model is released after use, every
    other model stays pinned (single source of truth for all four call sites)."""
    return KEEP_ALIVE_DEEP if model == DEEP_MODEL else KEEP_ALIVE_FAST

# Retrieval depth per tier: fast stays lean for speed, deep pulls more context
# for grounding. handle_question does its own tiered retrieval inline.
# TOP_K_FAST 6 -> 8. Measured on all 40 retrieval_gold labels: the binding
# constraint was the WINDOW, not the ranking. 5 of the 9 misses were target
# notes ranked 6, 6, 8, 8 and 9 -- ranked well, just outside the slots.
#
#   keep_k   in_context   assembled prompt (max)   questions over num_ctx 6144
#     6       31/40        ~4,547 tok               0/28
#     8       33/40        ~5,664 tok               0/28
#    10       34/40        ~6,720 tok               1/28
#    12       36/40        ~7,930 tok               8/28
#
# 8 is the largest window that still fits every gold question with headroom.
# The 36/40 at 12 is not real: those prompts exceed num_ctx, and Ollama
# truncates from the TOP, so the "retrieved" chunks that scored the win are
# partly discarded along with the system prompt. Counting them would be
# measuring the harness, not the system. fit_context() now guards the
# remainder, so the failure is a trimmed tail rather than a dropped contract.
TOP_K_FAST = 8   # was 3, then 4, then 6
TOP_K_DEEP = 8   # was 6


def _env_int(name, default, lo=1, hi=64):
    """Env override for a retrieval-depth constant, clamped. These exist so the
    window can be SWEPT by eval_retrieval.py (40s per arm, no judge) instead of
    guessed: measurement showed 5 of the 9 retrieval misses were target notes
    ranked just OUTSIDE the window (ranks 6, 6, 8, 8, 9 against keep_k 6 or 8),
    not ranking failures at all. Clamped because a window wider than the 6144
    context can hold silently truncates the prompt from the top, dropping the
    system prompt -- a much worse failure than a miss."""
    try:
        return max(lo, min(hi, int(os.environ[name])))
    except (KeyError, ValueError):
        return default


def _env_flag(name, default="0"):
    """Env override for a boolean feature flag, using the SAME truthiness table
    the hand-rolled flags in this file already use (BRAIN_HYDE, BRAIN_CTX_FIT,
    BRAIN_STREAM_PHASE1). Written once because every scholar flag below ships
    DEFAULT OFF and has to be flippable per eval arm: a second truthiness
    convention is exactly how an ablation ends up reporting an arm it never
    actually enabled. The empty string counts as OFF, since an env var exported
    with no value means "unset" to every shell that produced it."""
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


TOP_K_FAST = _env_int("BRAIN_TOP_K_FAST", TOP_K_FAST)
TOP_K_DEEP = _env_int("BRAIN_TOP_K_DEEP", TOP_K_DEEP)

# Retrieval section boost: when a query names a top-level vault folder
# (e.g. "CRPD"), add this to the BM25 score of chunks in that folder so
# canonical primary notes (CRPD/Article 2 Definitions) are not buried under
# dense secondary notes that merely repeat the query terms more often.
FOLDER_BOOST = 4.0

# Parent persona: additive boost for the curated parent-guidance notes, so a
# parent-mode question prefers the plain-language pathway notes over dense
# academic sources that share more query words. Applied to BM25 scores before
# fusion/rerank, so both retrieval paths inherit it.
PARENT_BOOST = 3.0
_PARENT_SOURCE_PREFIXES = (
    "LanguageDeprivationAgent/Parent Guidance/",
    "LanguageDeprivationAgent/Parents FAQ",
    "LanguageDeprivationAgent/Local Resources DMV/",
    "LanguageDeprivationAgent/Core/What Is Language Deprivation",
    "Concepts/EHDI Early Hearing Detection and Intervention",
    "Concepts/IFSP and Part C Early Intervention",
    "Organizations/Laurent Clerc National Deaf Education Center",
    "Organizations/Maryland School for the Deaf",
    "Organizations/National Deaf Center",
)

# Parent persona retrieval depth + diversity. Parent rubric questions span
# several notes (a mentors question needs the IFSP note; a schools question
# needs MSD and Clerc), so the fast tier gets deep-tier context, and no single
# source note may fill more than PARENT_MAX_PER_SOURCE of the kept slots (the
# boosted 64KB MSD note would otherwise crowd out every other source).
PARENT_TOP_K = _env_int("BRAIN_PARENT_TOP_K", 8)
# 3, not 2: single-canonical-note questions (national orgs, finding mentors)
# need three chunks of that one note, and the r2 eval showed a cap of 2
# displacing exactly those (national_orgs 100 -> 50). With TOP_K 8 a cap of 3
# still forces at least three distinct source notes into context.
PARENT_MAX_PER_SOURCE = _env_int("BRAIN_PARENT_MAX_PER_SOURCE", 3)

# ---- Scholar persona ------------------------------------------------------
# Third persona alongside default and parent. Its depth comes from RETRIEVAL
# BREADTH and a bounded second pass, never from prompting: the system prompt is
# byte-identical to the default persona's (see _shape_for_intent below), because
# a "prefer specifics, name every source" instruction on a 3B model is a
# citation-fabrication generator aimed at exactly the output class
# faithfulness_flags (two substring regexes) cannot see.
#
# 2.0, not PARENT_BOOST's 3.0: the parent boost has to beat dense academic notes
# that share more query words, while the scholar boost is pushing academic notes
# that ALREADY rank well on an academic question a little further up. Same
# additive shape, applied to BM25 before fusion/rerank, so both retrieval paths
# inherit it.
SCHOLAR_BOOST = 2.0
# The four prefixes that hold the language-deprivation literature. Verified
# against depth_retrieval_gold.jsonl, whose 8 labeled facets resolve entirely
# into "Sources/Language Deprivation/" and "LanguageDeprivationAgent/Core/".
_SCHOLAR_SOURCE_PREFIXES = ("Sources/Language Deprivation/",
                            "LanguageDeprivationAgent/Core/",
                            "Concepts/Language Deprivation",
                            "People/Wyatte C. Hall")

# 10, not 12. A depth question already sends the widest prompts in the system:
# at keep_k=12 half the LD battery lands over the 7,424-token budget at
# num_ctx=8192, and fit_context trims from the TAIL, so the widened window buys
# chunks that are dropped again before the model sees them. 10 is the largest
# window measured to survive the budget on this question set.
SCHOLAR_TOP_K          = _env_int("BRAIN_SCHOLAR_TOP_K", 10)
# 2, not PARENT_MAX_PER_SOURCE's 3: a depth answer is scored on how many
# distinct facets it discusses, so the cap exists to force distinct notes into
# the window. At keep_k=10 a cap of 2 guarantees at least five source notes.
SCHOLAR_MAX_PER_SOURCE = _env_int("BRAIN_SCHOLAR_MAX_PER_SOURCE", 2)
# Facets pursued by the second pass. Each one costs a retrieval round trip
# inside the deadline; the reconciling generation is a single call regardless.
SCHOLAR_FACETS_N       = _env_int("BRAIN_SCHOLAR_FACETS", 2, lo=1, hi=4)
# New chunks admitted per facet. Three is the smallest number that can carry a
# fact plus its surrounding claim; more crowds the reconciling prompt, which
# only ever sees the draft plus these chunks (never the pass-1 union).
SCHOLAR_FACET_CHUNKS   = 3
# Wall clock for the facet retrieval loop. Checked BETWEEN facets only, which is
# why chat_once grew a real timeout: without one the deadline cannot interrupt
# an in-flight generation and is decoration.
SCHOLAR_DEADLINE_S     = 60.0

# Each scholar addition was priced on its own arm against depth_gold (10 rows,
# temperature 0, seed 42) and ships ON only if it paid.
#
# PASS2 ships OFF. Measured: it carried 63% more sources than the default arm
# (12.9 vs 7.9) for no depth gain -- absolute facets covered went 4.70 -> 4.80,
# inside noise -- at +16s/turn (44.2s vs 27.8s), and it is what pushed the 8GB
# board into the OOM killer. Retrieval breadth demonstrably works here (LD
# sources/q 4.00 -> 5.12, max-chunks-from-one-source 2.12 -> 2.00); the 3B just
# spreads thinner over it rather than going deeper. Kept, not deleted: the
# footer rides on this persona and a larger answerer may yet use the breadth.
#
# CITE ships ON. It is the one intervention that moved a number: frontmatter-
# verified DOIs 0.00 -> 2.75 per answer with ungrounded_specifics staying at
# 0.00, i.e. real citations and zero fabrication. It costs no model call.
SCHOLAR_PASS2 = _env_flag("BRAIN_SCHOLAR_PASS2", "0")
SCHOLAR_CITE  = _env_flag("BRAIN_SCHOLAR_CITE",  "1")
# Defined, wired to NOTHING. The auto-detect hook below is unreached on purpose:
# promoting a question to scholar automatically is a change to what the DEFAULT
# persona does, and that only happens after an ablation prices it.
SCHOLAR_AUTO  = _env_flag("BRAIN_SCHOLAR_AUTO",  "0")

# Footer marker, imported rather than restated: the scorer, the judge and any
# ablation all strip the citation block with brain_depth's single constant, and
# a second copy here is how the two silently diverge.
try:
    from brain_depth import CITE_SENTINEL
except Exception:      # brain_depth missing -> the citation footer is unavailable
    CITE_SENTINEL = None

# Vault root, for the frontmatter reads the citation footer needs. Same
# BRAIN_VAULT override review_qa.py and research_agent.py already use.
SCHOLAR_VAULT_DIR = os.environ.get("BRAIN_VAULT", os.path.join(BRAIN_ROOT, "vault"))

# Facet seeds used ONLY when the wikilink graph yields nothing (an old v1 index
# has links_out={}). Not a topic table for the normal path: facets come from
# meta["links_out"], the same map expand_with_links consumes, so they stay
# correct as the vault grows. These eight are the labeled facet notes from
# depth_retrieval_gold.jsonl, so the fallback is measured rather than guessed.
LD_FALLBACK_FACETS = (
    "LanguageDeprivationAgent/Core/What Is Language Deprivation.md",
    "LanguageDeprivationAgent/Core/Critical Period and Brain.md",
    "LanguageDeprivationAgent/Core/Mental Health and Public Health.md",
    "LanguageDeprivationAgent/Core/Bilingual Bimodal Protective Factors.md",
    "LanguageDeprivationAgent/Core/Cochlear Implant Pathways and Risk.md",
    "LanguageDeprivationAgent/Core/CRPD Article 24 and Sign Language Peoples.md",
    "LanguageDeprivationAgent/Core/Research Gaps.md",
    "Sources/Language Deprivation/Hall Levin Anderson 2017 - Language Deprivation Syndrome.md",
)
_LD_FALLBACK_SEED = "__ld_fallback__"

# Query-time semantic rerank of BM25 candidates. After BM25 + section boost, the
# top hit is often a dense secondary note that merely repeats query words while
# the canonical primary note sits a few ranks down. Reranking the top-N BM25
# candidates by nomic-embed-text cosine to the query pulls the on-topic note up.
# Query-time only: no re-index, no dense vectors in the pickle. Reuses the SAME
# Ollama embed path as the semantic router (semantic_cache._embed/_normalize/_dot).
# Self-degrading: any embed failure falls back to BM25 order and never blocks.
RERANK_CAND_N = 20            # rerank this many top BM25 candidates
RERANK_CHUNK_CHARS = 500     # truncate chunk text before embedding
RERANK_EMBED_MODEL = "nomic-embed-text"
RERANK_TIMEOUT = 30
_RERANK_VECS = {}            # chunk key -> normalized embedding (in-process memo)
# BRAIN_RERANK=0 turns the embedding rerank off outright. This exists for one
# reason: BRAIN_HYBRID=0 is NOT a lexical baseline. It falls back to BM25 + this
# rerank, which re-embeds RERANK_CAND_N candidates per query with a dense model,
# so an arm labelled "hybrid off" still has an embedder inside it (and runs ~30x
# slower for it). Without a second gate there is no way to measure what BM25
# alone actually retrieves, which makes "dense vs BM25" unanswerable. Default
# stays on: this is a measurement gate, not a feature.
_RERANK_ENABLED = os.environ.get("BRAIN_RERANK", "1") not in ("0", "false", "False", "no")

# ---- Hybrid dense + BM25 retrieval (Reciprocal Rank Fusion) ----------------
# When the loaded index carries a dense block (build_index.py --embed), each
# query is ALSO scored by GLOBAL embedding cosine over every chunk, and the BM25
# ranking and the dense ranking are fused by RRF. This is a RECALL move: BM25
# alone cannot surface a chunk that shares no query words (a lay-worded question
# vs an academic note), and the query-time rerank only reorders BM25's shortlist,
# so it cannot recover such a chunk either. Dense retrieval runs over the whole
# corpus, so RRF can pull in the right note even with zero lexical overlap.
#
# Self-disabling: with no dense block, or if the query embed fails, or if
# BRAIN_HYBRID=0, handle_question falls back to the exact BM25 + rerank path, so
# an un-embedded index (and the A/B baseline) behaves precisely as before. The
# env gate lets one dense index be measured both ways: BRAIN_HYBRID=0 vs =1.
HYBRID_ENABLED = os.environ.get("BRAIN_HYBRID", "1") not in ("0", "false", "False", "no")
RRF_K = 60          # standard RRF damping constant; larger flattens rank weighting
RRF_POOL = 60       # fuse this many top docs from each ranker
DENSE_TIMEOUT = 30  # query-embed timeout (the doc vectors are prebuilt)

# ---- HyDE: retrieve with a hypothetical ANSWER, not the question -----------
# The gold questions are conversational and share almost no vocabulary with the
# notes that answer them: "a college only for deaf people" shares 4 of 20
# content tokens with the Gallaudet note, "the state school up in Frederick"
# reaches for an org note written in institutional prose. Both BM25 and a dense
# embedding of the QUESTION are searching from question-space; the documents
# live in answer-space. HyDE closes that gap by having the 3B write a short
# hypothetical answer and embedding THAT instead.
#
# It is fused as its own RRF ranker alongside BM25 and the real question's
# dense ranking, so it can only ADD evidence -- a wrong hypothesis costs a
# little rank dilution, never the literal question's ranking. And it never
# touches BM25, which is the crucial difference from expand_query_tokens: that
# merged answer-TYPE terms into the LEXICAL bag, where address-dense DMV notes
# outscored the real target and pushed it out of the window. It was measured
# worse and correctly removed. This is not that.
#
# Default ON: it earned it. Measured on all 40 retrieval labels, index-mac
# 1545 chunks, the only variable being this flag:
#
#            in_context   MRR      r@5  r@20  found
#   off        33/40      0.6437    31    38   39/40
#   on         36/40      0.6471    35    39   40/40
#
# Three questions entered the window (asl_full_language_2, bilingual_bimodal_2,
# gallaudet_2) and NONE left it. The mechanism matches the claim rather than
# just the number moving: gallaudet_1 ("a school in Washington, D.C. where
# every class is taught in sign language" -- never names Gallaudet) went from
# absent from the fused ranking entirely to rank 23, and interpreting_access_2
# from 36 to 12. Those are exactly the vocabulary-gap cases BM25 and a dense
# embedding of the QUESTION both structurally cannot reach.
#
# Three questions slipped -- critical_period_2 1->6, maryland_school_deaf_1
# 1->4, deaf_identity_audism_1 3->4 -- all still inside the window of 8, which
# is the whole of why r@1 dips 22->21 while every other metric rises. A third
# ranker dilutes rank slightly by construction; that is the price of the trade.
#
# Replicated byte-identical across two runs (0 rank differences on 40
# questions), which is the empirical check on the greedy-decoding claim below:
# if the hypothesis were sampled, this A/B could move for reasons unrelated to
# the change under test.
#
# The cost is real and is NOT visible in these numbers: 3.5s per question
# versus 0.07s, ~48x, entirely the extra 3B generation, measured on Metal. The
# 8GB Orin will be worse. Set BRAIN_HYDE=0 to buy that latency back at the cost
# of three questions.
HYDE_ENABLED = os.environ.get("BRAIN_HYDE", "1") not in ("0", "false", "False", "no")
HYDE_NUM_PREDICT = _env_int("BRAIN_HYDE_NUM_PREDICT", 96, lo=16, hi=512)
HYDE_TIMEOUT = 30
HYDE_SYSTEM = (
    "Write a short, plausible encyclopedia-style passage that would ANSWER the "
    "user's question. Two or three sentences. State it as fact, name the "
    "specific people, places, organizations, terms and years such a passage "
    "would contain, and do not hedge, ask questions, or mention that you are "
    "uncertain. This text is used only to search a document collection; it is "
    "never shown to anyone."
)

# ---- ANN (approximate nearest neighbor) dense search + non-indexed post-filter
# The default dense search is an EXACT full mat@q cosine: precise and, at a few
# thousand L2-normalized rows, already ~1 ms. ANN is an OPTIONAL SCALING path for
# when the self-updating KB grows large (tens of thousands of chunks), where an
# exhaustive scan per query stops being free. It builds an IVF (inverted-file)
# coarse quantizer: a small pure-numpy cosine k-means clusters the doc vectors
# into ~sqrt(n) cells, and each query only exact-scores the rows in its NPROBE
# nearest cells. That trades a little recall for a sub-linear scan; at the current
# corpus size ANN returns the same top docs as the exact scan, so it ships
# DEFAULT OFF (BRAIN_ANN=1 to engage) and the proven exact path is byte-for-byte
# unchanged. numpy-only: faiss/hnswlib are not installed and the Jetson's uplink
# is unreliable, so no compiled ANN dependency is introduced.
#
# POST-FILTER (non-indexed): an include/exclude predicate on a chunk's `source`
# path -- an attribute that is NOT part of the vector index. It is applied AFTER
# the ANN candidate set is gathered (the search over-fetches BRAIN_ANN_OVERFETCH x
# the fuse pool so filtering still leaves enough rows to rank), pruning candidates
# with no index rebuild. The same predicate is honored on the exact path too, so a
# filter behaves identically whichever dense search runs. Env-configured by
# default; callers may also pass an explicit predicate to _dense_scores.
ANN_ENABLED      = os.environ.get("BRAIN_ANN", "0") not in ("0", "false", "False", "no")
# nprobe default 20: on the current ~1.3k-row / ~36-cell index that is measured
# LOSSLESS for what reaches the LLM -- the dense top-8 (>= every persona's keep_k)
# matches the exact scan on all eval questions, and top-60 RRF-pool recall is
# ~0.998. Flipping BRAIN_ANN=1 on therefore cannot change answers at this scale.
# As the self-updating KB grows (cells scale with sqrt(n)), lower nprobe to trade
# a little recall for a smaller scanned fraction -- that speed/recall dial is the
# whole point of ANN, and it only starts paying off well above the current size.
ANN_NPROBE       = max(1, int(os.environ.get("BRAIN_ANN_NPROBE", "20") or 20))
ANN_MIN_DOCS     = max(1, int(os.environ.get("BRAIN_ANN_MIN_DOCS", "256") or 256))
ANN_OVERFETCH    = max(1, int(os.environ.get("BRAIN_ANN_OVERFETCH", "3") or 3))
ANN_KMEANS_ITERS = max(1, int(os.environ.get("BRAIN_ANN_KMEANS_ITERS", "10") or 10))
ANN_SEED         = int(os.environ.get("BRAIN_ANN_SEED", "0") or 0)


def _prefix_tuple(env_val):
    """Parse a comma-separated env value into a tuple of stripped prefixes."""
    return tuple(p.strip() for p in (env_val or "").split(",") if p.strip())


DENSE_INCLUDE = _prefix_tuple(os.environ.get("BRAIN_DENSE_INCLUDE"))
DENSE_EXCLUDE = _prefix_tuple(os.environ.get("BRAIN_DENSE_EXCLUDE"))

# ---- Obsidian Smart Connections Pro expansion -------------------------------
# A SEPARATE relationship signal from expand_with_links' explicit [[wikilinks]]:
# Obsidian's Smart Connections Pro plugin computes its own semantic similarity
# between notes (384-dim TaylorAI/bge-micro-v2 embeddings -- a different model
# and vector space than this codebase's own nomic-embed-text dense index, so
# NOT comparable by raw score, only usable as its own ranking -- see
# obsidian_rank.py's module docstring). When enabled, the retrieved notes seed a
# nearest-neighbor lookup (obsidian_rank.expand_from_seeds) against a
# precomputed vector export (export_obsidian_vectors.py), pulling in chunks
# from related-but-not-retrieved notes the SAME way expand_with_links does for
# wikilinks. No live query embedding is performed, so this adds no
# torch/transformers dependency on the Jetson (bge-micro-v2 inference there was
# checked and ruled out as impractical -- see
# [[project-jetson-brain-obsidian-signal]]). Default OFF pending an A/B pass
# against gold/heldout/parent evals, same discipline as WEB_SEMANTIC_ENABLED.
OBSIDIAN_RANK_ENABLED = os.environ.get("BRAIN_OBSIDIAN_RANK", "0") not in (
    "0", "false", "False", "no")
OBSIDIAN_TOP_K_PER_SEED = max(1, int(os.environ.get("BRAIN_OBSIDIAN_TOP_K", "5") or 5))
OBSIDIAN_MAX_EXPAND = max(1, int(os.environ.get("BRAIN_OBSIDIAN_MAX_EXPAND", "3") or 3))

# Paragraph-level refinement of the SAME signal. The plugin also embeds every
# block (paragraph run under a heading): 5,024 of them in the live vault against
# 306 note vectors. With this off, expand_with_obsidian_signal takes an expanded
# note's FIRST chunk ("cid = ids[0]") because it has no per-chunk score to pick
# by; with it on, the note's blocks are ranked against the top retrieved note's
# vector and the winning block is mapped back onto a chunk by heading breadcrumb
# (obsidian_rank.best_chunk_for_note). MEASURED on the real index: the block
# choice differs from ids[0] on 171 of 273 multi-chunk notes, so this is a real
# change in what reaches the context window and stays OFF until an A/B pass
# says otherwise -- same discipline as OBSIDIAN_RANK_ENABLED itself. It also
# costs deploy weight: the vector export grows from 1.3MB to 23MB, which is why
# export_obsidian_vectors.py gates the blocks separately (--blocks).
OBSIDIAN_BLOCKS_ENABLED = os.environ.get("BRAIN_OBSIDIAN_BLOCKS", "0") not in (
    "0", "false", "False", "no")
OBSIDIAN_BLOCK_CANDIDATES = max(1, int(
    os.environ.get("BRAIN_OBSIDIAN_BLOCK_CANDIDATES", "5") or 5))

# Skill loop (brain_skills.py): a bounded single-line-protocol tool loop that
# runs ONLY when the vault-first pass emitted INSUFFICIENT, before the phase-2
# web routing. Default OFF, and the flag is read here as well as inside
# brain_skills so that when it is off brain.py never even imports the module -
# off must cost nothing and must behave exactly like the pre-skills pipeline.
# All the loop's other knobs (BRAIN_SKILLS_MAX_STEPS, BRAIN_SKILLS_DEADLINE,
# BRAIN_SKILLS_NOTE_CHARS, BRAIN_SKILLS_KEEP_K, BRAIN_SKILLS_VERBOSE) live in
# brain_skills.py, which is their single source of truth.
SKILLS_ENABLED = os.environ.get("BRAIN_SKILLS", "0") not in (
    "0", "false", "False", "no")

# Self-updating knowledge base, now review-gated for web fetches.
#
# What this flag still does: when a note is persisted into the VAULT (a
# reviewer promoting a queued page, or any other write under vault/_WebCache),
# spawn update_index.py so the new pages are folded into bm25.pkl within
# seconds, and hot-reload the index when its mtime changes (see
# maybe_reload_index).
#
# What it no longer does: web fetches do NOT auto-learn any more. They are
# written to WEB_QUEUE_DIR, which is outside the vault root and therefore
# outside the index, and they stay there until a human promotes them with
# web_review.py. Unreviewed scrape was competing with peer-reviewed sources for
# context slots, which is the whole reason for the queue. The current question
# is still answered from the live excerpts exactly as before; only permanence
# waits for approval. BRAIN_AUTOLEARN=0 disables the index-update trigger
# entirely.
AUTOLEARN_ENABLED = os.environ.get("BRAIN_AUTOLEARN", "1") not in (
    "0", "false", "False", "no")

# Per-turn stage timing. BRAIN_TIMING=1 emits one machine-parseable line per
# question to stderr (stage seconds + Ollama token counters) so latency work is
# measured, not guessed. LAST_TIMING always holds the latest turn's numbers so
# eval_brain.py can record them per question without parsing stderr.
TIMING_ENABLED = os.environ.get("BRAIN_TIMING", "0") not in (
    "0", "false", "False", "no")
LAST_TIMING = {}

# Ollama call counters from the most recent chat_once/chat_stream response
# (prompt_eval_count = prefill tokens, eval_count = generated tokens,
# total_duration = ns inside Ollama). Best effort: empty when the client or
# response shape does not expose them.
LAST_CHAT_STATS = {}


def _capture_chat_stats(resp):
    """Record prefill/generation counters from an Ollama response object or
    final stream part into LAST_CHAT_STATS. Never raises."""
    global LAST_CHAT_STATS
    stats = {}
    try:
        for key in ("prompt_eval_count", "eval_count", "total_duration"):
            val = resp.get(key) if isinstance(resp, dict) else getattr(resp, key, None)
            if val is not None:
                stats[key] = int(val)
    except Exception:
        stats = {}
    if stats:
        LAST_CHAT_STATS = stats


def _reset_chat_stats():
    """Clear the per-call counters. Called before the phase-1 generation so an
    aborted or failed call records NO counters instead of silently reporting
    the previous call's (a refusal-aborted holdback stream never consumes the
    final Ollama part that carries them)."""
    global LAST_CHAT_STATS
    LAST_CHAT_STATS = {}

MAX_WEB_RESULTS = 5
MAX_WEB_FETCH = 3
WEB_FETCH_TIMEOUT = 8
# Prompt-facing cap: the most fetched page text any single excerpt contributes
# to a generation prompt. Deliberately unchanged at 2000. The persistence cap
# is STORE_CHAR_CAP (web_paths), which is larger, so a stored note keeps depth
# a prompt can never afford. Anything that builds a prompt slices to this.
PAGE_CHAR_CAP = 2000

# ---- Semantic web-fallback selection ---------------------------------------
# gather_web_context's DEFAULT behavior just takes DuckDuckGo's first
# MAX_WEB_FETCH results in search-engine order -- no relevance scoring at all.
# When enabled, this fetches all MAX_WEB_RESULTS candidates instead of stopping
# early, embeds the query once and each excerpt via the SAME nomic-embed-text
# pipeline already used for vault rerank (semantic_cache._embed/_normalize/_dot,
# see _rerank_by_embedding above), and keeps the MAX_WEB_FETCH excerpts closest
# to the query by cosine similarity instead of DDG's return order. Self-
# degrading: any embed failure (Ollama unreachable, timeout) falls back to the
# original first-N-in-order behavior rather than blocking or dropping results.
# Default OFF pending an A/B pass against heldout_gold.jsonl questions that hit
# the web fallback, matching this project's existing opt-in-until-measured
# discipline for retrieval changes (see BRAIN_ANN, cross-encoder rerank notes).
WEB_SEMANTIC_ENABLED = os.environ.get("BRAIN_WEB_SEMANTIC", "0") not in (
    "0", "false", "False", "no")
HISTORY_TURNS = 6  # keep the last N user+assistant exchanges

# Phase-1 grounding calls carry only the last 2 exchanges (4 messages). The
# full HISTORY_TURNS history in a 6-answer session can approach num_ctx=6144
# in the prefill, where Ollama silently truncates from the TOP -- dropping the
# system prompt and vault snippets, the exact content the grounding contract
# depends on. Follow-up resolution ("what about her other paper?") only needs
# the immediately preceding exchanges; the REPL history and phase-2 routing
# keep the full window.
PHASE1_HISTORY_MSGS = 4

# Stream the phase-1 vault answer with sentinel holdback (chat_stream_holdback)
# instead of blocking on the full generation. Output is byte-identical to the
# blocking path; only time-to-first-token changes. BRAIN_STREAM_PHASE1=0
# restores the blocking behavior.
STREAM_PHASE1 = os.environ.get("BRAIN_STREAM_PHASE1", "1") not in (
    "0", "false", "False", "no")

MAX_LINK_EXPAND = 3  # was 2 — one extra linked note now that context is 4096
MAX_DOC_URLS = 6     # cap on reference URLs offered to the model

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Terminal input hygiene.
#
# In a plain line REPL (input() over readline) the terminal is on the MAIN
# screen buffer, so many terminals (macOS Terminal.app, iTerm2, and anything
# that turned on "alternate scroll" or an xterm mouse-tracking mode) translate
# a mouse wheel tick or a two-finger trackpad scroll into arrow-key escape
# sequences (ESC [ A / ESC [ B). readline then treats those as history
# navigation, so scrolling silently walks through past commands -- the bug
# where "scrolling registered keystrokes when it shouldn't."
#
# The fix is to tell the terminal to stop reporting the mouse to us and to stop
# converting scroll into arrow keys, so a scroll moves the terminal's own
# scrollback buffer again and never reaches the program. We reset every mouse
# mode defensively; unsupported ones are ignored by the terminal.
# ---------------------------------------------------------------------------
# ESC [ ? 1000 l  disable X10/normal mouse click reporting
# ESC [ ? 1002 l  disable button-event (drag) tracking
# ESC [ ? 1003 l  disable any-event (motion) tracking
# ESC [ ? 1006 l  disable SGR extended mouse mode
# ESC [ ? 1015 l  disable urxvt extended mouse mode
# ESC [ ? 1007 l  disable ALTERNATE SCROLL -- the one that turns wheel into arrows
_DISABLE_MOUSE = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?1007l"

# The implementation moved to brain_input, which also owns the line editor and
# the cursor restore, so review_qa.py can share it without importing brain.py.
# These names stay bound here because callers already import them from brain.
if brain_input is not None:
    _DISABLE_MOUSE = brain_input._DISABLE_MOUSE
    quiet_scroll = brain_input.quiet_scroll
    restore_cursor = brain_input.restore_cursor
else:  # brain_input missing: keep the old one-shot behaviour rather than die.
    def quiet_scroll():
        try:
            if not sys.stdout.isatty():
                return
            sys.stdout.write(_DISABLE_MOUSE)
            sys.stdout.flush()
            atexit.register(_restore_scroll)
        except Exception:
            pass

    def restore_cursor():
        try:
            if sys.stdout.isatty():
                sys.stdout.write("\x1b[?25h")
                sys.stdout.flush()
        except Exception:
            pass


def _restore_scroll():
    try:
        sys.stdout.write(_DISABLE_MOUSE)
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tokenizer. MUST be identical to the one in build_index.py.
# ---------------------------------------------------------------------------
def _s_stem(w):
    # Conservative S-stemmer: fold only regular English plurals so a query and
    # the corpus agree on "implant"/"implants", "study"/"studies". MUST be
    # byte-identical to build_index.py._s_stem(). Guards keep it from mangling
    # acronyms (len<4), numbers (not isalpha), and -us/-ss/-is words. Irregular
    # plurals (child/children) are intentionally NOT folded.
    if len(w) < 4 or not w.isalpha():
        return w
    if w.endswith("ies") and not w.endswith(("eies", "aies")):
        return w[:-3] + "y"
    if w.endswith("es") and not w.endswith(("aes", "ees", "oes")):
        return w[:-1]
    if w.endswith("s") and not w.endswith(("us", "ss", "is")):
        return w[:-1]
    return w


def tokenize(s):
    # Shared tokenizer. MUST be byte-identical to build_index.py.tokenize();
    # any change requires rebuilding the index (bump EXPECTED_TOKENIZER_VERSION
    # and TOKENIZER_VERSION together). Order is load-bearing: NFKD accent-fold
    # (also lowercases via .lower()), collapse dotted acronyms
    # (A.S.L. -> asl, U.S. -> us), split, then S-stem each token.
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), s)
    return [_s_stem(t) for t in re.findall(r"[a-z0-9]+", s)]


# The tokenizer_version a loaded index pickle must carry. Mismatch means the
# pickle was built by a different tokenizer than this file queries with, which
# silently breaks retrieval -- load_index refuses rather than degrade quietly.
EXPECTED_TOKENIZER_VERSION = "tok-nfkd-acr-sstem"


# Grammatical filler dropped from the QUERY before BM25 scoring (query-side
# only -- the corpus tokenization and pickle are untouched). rank_bm25 floors
# the negative IDF of near-ubiquitous words to a small positive epsilon, so
# leaving "what/is/the" in a query adds a spurious bump to nearly every chunk
# and flattens the top-vs-rest gap the insufficiency gate depends on.
# Stemmed with _s_stem so the set matches the (now stemmed) tokens query_tokens
# filters -- e.g. tokenize("does") -> "doe", so "doe" must be the stored form.
QUERY_STOPWORDS = frozenset(
    _s_stem(w) for w in (
        "a an the of to in on for and or is are was were be been what which who "
        "whom when where why how do does did can could would should will this that "
        "these those with as at by from about it its into".split()
    )
)


def query_tokens(question):
    """Tokenize a question and drop filler, but never return an empty query
    (fall back to the full token list so an all-stopword question still scores)."""
    toks = tokenize(question)
    kept = [t for t in toks if t not in QUERY_STOPWORDS]
    return kept or toks


# ---------------------------------------------------------------------------
# Question intent: answer-type expansion and multi-clause decomposition.
#
# Wh-words are correctly dropped from BM25 scoring (their IDF is ~0, so keeping
# them bumps nearly every chunk and flattens the top-vs-rest gap). But dropping
# them also throws away the ONE thing that distinguishes "Where is Gallaudet
# University?" from "What is Gallaudet University?" -- both reduce to
# {gallaudet, university} and retrieve the same chunks. The fix is not to score
# the wh-word but to translate it into content terms that DO carry IDF, and to
# tell the generator what shape of answer the question wants.
# ---------------------------------------------------------------------------

# Answer-type terms appended to the query bag. Stemmed at module load so they
# match corpus tokens (_s_stem folds "outcomes" -> "outcome" etc.).
_INTENT_EXPANSIONS = {
    "where":  ("located", "location", "city", "address", "campus", "based"),
    "when":   ("year", "founded", "date", "established", "century"),
    "who":    ("founder", "founded", "director", "president", "author", "named"),
    "how many": ("number", "count", "total", "percent"),
    "how much": ("number", "cost", "percent", "rate"),
}
_INTENT_EXPANSIONS = {k: tuple(_s_stem(w) for w in v)
                      for k, v in _INTENT_EXPANSIONS.items()}

# A question this short with a concrete answer-type wants a fact, not an essay.
FACTUAL_MAX_TOKENS = 12

# Weight on the gated answer-type boost. Small on purpose: this reorders chunks
# that already match the question, it does not compete with matching it.
INTENT_BOOST_WEIGHT = 0.35

MULTI_TOP_K = _env_int("BRAIN_MULTI_TOP_K", 8)   # context slots for a multi-entity question

# Multiplier on a definition card when the question asks for a fact. Modest:
# the card should win among its OWN note's chunks, not outrank other notes.
CARD_BOOST = 0.5

# Max chunks from any one note for a multi-clause question. Tighter than the
# parent cap (3): these questions name two or three entities outright, so 6
# slots have to cover 3 notes.
MULTI_MAX_PER_SOURCE = _env_int("BRAIN_MULTI_MAX_PER_SOURCE", 2)
# The DEFAULT persona has never had a per-source cap: _diversify_order runs only
# for parent, scholar and intent == "multi". 0 keeps it that way, so this is
# inert until an arm sets it. Measured on the device 2026-08-16: at keep_k=8 the
# default persona averages 5.60 distinct sources across 8 slots and takes up to
# 2.58 chunks from a single note, while the four remaining retrieval misses sit
# at ranks 13-20 (recall@8 36, @10 36, @20 39). Widening the window cannot reach
# them -- keep_k=12 scored identically (36/40, MRR 0.6481) and keep_k=14 starts
# dropping chunks -- so freeing repeat slots is the only remaining lever.
DEFAULT_MAX_PER_SOURCE = _env_int("BRAIN_MAX_PER_SOURCE", 0)

# Per-source cap that applies ONLY to machine-fetched web notes, on every
# persona including the default one. It exists because the two caps above are
# both inert by default: DEFAULT_MAX_PER_SOURCE is 0, so nothing bounds a
# single note on a default question. That was safe while fetch_page truncated
# at 2000 chars (a web note was one or two chunks). Now that a note stores up
# to STORE_CHAR_CAP, one page is roughly 7-8 chunks at CHUNK_WORDS=220, which
# is enough to fill an 8192-token window on its own with one scraped page.
#
# Hand-authored vault sources are untouched: this caps _WebCache/ sources and
# nothing else, so a long scholarly note keeps every slot it earns.
WEB_SOURCE_PREFIXES = ("_WebCache/",)
WEB_MAX_PER_SOURCE = _env_int("BRAIN_WEB_MAX_PER_SOURCE", 2)

# Boost the reference notes on an ANALYTIC default-persona question. 0.0 keeps
# the ranker exactly as it is.
#
# Measured on the device 2026-08-16 over retrieval_gold: three of the four
# in_context misses share one cause, and it is not the window. critical_period_1
# (rank 11) wanted Core/Critical Period and Brain and got Parents FAQ, Ages 1 to
# 3 and Choosing a Communication Approach; interpreting_access_2 (rank 11)
# wanted US-Law/IDEA and CRPD/Article 24 and got the same two parent-facing
# notes. The parent/FAQ register outranks the reference register on exactly the
# questions that are asking for the reference register. The fourth, gallaudet_1
# (rank 23), is a known bad row -- retrieval_gold flags its premise as off and
# says the DMV school notes outrank the university note legitimately -- so the
# real ceiling on this set is 37/40, not 40/40.
#
# The prefix table is deliberately the scholar one plus the two legal sources:
# _apply_prefix_boost already exists for precisely this operation, and a second
# copy of that loop would be a second place for the numbers to drift.
#
# SWEPT 2026-08-16 against a frozen bm25.V4.pkl, seed pinned, temperature 0, one
# variable. The 1.0 this was going to ship at is INERT: both design targets hold
# rank 11 and 12 exactly, unmoved, and the set stays at 36/40. A flat +1.0 is
# simply too small to cross the parent-register notes stacked above them.
#
#   boost   in_context  MRR     gained                 lost
#   0.0     36/40       0.6351  --                     --
#   1.0     36/40       0.6380  --                     --      <- no-op
#   1.5     37/40       0.6399  critical_period_1      --
#   2.0     37/40       0.6410  critical_period_1      --      <- shipped
#   3.0     37/40       0.6429  critical_period_1      --
#   5.0     37/40       0.6539  + interpreting_access_2 gallaudet_2
#   8.0     36/40       0.6517  + interpreting_access_2 gallaudet_2, legal_recognition_1
#
# 1.5 to 3.0 is the band that gains without giving anything back: it lifts
# critical_period_1 from rank 11 to 4 and moves nothing else across the window
# edge. Past 5.0 the boost stops being a nudge and starts being an override --
# it drags interpreting_access_2 in but pushes gallaudet_2 (rank 4, a row that
# was already correct) out, and by 8.0 it evicts legal_recognition_1 too, back
# to where it started. 2.0 sits in the middle of the safe band rather than at
# its edge, so ordinary index churn cannot walk it into the lossy region.
DEFAULT_ANALYTIC_BOOST = 2.0
try:
    DEFAULT_ANALYTIC_BOOST = float(os.environ["BRAIN_ANALYTIC_BOOST"])
except (KeyError, ValueError):
    pass
_REFERENCE_SOURCE_PREFIXES = _SCHOLAR_SOURCE_PREFIXES + ("US-Law/", "CRPD/")

_WH_RE = re.compile(r"\b(how many|how much|where|when|who|whose|which|what|why|how)\b")

# Clause separators for decomposition. The wh-word is kept with the clause it
# introduces (lookahead, so only the "and" is consumed) -- splitting it off
# would strip each sub-question of the very word that says what it asks for.
_CLAUSE_SPLIT_RE = re.compile(
    r"\?|;|,\s+and\b|\s+and\s+(?=(?:in\s+|for\s+|by\s+|at\s+)?"
    r"(?:what|who|whom|whose|when|where|which|how|why)\b)|\s+and then\b")

# DEFAULT OFF as of the 3B optimization pass. Decomposition was on by
# construction, never measured against its own absence, and it COSTS
# retrieval. Measured on all 40 retrieval_gold labels, twice, on two
# independently built indexes:
#
#   index      arm          in_context   MRR      recall@20
#   1540-chunk DECOMPOSE=1  28/40        0.6135   36
#   1540-chunk DECOMPOSE=0  30/40        0.6406   37
#   1545-chunk DECOMPOSE=1  29/40        0.6164   37
#   1545-chunk DECOMPOSE=0  31/40        0.6437   38
#
# Same direction in the BM25-only arms too. Per question on the current index
# it recovers deaf_identity_audism_2 (rank 9 -> 2), deaf_world_visual_2
# (12 -> 3), asl_full_language_2 (34 -> 8), bilingual_bimodal_2 (19 -> 9),
# msd_founded (4 -> 1) and language_deprivation_2 (5 -> 4); the only loss is
# deaf_gain_1 (1 -> 2), which stays in context. The mechanism: each sub-query
# is its own RRF ranker, so a question that decomposes into three clauses
# dilutes the full question's ranking with two narrower ones, and the note
# that answers the WHOLE question ranks below notes that answer a fragment.
# BRAIN_DECOMPOSE=1 restores it.
DECOMPOSE_ENABLED = os.environ.get("BRAIN_DECOMPOSE", "0") != "0"
# Counted AFTER stopword stripping, so this is lower than a raw word count
# suggests: "Who founded Gallaudet University and in what year, and what law
# made it possible?" is 15 words but only 7 content tokens.
DECOMPOSE_MIN_TOKENS = 6   # below this a question is single-intent already
DECOMPOSE_MAX_SUBQ = 3     # cap the extra BM25 passes


def classify_intent(question):
    """Return (intent, wh) for one question.

    intent is 'factual_short' when the question asks for a single concrete fact
    (a place, date, person, or count) in few words, 'multi' when it has several
    clauses to answer, else 'analytic'. wh is the matched question word or "".
    Deterministic and LLM-free; runs on every question. Never raises."""
    try:
        q = re.sub(r"\s+", " ", (question or "").strip().lower())
        m = _WH_RE.search(q)
        wh = m.group(1) if m else ""
        n_content = len(query_tokens(q))
        if len(_split_clauses(q)) > 1 and n_content >= DECOMPOSE_MIN_TOKENS:
            return "multi", wh
        if wh in _INTENT_EXPANSIONS and n_content <= FACTUAL_MAX_TOKENS:
            return "factual_short", wh
        return "analytic", wh
    except Exception:
        return "analytic", ""


def intent_boost_terms(qtok, wh):
    """Answer-type terms for the question word, minus any already in the query.
    Returns () when the question word has no expansion."""
    extra = _INTENT_EXPANSIONS.get(wh)
    if not extra:
        return ()
    have = set(qtok)
    return tuple(t for t in extra if t not in have)


def _apply_card_boost(scores, chunks, intent):
    """Prefer a note's definition card when the question asks for a FACT.

    A card is short by construction, which is what makes it win a short query
    and lose a long one: BM25 sums over query terms, so a 63-word card matching
    four of twelve terms is beaten by a 315-word prose chunk matching nine --
    even though only the card carries the date being asked for. Measured on
    "which was founded first, Gallaudet or the Maryland School for the Deaf?":
    both MSD slots went to prose, and 1868 never reached the context.

    Applied only for fact-seeking intents. An analytic question wants the
    argument, which lives in the prose, so cards must not displace it there."""
    if intent not in ("factual_short", "multi") or not len(scores):
        return scores
    for i, c in enumerate(chunks):
        if scores[i] > 0 and c.get("card"):
            scores[i] *= (1.0 + CARD_BOOST)
    return scores


def _apply_intent_boost(scores, bm25, terms, weight=INTENT_BOOST_WEIGHT):
    """Add a fraction of the answer-type terms' BM25 score to chunks that ALREADY
    match the question, and to no others.

    The gate is the whole design. Merging these terms into the query bag instead
    is actively worse than not expanding at all: for "Where is Gallaudet
    University?" the added {located, city, address, campus} outscore
    {gallaudet, university} on notes that are wall-to-wall addresses, and the
    Gallaudet note falls out of the top 6 entirely -- measured, not theorized.
    Gating on scores[i] > 0 makes the expansion a TIE-BREAKER among chunks that
    match the entity ("of the Gallaudet chunks, prefer the one stating a
    location") and leaves it unable to promote a chunk the query never matched."""
    if not terms or not len(scores):
        return scores
    try:
        boost = bm25.get_scores(list(terms))
    except Exception:
        return scores
    if len(boost) != len(scores):
        return scores
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] += weight * boost[i]
    return scores


def entity_subqueries(qtok, chunks):
    """One sub-query per NOTE the question names outright, for comparison
    questions, else [].

    Clause splitting cannot reach these. "Gallaudet University and the Maryland
    School for the Deaf were both founded in the 1860s -- which came first?" is
    one clause naming two entities, so it stays a single 12-token bag, and a
    short definition card can only ever match the few terms it contains while a
    long prose chunk from either note matches more. Measured: six chunks came
    back holding 1864 twice and 1868 not at all, so "which came first" was
    unanswerable from context no matter how good the model is.

    Splitting BY ENTITY instead gives each named note its own focused query,
    carrying the question's shared predicate terms (founded, first, apart) but
    only ONE entity's terms, which is what lets each entity's card win its own
    sub-ranking. Returns [] unless at least two notes are named, since a single
    named entity is what ordinary retrieval already handles."""
    try:
        qset = set(qtok)
        if len(qset) < 4:
            return []
        matched = []
        seen = set()
        for c in chunks:
            title = c.get("title") or ""
            if title in seen:
                continue
            seen.add(title)
            ttok = _title_tokens(title)
            # The query must name the title in FULL. A partial match ("school"
            # against "Maryland School for the Deaf") is the ordinary topical
            # overlap that normal scoring already handles, and splitting on it
            # would fragment single-entity questions.
            if len(ttok) >= 2 and ttok <= qset:
                matched.append(ttok)
        if len(matched) < 2:
            return []
        matched.sort(key=len, reverse=True)
        matched = matched[:DECOMPOSE_MAX_SUBQ]
        # Terms belonging to no named entity are the shared predicate; every
        # sub-query keeps them so each stays a question, not a bare noun.
        entity_terms = set().union(*matched)
        shared = [t for t in qtok if t not in entity_terms]
        return [sorted(ttok) + shared for ttok in matched]
    except Exception:
        return []


def _split_clauses(question):
    """Split a question into candidate sub-question strings on clause
    boundaries. Returns the pieces that carry at least one content word."""
    parts = _CLAUSE_SPLIT_RE.split(question or "")
    out = []
    for p in parts:
        if not p:
            continue
        p = p.strip(" ,.;:!?-")
        if p and query_tokens(p):
            out.append(p)
    return out


def decompose(question):
    """Token lists for a multi-clause question's sub-questions, else [].

    A multi-clause question scored as ONE bag lets a term from clause B pull
    decoys that share no clause-A term: in "Who founded Gallaudet University and
    in what year, and what law made it possible?", `founded` and `law` are
    scored independently of `gallaudet`, so a note about a center "founded 2020"
    and a note about the NO FAKES Act both outrank the Gallaudet notes. Scoring
    each clause separately and fusing the rankings keeps each term next to the
    entity it belongs to.

    Every sub-query is ANCHORED with the first clause's content terms. Trailing
    clauses are usually subject-less ("in what year" -> just {year}), and an
    unanchored one-term query retrieves by that term alone -- which is the exact
    decoy problem this function exists to solve, reintroduced one level down."""
    if not DECOMPOSE_ENABLED:
        return []
    try:
        if len(query_tokens(question)) < DECOMPOSE_MIN_TOKENS:
            return []
        parts = _split_clauses(question)
        if len(parts) < 2:
            return []
        anchor = query_tokens(parts[0])
        subqs = [anchor]
        for p in parts[1:DECOMPOSE_MAX_SUBQ]:
            toks = query_tokens(p)
            merged = anchor + [t for t in toks if t not in set(anchor)]
            # A clause that adds nothing past the anchor is not a sub-question.
            if len(merged) > len(anchor):
                subqs.append(merged)
        return subqs if len(subqs) > 1 else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Index loading.
# ---------------------------------------------------------------------------
def load_index():
    """Load bm25.pkl and rebuild a BM25Okapi over the tokenized corpus.

    Returns (chunks, bm25, meta) where meta carries the v2 graph maps:
        meta = {
            "by_source": {source: [chunk_id, ...]},
            "link_map":  {normalized_key: source},
            "links_out": {source: [target_source, ...]},
            "urls_out":  {source: [external_url, ...]},
            "by_id":     {chunk_id: chunk_dict},
        }
    Backward compatible: an old (v1) pickle without the new keys is upgraded on
    the fly by backfilling by_source and link_map from the chunks, with
    links_out and urls_out left empty. Exits the process with a clear message
    if the index is missing or unreadable.
    """
    if not os.path.exists(INDEX_PATH):
        print("No index found at " + INDEX_PATH + ".")
        print("Build it first with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    try:
        with open(INDEX_PATH, "rb") as fh:
            data = pickle.load(fh)
    except Exception as exc:
        print("Could not read the index at " + INDEX_PATH + ": " + str(exc))
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    if not isinstance(data, dict):
        print("The index has an unexpected format (not a dict).")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    chunks = data.get("chunks") or []
    tokenized = data.get("tokenized")
    if not tokenized:
        # v3+ indexes omit the derivable 'tokenized' array to roughly halve the
        # pickle. Rebuild it from chunk text with the same tokenizer as build.
        # (build_index skips empty-token chunks, so no doc is empty here.)
        tokenized = [tokenize(c.get("text") or "") for c in chunks]
    if not chunks or len(chunks) != len(tokenized):
        print("The index looks empty or corrupt.")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    # Refuse a pickle built by a different tokenizer than this file queries with:
    # index tokens and query tokens must come from the same tokenize(), or
    # retrieval silently degrades. An old pre-versioning pickle has no field and
    # also fails closed here, prompting a rebuild.
    idx_tok_ver = data.get("tokenizer_version")
    if idx_tok_ver != EXPECTED_TOKENIZER_VERSION:
        print("Index tokenizer mismatch: index=%r, expected=%r." % (
            idx_tok_ver, EXPECTED_TOKENIZER_VERSION))
        print("The index was built by a different tokenizer version.")
        print("Rebuild it with:  python3 " + os.path.join(BRAIN_ROOT, "build_index.py"))
        sys.exit(1)

    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:
        print("The rank_bm25 package is not available: " + str(exc))
        print("Install it with:  python3 -m pip install --user rank_bm25")
        sys.exit(1)

    try:
        bm25 = BM25Okapi(tokenized)
    except Exception as exc:
        print("Could not rebuild the BM25 index: " + str(exc))
        sys.exit(1)

    meta = _build_meta(data, chunks)
    # Carry the build provenance through to callers. It is validated above and
    # then discarded, which left the eval harness printing "tok=None" for a
    # perfectly good index: a measurement tool that cannot name the artifact it
    # measured is how two runs get compared across a silent index change.
    meta["tokenizer_version"] = idx_tok_ver
    meta["index_version"] = data.get("version")
    # Hybrid retrieval is gated on the dense block EXISTING, not on the flag, so
    # an index built without --embed silently downgrades a "hybrid on" system to
    # BM25-only with no signal anywhere. That is how this vault ran for a full
    # rebuild cycle. Fail loud instead: the flag says hybrid, the data says no.
    if HYBRID_ENABLED and not meta.get("dense"):
        print("NOTE: BRAIN_HYBRID is on but this index carries no dense vectors, "
              "so retrieval is BM25-only.")
        print("      Rebuild with --embed to enable hybrid RRF:  python3 %s --embed"
              % os.path.join(BRAIN_ROOT, "build_index.py"))
    return chunks, bm25, meta


def index_mtime():
    """mtime of bm25.pkl, or None when unreadable. One stat call, cheap
    enough to run per question."""
    try:
        return os.path.getmtime(INDEX_PATH)
    except Exception:
        return None


def maybe_reload_index(chunks, bm25, meta, loaded_mtime):
    """Hot-reload the index when update_index.py (auto-learn) has replaced
    the pickle since it was loaded. Returns (chunks, bm25, meta, mtime,
    reloaded). Never raises and never kills the caller: any load failure
    keeps the current in-memory index and records the new mtime so a broken
    pickle is not retried on every question."""
    current = index_mtime()
    if current is None or (loaded_mtime is not None
                           and current <= loaded_mtime):
        return chunks, bm25, meta, loaded_mtime, False
    try:
        new_chunks, new_bm25, new_meta = load_index()
    except (SystemExit, Exception):
        return chunks, bm25, meta, current, False
    return new_chunks, new_bm25, new_meta, current, True


def _normalize_key(key):
    """Lowercase, strip, and collapse internal whitespace to single spaces."""
    try:
        return " ".join(str(key).strip().lower().split())
    except Exception:
        return ""


def _build_meta(data, chunks):
    """Assemble the v2 meta maps from the pickle, backfilling for old indexes."""
    by_source = data.get("by_source")
    link_map = data.get("link_map")
    links_out = data.get("links_out")
    urls_out = data.get("urls_out")

    need_backfill = not isinstance(by_source, dict) or not isinstance(link_map, dict) \
        or not by_source or not link_map

    if need_backfill:
        by_source = {}
        link_map = {}
        for c in chunks:
            src = c.get("source")
            cid = c.get("id")
            if src is None or cid is None:
                continue
            by_source.setdefault(src, []).append(cid)
            # filename stem without .md
            stem = os.path.splitext(os.path.basename(src))[0]
            skey = _normalize_key(stem)
            if skey and skey not in link_map:
                link_map[skey] = src
            # full rel path without .md
            rel = src[:-3] if src.endswith(".md") else src
            rkey = _normalize_key(rel)
            if rkey and rkey not in link_map:
                link_map[rkey] = src

    if not isinstance(links_out, dict):
        links_out = {}
    if not isinstance(urls_out, dict):
        urls_out = {}

    by_id = {}
    for c in chunks:
        cid = c.get("id")
        if cid is not None:
            by_id[cid] = c

    return {
        "by_source": by_source,
        "link_map": link_map,
        "links_out": links_out,
        "urls_out": urls_out,
        "by_id": by_id,
        "dense": _prep_dense(data.get("dense"), chunks),
    }


def _prep_dense(dense, chunks):
    """Validate and prepare the pickle's dense block for query-time scoring.

    Returns None (hybrid stays off) unless the block is well-formed: a model
    name, a positive dim, an ids list, and a flat float32 buffer of exactly
    len(ids)*dim floats. Maps each dense row to its chunk INDEX (position in
    `chunks`) so scoring is symmetric with the BM25 score array. Builds a numpy
    matrix when numpy is importable (one fast matvec per query); otherwise keeps
    the raw buffer for a pure-Python dot-product fallback. Never raises."""
    if not isinstance(dense, dict):
        return None
    try:
        model = dense.get("model")
        dim = int(dense.get("dim") or 0)
        ids = dense.get("ids")
        buf = dense.get("buf")
        if not model or dim <= 0 or not ids or buf is None:
            return None
        n = len(ids)
        if len(buf) != n * dim:
            return None
        idx_of_id = {}
        for i, c in enumerate(chunks):
            cid = c.get("id")
            if cid is not None:
                idx_of_id[cid] = i
        rows_idx = [idx_of_id.get(cid, -1) for cid in ids]
    except Exception:
        return None

    # prefix_version MUST survive into the prepared meta. _dense_scores calls
    # _query_prefix() on THIS dict, so dropping the key here silently queries a
    # prefixed index without its query prefix -- no crash, no warning, just
    # worse retrieval, and eval_retrieval.py reports prefix=None so the arm
    # looks identical to an unprefixed one. That is the exact failure
    # build_index.py's "make the artifact describe itself" comment guards
    # against, defeated one dict literal later.
    out = {"model": model, "dim": dim, "n": n, "rows_idx": rows_idx,
           "prefix_version": dense.get("prefix_version") or "none",
           "matrix": None, "buf": None}
    try:
        import numpy as np
        out["matrix"] = np.array(buf, dtype=np.float32).reshape(n, dim)
    except Exception:
        out["matrix"] = None
        out["buf"] = buf   # array.array('f'); sliced per row in the fallback
    return out


# Folder-name token sets are memoized: the vault has a handful of top-level
# folders but _apply_section_boost runs per chunk per question on the CPU.
_FOLDER_TOKENS = {}

# Note-title token sets, memoized for the same reason (a few hundred titles,
# scored per chunk per question).
_TITLE_TOKENS = {}

# Multiplier applied at a perfect title match. BM25 has no notion of fields, so
# a term in a note's TITLE counts exactly as much as one buried in its body --
# and chunk breadcrumbs inject the title path into every chunk, which makes it
# worse: sixteen "Summer Camps ... > Gallaudet University Hearing and Speech
# Center" resource listings all carry "gallaudet university" as body text and,
# being long, carry it several times over. Measured, they buried the actual
# Gallaudet note. Title matching is the field weight that separates "the note
# ABOUT X" from "a note that MENTIONS X".
TITLE_BOOST = 1.0


def _title_tokens(title):
    """Memoized content-token set for a note title."""
    toks = _TITLE_TOKENS.get(title)
    if toks is None:
        toks = frozenset(t for t in tokenize(title or "")
                         if t not in QUERY_STOPWORDS)
        _TITLE_TOKENS[title] = toks
    return toks


def _apply_title_boost(scores, chunks, qtok):
    """Scale up chunks whose NOTE TITLE the query names.

    The weight is the symmetric overlap between the query's content words and
    the title's, so it peaks when the question essentially IS the title ("Where
    is Gallaudet University?" against "Gallaudet University" = 1.0) and fades
    when either side carries much the other does not. Symmetry matters in both
    directions: a one-word title must not capture every query containing that
    word, and a long question must not fully boost a note it only brushes.

    Multiplicative, and applied only where the chunk already scores, so it
    reorders real matches rather than promoting unmatched ones."""
    if not len(scores):
        return scores
    qset = set(qtok)
    if not qset:
        return scores
    ratios = {}
    for i, c in enumerate(chunks):
        if scores[i] <= 0:
            continue
        title = c.get("title") or ""
        ratio = ratios.get(title)
        if ratio is None:
            ttok = _title_tokens(title)
            inter = len(qset & ttok)
            ratio = 0.0 if not inter else inter / float(len(qset | ttok))
            ratios[title] = ratio
        if ratio > 0:
            scores[i] *= (1.0 + TITLE_BOOST * ratio)
    return scores


def _apply_section_boost(scores, chunks, qtok):
    """Add FOLDER_BOOST to any chunk whose top-level vault folder is named
    in the query tokens, so canonical primary notes (e.g. CRPD/Article 2
    Definitions) are not buried under dense secondary notes. Returns scores
    unchanged when the query names no folder."""
    if not len(scores):
        return scores
    qset = set(qtok)
    boosted = None
    for i, c in enumerate(chunks):
        src = str(c.get("source") or "")
        top = src.split("/", 1)[0] if "/" in src else ""
        if not top:
            continue
        ftok = _FOLDER_TOKENS.get(top)
        if ftok is None:
            ftok = set(tokenize(top))
            _FOLDER_TOKENS[top] = ftok
        if ftok & qset:
            if boosted is None:
                boosted = list(scores)
            boosted[i] += FOLDER_BOOST
    return boosted if boosted is not None else scores


def _apply_prefix_boost(scores, chunks, prefixes, boost):
    """Add `boost` to the score of every chunk whose source starts with one of
    `prefixes`. Returns `scores` UNCHANGED (the same object) when nothing
    matches, which is what keeps a persona that owns no notes free.

    Generalized out of _apply_parent_boost when the scholar persona needed the
    identical operation over a different prefix table. Deliberately a refactor
    rather than a copy: the parent numbers are already measured, and a second
    copy of this loop is a second place for them to drift."""
    if not len(scores):
        return scores
    boosted = None
    for i, c in enumerate(chunks):
        src = str(c.get("source") or "")
        if src.startswith(prefixes):
            if boosted is None:
                boosted = list(scores)
            boosted[i] += boost
    return boosted if boosted is not None else scores


def _apply_parent_boost(scores, chunks):
    """Add PARENT_BOOST to chunks from the curated parent-guidance notes.
    Only called in parent persona; returns scores unchanged when nothing
    matches (e.g. the vault has no parent notes)."""
    return _apply_prefix_boost(scores, chunks, _PARENT_SOURCE_PREFIXES,
                               PARENT_BOOST)


def _diversify_order(order, chunks, keep_k, max_per_source):
    """Return up to keep_k chunk ids from `order`, keeping at most
    max_per_source chunks from any one source note and backfilling from the
    remaining ids in ranked order when the cap leaves slots open. Used by the
    parent and scholar personas and by every multi-intent question: those all
    need facts from several different notes, and one long boosted note can
    otherwise fill every slot."""
    picked, overflow, counts = [], [], {}
    for i in order:
        src = str((chunks[i] or {}).get("source") or "")
        if counts.get(src, 0) < max_per_source:
            counts[src] = counts.get(src, 0) + 1
            picked.append(i)
            if len(picked) >= keep_k:
                return picked
        else:
            overflow.append(i)
    for i in overflow:
        picked.append(i)
        if len(picked) >= keep_k:
            break
    return picked


def _cap_web_sources(order, chunks, max_per_source):
    """Demote the overflow chunks of any one WEB note to the end of `order`,
    keeping at most max_per_source of them in their original ranks. Returns a
    full-length reordering, never a truncation: nothing is dropped, so a
    question whose only usable material is one fat scraped page still gets it,
    just after everything else that ranked.

    Unlike _diversify_order this ignores hand-authored sources entirely and
    runs for every persona, because the default persona has no cap at all
    (DEFAULT_MAX_PER_SOURCE = 0) and is exactly where a 10k-char web note would
    otherwise take the whole window."""
    picked, overflow, counts = [], [], {}
    for i in order:
        src = str((chunks[i] or {}).get("source") or "").replace("\\", "/")
        if not src.startswith(WEB_SOURCE_PREFIXES):
            picked.append(i)
            continue
        if counts.get(src, 0) < max_per_source:
            counts[src] = counts.get(src, 0) + 1
            picked.append(i)
        else:
            overflow.append(i)
    return picked + overflow


def _rerank_by_embedding(question, chunks, cand_ids, qvec=None, qvec_model=None):
    """Reorder BM25 candidate chunk-indices by query<->chunk cosine using
    nomic-embed-text. Returns a reordered list of the same ids; on any failure
    returns cand_ids unchanged (BM25 order). Never raises, never blocks.

    qvec: an already-normalized question embedding (from the semantic router's
    miss) reused when qvec_model matches RERANK_EMBED_MODEL, saving the second
    embed of the same string; otherwise the question is embedded here."""
    global _RERANK_ENABLED
    if not _RERANK_ENABLED or len(cand_ids) < 2:
        return cand_ids
    try:
        from semantic_cache import _embed, _normalize, _dot
    except Exception:
        _RERANK_ENABLED = False
        return cand_ids
    try:
        if qvec and qvec_model == RERANK_EMBED_MODEL:
            qv = qvec
        else:
            raw = _embed(question, OLLAMA_HOST, RERANK_EMBED_MODEL, RERANK_TIMEOUT)
            if not raw:
                # embedder unreachable: disable for the rest of the run so we do
                # not pay a timeout per question, and keep BM25 order.
                _RERANK_ENABLED = False
                return cand_ids
            qv = _normalize(raw)
        scored = []
        any_vec = False
        for i in cand_ids:
            c = chunks[i]
            cid = c.get("id")
            key = cid if cid is not None else ("_idx", i)
            cv = _RERANK_VECS.get(key, False)   # False = not seen; None = un-embeddable
            if cv is False:
                vec = _embed((c.get("text") or "")[:RERANK_CHUNK_CHARS],
                             OLLAMA_HOST, RERANK_EMBED_MODEL, RERANK_TIMEOUT)
                cv = _normalize(vec) if vec else None
                _RERANK_VECS[key] = cv
            if cv is None:
                scored.append((float("-inf"), i))   # sinks below embedded ones
            else:
                any_vec = True
                scored.append((_dot(qv, cv), i))
        if not any_vec:
            return cand_ids
        scored.sort(key=lambda p: p[0], reverse=True)
        return [i for _, i in scored]
    except Exception:
        return cand_ids


def _default_dense_predicate():
    """Build a per-chunk keep() predicate from BRAIN_DENSE_INCLUDE / _EXCLUDE, or
    return None when neither is set (no filtering). keep(chunk) is True to keep the
    chunk. NON-INDEXED: it reads chunk['source'] (a metadata attribute), never the
    vector index, so include/exclude changes take effect with no rebuild."""
    inc, exc = DENSE_INCLUDE, DENSE_EXCLUDE
    if not inc and not exc:
        return None

    def keep(chunk):
        src = str((chunk or {}).get("source") or "")
        if inc and not src.startswith(inc):
            return False
        if exc and src.startswith(exc):
            return False
        return True

    return keep


def _build_ivf(mat, iters=ANN_KMEANS_ITERS, seed=ANN_SEED):
    """Cluster the (n, dim) L2-normalized doc matrix into ~sqrt(n) cells with a
    small cosine k-means (pure numpy). Returns {'centroids': (k, dim) float32
    L2-normalized, 'cells': list of int32 row-index arrays} or None on any failure
    (caller then falls back to the exact scan). Deterministic for a fixed seed so
    eval A/Bs are reproducible. Since the doc rows are L2-normalized, cosine is a
    dot product, so assignment is argmax of mat @ centroids.T."""
    try:
        import numpy as np
    except Exception:
        return None
    try:
        n = int(mat.shape[0])
        if n <= 0:
            return None
        k = min(max(1, int(round(n ** 0.5))), n)
        rng = np.random.default_rng(seed)
        cent = mat[rng.choice(n, size=k, replace=False)].astype(np.float32).copy()
        for _ in range(iters):
            labels = (mat @ cent.T).argmax(axis=1)
            for j in range(k):
                members = mat[labels == j]
                if members.shape[0]:
                    cent[j] = members.mean(axis=0)
            norms = np.linalg.norm(cent, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            cent = (cent / norms).astype(np.float32)
        labels = (mat @ cent.T).argmax(axis=1)
        cells = [np.where(labels == j)[0].astype(np.int32) for j in range(k)]
        return {"centroids": cent, "cells": cells}
    except Exception:
        return None


def _get_ivf(dm, mat):
    """Lazily build and cache the IVF on the dense-block dict. A hot index reload
    hands _prep_dense a fresh dm with no cache, so the IVF rebuilds automatically
    when the vectors change. Caches False on build failure to avoid retrying every
    query; returns the IVF dict or None."""
    ivf = dm.get("_ivf")
    if ivf is None:
        ivf = _build_ivf(mat)
        dm["_ivf"] = ivf or False
    return ivf or None


def _ann_candidates(qa, ivf, nprobe, want):
    """Return row indices from the query's nearest IVF cells: at least `nprobe`
    cells, and keep probing (in centroid-similarity order) until at least `want`
    rows are gathered or the cells are exhausted. The `want` over-fetch gives the
    non-indexed post-filter headroom so it still leaves enough rows to fuse.
    Returns a python list of row indices, or None if the IVF is unusable."""
    try:
        import numpy as np
        order = np.argsort(-(ivf["centroids"] @ qa))
        cells = ivf["cells"]
        cand, probed = [], 0
        for c in order:
            if probed >= nprobe and len(cand) >= want:
                break
            cand.extend(int(r) for r in cells[int(c)])
            probed += 1
        return cand
    except Exception:
        return None


def _post_filter_rows(cand_rows, rows_idx, chunks, predicate):
    """Drop candidate dense-rows whose chunk fails the (non-indexed) predicate.
    rows_idx maps a dense row to its chunk index; rows with no chunk (-1) are
    dropped. Dedupes while preserving first-seen order. predicate=None keeps every
    mapped row."""
    out, seen = [], set()
    for r in cand_rows:
        if r in seen:
            continue
        seen.add(r)
        ci = rows_idx[r] if 0 <= r < len(rows_idx) else -1
        if ci < 0:
            continue
        if predicate is not None and not predicate(chunks[ci]):
            continue
        out.append(r)
    return out


# prefix_version -> query-side prefix. Mirrors build_index.EMBED_PREFIXES, but
# kept as a literal so brain.py never has to import the builder at query time.
# An index whose prefix_version is missing or unrecognized is treated as "none",
# which is what every index built before the option existed actually is.
_QUERY_PREFIXES = {
    "none": "",
    "nomic-v1": "search_query: ",
}


def _query_prefix(dense_meta):
    """Query-side task prefix for this index's dense block. Never raises."""
    try:
        return _QUERY_PREFIXES.get(dense_meta.get("prefix_version") or "none", "")
    except Exception:
        return ""


def _dense_scores(question, chunks, meta, predicate=None, qvec=None,
                  qvec_model=None):
    """Global embedding-cosine score per chunk INDEX for hybrid retrieval.

    Reuses `qvec` (an already-normalized question embedding, e.g. from the
    semantic router's miss) when qvec_model matches the index's dense model,
    saving one Ollama embeddings round-trip per fresh turn; otherwise embeds
    the question here (same Ollama embed path as the router/rerank) with
    the index's own dense model, then scores every chunk that has a dense vector
    by cosine (a dot product, since both sides are L2-normalized). Returns a list
    aligned to `chunks`: the cosine for chunks with a dense vector, -inf for
    those without (so they sink in the dense ranking and are filtered from the
    RRF pool). Returns None if there is no dense block or the query embed fails,
    which makes handle_question fall back to the BM25 + rerank path. Never raises."""
    dm = meta.get("dense")
    if not dm:
        return None
    try:
        from semantic_cache import _embed, _normalize, _dot
    except Exception:
        return None
    try:
        # The index records the task-prefix scheme it was embedded under, so the
        # query side follows the ARTIFACT rather than a local default. Querying
        # a prefixed index without the prefix (or the reverse) does not fail
        # loudly, it just retrieves worse, so the pairing must not be a thing
        # anyone can set independently on the two sides.
        qprefix = _query_prefix(dm)
        if qvec and qvec_model == dm["model"] and not qprefix:
            qv = qvec
        else:
            # A cached qvec from the router was embedded without a prefix, so it
            # is not reusable against a prefixed index: re-embed instead.
            raw = _embed(qprefix + question, OLLAMA_HOST, dm["model"], DENSE_TIMEOUT)
            if not raw:
                return None
            qv = _normalize(raw)
        dim = dm["dim"]
        if len(qv) != dim:
            return None
        n = dm["n"]
        rows_idx = dm["rows_idx"]
        scores = [float("-inf")] * len(chunks)
        pred = predicate if predicate is not None else _default_dense_predicate()
        mat = dm.get("matrix")
        if mat is not None:
            import numpy as np
            qa = np.array(qv, dtype=np.float32)
            rows = None
            # ANN path: restrict scoring to the post-filtered candidates in the
            # query's nearest IVF cells. Only above ANN_MIN_DOCS, where a coarse
            # quantizer beats an exhaustive scan; below it the exact scan wins.
            if ANN_ENABLED and n >= ANN_MIN_DOCS:
                ivf = _get_ivf(dm, mat)
                if ivf is not None:
                    cand = _ann_candidates(qa, ivf, ANN_NPROBE,
                                           RRF_POOL * ANN_OVERFETCH)
                    if cand is not None:
                        rows = _post_filter_rows(cand, rows_idx, chunks, pred)
            if rows is not None:
                # ANN + post-filter: exact-cosine ONLY the surviving candidate rows
                # (ranking among them stays exact). Rows outside the candidate set
                # keep -inf and sink out of the dense pool. An empty survivor set
                # means dense contributes nothing this query and BM25 carries it.
                if rows:
                    cr = np.asarray(sorted(set(rows)), dtype=np.int64)
                    sims = mat[cr] @ qa
                    for j in range(cr.shape[0]):
                        ci = rows_idx[int(cr[j])]
                        if ci >= 0:
                            scores[ci] = float(sims[j])
            else:
                # Exact full scan (default). Honor the non-indexed post-filter here
                # too, so a filter behaves the same with ANN off as with it on.
                sims = mat @ qa                           # (n,) cosine per row
                for r in range(n):
                    ci = rows_idx[r]
                    if ci < 0:
                        continue
                    if pred is not None and not pred(chunks[ci]):
                        continue
                    scores[ci] = float(sims[r])
        else:
            # Pure-Python fallback (numpy unavailable): exact scan + post-filter.
            # ANN needs numpy, so this path stays exhaustive.
            buf = dm.get("buf")
            if buf is None:
                return None
            for r in range(n):
                ci = rows_idx[r]
                if ci < 0:
                    continue
                if pred is not None and not pred(chunks[ci]):
                    continue
                base = r * dim
                scores[ci] = _dot(qv, buf[base:base + dim])
        return scores
    except Exception:
        return None


def hyde_hypothesis(question, model=None, tm=None):
    """Ask the fast model for a short hypothetical answer to `question`.

    Returns the text, or None on any failure. Deliberately total: HyDE is an
    optional extra ranker, so a dead or slow generation must degrade retrieval
    to exactly the non-HyDE path rather than take the turn down with it."""
    try:
        opts = dict(GEN_OPTIONS)
        opts["num_predict"] = HYDE_NUM_PREDICT
        # Greedy. A hypothesis is a retrieval probe, not prose for a human, and
        # sampling it would make the whole retrieval ranking nondeterministic
        # even with BRAIN_SEED set on the answer call.
        opts["temperature"] = 0.0
        text = chat_once(
            model or FAST_MODEL,
            [{"role": "system", "content": HYDE_SYSTEM},
             {"role": "user", "content": question}],
            options=opts,
        )
        text = (text or "").strip()
        if not text:
            return None
        # A refusal or a clarifying question is worse than nothing: it would
        # embed near notes about not knowing things. Cheap shape check.
        if _is_insufficient(text):
            return None
        return text
    except Exception:
        return None


def _hyde_scores(question, chunks, meta, tm=None, model=None):
    """Dense scores for a HYPOTHETICAL ANSWER to the question, or None.

    Reuses _dense_scores so the hypothesis goes through the identical embed
    path, index prefix handling, ANN branch and source predicate as the real
    query. Adding a second scoring implementation for this is how the two
    quietly diverge."""
    _t = time.perf_counter()
    try:
        hyp = hyde_hypothesis(question, model=model, tm=tm)
        if not hyp:
            return None
        # Anchor the hypothesis with the question. An unanchored hypothesis
        # that guesses the wrong entity retrieves confidently for that wrong
        # entity; keeping the question in the embedded text bounds how far the
        # probe can wander from what was actually asked.
        return _dense_scores(question + "\n" + hyp, chunks, meta)
    except Exception:
        return None
    finally:
        if tm is not None:
            tm["hyde"] = time.perf_counter() - _t


def _rrf_fuse(rankings, k=RRF_K, pool=RRF_POOL):
    """Reciprocal Rank Fusion of several ranked chunk-index lists.

    Each ranker contributes 1/(k + rank) to the docs in its top `pool` (rank is
    0-based, so the top doc contributes 1/(k+1)). A doc's fused score is the sum
    across the rankers it appears in, so agreement between BM25 and dense is
    rewarded while either alone can still lift a doc the other missed. Returns
    the fused chunk indices sorted by descending score."""
    agg = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked[:pool]):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(agg.keys(), key=lambda i: agg[i], reverse=True)


# ---------------------------------------------------------------------------
# Ollama helpers. Prefer the python client, fall back to REST via requests.
# ---------------------------------------------------------------------------
def list_models():
    """Return a list of installed model names. Best effort, never raises."""
    names = []
    try:
        import ollama
        data = ollama.list()
        models = getattr(data, "models", None)
        if models is None and isinstance(data, dict):
            models = data.get("models", [])
        for m in models or []:
            name = getattr(m, "model", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        if names:
            return names
    except Exception:
        pass

    # REST fallback.
    try:
        import requests
        resp = requests.get(OLLAMA_HOST + "/api/tags", timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        for m in payload.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
    except Exception:
        pass
    return names


def chat_once(model, messages, options=None, force_rest=False, timeout=None):
    """Non-streaming chat. Returns the full text, or an error string prefixed
    with '[error]' so callers can detect failure without a crash.

    `options` overrides GEN_OPTIONS for one call and defaults to None, so every
    pre-existing call site keeps the exact same request body. It exists for the
    skill loop's selection call, which wants temperature 0 and a tiny
    num_predict (brain_skills.SELECT_GEN_OPTIONS) while the answer calls keep
    the 0.2 default. (Rejected: swapping the GEN_OPTIONS global around the call
    - the UI animator runs on its own thread, so a mutated global is a race.)

    `force_rest` and `timeout` both default to inert. They exist for the scholar
    second pass, which runs under a wall-clock deadline: the ollama client path
    below takes NO timeout at all, so a stalled generation there is unbounded and
    SCHOLAR_DEADLINE_S -- checked only between facets -- cannot interrupt it.
    force_rest=True skips straight to the requests path, where a timeout is a
    real one; timeout overrides that path's 120s default. Every pre-existing
    call site passes neither and gets the exact same two-path behaviour.
    """
    _opts = options if options is not None else GEN_OPTIONS
    if not force_rest:
        try:
            import ollama
            _ka = _keep_alive(model)
            resp = ollama.chat(model=model, messages=messages, stream=False, options=_opts, keep_alive=_ka)
            _capture_chat_stats(resp)
            msg = resp.get("message") if isinstance(resp, dict) else getattr(resp, "message", None)
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            content = getattr(msg, "content", None)
            if content is not None:
                return content
        except Exception:
            pass

    # REST fallback.
    try:
        import requests
        resp = requests.post(
            OLLAMA_HOST + "/api/chat",
            json={"model": model, "messages": messages, "stream": False, "options": _opts, "keep_alive": _keep_alive(model)},
            timeout=(120 if timeout is None else timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        _capture_chat_stats(data)
        return (data.get("message") or {}).get("content", "") or ""
    except Exception as exc:
        return "[error] could not reach the model: " + str(exc)


def chat_stream(model, messages, sink=None):
    """Stream a chat reply and return the accumulated text.

    Each token is passed to `sink` when given (the UI answer emitter); otherwise
    it is written straight to stdout. On failure prints a short note and returns
    what it has (possibly empty).
    """
    collected = []

    def _emit(piece):
        if sink is not None:
            sink(piece)
        else:
            sys.stdout.write(piece)
            sys.stdout.flush()

    try:
        import ollama
        _ka = _keep_alive(model)
        for part in ollama.chat(model=model, messages=messages, stream=True, options=GEN_OPTIONS, keep_alive=_ka):
            piece = ""
            _capture_chat_stats(part)   # the final part carries the counters
            msg = part.get("message") if isinstance(part, dict) else getattr(part, "message", None)
            if isinstance(msg, dict):
                piece = msg.get("content", "") or ""
            elif msg is not None:
                piece = getattr(msg, "content", "") or ""
            if piece:
                _emit(piece)
                collected.append(piece)
        if sink is None:
            print()
        return "".join(collected)
    except Exception:
        # A mid-stream failure already emitted tokens; retrying over REST would
        # regenerate from scratch and duplicate them in the sink, the history,
        # and the cache. Keep the partial answer instead.
        if collected:
            _emit("\n[stream interrupted; the answer may be incomplete]")
            return "".join(collected)

    # REST streaming fallback (only reached when nothing was emitted).
    try:
        import json
        import requests
        with requests.post(
            OLLAMA_HOST + "/api/chat",
            json={"model": model, "messages": messages, "stream": True, "options": GEN_OPTIONS, "keep_alive": _keep_alive(model)},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                piece = (obj.get("message") or {}).get("content", "") or ""
                if piece:
                    _emit(piece)
                    collected.append(piece)
        if sink is None:
            print()
        return "".join(collected)
    except Exception as exc:
        msg = "\n[the model could not stream a reply: " + str(exc) + "]"
        if sink is not None:
            sink(msg)
        else:
            print(msg)
        return "".join(collected)


# NORMALIZED characters buffered before the phase-1 INSUFFICIENT decision is
# made on a streaming reply. _is_insufficient collapses runs of whitespace and
# then inspects only the opening ~200 chars of that normalized text, so the
# buffer is measured the same way: once 210 normalized chars exist, the head
# verdict provably equals the full-text verdict (raw-char counting undercounts
# whitespace-heavy heads and could let a late-completing refusal stream).
STREAM_HOLDBACK_CHARS = 210


def _normalized_len(text):
    # Length of text as _is_insufficient sees it (whitespace runs collapsed).
    return len(re.sub(r"\s+", " ", text.strip()))


def chat_stream_holdback(model, messages, decide, begin_stream,
                         holdback_chars=STREAM_HOLDBACK_CHARS):
    """Stream a chat reply, holding back the head until `holdback_chars`
    NORMALIZED characters exist (see _normalized_len).

    The head of the generation is buffered, then decide(head_text) picks the
    branch:
      - decide truthy (a refusal): generation is stopped immediately and
        (partial_text, False) returned with NOTHING emitted, so the caller
        runs its non-streamed branch exactly as if chat_once had returned the
        sentinel (and the refusal's remaining tokens are never generated).
      - decide falsy: begin_stream() is called ONCE for the sink, the buffer
        is flushed to it, and the rest streams live. Returns (full_text, True).

    A reply that ends before the threshold is decided on its complete text.
    A transport error before anything was emitted returns (None, False) so the
    caller can fall back to blocking chat_once; after emission has started, a
    short interruption note is emitted and the partial text returned (same
    policy as chat_stream: never regenerate into a sink that already has
    text)."""
    collected = []
    sink = None
    decided = False

    def _decide_and_flush():
        # True = refusal (stop generating). Otherwise open the sink and flush.
        nonlocal sink, decided
        decided = True
        if decide("".join(collected)):
            return True
        sink = begin_stream()
        sink("".join(collected))
        return False

    try:
        import ollama
        _ka = _keep_alive(model)
        stream = ollama.chat(model=model, messages=messages, stream=True,
                             options=GEN_OPTIONS, keep_alive=_ka)
        try:
            for part in stream:
                _capture_chat_stats(part)   # the final part carries the counters
                msg = part.get("message") if isinstance(part, dict) \
                    else getattr(part, "message", None)
                if isinstance(msg, dict):
                    piece = msg.get("content", "") or ""
                elif msg is not None:
                    piece = getattr(msg, "content", "") or ""
                else:
                    piece = ""
                if not piece:
                    continue
                collected.append(piece)
                if decided:
                    sink(piece)
                elif _normalized_len("".join(collected)) >= holdback_chars:
                    if _decide_and_flush():
                        return "".join(collected), False
        finally:
            # Break/return closes the generator here so Ollama sees the
            # disconnect and stops generating the refusal's tail.
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if not decided:
            # Whole reply shorter than the holdback: decide on the full text.
            if _decide_and_flush():
                return "".join(collected), False
        return "".join(collected), True
    except Exception:
        if sink is not None and collected:
            try:
                sink("\n[stream interrupted; the answer may be incomplete]")
            except Exception:
                pass
            return "".join(collected), True
        return None, False


# ---------------------------------------------------------------------------
# Web search + fetch.
# ---------------------------------------------------------------------------
def web_search(query):
    """Run a DuckDuckGo text search. Returns a list of result dicts or []."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=MAX_WEB_RESULTS)
        return list(results) if results else []
    except Exception:
        return []


def fetch_page(url):
    """Fetch a URL and return cleaned readable text capped at STORE_CHAR_CAP.
    Returns '' on any failure.

    The cap here is the PERSISTENCE cap, not the prompt cap: a stored note is
    worth keeping at full depth, while a prompt still only ever sees
    PAGE_CHAR_CAP. Every prompt-facing caller must therefore slice to
    PAGE_CHAR_CAP itself, since this function no longer does it for them
    (gather_web_context and the direct-FETCH branch both do)."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=WEB_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:STORE_CHAR_CAP]
    except Exception:
        return ""


def _spawn_index_update():
    """Fire-and-forget update_index.py so pages fetched for THIS question
    become retrievable on the next one (main/brain_web hot-reload on index
    mtime). Detached and silent: the child owns no TTY, survives REPL exit,
    and any spawn failure is swallowed (the notes still land in the next full
    build_index.py run). Gated by BRAIN_AUTOLEARN."""
    if not AUTOLEARN_ENABLED:
        return
    try:
        import subprocess
        subprocess.Popen(
            [sys.executable or "python3",
             os.path.join(BRAIN_ROOT, "update_index.py"), "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass


def _rank_excerpts_by_relevance(query, candidates):
    """Reorder (url, text) candidates by query<->excerpt cosine using the same
    nomic-embed-text pipeline as _rerank_by_embedding. Returns candidates
    unchanged (DDG order) on any embed failure -- never raises, never blocks.
    Caller is responsible for slicing to MAX_WEB_FETCH afterward."""
    if len(candidates) < 2:
        return candidates
    try:
        from semantic_cache import _embed, _normalize, _dot
    except Exception:
        return candidates
    qraw = _embed(query, OLLAMA_HOST, RERANK_EMBED_MODEL, RERANK_TIMEOUT)
    if not qraw:
        return candidates
    qv = _normalize(qraw)
    scored = []
    any_vec = False
    for href, text in candidates:
        vraw = _embed(text[:RERANK_CHUNK_CHARS], OLLAMA_HOST, RERANK_EMBED_MODEL,
                       RERANK_TIMEOUT)
        if vraw:
            any_vec = True
            scored.append((_dot(qv, _normalize(vraw)), (href, text)))
        else:
            scored.append((float("-inf"), (href, text)))  # sinks below embedded ones
    if not any_vec:
        return candidates
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [pair for _, pair in scored]


# Set by gather_web_context to whether the last search's pages were written to
# the review queue, so the caller's progress line can say so without threading
# a second return value through every call site.
LAST_WEB_QUEUED = False


def _persist_web(query, pairs, statuses=None, notes_dir=None):
    """Persist a batch of (url, full_text) pairs fetched for `query`.

    The single persistence path for every web fetch in this file: the SEARCH
    path (gather_web_context) and the direct-FETCH branch both come here, so a
    fetch cannot land in one and be dropped by the other. Notes go to
    WEB_QUEUE_DIR by default, which is outside the vault root and therefore
    outside the index, so the page waits for web_review.py instead of joining
    the retrieval pool unreviewed.

    `statuses` is an optional {url: 'snippet'} side dict marking notes whose
    body is a DuckDuckGo blurb rather than real page text, so review can reject
    thin notes on sight. It is a SIDE dict on purpose: excerpts are unpacked as
    'for u, t in excerpts' in several places, so a third tuple element would
    crash them. Passed through only if web_cache accepts it.

    Best-effort and strictly additive: a no-op when nothing was fetched or the
    module is unavailable, and any failure is swallowed so answering the user's
    question is never affected. These are a handful of small atomic disk writes
    on the answer path (a few ms on NVMe); trivial next to the network fetches,
    but not literally free.

    Returns True when the write was attempted and did not blow up, so a caller
    can honestly tell the user the fetch was queued."""
    if not pairs or persist_web_fetch is None:
        return False
    target = notes_dir or WEB_QUEUE_DIR
    try:
        if statuses:
            try:
                persist_web_fetch(query, pairs, WEB_CACHE_DIR, target,
                                  statuses=statuses)
            except TypeError:
                # Older web_cache without the statuses parameter: the notes are
                # still worth writing, they just carry no snippet marker.
                persist_web_fetch(query, pairs, WEB_CACHE_DIR, target)
        else:
            persist_web_fetch(query, pairs, WEB_CACHE_DIR, target)
        # Learn what we just looked up, but ONLY when the note actually landed
        # somewhere the indexer looks. A queued note is not under the vault
        # root, so spawning update_index.py for it would pay for a subprocess
        # and contend for the .update-lock to index nothing.
        if target == WEB_VAULT_DIR:
            _spawn_index_update()
        return True
    except Exception:
        return False


def gather_web_context(query):
    """Search, fetch hrefs, and return a list of (url, excerpt) capped at
    MAX_WEB_FETCH.

    Default: takes DuckDuckGo's first MAX_WEB_FETCH results in search-engine
    order (original behavior, unchanged fetch cost). With BRAIN_WEB_SEMANTIC=1,
    fetches all MAX_WEB_RESULTS candidates and keeps the MAX_WEB_FETCH closest
    to the query by embedding cosine instead -- see WEB_SEMANTIC_ENABLED above.
    """
    results = web_search(query)
    fetch_cap = MAX_WEB_RESULTS if WEB_SEMANTIC_ENABLED else MAX_WEB_FETCH
    candidates = []
    statuses = {}
    for r in results:
        if len(candidates) >= fetch_cap:
            break
        href = r.get("href") or r.get("url") or r.get("link")
        if not href:
            continue
        text = fetch_page(href)
        if not text:
            # Fall back to the search snippet so the model still has something.
            # Mark it, so review can tell a real page from a two-line blurb.
            text = (r.get("body") or "").strip()
            if text:
                statuses[href] = "snippet"
        if text:
            candidates.append((href, text))
    if WEB_SEMANTIC_ENABLED:
        candidates = _rank_excerpts_by_relevance(query, candidates)
    kept = candidates[:MAX_WEB_FETCH]
    # Persist the FULL fetched text (up to STORE_CHAR_CAP), then return the
    # prompt-capped view. The two caps are decoupled here and nowhere else: a
    # stored note keeps the depth a reviewer and a later retrieval want, while
    # every existing consumer of this function's return value stays capped at
    # PAGE_CHAR_CAP without being touched.
    kept_urls = set(u for u, _t in kept)
    global LAST_WEB_QUEUED
    LAST_WEB_QUEUED = _persist_web(query, kept, statuses=dict(
        (u, s) for u, s in statuses.items() if u in kept_urls))
    return [(u, t[:PAGE_CHAR_CAP]) for u, t in kept]


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------
# Slim routing prompt for Phase 2 (Q1). It does NOT reference vault snippets,
# because the routing call never receives them: Phase 1 already judged the
# vault insufficient. It only needs the question and any reference URLs to
# pick FETCH vs SEARCH.
ROUTE_SYSTEM = (
    "The user's question could not be answered from a local thesis vault, so "
    "you must reach the web. You are given the question and possibly a list of "
    "reference links found in the vault notes. Reply with EXACTLY ONE line and "
    "nothing else, choosing between:\n"
    "- 'FETCH: <url>' when the answer lives on ONE specific web page: the "
    "question names or clearly implies a particular website, document, paper, "
    "or organization, or one of the reference links is the right source. Give "
    "the single most relevant full URL.\n"
    "- 'SEARCH: <query>' when no single page is obvious, or for anything current "
    "or time sensitive (weather, prices, latest news, recent events). Give a "
    "good web search query.\n"
    "Output only that one FETCH or SEARCH line with no other text."
)

WEB_SYSTEM = (
    "You are a careful research assistant. You are given excerpts fetched from "
    "the web, each labeled with its source URL, plus the user's question. "
    "Answer the question using these excerpts and cite the URLs you relied on."
)

DOC_SYSTEM = (
    "You are a careful research assistant. You are given the readable text of a "
    "single reference page fetched from a URL, plus the user's question. Answer "
    "the question from this fetched reference and cite the URL you relied on."
)

NO_WEB_SYSTEM = (
    "You are a careful research assistant grounded in a local thesis vault. "
    "Web search is disabled, so answer the question as fully as the vault "
    "snippets allow, covering every relevant point and naming the specific "
    "scholars, articles, and terms that appear in the snippets. Cite vault "
    "sources by filename when you use them, aim for about 5 to 8 sentences, "
    "and note plainly if you are unsure."
)

# Phase 1: the vault is ALWAYS consulted for an answer first. The model may only
# reach the web after this phase reports the vault cannot answer.
VAULT_FIRST_SYSTEM = (
    "You are a careful research assistant grounded in a local thesis vault. "
    "You are given retrieved snippets from the vault, each labeled with its "
    "source filename. Some snippets may be labeled '(linked)' because they come "
    "from notes that the top notes link to.\n"
    "Answer the question USING ONLY the vault snippets provided. Cite the "
    "sources by filename when you use them.\n"
    "If, and only if, the vault snippets do not contain enough information to "
    "answer the question, reply with EXACTLY the single word INSUFFICIENT and "
    "nothing else. Do not guess, do not use outside knowledge, and do not "
    "mention the web. Only emit INSUFFICIENT when the vault genuinely cannot "
    "answer; if the snippets let you answer even partially, answer instead. "
    "Never supply article numbers, statutes, dates, statistics, author-year "
    "citations, filenames, or URLs from your own memory: include such "
    "specifics ONLY if they appear verbatim in the snippets. If the snippets "
    "discuss the topic but lack the exact fact asked for, state what the "
    "snippets DO support and say plainly that the specific detail is not in "
    "the vault. "
    "Answer thoroughly: cover every relevant point the snippets support. "
    "Aim for a clear, well organized answer of about 5 to 8 sentences. "
    "Use only the provided context.\n"
    # The attribution requirement used to live mid-paragraph, phrased as "name
    # the specific scholars, articles, terms and definitions ... whenever the
    # snippets provide them", and the 3B ignored it. The judged failure
    # taxonomy is unambiguous about the cost: of 13 failures on gold_questions,
    # 10 scored 80-86, which under --isolate per-fact grading is ONE missing
    # fact out of five to seven, and the missing fact was a name almost every
    # time -- Bauman and Murray for Deaf Gain, Tom Humphries for audism,
    # Bahan/Lane/Hoffmeister for DEAF-WORLD, Stokoe and Battison for the sign
    # parameters, Wyatte Hall for language deprivation, Frederick and Columbia
    # for MSD's campuses. Retrieval had already delivered 6-11 sources on every
    # one of those, so the snippets contained the name and the answer left it
    # out.
    #
    # This is now a separate, terse, FINAL instruction rather than a clause
    # buried in a long paragraph. A 3B follows the last thing it read far more
    # reliably than the middle of a wall of text, and lengthening the prompt is
    # itself a known hazard here (a 700-word filler prompt measurably broke the
    # grounding contract), so this replaces the old clause rather than adding
    # to it.
    "One more requirement: when a snippet attributes something to a named "
    "person, school, place, or year -- who coined a term, who wrote a work, "
    "where an institution sits, when it was founded -- name them explicitly. "
    "Naming who or where a claim comes from is part of answering, not "
    "optional detail."
)

# Parent persona: same grounding contract as VAULT_FIRST_SYSTEM (snippets-only,
# filename citations, the exact bare INSUFFICIENT sentinel, no facts from model
# memory) so _is_insufficient and faithfulness_flags work unchanged. Only the
# register shifts: warm plain language for a hearing parent new to deafness,
# Deaf-community-affirming framing, and concrete next steps.
PARENT_SYSTEM = (
    "You are a warm, knowledgeable guide for hearing parents who have just "
    "learned their child is deaf, grounded in a local thesis vault. You are "
    "given retrieved snippets from the vault, each labeled with its source "
    "filename. Some snippets may be labeled '(linked)' because they come from "
    "notes that the top notes link to.\n"
    "Answer the question USING ONLY the vault snippets provided. Cite the "
    "sources by filename when you use them.\n"
    "If, and only if, the vault snippets do not contain enough information to "
    "answer the question, reply with EXACTLY the single word INSUFFICIENT and "
    "nothing else. Do not guess, do not use outside knowledge, and do not "
    "mention the web. Only emit INSUFFICIENT when the vault genuinely cannot "
    "answer; if the snippets let you answer even partially, answer instead. "
    "Never supply article numbers, statutes, dates, statistics, author-year "
    "citations, filenames, or URLs from your own memory: include such "
    "specifics ONLY if they appear verbatim in the snippets.\n"
    "Speak to the parent directly and plainly. Explain any technical term the "
    "first time you use it. Never shame or blame a parent for past choices; "
    "treat every question as a fresh start, and when the snippets support it, "
    "say plainly that past choices were made with good intentions, that it is "
    "not the parent's fault, and that it is not too late to act. Follow the "
    "Deaf-community-affirming framing in the snippets: deafness is a "
    "difference with its own language and community, not only a medical "
    "problem, and early access to sign language protects a child's "
    "development without harming speech.\n"
    "Be thorough: cover every point the snippets support that helps answer "
    "the question, and name each organization, program, school, law, or "
    "defined term the snippets provide (for example IDEA Part C, an IFSP, a "
    "named school or mentor program) with a plain one-line explanation of "
    "what it does for the family. Do not stop at the first good point; a "
    "parent deciding what to do next needs the whole picture the snippets "
    "give. When the snippets name organizations, programs, schools, or next "
    "steps, close with the concrete actions the parent can take, citing the "
    "source filenames. Aim for a clear, warm answer of about 7 to 12 "
    "sentences, and use short paragraphs or a short list when that is easier "
    "for a stressed parent to follow. Use only the provided context."
)


# Free-form refusals a small model emits INSTEAD of the exact INSUFFICIENT
# sentinel that VAULT_FIRST_SYSTEM asks for. llama3.2:3b cannot be reliably
# forced to emit a bare sentinel, so it paraphrases ("I can't answer this
# question."), which the old literal-token check let through as a real answer.
# Phrases are curated to be specific enough that a genuine substantive vault
# answer is very unlikely to contain them, and are only matched inside the
# reply's opening clause (see _is_insufficient) so a long, real answer that
# merely quotes one deep in its body is not misclassified.
_REFUSAL_PHRASES = (
    "i can't answer",
    "i cannot answer",
    "i can not answer",
    "can't answer this",
    "cannot answer this",
    "unable to answer",
    "not able to answer",
    "cannot be answered",
    "can't be answered",
    "i don't know",
    "i do not know",
    "i'm unable to",
    "i am unable to",
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "there is no information",
    "no relevant information",
    "does not contain enough",
    "doesn't contain enough",
    "do not contain enough",
    "don't contain enough",
    "cannot provide an answer",
    "can't provide an answer",
)


def _is_insufficient(text):
    """True when the vault-first pass could not answer from the vault.

    Recognizes BOTH the exact INSUFFICIENT sentinel from VAULT_FIRST_SYSTEM and
    the free-form refusals small models emit instead ("I can't answer this
    question.", "I don't know", "the vault does not contain enough ..."). The
    refusal phrases are only searched inside the reply's opening ~200 chars,
    where a genuine refusal always sits, so a long real answer that happens to
    contain such a phrase later is not treated as a refusal."""
    if not text:
        return True
    # Normalize: collapse internal whitespace, strip wrapping quotes/punct,
    # lowercase. re is already imported at module top.
    t = re.sub(r"\s+", " ", text.strip()).strip(".!?\"'` ").lower()
    if not t:
        return True
    # Exact sentinel (case-insensitive), tolerant of a trailing clause like
    # "INSUFFICIENT - the snippets do not ...". Preserves original semantics.
    if t == "insufficient" or t.startswith("insufficient"):
        return True
    # Free-form refusal: only look at the opening clause.
    if any(p in t[:200] for p in _REFUSAL_PHRASES):
        return True
    return False


_CITE_URL_RE = re.compile(r"https?://\S+")
_CITE_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Za-z]?\b")


def faithfulness_flags(answer, context_text):
    """Citation-like tokens asserted in `answer` that do NOT appear in the
    retrieved vault context. High precision by design: only URLs and
    'Article N' references, the two fabrication types small local models most
    reliably invent. Surfaced as a caution so unverifiable specifics are
    flagged rather than silently trusted."""
    if not answer:
        return []
    ctx = (context_text or "").lower()
    flags, seen = [], set()
    for m in _CITE_URL_RE.findall(answer):
        tok = m.rstrip(".,);]")
        k = tok.lower()
        if k not in ctx and k not in seen:
            seen.add(k)
            flags.append(tok)
    for m in _CITE_ARTICLE_RE.findall(answer):
        k = m.lower()
        if k not in ctx and k not in seen:
            seen.add(k)
            flags.append(m)
    return flags


def format_vault_context(chunks, linked_ids=None):
    if not chunks:
        return "(no relevant vault snippets were found)"
    linked_ids = linked_ids or set()
    blocks = []
    for c in chunks:
        src = c.get("source") or c.get("title") or "unknown"
        label = src
        if c.get("id") in linked_ids:
            label = src + " (linked)"
        text = c.get("text") or ""
        blocks.append("[source: " + label + "]\n" + text)
    return "\n\n".join(blocks)


def format_web_context(excerpts):
    blocks = []
    for url, text in excerpts:
        blocks.append("[url: " + url + "]\n" + text)
    return "\n\n".join(blocks)


# Context budget. Nothing measured the assembled prompt against num_ctx before
# this: on overflow Ollama truncates from the TOP, which drops the system prompt
# and the highest-ranked vault snippets -- precisely the two things the
# grounding contract depends on -- and the model then answers ungrounded with no
# error anywhere. Widening the retrieval window makes that reachable, so the
# guard ships with the wider window rather than after it.
#
# CHARS_PER_TOKEN is deliberately pessimistic. This vault's chunks average ~3.9
# chars/token, but they are dense with proper nouns, wikilink syntax and heading
# breadcrumbs that tokenize short, so assuming 3.4 leaves the estimate on the
# safe side. The cost of overestimating is one dropped tail chunk; the cost of
# underestimating is a silently ungrounded answer.
CHARS_PER_TOKEN = 3.4
# Measured on the device, 2026-08-16, over the 40 retrieval_gold prompts through
# the real path, against Ollama's own prompt_eval_count (num_predict=1, so
# prefill only): the true ratio is 4.25 min / 4.61 mean / 4.77 max. 3.4 is
# therefore ~35% pessimistic, and the pessimism is not free -- it is spent
# dropping tail chunks that would have fit. Overridable so an arm can be priced
# without an edit; see BRAIN_TOP_K_FAST, whose ceiling was set by this estimator.
try:
    CHARS_PER_TOKEN = float(os.environ["BRAIN_CHARS_PER_TOKEN"])
except (KeyError, ValueError):
    pass
# Leave room for the answer (num_predict) plus chat-template overhead, so the
# guard trims before llama.cpp has to.
CTX_RESERVE_TOKENS = 768
CTX_FIT_ENABLED = os.environ.get("BRAIN_CTX_FIT", "1") not in ("0", "false", "False", "no")
LAST_CTX_FIT = {}


def _estimate_tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN)


def fit_context(question, chunks, system_prompt, linked_ids=None,
                num_ctx=None, reserve=CTX_RESERVE_TOKENS):
    """Drop the LOWEST-ranked chunks until the assembled prompt fits num_ctx.

    Returns (kept_chunks, info). Trimming from the tail is the whole point: the
    alternative is Ollama trimming from the head, which takes the system prompt
    first. A ranked list that has to lose something should lose its weakest
    member, not its contract.

    Never drops below one chunk -- a question with no context at all should
    reach the INSUFFICIENT path honestly, not be silently starved here."""
    kept = list(chunks or [])
    budget = (num_ctx if num_ctx is not None else GEN_OPTIONS.get("num_ctx", 6144)) - reserve
    info = {"dropped": 0, "budget_tokens": budget,
            "est_tokens": 0, "fit": True}
    if not CTX_FIT_ENABLED or budget <= 0:
        info["est_tokens"] = _estimate_tokens(
            system_prompt + build_first_pass_user(question, kept, linked_ids))
        return kept, info
    while True:
        est = _estimate_tokens(
            (system_prompt or "")
            + build_first_pass_user(question, kept, linked_ids))
        info["est_tokens"] = est
        if est <= budget or len(kept) <= 1:
            info["fit"] = est <= budget
            return kept, info
        kept.pop()
        info["dropped"] += 1


def build_first_pass_user(question, chunks, linked_ids=None):
    parts = ["Vault context:\n" + format_vault_context(chunks, linked_ids)]
    parts.append("Question: " + question)
    return "\n\n".join(parts)


def normalize_url(raw):
    """Pull a single URL out of a FETCH line and normalize it. Adds an https://
    scheme when it is missing so the model can name a bare domain. Returns '' if
    the text does not look like a URL."""
    raw = (raw or "").strip().strip("<>\"'")
    if not raw:
        return ""
    # Take the first whitespace-delimited token in case of trailing text.
    url = raw.split()[0].strip("<>\"'").rstrip(".,;:!?)]}")
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    # Must look like host.tld to be worth a fetch.
    if not re.search(r"[a-z0-9][a-z0-9\-]*\.[a-z]{2,}", url, re.IGNORECASE):
        return ""
    return url


def vault_sources(chunks):
    """Ordered, de-duplicated list of source names for the given chunks."""
    sources = []
    for c in chunks:
        src = c.get("source") or c.get("title")
        if src and src not in sources:
            sources.append(src)
    return sources


# ---------------------------------------------------------------------------
# Model menu.
# ---------------------------------------------------------------------------
def choose_model():
    """Show a numbered menu of installed models and return one name.
    Enter with no input selects DEFAULT_MODEL. Single selection, no looping."""
    models = list_models()
    if not models:
        print("Could not list installed models. Falling back to " + DEFAULT_MODEL + ".")
        return DEFAULT_MODEL

    print("\nInstalled models:")
    for idx, name in enumerate(models, start=1):
        marker = "  (default)" if name == DEFAULT_MODEL else ""
        print("  " + str(idx) + ". " + name + marker)
    if DEFAULT_MODEL not in models:
        print("  (Enter selects the default " + DEFAULT_MODEL + ", even though it is not listed above.)")

    try:
        raw = input("Pick a model number, or press Enter for the default: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return DEFAULT_MODEL

    if raw == "":
        return DEFAULT_MODEL
    if raw.isdigit():
        i = int(raw)
        if 1 <= i <= len(models):
            return models[i - 1]
    print("Unrecognized selection, using the default " + DEFAULT_MODEL + ".")
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# History management.
# ---------------------------------------------------------------------------
def trim_history(history):
    """Keep only the last HISTORY_TURNS user+assistant exchanges."""
    max_msgs = HISTORY_TURNS * 2
    if len(history) > max_msgs:
        del history[: len(history) - max_msgs]


# ---------------------------------------------------------------------------
# Retrieval + wikilink expansion.
# ---------------------------------------------------------------------------
def _score_at(scores, cid):
    """Safe lookup into the score array by chunk id."""
    try:
        if 0 <= cid < len(scores):
            return scores[cid]
    except Exception:
        pass
    return 0


def expand_with_links(retrieved, scores, meta):
    """Pull in a representative chunk for each note linked from the retrieved
    notes. Returns (added_chunks, linked_ids). Capped at MAX_LINK_EXPAND and
    de-duplicated by chunk id. Never raises."""
    added = []
    linked_ids = set()
    try:
        by_source = meta.get("by_source") or {}
        by_id = meta.get("by_id") or {}
        links_out = meta.get("links_out") or {}

        represented = set()
        seen_ids = set()
        for c in retrieved:
            src = c.get("source")
            if src:
                represented.add(src)
            cid = c.get("id")
            if cid is not None:
                seen_ids.add(cid)

        for c in list(retrieved):
            if len(added) >= MAX_LINK_EXPAND:
                break
            src = c.get("source")
            if not src:
                continue
            for target in links_out.get(src, []):
                if len(added) >= MAX_LINK_EXPAND:
                    break
                if not target or target in represented:
                    continue
                ids = by_source.get(target) or []
                if not ids:
                    continue
                # Pick the target note's best-scoring chunk; if all zero, first.
                best_id = None
                best_score = None
                for cid in ids:
                    sc = _score_at(scores, cid)
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_id = cid
                if best_id is None or best_score is None or best_score <= 0:
                    best_id = ids[0]
                chunk = by_id.get(best_id)
                if chunk is None:
                    continue
                cid = chunk.get("id")
                if cid in seen_ids:
                    continue
                added.append(chunk)
                represented.add(target)
                seen_ids.add(cid)
                if cid is not None:
                    linked_ids.add(cid)
    except Exception:
        return added, linked_ids
    return added, linked_ids


def expand_with_obsidian_signal(retrieved, context_chunks, meta):
    """Pull in a representative chunk for each note the Obsidian Smart
    Connections Pro plugin has scored as related to the retrieved notes.

    Mirrors expand_with_links' shape and dedup rules, but the relationship
    source is different: AI-computed embedding similarity the plugin already
    persisted to disk (obsidian_rank.py), not an explicit [[wikilink]] a human
    wrote. `retrieved` (the top BM25/dense hits) supplies the seed note paths;
    `context_chunks` (retrieved + any wikilink expansion already applied) is
    used only for dedup, so this never re-adds a chunk already present. Capped
    at OBSIDIAN_MAX_EXPAND. Never raises: any failure (missing module, empty
    vector export, bad seed) degrades to returning whatever succeeded so far,
    or an empty list."""
    added = []
    if not OBSIDIAN_RANK_ENABLED:
        return added
    try:
        import obsidian_rank
        vectors = obsidian_rank.load_note_vectors_from_export()
        if not vectors:
            return added
        by_source = meta.get("by_source") or {}
        by_id = meta.get("by_id") or {}

        seed_paths = []
        seen_seed = set()
        for c in retrieved:
            src = c.get("source")
            if src and src not in seen_seed:
                seen_seed.add(src)
                seed_paths.append(src)
        if not seed_paths:
            return added

        already_present = {c.get("id") for c in context_chunks
                           if c.get("id") is not None}
        already_sources = {c.get("source") for c in context_chunks
                           if c.get("source")}

        expanded = obsidian_rank.expand_from_seeds(
            vectors, seed_paths, top_k_per_seed=OBSIDIAN_TOP_K_PER_SEED,
            max_results=OBSIDIAN_MAX_EXPAND)

        # Block-level chunk pick (BRAIN_OBSIDIAN_BLOCKS). The query-side signal
        # is the TOP retrieved note's own vector, not a live query embedding:
        # bge-micro-v2 inference on the Jetson was already ruled out, and the
        # best BM25/dense hit is the closest in-space proxy for "what this
        # question is about" that costs nothing. Falls back to the next seed
        # when the top one has no plugin vector. blocks_by_note() is built once
        # per turn, not per note (it is a full rescan of ~5k blocks otherwise).
        blocks_index = None
        signal_vec = None
        if OBSIDIAN_BLOCKS_ENABLED:
            block_vectors = obsidian_rank.load_block_vectors_from_export()
            if block_vectors:
                for seed in seed_paths:
                    signal_vec = vectors.get(seed)
                    if signal_vec:
                        break
                if signal_vec:
                    blocks_index = obsidian_rank.blocks_by_note(block_vectors)

        for path, _score in expanded:
            if path in already_sources:
                continue
            ids = by_source.get(path) or []
            if not ids:
                continue
            cid = ids[0]  # note's first/primary chunk -- no BM25 score here to
                          # pick a better one by, unlike expand_with_links.
            if blocks_index:
                # best_chunk_for_note returns None when no block maps onto a
                # chunk, which deliberately leaves ids[0] in charge.
                chunk_texts = [(i, (by_id.get(i) or {}).get("text") or "")
                               for i in ids]
                better = obsidian_rank.best_chunk_for_note(
                    blocks_index, path, signal_vec, chunk_texts,
                    max_blocks=OBSIDIAN_BLOCK_CANDIDATES)
                if better is not None:
                    cid = better
            if cid in already_present:
                continue
            chunk = by_id.get(cid)
            if chunk is None:
                continue
            added.append(chunk)
            already_present.add(cid)
            already_sources.add(path)
    except Exception:
        return added
    return added


def collect_reference_urls(chunks, meta):
    """De-duplicated list of external URLs found in the given notes, capped."""
    urls = []
    try:
        urls_out = meta.get("urls_out") or {}
        for c in chunks:
            src = c.get("source")
            if not src:
                continue
            for u in urls_out.get(src, []):
                if u and u not in urls:
                    urls.append(u)
                if len(urls) >= MAX_DOC_URLS:
                    return urls
    except Exception:
        return urls
    return urls


# ---------------------------------------------------------------------------
# One question, routed. Returns (answer_text, sources_list).
# ---------------------------------------------------------------------------
def _short_srcs(sources, n=3):
    """First n source basenames, comma-joined, for a compact tree branch."""
    names = []
    for s in sources[:n]:
        try:
            names.append(os.path.splitext(os.path.basename(str(s)))[0])
        except Exception:
            names.append(str(s))
    tail = ", …" if len(sources) > n else ""
    return ", ".join(names) + tail


def _log_turn(question, answer, sources):
    """Append one answered turn to the persistent transcript, if armed.

    A no-op unless the REPL called brain_transcript.enable(), which is what
    keeps the :8899 web server and the :8000 advisor -- both of which reach
    _finish through handle_question -- from writing into the REPL's history."""
    if brain_transcript is None:
        return
    try:
        brain_transcript.append(question, answer, sources)
    except Exception:
        pass


def _finish(run, history, question, answer, sources, history_text=None):
    """Common tail: append to history, print the sources branch, return.

    `history_text` stores something OTHER than the returned answer in history,
    and defaults to None so every existing caller is unchanged. It exists for
    the scholar second pass: a merged two-pass answer plus a citation footer is
    roughly 1,600 estimated tokens, a fifth of the 7,424-token budget, and
    PHASE1_HISTORY_MSGS carries it into the NEXT question's prefill where
    fit_context pays for it by dropping retrieval chunks from the tail. The
    pass-1 draft is the part of the turn worth remembering; the expansion and
    the citations are for the reader."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant",
                    "content": answer if history_text is None else history_text})
    trim_history(history)
    # The transcript stores the DISPLAYED answer, never history_text. The two
    # diverge on purpose: history_text is the shrunken version kept for the next
    # prompt's context budget, while /history is read by a human who wants back
    # exactly what was on screen.
    _log_turn(question, answer, sources)
    run.sources(sources, vault_name=VAULT_NAME)
    return answer, sources


def _answer_no_web(model, question, context_chunks, linked_ids, history, run):
    """Best-effort vault answer for every no-web path (web off, no usable web
    results, direct fetch failed)."""
    messages = [{"role": "system", "content": NO_WEB_SYSTEM}]
    messages.extend(history)
    messages.append(
        {"role": "user", "content": build_first_pass_user(
            question, context_chunks, linked_ids)}
    )
    answer = chat_stream(model, messages, sink=run.begin_answer())
    return _finish(run, history, question, answer, vault_sources(context_chunks))


def _answer_from_excerpts(model, question, excerpts, history, run):
    """Answer from fetched web excerpts and cite their URLs."""
    messages = [{"role": "system", "content": WEB_SYSTEM}]
    messages.extend(history)
    messages.append(
        {"role": "user", "content": (
            "Web excerpts:\n" + format_web_context(excerpts)
            + "\n\nQuestion: " + question)}
    )
    answer = chat_stream(model, messages, sink=run.begin_answer())
    return _finish(run, history, question, answer, [u for u, _ in excerpts])


def _run_skill_loop(model, question, bm25, chunks, meta, web_enabled, persona,
                    intent="analytic", history=None):
    """One bounded skill-loop attempt at an INSUFFICIENT question.

    Returns a brain_skills.SkillResult, or None when the loop is unavailable or
    blew up. Everything here is deliberately defensive: the skill loop is an
    ADDITION to a pipeline that already answers, so any failure has to leave the
    existing phase-2 web path untouched rather than take the turn down with it.
    The import is lazy and inside the try for the same reason
    expand_with_obsidian_signal imports obsidian_rank that way: a missing or
    broken module must degrade at question time, never at startup.

    Prompts are passed VERBATIM (vault_system/web_system/doc_system) together
    with _is_insufficient, so the grounding contract the loop's sub-answers are
    judged by is the same text this file's own phase-1 call uses, per-intent
    directive included (_shape_for_intent APPENDS, so the INSUFFICIENT sentinel
    and the citation rules survive untouched). Substituting a paraphrase there
    would silently detach _is_insufficient from what the model was told to say.

    `history` is only ever rendered into a short capped blob for the loop's
    SELECTION call, so a follow-up question has an antecedent to route on. It
    never becomes chat turns, which on a 6144-token window is how a system
    prompt gets truncated away.
    """
    try:
        import brain_skills
    except Exception:
        return None
    try:
        def _retrieve(query, keep_k):
            # Wider than the phase-1 window on purpose: the loop only runs
            # because keep_k chunks were not enough. order_full is the
            # PRE-diversify ranking (retrieve_context keeps it exactly so a
            # deeper slice is possible), and qvec is NOT reused because the
            # skill rewrote the query - a stale embedding would score the wrong
            # question.
            r = retrieve_context(query, bm25, chunks, meta, model=model,
                                 persona=persona, tm={})
            order = r.get("order_full") or r.get("order") or []
            return [chunks[i] for i in order[:keep_k]]

        def _chat_select(m, messages):
            opts = dict(GEN_OPTIONS)
            opts.update(brain_skills.SELECT_GEN_OPTIONS)
            return chat_once(m, messages, options=opts)

        # Last exchange only, ~200 chars a side. One turn is what a pronoun
        # actually needs, and the cap keeps this near 100 tokens so the
        # selection prompt stays the tiny thing brain_skills' header argues for.
        _prev = []
        for _role in ("user", "assistant"):
            for _msg in reversed(history or []):
                if _msg.get("role") == _role:
                    _prev.append(_role + ": " + (_msg.get("content") or "")[:200])
                    break
        prev_turn = "\n".join(_prev)

        ctx = brain_skills.SkillContext(
            model=model,
            chat=chat_once,
            chat_select=_chat_select,
            retrieve=_retrieve,
            gather_web=gather_web_context,
            fetch_page=fetch_page,
            meta=meta,
            web_enabled=web_enabled,
            vault_system=_shape_for_intent(
                PARENT_SYSTEM if persona == "parent" else VAULT_FIRST_SYSTEM,
                intent),
            web_system=WEB_SYSTEM,
            doc_system=DOC_SYSTEM,
            # reason_system is deliberately NOT injected: brain.py's nearest
            # prompt (NO_WEB_SYSTEM) tells the model to answer from the vault
            # snippets, which is the opposite of what the reason skill does
            # (no sources at all). brain_skills.DEFAULT_REASON_SYSTEM already
            # says exactly that and forbids inventing a citation.
            is_insufficient=_is_insufficient,
            format_vault_context=format_vault_context,
            vault_sources=vault_sources,
            prev_turn=prev_turn,
        )
        return brain_skills.answer_with_skills(ctx, question)
    except Exception:
        return None


def _format_timing(tm):
    """One machine-parseable stderr line from a turn's timing dict."""
    parts = ["[timing]"]
    for key in ("mode", "persona"):
        if tm.get(key):
            parts.append("%s=%s" % (key, tm[key]))
    for key in ("bm25", "dense", "hyde", "rerank", "expand", "obsidian", "ttft",
                "phase1", "skills", "route", "web", "total"):
        if key in tm:
            parts.append("%s=%.3f" % (key, tm[key]))
    # ctx_dropped is absent on a turn that fit, so it appears in the timing line
    # ONLY when the budget guard actually trimmed something. A silent overflow
    # was previously invisible; this makes trimming a thing you can grep for.
    for key in ("prompt_eval_count", "eval_count", "ctx_dropped"):
        if key in tm:
            parts.append("%s=%d" % (key, tm[key]))
    return " ".join(parts)


def handle_question(model, question, history, bm25, chunks, meta, web_enabled, run,
                    persona="default", qvec=None, qvec_model=None):
    """Route a single question and stream the final answer through the run's UI.
    Updates history and returns (answer_text, sources_list). On error paths
    returns ("", []). `run` drives the ultracode-style progress tree; it always
    exists (it degrades to plain output when stdout is not a TTY).
    persona="parent" answers in the warm parent-guidance register (PARENT_SYSTEM)
    and boosts the curated parent-guidance notes in retrieval; grounding rules
    are identical to the default persona.

    qvec/qvec_model: an already-normalized question embedding from the semantic
    router's miss (SemanticCache.match_with_vec), reused by dense scoring and
    the rerank fallback when the model names match, so one fresh turn embeds
    the question once instead of twice.

    Thin timing shell around _handle_question_impl: stage seconds land in
    LAST_TIMING on every return path (BRAIN_TIMING=1 also prints them), so
    latency work is measured on the real pipeline, not a mock."""
    global LAST_TIMING
    tm = {"persona": persona}
    t0 = time.perf_counter()
    try:
        return _handle_question_impl(model, question, history, bm25, chunks,
                                     meta, web_enabled, run, persona, tm,
                                     qvec, qvec_model)
    finally:
        tm["total"] = time.perf_counter() - t0
        LAST_TIMING = tm
        if TIMING_ENABLED:
            try:
                sys.stderr.write(_format_timing(tm) + "\n")
                sys.stderr.flush()
            except Exception:
                pass


# Appended to the system prompt for the question shapes the base prompts serve
# badly. The base prompts ask for "about 5 to 8 sentences" unconditionally, so a
# 3B model answers "Where is Gallaudet University?" with an essay that buries
# the city in a subordinate clause. These directives are APPENDED, never
# substituted: the INSUFFICIENT contract, the citation rules, and the
# no-outside-knowledge rules must survive verbatim or _is_insufficient,
# faithfulness_flags, and chat_stream_holdback all stop working.
_INTENT_DIRECTIVES = {
    "multi": (
        " This question has several parts. Answer EVERY part, in the order "
        "asked, and say plainly which parts the snippets do not cover rather "
        "than answering only the part you have the most material for."
    ),
}

# The length instruction, SUBSTITUTED rather than contradicted.
#
# The base prompts end with an unconditional "Aim for ... about 5 to 8
# sentences" (7 to 12 for the parent persona), and the factual_short directive
# used to be appended immediately after it saying "OVERRIDE the length guidance
# above ... at most two short sentences. Do not write an essay." So a 3B model
# received two directly opposed length instructions in adjacent sentences of
# one system message and had to resolve the conflict by meta-instruction.
# Recency helps but does not guarantee it, and on the parent persona the gap is
# wider still (7-12 sentences vs at most two).
#
# Substituting the sentence removes the contradiction instead of asking the
# model to arbitrate it. The grounding contract is untouched BY CONSTRUCTION:
# only this one trailing sentence is replaced, so the INSUFFICIENT sentinel,
# the cite-by-filename rule and the no-outside-knowledge clause stay
# byte-identical -- which _is_insufficient, faithfulness_flags and
# chat_stream_holdback all depend on.
_LENGTH_SENTENCES = (
    "Aim for a clear, well organized answer of about 5 to 8 sentences.",
    "Aim for a clear, warm answer of about 7 to 12 sentences, and use short "
    "paragraphs or a short list when that is easier for a stressed parent to "
    "follow.",
)

_LENGTH_OVERRIDES = {
    "factual_short": (
        "This question asks for one specific fact. State that fact in your "
        "FIRST sentence, directly and completely, then add at most two short "
        "sentences of supporting context from the snippets. Do not open with "
        "background. Do not write an essay."
    ),
}


def _shape_for_intent(system_prompt, intent):
    """Apply the intent's shaping to a base system prompt.

    A length override REPLACES the base length sentence; other directives are
    appended. Unknown or 'analytic' intents get the prompt back unchanged.
    If the expected length sentence is ever reworded and no longer matches,
    this falls back to appending, which is the old behavior -- degraded, but
    never silently dropping the instruction."""
    out = system_prompt
    override = _LENGTH_OVERRIDES.get(intent)
    if override:
        for sentence in _LENGTH_SENTENCES:
            if sentence in out:
                out = out.replace(sentence, override, 1)
                break
        else:
            out = out + " " + override
    directive = _INTENT_DIRECTIVES.get(intent, "")
    if directive:
        out = out + directive
    return out


def retrieve_context(question, bm25, chunks, meta, model=None,
                     persona="default", tm=None, qvec=None, qvec_model=None,
                     hyde=None):
    """Run the retrieval half of a turn and return the ordered chunks.

    Extracted verbatim from _handle_question_impl so the retrieval eval harness
    (eval_retrieval.py) scores THE SAME code production answers with. Adding a
    second implementation for measurement is how a harness starts lying: the
    build_note_chunks() seam exists for the same reason on the index side.

    Returns a dict with keys: intent, wh, qtok, scores, order, keep_k,
    retrieved, mode, top_score, n_subq. `tm` is filled in place when given.

    `hyde` defaults to None, meaning "use the HYDE_ENABLED module global" --
    i.e. exactly what every caller got before the parameter existed. It is an
    explicit per-call override for the scholar second pass, whose facet queries
    are ALREADY hypothetical-answer text: running HyDE over them pays a full
    extra 3B generation (measured at 4.8s) to hypothesize about a hypothesis,
    inside a wall-clock deadline. That is a latency requirement, not a tuning
    preference.
    """
    if tm is None:
        tm = {}
    # None means "module default", so an omitted kwarg reproduces the old gate.
    _hyde_on = HYDE_ENABLED if hyde is None else bool(hyde)
    _t = time.perf_counter()
    intent, wh = classify_intent(question)
    qtok = query_tokens(question)

    # A question that names two or more notes outright is a comparison and needs
    # multi handling even when it is a single clause ("Which is older, X or Y?"
    # has nothing to split on), so entity sub-queries can promote the intent.
    # Resolved BEFORE the boosts and keep_k, which both read the final intent.
    ent_subs = entity_subqueries(qtok, chunks)
    if ent_subs and intent != "multi":
        intent = "multi"

    try:
        scores = bm25.get_scores(qtok)
    except Exception:
        scores = []
    scores = _apply_section_boost(scores, chunks, qtok)
    scores = _apply_title_boost(scores, chunks, qtok)
    scores = _apply_intent_boost(scores, bm25, intent_boost_terms(qtok, wh))
    scores = _apply_card_boost(scores, chunks, intent)
    if persona == "parent":
        scores = _apply_parent_boost(scores, chunks)
    elif persona == "scholar":
        scores = _apply_prefix_boost(scores, chunks, _SCHOLAR_SOURCE_PREFIXES,
                                     SCHOLAR_BOOST)
    elif DEFAULT_ANALYTIC_BOOST and intent == "analytic":
        # Live at the shipped 2.0, worth +1 in_context. See the swept table at
        # DEFAULT_ANALYTIC_BOOST: 1.0 is a no-op, past 5.0 it starts evicting
        # rows that were already right.
        scores = _apply_prefix_boost(scores, chunks,
                                     _REFERENCE_SOURCE_PREFIXES,
                                     DEFAULT_ANALYTIC_BOOST)

    # Per-sub-question BM25 rankings, fused with the whole-question ranking
    # below so a chunk answering ANY one part well can reach the context window.
    sub_rankings = []
    if intent == "multi":
        for sub in decompose(question) + ent_subs:
            try:
                s = bm25.get_scores(sub)
            except Exception:
                continue
            if not len(s):
                continue
            sub_rankings.append(
                [i for i in sorted(range(len(chunks)), key=lambda i: s[i],
                                   reverse=True) if s[i] > 0][:RRF_POOL])
    tm["bm25"] = time.perf_counter() - _t
    tm["subq"] = len(sub_rankings)

    retrieved = []
    top_score = 0.0
    mode = "bm25"
    order = None
    order_full = None
    keep_k = TOP_K_FAST
    if len(scores):
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        if model == DEEP_MODEL:
            keep_k = TOP_K_DEEP
        elif persona == "parent":
            keep_k = PARENT_TOP_K
        elif persona == "scholar":
            # Depth comes from breadth of evidence, not from asking for depth.
            keep_k = SCHOLAR_TOP_K
        elif intent == "multi":
            # Every named entity needs its own facts in context, and at 2 chunks
            # per source TOP_K_FAST=6 covers only three notes -- with the card
            # carrying the date competing against its own note's prose for those
            # two slots. Same size as TOP_K_DEEP, which num_ctx=6144 already
            # holds for the 7B path.
            keep_k = MULTI_TOP_K
        else:
            keep_k = TOP_K_FAST
        order = None
        # Hybrid path: fuse BM25 with GLOBAL dense retrieval via RRF. Only when a
        # dense block is present, hybrid is enabled, and the query embed succeeds;
        # otherwise order stays None and we take the BM25 + rerank path below.
        if HYBRID_ENABLED and meta.get("dense"):
            _t = time.perf_counter()
            dscore = _dense_scores(question, chunks, meta, qvec=qvec,
                                   qvec_model=qvec_model)
            tm["dense"] = time.perf_counter() - _t
            if dscore is not None:
                dense_ranked = sorted(range(len(chunks)),
                                      key=lambda i: dscore[i], reverse=True)
                # Lexical side contributes only positive-BM25 docs; dense side
                # contributes its top pool regardless of lexical overlap (the
                # recall win). RRF fuses the two rank lists.
                bm25_pool = [i for i in ranked if scores[i] > 0][:RRF_POOL]
                dense_pool = [i for i in dense_ranked
                              if dscore[i] > float("-inf")][:RRF_POOL]
                pools = [bm25_pool, dense_pool]
                # HyDE contributes a THIRD ranker, from a hypothetical answer
                # rather than the question, for the vocabulary gap neither of
                # the other two can cross. Additive by construction: a bad
                # hypothesis dilutes rank slightly, it cannot overwrite the
                # literal question's ranking.
                suffix = ""
                if _hyde_on:
                    hscore = _hyde_scores(question, chunks, meta, tm)
                    if hscore is not None:
                        hyde_ranked = sorted(range(len(chunks)),
                                             key=lambda i: hscore[i], reverse=True)
                        pools.append([i for i in hyde_ranked
                                      if hscore[i] > float("-inf")][:RRF_POOL])
                        suffix = "+hyde"
                order = _rrf_fuse(pools + sub_rankings)
                order_full = order
                mode = ("hybrid+subq" if sub_rankings else "hybrid") + suffix
        if order is None:
            # BM25 + query-time rerank (un-embedded index, hybrid off, or the
            # embedder was unreachable). Identical to the pre-hybrid behavior.
            if sub_rankings:
                # Fuse the whole-question and per-clause rankings FIRST, then
                # rerank that shortlist, so a chunk that only a sub-question
                # surfaces is still a rerank candidate.
                whole_pool = [i for i in ranked if scores[i] > 0][:RRF_POOL]
                cand_ids = _rrf_fuse([whole_pool] + sub_rankings)[:RERANK_CAND_N]
                mode = "bm25+subq"
            else:
                cand_ids = [i for i in ranked if scores[i] > 0][:RERANK_CAND_N]
            _t = time.perf_counter()
            order = _rerank_by_embedding(question, chunks, cand_ids, qvec=qvec,
                                         qvec_model=qvec_model)
            order_full = order
            tm["rerank"] = time.perf_counter() - _t
        if order and WEB_MAX_PER_SOURCE > 0:
            # Bound scraped pages before any persona cap runs. order_full is
            # deliberately left pointing at the pre-cap ranking so retrieval
            # measurement still sees where a chunk actually ranked.
            order = _cap_web_sources(order, chunks, WEB_MAX_PER_SOURCE)
        if order and (persona in ("parent", "scholar") or intent == "multi"
                      or DEFAULT_MAX_PER_SOURCE > 0):
            # A multi-entity question needs facts from several different notes,
            # and the boosts that make the FIRST entity rank well let its note
            # take every slot: "which was founded first, Gallaudet or the
            # Maryland School for the Deaf?" retrieved six Gallaudet chunks and
            # zero MSD ones, so half the answer could not be in context at any
            # generation quality. Same failure the parent rubric hit, same fix.
            # Keep the pre-diversify ranking for measurement. _diversify_order
            # returns AT MOST keep_k ids, so scoring "where did the target
            # rank" against its output silently caps every multi-intent
            # question at keep_k and reports anything past the window as "not
            # retrieved at all". That conflates a ranking miss with a window
            # miss, which are different problems with different fixes.
            # A scholar question has the same shape as the multi case for this
            # purpose: the Gulati note alone is long enough to fill a 10-slot
            # window on its own, and an answer sourced from one note cannot
            # cover more than one facet however good it is.
            order_full = list(order)
            order = _diversify_order(
                order, chunks, keep_k,
                PARENT_MAX_PER_SOURCE if persona == "parent"
                else SCHOLAR_MAX_PER_SOURCE if persona == "scholar"
                else MULTI_MAX_PER_SOURCE if intent == "multi"
                else DEFAULT_MAX_PER_SOURCE)
        for i in order[:keep_k]:
            retrieved.append(chunks[i])
        if ranked:
            try:
                top_score = float(scores[ranked[0]])
            except Exception:
                top_score = 0.0
    return {"intent": intent, "wh": wh, "qtok": qtok, "scores": scores,
            "order": order, "order_full": order_full if order_full is not None
            else order, "keep_k": keep_k, "retrieved": retrieved,
            "mode": mode, "top_score": top_score, "n_subq": len(sub_rankings)}


def _scholar_auto_hook(question, intent):
    """UNREACHED ON PURPOSE. Nothing calls this.

    The place a future BRAIN_SCHOLAR_AUTO would decide to promote a question to
    the scholar persona on its own. It is a stub rather than an implementation
    because auto-promotion changes what the DEFAULT persona does on questions
    nobody asked to be deep, and the only honest way to buy that is an ablation
    that prices scholar against default on the same battery. The flag is defined
    (so an arm can record it) and wired to nothing (so it cannot silently move a
    number). Delete this function or call it, but do not let it half-exist."""
    raise NotImplementedError("BRAIN_SCHOLAR_AUTO is not implemented")


def _scholar_read_frontmatter(source):
    """Frontmatter fields for a vault-relative source path, or {}.

    Injected into brain_depth.citation_footer so that module keeps its
    no-brain-import, no-filesystem guarantee. Frontmatter ONLY, and only the
    head of the file: the Gulati 2018 note has no frontmatter doi: while its
    reference list carries 18 other papers' DOIs, so anything that reads the
    body will confidently attribute someone else's DOI to it."""
    try:
        import build_index
        path = os.path.join(SCHOLAR_VAULT_DIR, str(source))
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(8192)
        fm, _body = build_index.split_frontmatter(head)
        return build_index.parse_frontmatter_fields(fm) or {}
    except Exception:
        return {}


def _scholar_citation_footer(answer, context_chunks):
    """Answer with the retrieved-sources footer appended. Never raises.

    Called AFTER faithfulness_flags has graded the model-authored body, which is
    load-bearing in both directions: the flags must never see our own footer
    (every name in it would read as an unsupported specific), and the footer's
    citations are built from frontmatter with the url: field deliberately
    dropped, so _CITE_URL_RE can never flag a citation we verified ourselves."""
    if not SCHOLAR_CITE or not answer:
        return answer
    try:
        import brain_depth
        if CITE_SENTINEL and CITE_SENTINEL.strip() in answer:
            return answer            # already footed; never double-append
        footer = brain_depth.citation_footer(vault_sources(context_chunks),
                                             _scholar_read_frontmatter)
    except Exception:
        return answer
    return answer + footer if footer else answer


def _scholar_second_pass(model, question, bm25, chunks, meta, context_chunks,
                         linked_ids, system_prompt, draft, tm):
    """One bounded second retrieval+generation pass for the scholar persona.

    Returns (merged_answer, merged_chunks, merged_linked_ids). NEVER raises: the
    first pass has already produced a usable answer by the time this runs, so
    every failure mode returns the draft untouched and records tm['scholar_error']
    rather than taking the turn down.

    Shape, and why it is this shape:

    * Facets come from the vault's own [[wikilink]] graph (meta["links_out"], the
      map expand_with_links already consumes), ranked by brain_depth.rank_facets
      for what the draft says LEAST about. LD_FALLBACK_FACETS is used only when
      the graph yields nothing at all.
    * Facet retrieval runs with hyde=False and persona="scholar" inside
      SCHOLAR_DEADLINE_S, and admits at most SCHOLAR_FACET_CHUNKS chunks the
      first pass has NOT already seen.
    * Exactly ONE reconciling generation, not one per facet. Per-facet calls
      re-answer the original question from three narrow chunks, frequently trip
      _is_insufficient, and when they do not they yield two partial restatements
      with nothing reconciling them.
    * The reconciling call sends the draft plus ONLY THE NEW CHUNKS, never the
      union. fit_context pops from the TAIL, and at keep_k=10 the pass-1 context
      alone estimates 6,300-7,600 tokens against a 7,424 budget -- a union
      payload lands ~50% over and sheds exactly the new evidence it exists to
      deliver.
    * The system string is byte-identical to pass 1. No "prefer specifics, name
      every source" instruction: on a 3B that is a citation-fabrication
      generator, aimed at exactly the output class faithfulness_flags cannot see.

    tm gains scholar_facets_selected / scholar_new_chunks / scholar_pass2_s,
    which reach results[]["timing"] through LAST_TIMING. They are a free,
    judge-free, per-arm proof that pass 2 contributed real NEW evidence rather
    than re-sending what pass 1 already had.
    """
    t0 = time.perf_counter()
    try:
        import brain_depth
    except Exception as exc:
        tm["scholar_error"] = "brain_depth unavailable: " + repr(exc)[:120]
        return draft, context_chunks, linked_ids

    merged, merged_chunks = draft, context_chunks
    try:
        ctx_sources = vault_sources(context_chunks)
        top_source = ctx_sources[0] if ctx_sources else None
        seen_ids = {c.get("id") for c in context_chunks
                    if isinstance(c, dict) and c.get("id") is not None}

        cands = brain_depth.graph_facets(meta.get("links_out") or {},
                                         top_source, ctx_sources)
        if not cands:
            # v1 index (links_out backfilled empty). Same call, synthetic graph,
            # so the fallback produces facet dicts through the one code path.
            cands = brain_depth.graph_facets(
                {_LD_FALLBACK_SEED: list(LD_FALLBACK_FACETS)},
                _LD_FALLBACK_SEED, ctx_sources)
        facets = brain_depth.rank_facets(cands, draft, question,
                                         SCHOLAR_FACETS_N)
        tm["scholar_facets_selected"] = [f.get("source") for f in facets]

        new_chunks, known = [], set(seen_ids)
        for f in facets:
            if time.perf_counter() - t0 >= SCHOLAR_DEADLINE_S:
                tm["scholar_deadline"] = 1
                break
            # Label first, question second: the label is what makes this query
            # different from pass 1's, and the question keeps the facet's
            # retrieval anchored to what was actually asked.
            fq = (f.get("label") or "") + " " + question
            try:
                _fr = retrieve_context(fq, bm25, chunks, meta, model=model,
                                       persona="scholar", tm={},
                                       hyde=False)
            except Exception:
                continue
            got = brain_depth.new_chunks(_fr.get("retrieved") or [], known,
                                         SCHOLAR_FACET_CHUNKS)
            for c in got:
                cid = c.get("id")
                if cid is not None:
                    known.add(cid)
            new_chunks.extend(got)
        tm["scholar_new_chunks"] = len(new_chunks)
        if not new_chunks:
            return draft, context_chunks, linked_ids

        # Budget the reconciling prompt against the draft, which is prefill the
        # estimator inside fit_context cannot see from the chunk list alone.
        tail = ("\n\nDRAFT SO FAR:\n" + draft
                + "\n\nAdd only what the draft does not already state.")
        send_chunks, _fit2 = fit_context(question, new_chunks,
                                         (system_prompt or "") + tail,
                                         linked_ids)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": build_first_pass_user(question, send_chunks,
                                                      linked_ids) + tail}]
        # Same num_ctx as pass 1: a second sequential call is free in VRAM only
        # at the same window, because the KV cache is already allocated for it.
        expansion = (chat_once(model, messages,
                               options=dict(GEN_OPTIONS, num_predict=320),
                               force_rest=True, timeout=90) or "").strip()
        if (not expansion or expansion.startswith("[error]")
                or _is_insufficient(expansion)):
            return draft, context_chunks, linked_ids

        merged = brain_depth.merge(draft, expansion)
        # Interleave rather than append: the new evidence lands where a reader
        # (and any downstream fit) sees it before the weakest pass-1 chunks.
        merged_chunks = (list(context_chunks[:6]) + list(new_chunks)
                         + list(context_chunks[6:]))
    except Exception as exc:
        tm["scholar_error"] = repr(exc)[:200]
        return draft, context_chunks, linked_ids
    finally:
        tm["scholar_pass2_s"] = time.perf_counter() - t0
    return merged, merged_chunks, linked_ids


def _handle_question_impl(model, question, history, bm25, chunks, meta,
                          web_enabled, run, persona, tm, qvec=None,
                          qvec_model=None):
    # --- Retrieval: BM25 scoring -----------------------------------------
    run.running("bm25")
    _r = retrieve_context(question, bm25, chunks, meta, model=model,
                          persona=persona, tm=tm, qvec=qvec,
                          qvec_model=qvec_model)
    intent = _r["intent"]
    scores = _r["scores"]
    retrieved = _r["retrieved"]
    mode = _r["mode"]
    top_score = _r["top_score"]
    vsrc = vault_sources(retrieved)
    tm["mode"] = mode
    run.done("bm25", "%d chunks · %s · top score %.1f · %s" % (
        len(retrieved), mode, top_score, _short_srcs(vsrc)) if retrieved
        else "no lexical matches")

    # --- Retrieval: wikilink expansion -----------------------------------
    run.running("expand")
    _t = time.perf_counter()
    added, linked_ids = expand_with_links(retrieved, scores, meta)
    tm["expand"] = time.perf_counter() - _t
    context_chunks = retrieved + added
    if added:
        run.done("expand", "+%d linked chunks (%d total)" % (
            len(added), len(context_chunks)))
    else:
        run.skip("expand", "no wikilinks to follow")

    # --- Retrieval: Obsidian Smart Connections expansion ------------------
    # A second, independent expansion signal alongside wikilinks -- see
    # OBSIDIAN_RANK_ENABLED / expand_with_obsidian_signal above. Reuses
    # linked_ids' existing "(linked)" context label rather than adding a
    # separate label, since both mean the same thing to the model: this chunk
    # was pulled in via a relationship signal, not direct BM25/dense retrieval.
    if OBSIDIAN_RANK_ENABLED:
        run.running("obsidian")
        _t = time.perf_counter()
        obsidian_added = expand_with_obsidian_signal(retrieved, context_chunks, meta)
        tm["obsidian"] = time.perf_counter() - _t
        if obsidian_added:
            linked_ids |= {c.get("id") for c in obsidian_added
                          if c.get("id") is not None}
            context_chunks = context_chunks + obsidian_added
            run.done("obsidian", "+%d related chunks (%d total)" % (
                len(obsidian_added), len(context_chunks)))
        else:
            run.skip("obsidian", "no related notes found")

    # Reference URLs offered to the model for optional FETCH.
    ref_urls = collect_reference_urls(context_chunks, meta)

    # Default the scholar depth stage to SKIPPED before phase 1. The stage is
    # REGISTERED by the caller on `scholar_once and SCHOLAR_PASS2`, but it is
    # only DRIVEN inside the vault_ok branch below. Every other exit -- phase 1
    # INSUFFICIENT into the skill loop, _answer_no_web, or the phase-2 web route
    # -- would otherwise leave "Depth pass (queued)" frozen in the final tree,
    # which is the never-resolving stage this whole change exists to remove.
    # Ordering is load-bearing: running()/done() below overwrite this, and
    # by_key makes it a silent no-op whenever the stage was not registered.
    run.skip("depth", "vault answer needed no second pass")

    # === PHASE 1: vault-first answer attempt (ALWAYS) ====================
    # The thesis vault is consulted for an actual answer before any web access.
    # Strict grounding (VAULT_FIRST_SYSTEM): answer only from the snippets, or
    # emit INSUFFICIENT. This keeps the common vault-answerable case at ONE
    # grounded call. (A merged answer-or-route call was tried and rejected: it
    # made the small model leave the vault for the web on questions the vault
    # answers well -- slower and less grounded.)
    run.running("model")
    # Scholar deliberately falls through to VAULT_FIRST_SYSTEM, byte-identical
    # to the default persona: _is_insufficient and faithfulness_flags key off
    # this exact contract, and its depth is bought with retrieval breadth and a
    # second pass, never with a "be thorough, cite everything" instruction.
    system_prompt = PARENT_SYSTEM if persona == "parent" else VAULT_FIRST_SYSTEM
    system_prompt = _shape_for_intent(system_prompt, intent)
    vault_messages = [{"role": "system", "content": system_prompt}]
    _hist = history[-PHASE1_HISTORY_MSGS:] if history else []
    vault_messages.extend(_hist)
    # Fit BEFORE sending. The history is charged against the budget too, since
    # it is part of the same prefill: a long session was the original way this
    # overflowed. context_chunks here is post-link and post-obsidian expansion,
    # so this is the real, final list.
    _hist_text = "".join((m.get("content") or "") for m in _hist)
    context_chunks, _fit = fit_context(
        question, context_chunks, system_prompt + _hist_text, linked_ids)
    LAST_CTX_FIT.clear()
    LAST_CTX_FIT.update(_fit)
    if _fit["dropped"]:
        tm["ctx_dropped"] = _fit["dropped"]
    vault_messages.append(
        {"role": "user", "content": build_first_pass_user(
            question, context_chunks, linked_ids)}
    )
    _t = time.perf_counter()
    _reset_chat_stats()
    first = None
    streamed = False
    _live_sink = []
    if STREAM_PHASE1 and context_chunks and persona != "scholar":
        # Scholar never streams. vault_ok below commits the turn the moment
        # tokens are visible, so a streamed draft would render in full and then
        # visibly GROW as the second pass merged into it. The blocking chat_once
        # a few lines down is the existing fall-through; no new path is needed.
        #
        # Holdback streaming: the head is buffered until the INSUFFICIENT
        # decision is safe, then the answer renders live. A refusal aborts the
        # stream with nothing shown and falls through to the branches below,
        # exactly like the blocking path. The sink is kept so the grounding
        # footer can be appended after the stream ends.
        def _begin_stream():
            tm["ttft"] = time.perf_counter() - _t
            run.done("model", "streaming from vault")
            run.skip("web", "vault answer sufficient")
            s = run.begin_answer()
            _live_sink.append(s)
            return s

        text, streamed = chat_stream_holdback(
            model, vault_messages, _is_insufficient, _begin_stream)
        if text is not None:
            first = text.strip()
    if first is None:
        # Streaming off, no context chunks, or the stream failed before
        # anything was emitted: the original blocking call.
        streamed = False
        first = chat_once(model, vault_messages).strip()
    tm["phase1"] = time.perf_counter() - _t
    tm.update(LAST_CHAT_STATS)

    if first.startswith("[error]"):
        run.fail("model", first[:80])
        run.skip("web")
        run.finish()
        return "", []

    # Once tokens have streamed, the turn is committed to the vault answer:
    # decide() already ruled out a refusal on the text's decision window, and
    # re-branching to the web after visible output would corrupt the UI.
    vault_ok = streamed or (bool(context_chunks) and not _is_insufficient(first))

    # Case C: the vault answered. Stream the answer we already computed.
    if vault_ok:
        # Scholar second pass, FIRST statement here and before the flags. The
        # order is load-bearing: faithfulness_flags must grade the whole
        # model-authored body (draft + expansion) and nothing else, so the pass
        # runs before it and the citation footer is appended after it.
        _draft = first
        if persona == "scholar" and SCHOLAR_PASS2 and context_chunks:
            # Drive the "depth" stage so the extra ~30-45s generation is visible
            # instead of a frozen spinner. The stage is only registered when the
            # run was started with depth=True, and by_key makes it a no-op
            # otherwise, so this is safe on every other path.
            run.running("depth")
            first, context_chunks, linked_ids = _scholar_second_pass(
                model, question, bm25, chunks, meta, context_chunks,
                linked_ids, system_prompt, first, tm)
            run.done("depth", tm.get("scholar_error") or "")
        _flags = faithfulness_flags(
            first, format_vault_context(context_chunks, linked_ids))
        footer = ""
        if _flags:
            footer = ("\n\n[grounding check - not found in the retrieved "
                      "vault sources, treat as unverified: "
                      + ", ".join(_flags) + "]")
            first = first.rstrip() + footer
        if persona == "scholar":
            # After the flags, never before: see _scholar_citation_footer.
            # No-op unless BRAIN_SCHOLAR_CITE=1.
            first = _scholar_citation_footer(first, context_chunks)
        if streamed and _live_sink:
            # The answer body already rendered live; only the footer remains.
            if footer:
                _live_sink[0](footer)
        else:
            run.done("model", "%d tokens · from vault" % len(tokenize(first)))
            run.skip("web", "vault answer sufficient")
            sink = run.begin_answer()
            sink(first)
        return _finish(run, history, question, first,
                       vault_sources(context_chunks),
                       # Scholar only, and only when the answer actually grew:
                       # for every other persona this stays None, so the
                       # grounding footer keeps reaching history exactly as
                       # before.
                       history_text=(_draft if persona == "scholar"
                                     and _draft != first else None))

    # === SKILL LOOP: bounded tool selection before the fixed web route =====
    # Only reached because phase 1 said INSUFFICIENT. The existing phase-2 path
    # is a ONE-shot decision (FETCH or SEARCH, then answer); the skill loop can
    # instead re-search the vault with a rewritten query, open a named note, or
    # take two steps. Same single-line prefix protocol as ROUTE_SYSTEM, same
    # permissive fallback, because a 3B model cannot emit JSON tool calls.
    # Placed BEFORE the web-off return so it also helps offline turns (its
    # menu drops the web skills when web is off). Default OFF; when off nothing
    # here executes and the turn is byte-identical to the pre-skills pipeline.
    if SKILLS_ENABLED:
        run.running("skills")
        _t = time.perf_counter()
        _skill = _run_skill_loop(model, question, bm25, chunks, meta,
                                 web_enabled, persona, intent, history)
        tm["skills"] = time.perf_counter() - _t
        if _skill is not None and _skill.ok:
            run.done("model", "vault insufficient · answered by skill loop")
            run.done("skills", "%s · %d step(s)" % (_skill.skill, _skill.steps))
            run.skip("web", "skill loop answered")
            sink = run.begin_answer()
            sink(_skill.answer)
            return _finish(run, history, question, _skill.answer,
                           _skill.sources)
        run.skip("skills", _skill.reason if _skill is not None
                 else "skill loop unavailable")

    # Vault could not answer. With web off, answer best-effort without it.
    if not web_enabled:
        run.done("model", "vault insufficient")
        run.skip("web", "web search is off")
        return _answer_no_web(model, question, context_chunks, linked_ids,
                              history, run)

    # === PHASE 2: web routing (only because the vault was insufficient) ===
    # Q1: the routing call does NOT re-send the vault context. Phase 1 already
    # judged it insufficient, and the FETCH-vs-SEARCH choice only needs the
    # question plus any reference URLs from the notes. Dropping the ~1.5-2k
    # vault-context tokens here removes most of this call's prefill cost.
    route_user = "Question: " + question
    if ref_urls:
        route_user = ("Reference links found in the vault notes:\n"
                      + "\n".join(ref_urls) + "\n\n" + route_user)
    route_messages = [{"role": "system", "content": ROUTE_SYSTEM}]
    route_messages.extend(history)
    route_messages.append({"role": "user", "content": route_user})
    _t = time.perf_counter()
    route = chat_once(model, route_messages).strip()
    tm["route"] = time.perf_counter() - _t
    if route.startswith("[error]"):
        route = "SEARCH: " + question

    wants_fetch = route.startswith("FETCH:")
    # Anything that is not an explicit FETCH becomes a web SEARCH.
    first = route if wants_fetch or route.startswith("SEARCH:") \
        else ("SEARCH: " + question)
    run.done("model", "vault insufficient · going to web")

    # Case F: model asked to fetch one specific reference page.
    if wants_fetch:
        url = normalize_url(first[len("FETCH:"):])
        if url:
            run.running("web", "fetching " + url)
            _t = time.perf_counter()
            text = fetch_page(url)
            tm["web"] = tm.get("web", 0.0) + (time.perf_counter() - _t)
            if text:
                run.done("web", "fetched " + url)
                # Persist BEFORE generating. This branch used to answer and
                # return without writing anything, so a direct FETCH taught the
                # brain nothing at all. It has to happen here rather than in
                # _finish: _finish is not the only way out of this function
                # (the answer-cache hit and the semantic-router hit both return
                # earlier), so persistence hung off it would run only sometimes.
                if _persist_web(question, [(url, text)]):
                    run.done("web", "fetched " + url
                             + " · queued for review")
                doc_messages = [{"role": "system", "content": DOC_SYSTEM}]
                doc_messages.extend(history)
                doc_messages.append(
                    {"role": "user", "content": (
                        # Prompt-only consumer: fetch_page now returns up to
                        # STORE_CHAR_CAP, so cap explicitly or a 10k page walks
                        # straight into num_ctx.
                        "Reference page (" + url + "):\n" + text[:PAGE_CHAR_CAP]
                        + "\n\nQuestion: " + question)}
                )
                answer = chat_stream(model, doc_messages, sink=run.begin_answer())
                return _finish(run, history, question, answer, [url])
        # Direct fetch failed: fall back to a web search, then the vault.
        run.running("web", "direct fetch failed, searching")
        _t = time.perf_counter()
        excerpts = gather_web_context(question)
        tm["web"] = tm.get("web", 0.0) + (time.perf_counter() - _t)
        if excerpts:
            run.done("web", "DuckDuckGo · %d results%s"
                     % (len(excerpts),
                        " · queued for review" if LAST_WEB_QUEUED else ""))
            return _answer_from_excerpts(model, question, excerpts,
                                         history, run)
        run.skip("web", "no usable web results")
        return _answer_no_web(model, question, context_chunks, linked_ids,
                              history, run)

    # Case A: the model chose SEARCH (every non-FETCH route; web is on, since
    # the web-off case already returned above).
    query = first[len("SEARCH:"):].strip() or question
    run.running("web", "searching: " + query[:40])
    _t = time.perf_counter()
    excerpts = gather_web_context(query)
    tm["web"] = tm.get("web", 0.0) + (time.perf_counter() - _t)
    if not excerpts:
        run.skip("web", "no usable web results")
        return _answer_no_web(model, question, context_chunks, linked_ids,
                              history, run)
    run.done("web", "DuckDuckGo · %d results%s"
                     % (len(excerpts),
                        " · queued for review" if LAST_WEB_QUEUED else ""))
    return _answer_from_excerpts(model, question, excerpts, history, run)


# ---------------------------------------------------------------------------
# REPL.
# ---------------------------------------------------------------------------
HELP_TEXT = """Commands:
  /deep <q>      re-ask one question with the deep 7B model (slower, more grounded)
  /parent <q>    answer one question in parent mode (warm, plain-language,
                 Deaf-community-affirming guidance for parents of deaf children)
  /scholar <q>   answer one question in scholar mode (wider retrieval over the
                 language-deprivation literature; never cached)
  /model         re-pick the model
  /web on|off    allow / disallow web search
  /cache         show the knowledge-bank stats (entries, reuses, balance, size)
  /cache clear   delete all cached answers
  /cache on|off  enable / disable the answer cache
  /clear         forget the conversation context (the transcript is kept)
  /history [n]   scroll back through past Q&A, including previous runs
  /history all   show the whole saved transcript
  /history <t>   show only saved turns matching the text <t>
  /history clear delete the saved transcript
  /help          show this help
  /exit          quit (Ctrl-D and Ctrl-C also quit)
Just type a question to ask the vault."""


def parse_args(argv):
    model = None
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        else:
            i += 1
    return model


def _history_text(entries, ui):
    """Render saved turns into one scrollable block of text."""
    def c(text, *codes):
        if ui is None:
            return text
        return ui.c(text, *codes)

    p = ui.p if ui is not None else None
    out = []
    total = len(entries)
    for i, rec in enumerate(entries, 1):
        try:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.get("ts", 0)))
        except Exception:
            when = "?"
        head = "[%d/%d] %s · %s" % (i, total, when, rec.get("model") or "?")
        mode = rec.get("mode") or "fast"
        if mode and mode != "fast":
            head += " · " + mode
        out.append(c(head, p.MUTED_DARK) if p else head)
        out.append(c("you> ", p.ACCENT, p.BOLD) if p else "you> ")
        out[-1] += rec.get("q", "")
        out.append("")
        out.append(rec.get("a", "").rstrip())
        srcs = rec.get("sources") or []
        if srcs:
            line = "sources: " + _short_srcs(srcs)
            out.append(c(line, p.MUTED) if p else line)
        out.append("")
    return "\n".join(out)


def _page(text, ui):
    """Show `text` in a pager so it can actually be scrolled.

    Delegating to less buys arrow/PageUp scrolling, / search and a clean exit
    for free, which is the whole feature. -R keeps our color escapes, -F skips
    the pager when the output already fits one screen, -X leaves the text in the
    terminal's own scrollback afterwards instead of wiping it.

    Falls back to a plain print whenever paging is not appropriate (not a TTY,
    no color/cursor control, no pager on PATH, spawn failed) -- piping `brain`
    into a file must never block on an interactive pager.
    """
    plain = ui is None or getattr(ui, "plain", True)
    try:
        tty = sys.stdout.isatty()
    except Exception:
        tty = False
    if plain or not tty:
        print(text)
        return
    import shutil
    import subprocess
    pager = os.environ.get("PAGER") or "less"
    cmd = [pager]
    if os.path.basename(pager) == "less":
        cmd = [pager, "-R", "-F", "-X"]
    if shutil.which(cmd[0]) is None:
        print(text)
        return
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.communicate(text.encode("utf-8", "replace"))
    except KeyboardInterrupt:
        pass
    except Exception:
        print(text)
    finally:
        # REQUIRED, not tidiness. Mouse reporting is terminal state, not ours,
        # and less turns it back on behind our back; without this re-arm the
        # next trackpad scroll at the prompt comes back as arrow-key junk. See
        # the quiet_scroll() docstring in brain_input.py.
        try:
            quiet_scroll()
            restore_cursor()
        except Exception:
            pass


HISTORY_DEFAULT = 20   # turns shown by a bare /history


def _cmd_history(arg, ui):
    """Handle /history and its subcommands. `arg` is everything after the word."""
    if brain_transcript is None or not brain_transcript.enabled():
        print("The saved transcript is unavailable "
              "(could not open the cache directory).")
        return
    arg = (arg or "").strip()

    if arg == "clear":
        n = brain_transcript.clear()
        print("Deleted %d saved turn%s." % (n, "" if n == 1 else "s"))
        return

    limit = HISTORY_DEFAULT
    search = None
    if arg == "all":
        limit = None
    elif arg.isdigit():
        limit = max(1, int(arg))
    elif arg:
        # Anything else is a search term, and a search defaults to showing
        # every match rather than the last 20 of them.
        search = arg
        limit = None

    entries = brain_transcript.read(limit=limit, search=search)
    if not entries:
        if search:
            print("No saved turns match \"" + search + "\".")
        else:
            print("No saved turns yet. Ask a question and it will be here "
                  "next time, too.")
        return
    _page(_history_text(entries, ui), ui)


def _cache_stats_line(cache, cache_enabled):
    """Human-readable one-liner for /cache including the knowledge-bank balance."""
    try:
        st = cache.stats()
        kb = st.get("bytes", 0) / 1024.0
        return (
            "Knowledge bank: %d entries · %d reuses · balance %.1f · %.0f KB on disk · %s"
            % (st.get("count", 0), st.get("hits", 0), st.get("bank", 0.0), kb,
               "ON" if cache_enabled else "OFF")
        )
    except Exception as exc:
        return "[could not read cache stats: " + str(exc) + "]"


class _NullRun:
    """No-op stand-in for a UI Run, used only when brain_ui failed to import
    (ui is None). handle_question and the cache path call run.* unconditionally;
    without this they would crash the plain no-UI fallback with AttributeError.
    Stage methods do nothing (there is no tree to render); the answer sink and
    sources() write plainly to stdout so the fallback still produces output."""

    def running(self, *args, **kwargs):
        pass

    def done(self, *args, **kwargs):
        pass

    def skip(self, *args, **kwargs):
        pass

    def fail(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def _emit(self, piece):
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()

    def begin_answer(self):
        sys.stdout.write("\n")
        return self._emit

    def sources(self, srcs, vault_name=None):
        if srcs:
            sys.stdout.write(
                "\nsources: " + ", ".join(str(s) for s in srcs) + "\n")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    model = parse_args(sys.argv[1:])

    # Install line editing (arrow keys, history) and stop mouse-wheel /
    # trackpad scrolls leaking in as arrow keys. Importing readline is what
    # makes input() do line editing at all; without it the tty is in canonical
    # mode and an arrow key echoes as the literal junk "^[[A".
    if brain_input is not None:
        brain_input.setup(os.path.join(CACHE_DIR, ".brain_history"))
    else:
        quiet_scroll()

    # Arm the persistent Q&A transcript that /history reads. Deliberately here
    # and nowhere else: brain_web.py and advisor_server.py import this module
    # for handle_question, and never calling enable() is what keeps their
    # traffic out of your personal history.
    if brain_transcript is not None:
        brain_transcript.enable(os.path.join(CACHE_DIR, "transcript.jsonl"))

    ui = UI() if UI is not None else None

    if ui is None or ui.plain:
        print("Loading the vault index...")
    chunks, bm25, meta = load_index()
    index_mtime_loaded = index_mtime()

    if not model:
        model = choose_model()

    # Answer cache. Degrade gracefully if the module is missing or unusable.
    cache = None
    cache_enabled = True
    if AnswerCache is not None:
        try:
            cache = AnswerCache(CACHE_DIR)
        except Exception:
            cache = None
    if cache is None:
        cache_enabled = False

    # Semantic concept-router: widens coverage past the keyword cache's lexical
    # match (a long natural question rarely shares enough content words with a
    # short seeded phrase to hit the Jaccard cache, but it embeds close to the
    # right concept). Optional and self-degrading: if prime_cache/semantic_cache
    # or the embedding model are unavailable it stays None and the brain just
    # uses RAG. The margin gate (0.08) keeps it from serving an adjacent concept
    # on an ambiguous question; those fall through to a fresh grounded answer.
    semantic = None
    try:
        from semantic_cache import SemanticCache
        import prime_cache as _pc
        semantic = SemanticCache(
            _pc.KNOWLEDGE, host=OLLAMA_HOST, margin=0.08,
            cache_path=os.path.join(CACHE_DIR, ".semantic_vecs.json"))
        if not getattr(semantic, "ready", False):
            semantic = None
    except Exception:
        semantic = None

    # Welcome banner (Claude Code style) or a plain header.
    cache_count = 0
    if cache is not None:
        try:
            cache_count = cache.stats().get("count", 0)
        except Exception:
            cache_count = 0
    if ui is not None:
        ui.welcome(model, len(chunks), cache_count, VAULT_NAME, CACHE_DIR)
        if cache is None:
            ui.note("[answer cache unavailable, running without it]")
    else:
        print("Loaded %d chunks. Using model: %s" % (len(chunks), model))
        print("Type /help for commands. Ctrl-D or /exit to quit.")

    web_enabled = True
    history = []

    while True:
        try:
            line = (ui.read_prompt() if ui is not None
                    else (brain_input.read_line("\nyou> ")
                          if brain_input is not None else input("\nyou> ")))
        except EOFError:
            print("\nbye")
            break
        except KeyboardInterrupt:
            # Ctrl-C at the prompt abandons the half-typed line and re-prompts,
            # as every other REPL does. Ctrl-D (EOFError, above) is the way out.
            print("")
            continue

        question = line.strip()
        if not question:
            continue

        # Hot-reload: when the pickle has changed on disk (a promoted web note,
        # a rebuild, a research-agent run), swap the new index in before
        # answering. The banner no longer claims "web knowledge folded in":
        # web fetches now go to the review queue, so a reload here is not
        # evidence that anything web-sourced was learned.
        chunks, bm25, meta, index_mtime_loaded, _reloaded = \
            maybe_reload_index(chunks, bm25, meta, index_mtime_loaded)
        if _reloaded:
            note = "[index reloaded: %d chunks]" % len(chunks)
            if ui is not None:
                ui.note(note)
            else:
                print(note)

        # Per-question tier: default fast; "/deep <q>" escalates to the deep model.
        deep_once = False
        parent_once = False
        scholar_once = False

        # Commands.
        if question in ("/exit", "/quit"):
            print("bye")
            break
        if question == "/help":
            print(HELP_TEXT)
            continue
        if question in ("/clear",):
            history = []
            print("Conversation context cleared. The saved transcript is "
                  "untouched -- use /history clear to delete that.")
            continue
        if question == "/history" or question.startswith("/history "):
            _cmd_history(question[len("/history"):], ui)
            continue
        if question == "/model":
            model = choose_model()
            print("Using model: " + model)
            continue
        if question == "/web on":
            web_enabled = True
            print("Web search is ON.")
            continue
        if question == "/web off":
            web_enabled = False
            print("Web search is OFF.")
            continue
        if question == "/cache":
            if cache is None:
                print("The answer cache is unavailable.")
            else:
                print(_cache_stats_line(cache, cache_enabled))
            continue
        if question == "/cache clear":
            if cache is None:
                print("The answer cache is unavailable.")
            else:
                try:
                    n = cache.clear()
                    print("Cleared " + str(n) + " cached answers.")
                except Exception as exc:
                    print("[could not clear the cache: " + str(exc) + "]")
            continue
        if question == "/cache on":
            if cache is None:
                print("The answer cache is unavailable, cannot enable it.")
            else:
                cache_enabled = True
                print("Answer cache is ON.")
            continue
        if question == "/cache off":
            cache_enabled = False
            print("Answer cache is OFF.")
            continue
        if question == "/deep" or question.startswith("/deep "):
            rest = question[len("/deep"):].strip()
            if not rest:
                print("Usage: /deep <question>  (re-ask with the "
                      + DEEP_MODEL + " model for a more careful, grounded answer)")
                continue
            question = rest
            deep_once = True
        elif question == "/parent" or question.startswith("/parent "):
            rest = question[len("/parent"):].strip()
            if not rest:
                print("Usage: /parent <question>  (warm, plain-language "
                      "guidance for parents of deaf children)")
                continue
            question = rest
            parent_once = True
        elif question == "/scholar" or question.startswith("/scholar "):
            # MUST stay above the unknown-command catch-all below, or /scholar
            # is swallowed as an unknown command.
            rest = question[len("/scholar"):].strip()
            if not rest:
                print("Usage: /scholar <question>  (wider retrieval over the "
                      "language-deprivation literature for a research answer)")
                continue
            question = rest
            scholar_once = True
        elif question.startswith("/"):
            print("Unknown command. Type /help for the list.")
            continue

        # Per-question tier selection.
        q_model = DEEP_MODEL if deep_once else model

        # Tell the transcript which model and mode this turn belongs to, so
        # _finish (which is shared by every answer path and knows neither) does
        # not need two more parameters threaded through a dozen call sites.
        if brain_transcript is not None:
            brain_transcript.set_turn(
                q_model,
                "deep" if deep_once else
                "parent" if parent_once else
                "scholar" if scholar_once else "fast")

        # Start a run so the whole question renders like an ultracode pipeline.
        # When brain_ui is unavailable (ui is None), fall back to a no-op run so
        # the plain path still answers instead of crashing on run.* calls.
        run = (ui.start_run(q_model, depth=bool(scholar_once and SCHOLAR_PASS2))
               if ui is not None else _NullRun())

        # Question embedding captured from the semantic router's miss, reused by
        # hybrid dense retrieval so a fresh turn embeds the question ONCE.
        sem_qvec = None

        # Everything from here to the matching `finally` runs with the cursor
        # hidden by Run.__init__, so run.finish() -- which restores it and stops
        # the animator -- must be unconditional. It used to be repeated by hand
        # in each handler, which left the cache fast path below completely
        # unguarded: its `except Exception` clauses do not catch KeyboardInterrupt
        # (a BaseException), so a Ctrl-C during cache.lookup or the semantic
        # match unwound straight out of main() and exited with the cursor still
        # hidden, leaving the user's shell cursorless.
        try:
            # Fast path: serve from the answer cache before retrieval or a model
            # call. Deep (/deep) answers bypass the cache so they never collide
            # with a fast draft.
            if cache_enabled and cache is not None and not deep_once \
                    and not parent_once and not scholar_once:
                try:
                    run.running("cache")
                    hit = cache.lookup(question)
                except Exception:
                    hit = None
                if hit:
                    cached_answer = hit.get("answer", "") or ""
                    age = str(hit.get("created", "?"))
                    run.done("cache", "hit · served from cache (saved " + age + ")")
                    for k in ("bm25", "expand", "model", "web"):
                        run.skip(k, "cache hit")
                    sink = run.begin_answer()
                    sink(cached_answer)
                    run.sources(hit.get("sources") or [], vault_name=VAULT_NAME)
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": cached_answer})
                    trim_history(history)
                    # The cache fast path returns without ever reaching
                    # _finish, so it needs its own transcript append or a
                    # cached answer would be missing from /history entirely.
                    _log_turn(question, cached_answer, hit.get("sources") or [])
                    continue
                # Keyword cache missed. Try the semantic concept-router before
                # RAG: it serves a vetted concept answer when the question embeds
                # close to exactly one concept, otherwise returns None and we
                # compute fresh.
                if semantic is not None:
                    try:
                        m, sem_qvec = semantic.match_with_vec(question)
                    except Exception:
                        m, sem_qvec = None, None
                    if m and (m.get("answer") or "").strip():
                        sem_answer = m["answer"]
                        run.done("cache", "semantic hit · %s (cos %.2f)" % (
                            m.get("concept", ""), m.get("score", 0.0)))
                        for k in ("bm25", "expand", "model", "web"):
                            run.skip(k, "semantic hit")
                        sink = run.begin_answer()
                        sink(sem_answer)
                        run.sources(m.get("sources") or [], vault_name=VAULT_NAME)
                        history.append({"role": "user", "content": question})
                        history.append({"role": "assistant", "content": sem_answer})
                        trim_history(history)
                        _log_turn(question, sem_answer, m.get("sources") or [])
                        continue
                run.done("cache", "miss · computing fresh")
            else:
                run.skip("cache", "bypassed for /deep" if deep_once
                         else ("bypassed for /parent" if parent_once
                               else ("bypassed for /scholar" if scholar_once
                                     else "cache disabled")))

            try:
                answer, sources = handle_question(
                    q_model, question, history, bm25, chunks, meta, web_enabled, run,
                    persona=("parent" if parent_once
                             else "scholar" if scholar_once else "default"),
                    qvec=sem_qvec,
                    qvec_model=(getattr(semantic, "model", None)
                                if semantic is not None else None),
                )
            except KeyboardInterrupt:
                # Halt the animator BEFORE printing. The `finally` below would
                # do it microseconds later anyway, but only after the message is
                # already on screen, leaving a (vanishingly small) window for a
                # spinner frame to rewind over it. finish() is idempotent.
                run.finish()
                print("\n[interrupted, back to the prompt]")
                continue
            except Exception as exc:
                run.finish()
                print("[something went wrong handling that: " + str(exc) + "]")
                continue
        finally:
            run.finish()

        # Store fresh answers in the cache (triggers the storage-optimization
        # pass and compounds the knowledge bank).
        # A scholar answer is NEVER cached, and this guard is not optional.
        # Without it a long two-pass answer is stored under the user's question
        # and later served verbatim to any plain question that clears
        # FUZZY_THRESHOLD 0.70 -- which is exactly how the 286-word cached
        # answer already sitting in the cache came to exist. Its latency is
        # deliberately never amortized.
        if cache_enabled and cache is not None and answer and not deep_once \
                and not parent_once and not scholar_once:
            try:
                cache.store(question, answer, sources, model)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        # Last line of defence for the cursor. brain_input registers an atexit
        # for this too, but atexit does not run on SIGTERM/SIGHUP and this costs
        # nothing on the paths where it does.
        restore_cursor()
