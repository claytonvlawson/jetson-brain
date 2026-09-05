"""LangChain orchestration layer over brain.py, plus an optional cheap/fast
cloud tier (Groq, OpenAI-compatible) for simple turns.

This module is ADDITIVE: it imports brain.py and calls its public functions
(chat_once, chat_stream, retrieve_context, load_index) exactly as every other
caller does. It does not reimplement or modify retrieval, generation, or the
handle_question()/retrieve_context() signatures eval_brain.py and
eval_retrieval.py depend on -- those two harnesses never import this file.

Env vars (matching brain.py's BRAIN_* convention):
    GROQ_API_KEY      Required for the cloud tier. Absent -> auto mode falls
                       back to local silently; cloud_speed mode raises.
    BRAIN_CLOUD_MODEL Groq model id. Default: "openai/gpt-oss-20b".
    BRAIN_ROUTE_MODE  "auto" (default) | "local_only" | "cloud_speed".
    BRAIN_PRIVACY_SCAN "0" (default) | "1" -- gate a pre-response privacy
                       scan of the retrieved context (see privacy_scan below).
    BRAIN_TEMPERATURE / BRAIN_SEED  When exported (brain.py's own eval
                       convention) they ALSO apply to the cloud tier, so the
                       two generators can be A/B'd on equal footing. Unset,
                       the cloud tier keeps its 0.2 / unseeded defaults.

Measuring the two tiers: eval_route.py freezes one retrieval per question
via build_messages() and runs generate(strict=True) per arm, so a cloud
failure is recorded as "cloud-error" instead of silently becoming a local
answer. answer() itself keeps the silent fall-back -- that is the right
behavior for a REPL, and the wrong one for a measurement.

Usage (library):
    from langchain_agent import answer
    text = answer("When did Gallaudet found the Clerc Center?")

Usage (CLI, same shape as ThesisVault's vault_agent.py):
    python3 langchain_agent.py "When did Gallaudet found the Clerc Center?"
    python3 langchain_agent.py --interactive
    python3 langchain_agent.py --interactive --persona parent
"""

import argparse
import os
import sys
import time

import brain
import vault_tools_local

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# llama-3.1-8b-instant was retired from Groq's catalog; gpt-oss-20b is the
# current fast/cheap chat model there (checked against GET /v1/models on
# 2026-09-02). BRAIN_CLOUD_MODEL overrides this if Groq's lineup moves again.
DEFAULT_CLOUD_MODEL = "openai/gpt-oss-20b"

# The Jetson's installed Brotli 1.0.9 (Debian's python3-brotli, /usr/lib, not
# pip-managed) exposes decompressor.process() without httpx2's expected
# output_buffer_limit kwarg, so any brotli-encoded response crashes with
# "TypeError: process() takes no keyword arguments" deep in the transport
# layer -- surfaced to callers as a generic openai.APIConnectionError, which
# looks like a network failure and is not one. Declining brotli (and zstd,
# same class of risk) up front is simpler and safer than patching or
# uninstalling a system package outside pip's control.
_ACCEPT_ENCODING_NO_BROTLI = {"Accept-Encoding": "gzip, deflate"}

# Measured live: Groq's on_demand service tier occasionally stalls a request
# for 40+ seconds with no error at all (not a slow answer -- a hang), while a
# normal call completes in well under a second. The whole point of the cloud
# tier is speed, so a call that isn't fast should fail fast and let answer()
# fall back to local rather than leave the REPL stuck. BRAIN_CLOUD_TIMEOUT_S
# overrides for a slower/less reliable link.
def _cloud_timeout_s():
    try:
        return float(os.environ.get("BRAIN_CLOUD_TIMEOUT_S") or 8)
    except ValueError:
        return 8.0


def _cloud_temperature():
    try:
        return float(os.environ.get("BRAIN_TEMPERATURE") or 0.2)
    except ValueError:
        return 0.2


def _route_mode():
    return (os.environ.get("BRAIN_ROUTE_MODE") or "auto").strip().lower()


def _cloud_model_name():
    return os.environ.get("BRAIN_CLOUD_MODEL") or DEFAULT_CLOUD_MODEL


def _groq_key():
    return os.environ.get("GROQ_API_KEY")


def _lc_messages_to_ollama(messages):
    """Convert a list of langchain_core BaseMessage into brain.chat_once's
    plain [{"role", "content"}] shape -- brain.py's own request format, so
    this is the one place the two message conventions meet."""
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    out = []
    for m in messages:
        role = role_map.get(getattr(m, "type", "human"), "user")
        out.append({"role": role, "content": m.content})
    return out


if _LANGCHAIN_AVAILABLE:

    class BrainChatModel(BaseChatModel):
        """Wraps brain.chat_once()/chat_stream() for one model tier (fast or
        deep). Does not touch Ollama directly -- every call goes through
        brain.py's existing client+REST fallback, so tier behavior (timeouts,
        keep_alive, num_ctx, etc.) stays exactly what brain.py already ships."""

        model_name: str = brain.DEFAULT_MODEL

        @property
        def _llm_type(self):
            return "brain-local-" + self.model_name

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            ollama_messages = _lc_messages_to_ollama(messages)
            text = brain.chat_once(self.model_name, ollama_messages)
            gen = ChatGeneration(message=AIMessage(content=text))
            return ChatResult(generations=[gen])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            ollama_messages = _lc_messages_to_ollama(messages)

            def _sink(piece):
                if run_manager is not None:
                    run_manager.on_llm_new_token(piece)

            text = brain.chat_stream(self.model_name, ollama_messages, sink=_sink)
            yield ChatGeneration(message=AIMessage(content=text))

    def available_chat_models():
        """Installed Ollama models minus embedding models -- brain.list_models()
        returns everything `ollama list` shows, including nomic-embed-text,
        which cannot serve a /api/chat request. Any chat-capable model this
        returns is safe to hand to make_local_chat_model(model=...)."""
        return [m for m in brain.list_models() if "embed" not in m.lower()]

    def make_local_chat_model(tier="fast", model=None):
        """A BrainChatModel for one Ollama model.

        Pass `model` (any name from available_chat_models(), or any tag
        `ollama pull`ed on the device -- llama3.2:3b, qwen2.5:7b,
        qwen2.5:0.5b, or a future addition) to address it directly; every
        per-model behavior -- context window, keep_alive, memory headroom --
        comes from brain.py's own _gen_options()/_keep_alive()/_before_chat()
        dispatch inside chat_once()/chat_stream(), so a new model needs no
        change here, only (optionally) a case in that dispatch if it wants
        tuning different from the fast-tier default it inherits today.

        `tier` ("fast" | "deep") is kept as the zero-config default so
        existing callers (answer(), choose_tier()) are unaffected; it is
        ignored when `model` is given."""
        if model is not None:
            return BrainChatModel(model_name=model)
        resolved = brain.DEEP_MODEL if tier == "deep" else brain.DEFAULT_MODEL
        return BrainChatModel(model_name=resolved)

    def make_cloud_chat_model():
        """A Groq-backed ChatModel via LangChain's OpenAI-compatible client.
        Import is local (not at module top) so a Jetson without
        langchain-openai installed can still use local-only mode."""
        from langchain_openai import ChatOpenAI

        key = _groq_key()
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set -- cloud tier unavailable.")
        # Sampling knobs follow brain.py's own eval convention so the two
        # tiers can be A/B'd on equal footing: BRAIN_TEMPERATURE overrides the
        # 0.2 default and BRAIN_SEED pins the sampler (Groq honors OpenAI's
        # `seed`, best-effort). Unset -> exactly the previous request body.
        kwargs = {}
        if os.environ.get("BRAIN_SEED"):
            try:
                kwargs["seed"] = int(os.environ["BRAIN_SEED"])
            except ValueError:
                pass
        return ChatOpenAI(model=_cloud_model_name(), api_key=key,
                           base_url=GROQ_BASE_URL,
                           temperature=_cloud_temperature(),
                           default_headers=_ACCEPT_ENCODING_NO_BROTLI,
                           timeout=_cloud_timeout_s(),
                           # answer()'s own loop already retries with a fresh
                           # client on a bad attempt; the SDK's built-in retry
                           # would just re-extend a stall inside one attempt.
                           max_retries=0, **kwargs)


def _looks_simple(question, persona):
    """Heuristic only -- no extra model call, so it costs nothing on the
    latency path it exists to shorten. Anything that wants citation
    discipline or the deep model stays local; short single-fact questions
    with no such signal go to the cloud speed tier."""
    if persona in ("parent", "scholar"):
        return False
    q = question.strip()
    if len(q) > 220:
        return False
    if q.lower().startswith("/deep"):
        return False
    return True


def choose_tier(question, persona="default", deep=False):
    """Return one of "local-fast", "local-deep", "cloud-speed" for this turn.

    BRAIN_ROUTE_MODE overrides the heuristic entirely: "local_only" never
    routes to the cloud, "cloud_speed" always tries the cloud (falling back
    to local-fast if the key is missing). "auto" (default) uses
    _looks_simple() and always keeps deep/parent/scholar turns local, since
    those are exactly the turns that want the citation tools in
    vault_tools_local.py and the persona-tuned retrieval boosts in
    brain.retrieve_context()."""
    mode = _route_mode()
    if deep:
        return "local-deep"
    if mode == "local_only":
        return "local-fast"
    if mode == "cloud_speed":
        return "cloud-speed" if _groq_key() else "local-fast"
    # auto
    if _looks_simple(question, persona) and _groq_key():
        return "cloud-speed"
    return "local-fast"


def build_messages(question, persona="default", index=None, k=8):
    """Retrieve ONCE and build the exact prompt both tiers share.

    Returns (messages, hits). `index` is the (chunks, bm25, meta) tuple from
    brain.load_index(); pass it in from a loop so the index is not reloaded
    per call. Retrieval goes through vault_tools_local.search_vault(), a thin
    adapter over brain.retrieve_context(), so it is identical whichever tier
    generates -- and because it runs here, before any tier is chosen, an eval
    can freeze one context per question and hand the same messages to both
    generators. Retrieval can otherwise differ between two calls: HyDE inside
    retrieve_context() is itself a 3B generation."""
    if index is None:
        index = brain.load_index()
    chunks, bm25, meta = index

    hits = vault_tools_local.search_vault(question, bm25, chunks, meta,
                                           k=k, persona=persona)
    context = "\n\n".join(
        "[%s] %s" % (h["source"], h["text"]) for h in hits)

    system = brain.PARENT_SYSTEM if persona == "parent" else brain.VAULT_FIRST_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Context:\n%s\n\nQuestion: %s" % (context, question)},
    ]

    if os.environ.get("BRAIN_PRIVACY_SCAN", "0") == "1":
        from privacy_scan_local import scan_text
        flags = scan_text(context)
        if flags:
            messages[0]["content"] += (
                "\n\nSome retrieved context was flagged by the privacy "
                "scanner and must not be repeated verbatim: %r" % flags)
    return messages, hits


def _content_text(result):
    """langchain's AIMessage.content is a str for the chat-completions API,
    but newer versions can return a list of content blocks. Normalize to the
    visible text so an answer is never judged against a repr of a list."""
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text"):
            parts.append(block.get("text") or "")
    return "".join(parts)


def _retry_after_s(exc, default=2.0):
    """Seconds to wait after a rate-limit error. Groq puts the number in the
    error body ("Please try again in 7.35s") and in a Retry-After header the
    SDK exposes on exc.response; fall back to `default`."""
    import re
    resp = getattr(exc, "response", None)
    try:
        ra = resp.headers.get("retry-after") if resp is not None else None
        if ra:
            return max(1.0, float(ra))
    except Exception:
        pass
    m = re.search(r"try again in ([0-9.]+)\s*(ms|s)", str(exc))
    if m:
        v = float(m.group(1))
        return max(1.0, v / 1000.0 if m.group(2) == "ms" else v)
    return default


def _is_rate_limit(exc):
    name = type(exc).__name__
    return name == "RateLimitError" or "429" in str(exc) or "rate limit" in str(exc).lower()


def generate(messages, tier, strict=False, fallback="local-fast", rate_limit_wait_s=0.0):
    """Run ONE generation on an already-built prompt and say who served it.

    Returns a dict:
        text        the answer ("" on a strict-mode error)
        served_by   "cloud" | "local-fast" | "local-deep" | "cloud-error" | "local-error"
        model       the model id that actually produced `text`
        seconds     wall clock for this tier's attempt(s)
        attempts    cloud attempts made (0 for a local tier)
        error       None, or a one-line reason
        finish_reason / usage   cloud only, when the provider reports them

    strict=False is production behavior, unchanged: a cloud call that raises,
    times out, or comes back empty twice falls back to `fallback` and the
    caller gets a local answer. `rate_limit_wait_s` (default 0 = never wait)
    lets a MEASUREMENT sit out Groq 429s for up to that many seconds total,
    honoring Retry-After, so a rate limit is recorded as `rate_limited` waits
    instead of masquerading as a generation failure; a REPL should keep 0 and
    fall back to local at once. strict=True is for measurement: a cloud
    failure returns served_by="cloud-error" with text "" and NEVER touches the
    local model, so an eval can never mislabel a local answer as a cloud one
    -- the silent fallback that made the old answer() unmeasurable."""
    t0 = time.perf_counter()
    if tier == "cloud-speed":
        attempts = 0
        err = None
        meta = {}
        waited = 0.0
        rate_limited = 0
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_in = [SystemMessage(content=messages[0]["content"]),
                     HumanMessage(content=messages[1]["content"])]
            # Measured live against Groq's on_demand service tier: an
            # occasional call comes back with finish_reason "stop" and a
            # genuinely empty .content -- not an exception, so the except
            # below never sees it, and a naive caller would hand the user a
            # blank answer. One retry (a fresh ChatOpenAI instance, not the
            # same one) clears it every time it was reproduced; only fall
            # through if BOTH attempts come back empty.
            empties = 0
            while empties < 2:
                attempts += 1
                try:
                    result = make_cloud_chat_model().invoke(lc_in)
                except Exception as e:
                    if _is_rate_limit(e) and waited < rate_limit_wait_s:
                        wait = min(_retry_after_s(e), rate_limit_wait_s - waited)
                        rate_limited += 1
                        waited += wait
                        time.sleep(wait)
                        continue
                    raise
                rm = getattr(result, "response_metadata", None) or {}
                meta = {"finish_reason": rm.get("finish_reason"),
                        "usage": getattr(result, "usage_metadata", None) or {}}
                text = _content_text(result)
                if text and text.strip():
                    out = {"text": text, "served_by": "cloud",
                           "model": _cloud_model_name(),
                           # generation time only: rate-limit waits are
                           # reported separately, not folded into latency
                           "seconds": time.perf_counter() - t0 - waited,
                           "attempts": attempts, "error": None,
                           "rate_limited": rate_limited, "waited_s": round(waited, 2)}
                    out.update(meta)
                    return out
                empties += 1
            err = "empty content on %d attempt(s)" % empties
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, str(e)[:200])
        if strict:
            out = {"text": "", "served_by": "cloud-error",
                   "model": _cloud_model_name(),
                   "seconds": time.perf_counter() - t0 - waited,
                   "attempts": attempts, "error": err,
                   "rate_limited": rate_limited, "waited_s": round(waited, 2)}
            out.update(meta)
            return out
        # Groq unreachable/misconfigured, or returned empty twice in a row --
        # fall back to local rather than surface a cloud error (or blank
        # answer) for a question the 3B/7B can already answer.
        tier = fallback
        t0 = time.perf_counter()

    model_name = brain.DEEP_MODEL if tier == "local-deep" else brain.DEFAULT_MODEL
    text = brain.chat_once(model_name, messages)
    failed = (text or "").startswith("[error]")
    return {"text": text, "served_by": (tier + "-error") if failed else tier,
            "model": model_name, "seconds": time.perf_counter() - t0,
            "attempts": 0, "error": text[:200] if failed else None}


def answer(question, persona="default", deep=False):
    """One-shot: route, retrieve via brain.py, generate, return text.

    This is a convenience entry point for the new LangChain path only. It
    does not replace brain.handle_question() -- the CLI/UI/eval harness keep
    calling that directly and are unaffected by anything in this file.
    Composes build_messages() + generate(strict=False), so behavior is what
    it always was: retrieval first, then the chosen tier, with a silent
    fall-back to local when the cloud tier fails."""
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "langchain-core is not installed. `pip install -r requirements.txt`.")

    index = brain.load_index()
    tier = choose_tier(question, persona=persona, deep=deep)
    messages, _hits = build_messages(question, persona=persona, index=index)
    fallback = "local-deep" if deep else "local-fast"
    return generate(messages, tier, strict=False, fallback=fallback)["text"]


def interactive(persona="default"):
    """A REPL that just takes typed text -- same shape as ThesisVault's
    vault_agent.interactive(): a bare `> ` prompt, no Python quoting, Ctrl+C/
    EOF to exit. `/deep <question>` forces the deep tier for that one turn,
    matching brain.py's own `/deep` convention elsewhere in this repo."""
    groq_status = "loaded" if _groq_key() else "not set (local only)"
    print("Jetson Brain (LangChain) -- persona=%s, GROQ_API_KEY=%s. "
          "Ctrl+C to exit.\n" % (persona, groq_status))
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        deep = False
        if q.lower().startswith("/deep "):
            deep = True
            q = q[len("/deep "):].strip()
        tier = choose_tier(q, persona=persona, deep=deep)
        print("[route] %s" % tier, file=sys.stderr)
        print()
        print(answer(q, persona=persona, deep=deep))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--persona", default="default",
                     choices=["default", "parent", "scholar"])
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        interactive(persona=args.persona)
        return
    if args.question:
        print(answer(args.question, persona=args.persona, deep=args.deep))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
