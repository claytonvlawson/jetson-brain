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
        return ChatOpenAI(model=_cloud_model_name(), api_key=key,
                           base_url=GROQ_BASE_URL, temperature=0.2,
                           default_headers=_ACCEPT_ENCODING_NO_BROTLI,
                           timeout=_cloud_timeout_s(),
                           # answer()'s own loop already retries with a fresh
                           # client on a bad attempt; the SDK's built-in retry
                           # would just re-extend a stall inside one attempt.
                           max_retries=0)


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


def answer(question, persona="default", deep=False):
    """One-shot: route, retrieve via brain.py, generate, return text.

    This is a convenience entry point for the new LangChain path only. It
    does not replace brain.handle_question() -- the CLI/UI/eval harness keep
    calling that directly and are unaffected by anything in this file."""
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "langchain-core is not installed. `pip install -r requirements.txt`.")

    chunks, bm25, meta = brain.load_index()
    tier = choose_tier(question, persona=persona, deep=deep)

    hits = vault_tools_local.search_vault(question, bm25, chunks, meta,
                                           persona=persona)
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

    if tier == "cloud-speed":
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
            # through to local if BOTH attempts come back empty.
            for _attempt in range(2):
                result = make_cloud_chat_model().invoke(lc_in)
                if result.content and result.content.strip():
                    return result.content
        except Exception:
            pass
        # Groq unreachable/misconfigured, or returned empty twice in a row --
        # fall back to local rather than surface a cloud error (or blank
        # answer) for a question the 3B/7B can already answer.
        tier = "local-deep" if deep else "local-fast"

    model_name = brain.DEEP_MODEL if tier == "local-deep" else brain.DEFAULT_MODEL
    return brain.chat_once(model_name, messages)


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
