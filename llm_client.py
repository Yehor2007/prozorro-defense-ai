"""
llm_client.py
Single point of LLM access for rag_query.py and investigations.py — reads
LLM_PROVIDER from the environment ("ollama" or "openai") and dispatches
chat calls accordingly, so the rest of the codebase never imports `ollama`
or `openai` directly. Configured entirely via .env:

    LLM_PROVIDER=ollama          # default — local, free, needs `ollama serve`
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...        # required if LLM_PROVIDER=openai
    OPENAI_MODEL=gpt-4o-mini     # optional, this is already the default

Why a plain `openai` client rather than LangChain: every call site in
this project needs exactly one thing — send a system+user message pair,
get back the text — a single HTTP request either way. LangChain adds an
abstraction layer, its own dependency tree, and a different calling
convention for a capability the `openai` package already exposes
directly; this project has no chains, agents, or LLM-orchestrated
retrieval to justify that (retrieval here is deterministic Python code in
rag_query.py/investigations.py, not something an LLM framework drives).
If a future need genuinely calls for LangChain, this is the one file that
would change — every caller already goes through chat()/health_check(),
never `ollama`/`openai` directly.
"""

import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_openai_client = None


def chat(messages: list[dict], model: str | None = None) -> str:
    """`model` is only meaningful for the ollama provider (an Ollama model
    tag, e.g. "qwen2.5:14b-instruct"). The openai provider always uses
    OPENAI_MODEL from the environment — the point of choosing it is a
    specific hosted model, not swapping tags at runtime."""
    if LLM_PROVIDER == "openai":
        return _chat_openai(messages)
    if LLM_PROVIDER == "ollama":
        return _chat_ollama(messages, model)
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} — must be 'ollama' or 'openai'")


def _chat_ollama(messages: list[dict], model: str | None) -> str:
    import ollama
    if not model:
        raise ValueError("model is required when LLM_PROVIDER=ollama")
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def _chat_openai(messages: list[dict]) -> str:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set — add it to .env")
        _openai_client = OpenAI(api_key=api_key)
    response = _openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return response.choices[0].message.content


def health_check() -> None:
    """Fail fast at startup with a clear message rather than a raw
    exception on the first user query — same reasoning as this project's
    existing ollama.list() startup check."""
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env")
        return
    if LLM_PROVIDER == "ollama":
        import ollama
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama ({e}). Run `ollama serve` in another terminal, "
                "and confirm the model is pulled with `ollama list`."
            ) from e
        return
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} — must be 'ollama' or 'openai'")
