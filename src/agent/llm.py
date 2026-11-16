"""LLM client — one place to talk to the model, provider-agnostic (OpenAI-compatible API).

Default: local Ollama (free, no key):  base_url http://localhost:11434/v1
Optional: OpenRouter or any OpenAI-compatible provider via environment variables (e.g. in a .env file
that is NEVER committed):
    LLM_BASE_URL=https://openrouter.ai/api/v1
    LLM_API_KEY=...

Reasoning ("thinking"): only some models (e.g. qwen3) have a built-in thinking mode.
  think=True  → let the model think (slower, often more accurate)
  think=False → switch it off (Ollama: reasoning_effort="none"; the "/no_think" text switch did NOT work —
                measured 14.5 s vs 1.4 s for the same question)
  think=None  → model default (models without thinking just answer)
"""

import os
import re
import time

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"
THINKING_MODELS = ("qwen3",)


def client() -> OpenAI:
    return OpenAI(base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
                  api_key=os.getenv("LLM_API_KEY", "ollama"))


def chat(model: str, messages: list, think: bool | None = None, **kw) -> tuple[str, float]:
    """Return (answer text, seconds). Temperature 0 for reproducibility."""
    if think is False and model.startswith(THINKING_MODELS):
        kw["reasoning_effort"] = "none"
    t0 = time.time()
    r = client().chat.completions.create(model=model, messages=messages, temperature=0, **kw)
    text = r.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text, time.time() - t0
