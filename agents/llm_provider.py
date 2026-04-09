"""
LLM Provider Abstraction Layer.
Supports four free backends:
  1. Groq        – free-tier Llama models (needs API key from console.groq.com)
  2. Gemini      – free-tier Google Gemini Flash (needs API key from aistudio.google.com)
  3. HuggingFace – free Inference API, large models on HF servers
                   (needs free token from huggingface.co/settings/tokens)
  4. Ollama      – powerful local LLMs, NO API key required
                   (install from ollama.com, then: ollama pull llama3.1)
"""

from __future__ import annotations


# ──────────────────────────── base class ────────────────────────────
class LLMProvider:
    """Unified interface for all LLM backends."""

    # Subclasses can override these to signal their token budget limits
    max_input_words: int = 5000    # max words to feed as input
    max_output_tokens: int = 3000  # max tokens to request as output

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        raise NotImplementedError


# ──────────────────────────── Groq ──────────────────────────────────
class GroqProvider(LLMProvider):
    """Groq provider with automatic fallback across all available models.

    If the primary model fails (rate limit, overload, etc.), it tries the
    next model in the chain immediately — no waiting, no wasted time.

    Fallback order (best balance of speed, context & daily quota):
      1. llama-3.3-70b-versatile        – 12K ctx, 100K/day
      2. llama-3.1-8b-instant           – 6K ctx, 500K/day  (fast)
      3. meta-llama/llama-4-scout-17b-16e-instruct – 30K ctx, 500K/day
      4. groq/compound                  – 70K ctx, No limit
      5. groq/compound-mini             – 70K ctx, No limit
      6. qwen/qwen3-32b                 – 6K ctx, 500K/day
      7. moonshotai/kimi-k2-instruct    – 10K ctx, 300K/day
      8. openai/gpt-oss-120b            – 8K ctx, 200K/day
      9. allam-2-7b                     – 6K ctx, 500K/day
    """

    FALLBACK_CHAIN = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "groq/compound",
        "groq/compound-mini",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct",
        "openai/gpt-oss-120b",
        "allam-2-7b",
    ]

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model
        # Build fallback list: primary model first, then the rest in order
        self._models = [self.model]
        for m in self.FALLBACK_CHAIN:
            if m != self.model:
                self._models.append(m)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        last_error = None
        for model in self._models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content
                if text and text.strip():
                    # Stick with whichever model worked for future calls
                    if model != self._models[0]:
                        self._models.remove(model)
                        self._models.insert(0, model)
                    return text.strip()
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"All Groq models failed. Last error: {last_error}. "
            f"Check your API key or try again in a minute."
        )


# ──────────────────────────── Gemini ────────────────────────────────
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return response.text


# ──────────────────────────── HuggingFace Inference API ─────────────
class HuggingFaceProvider(LLMProvider):
    """Uses HuggingFace Inference API — free tier, runs powerful models
    on HuggingFace servers.  No local GPU required.

    Includes automatic retry with backoff and fallback model support
    to handle cold starts, rate limits, and model loading delays.
    """

    # HuggingFace free-tier limits: ~8k total context, ~1k output
    max_input_words: int = 2000
    max_output_tokens: int = 1024

    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        from huggingface_hub import InferenceClient

        if not api_key or not api_key.strip():
            raise RuntimeError(
                "HuggingFace API key is empty. "
                "Enter your token in the sidebar or set HF_API_KEY in .env "
                "(free at huggingface.co/settings/tokens)."
            )
        self.api_key = api_key.strip()
        self.client = InferenceClient(token=self.api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Single direct call — no silent retries, no fallback models.
        Fails fast with a clear error message."""
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text.strip()
            raise RuntimeError(
                f"HuggingFace model '{self.model}' returned an empty response. "
                f"Try Groq or Gemini instead."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            err = str(exc)
            # Map common errors to clear messages
            err_lower = err.lower()
            if "api_key" in err_lower or "token" in err_lower or "401" in err:
                raise RuntimeError(
                    "HuggingFace API key is invalid. "
                    "Get a free token at huggingface.co/settings/tokens."
                ) from exc
            if "402" in err or "payment" in err_lower or "quota" in err_lower or "credit" in err_lower:
                raise RuntimeError(
                    "HuggingFace free credits exhausted. "
                    "Switch to Groq or Gemini in the sidebar (both free)."
                ) from exc
            if "loading" in err_lower or "503" in err:
                raise RuntimeError(
                    f"HuggingFace model '{self.model}' is loading (cold start). "
                    f"Wait 30 seconds and try again, or switch provider."
                ) from exc
            raise RuntimeError(
                f"HuggingFace error: {err[:200]}. "
                f"Try Groq or Gemini instead."
            ) from exc


# ──────────────────────────── Ollama (local, no API key) ────────────
class OllamaProvider(LLMProvider):
    """Uses Ollama to run powerful LLMs locally — NO API key required.
    Install from ollama.com, pull a model (e.g. ollama pull llama3.1),
    and make sure the Ollama server is running."""

    max_input_words: int = 2500
    max_output_tokens: int = 1500

    def __init__(self, model: str = "llama3.1"):
        import requests

        self._base_url = "http://localhost:11434"
        self.model = model
        # Verify Ollama is running
        try:
            requests.get(f"{self._base_url}/api/tags", timeout=5)
        except Exception:
            raise ConnectionError(
                "Ollama is not running. Install from ollama.com and start it, "
                "then pull a model with: ollama pull llama3.1"
            )

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        import requests

        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": max(temperature, 0.01),
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
            timeout=600,
        )
        if response.status_code != 200:
            error_msg = ""
            try:
                error_msg = response.json().get("error", "")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama error ({self.model}): {error_msg or response.text[:200]}"
            )
        return response.json()["response"]


# ──────────────────────────── Factory ───────────────────────────────
def get_llm_provider(provider_name: str, api_key: str = "", model: str = "") -> LLMProvider:
    """Create a fresh LLM provider instance.

    NOT cached — providers are lightweight (no model loading), and caching
    would leak API keys in Streamlit's server-wide cache across sessions.
    The heavy embedding model is cached separately in agents/embeddings.py.
    """
    if provider_name == "groq":
        return GroqProvider(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    elif provider_name == "gemini":
        return GeminiProvider(api_key=api_key, model=model or "gemini-2.0-flash")
    elif provider_name == "huggingface":
        return HuggingFaceProvider(api_key=api_key, model=model or "Qwen/Qwen2.5-7B-Instruct")
    else:
        return OllamaProvider(model=model or "llama3")
