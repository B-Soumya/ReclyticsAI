"""
Summarizer Agent — Comprehensive overall summary.

Reads the full article text across all pages and produces a single
in-depth summary covering objectives, methods, findings, math, and impact.
"""

from __future__ import annotations

from agents.llm_provider import LLMProvider


SUMMARY_PROMPT = """You are an expert research analyst. Read the ENTIRE article text below
and write a comprehensive, well-structured summary covering ALL of the following:

1. **Research Objective** — What problem does this paper address and why does it matter?
2. **Background & Context** — How does this work fit into the broader field?
3. **Methodology** — What approach, techniques, datasets, or experimental setup do the authors use?
4. **Key Findings & Results** — What are the main outcomes? Include specific numbers, metrics, and comparisons.
5. **Mathematical / Quantitative Framework** — What equations, statistical methods, or quantitative
   techniques are used? Why were they chosen? What assumptions do they rely on?
   If the article has no equations, analyze the quantitative reasoning and data analysis instead.
6. **Contributions & Impact** — What are the most significant contributions?
7. **Limitations & Future Work** — What are the acknowledged or apparent limitations?
   What directions for future research are suggested?

IMPORTANT RULES:
- Write in your OWN analytical voice. Do NOT copy sentences from the article.
- Be thorough — aim for 800-1200 words minimum.
- Include specific numbers, results, and technical details from the paper.
- Use Markdown with clear headers (##) for each section above.
- Ground every claim in the article content — no hallucination.
{lang_instruction}
Article Text:
---
{text}
---

Comprehensive Summary:"""


class SummarizerAgent:
    """Produces one comprehensive overall summary from the full article text."""

    name = "Summarizer Agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _truncate(self, text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    def _lang(self, language: str) -> str:
        if language == "English":
            return ""
        return f"\nIMPORTANT: Write your ENTIRE response in {language}.\n"

    def process(self, full_text: str, language: str = "English") -> dict:
        li = self._lang(language)

        # Use provider-advertised limits (respects HF free-tier, Ollama, etc.)
        max_words = getattr(self.llm, "max_input_words", 5000)
        max_tokens = getattr(self.llm, "max_output_tokens", 3000)

        text_for_llm = self._truncate(full_text, max_words)

        prompt = SUMMARY_PROMPT.format(text=text_for_llm, lang_instruction=li)

        # Direct call — no fallback, no silent swallow. Fail loud.
        result = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.35)

        if not result or not result.strip():
            raise RuntimeError(
                "Summarizer received an empty response from the LLM. "
                "Check your API key and provider status, then try again."
            )

        return {"summary": result.strip()}
