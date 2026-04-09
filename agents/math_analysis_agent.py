"""
Math Analysis Agent.

Identifies every mathematical equation, formula, and quantitative technique
in a research paper, then performs a deep analysis explaining:
  - What each equation does and what every variable means
  - WHY the authors chose this formulation over alternatives
  - HOW equations connect to each other and to the research objectives
  - What assumptions they rely on and when they break down

Supports multi-language output.
"""

from __future__ import annotations

from agents.llm_provider import LLMProvider


# ── Step 1: Identify all math elements ───────────────────────────────

IDENTIFY_PROMPT = """You are a mathematics and research methodology expert.

Carefully read the following research article and identify EVERY mathematical
element present. This includes:
  - Explicit equations and formulas
  - Loss functions, objective functions, optimization targets
  - Algorithms described with mathematical notation
  - Mathematical definitions (e.g., "let X be...")
  - Metrics and scoring functions (e.g., F1 = 2PR/(P+R))
  - Transformations, mappings, or function definitions

For EACH element found, output in this exact format:

ITEM: <name or description>
EQUATION: <the equation in plain-text math notation, e.g. "L = -sum(y_i * log(p_i))">
LOCATION: <where it appears, e.g. "Section 3.2 — Methodology">
PURPOSE: <one sentence on what it computes or represents>

If the article contains NO mathematical equations at all, respond with exactly:
NO_MATH_FOUND

Then list the key quantitative claims or numerical results instead, in this format:
CLAIM: <the quantitative claim>
VALUE: <the number or metric>
LOCATION: <where it appears>
{lang_instruction}

Article Text:
---
{text}
---

Identify all mathematical elements:"""


# ── Step 2: Deep equation-by-equation analysis ───────────────────────

ANALYSIS_PROMPT = """You are a senior research mathematician reviewing a paper.

You identified these mathematical elements in the article:
---
{equations}
---

Now write a **deep mathematical analysis** covering:

## Mathematical Framework Overview
What family of methods does this paper use? How do the equations form a coherent
mathematical system? What is the high-level mathematical pipeline from input to output?

## Equation-by-Equation Analysis

For EACH equation or formula identified above, write a subsection:

### [Name of Equation/Method]
- **Definition**: State the equation clearly. Define every variable, subscript, and symbol.
- **Why this formulation?**: Why did the authors choose THIS specific equation over known
  alternatives? What mathematical properties make it suitable — convexity, differentiability,
  computational tractability, interpretability, convergence guarantees?
- **Role in the research**: How does this equation serve the paper's research objectives?
  What would change if it were removed or replaced?
- **Connections**: Which other equations in the paper does this feed into or depend on?
  Trace the data flow.
- **Alternatives**: What other formulations could have been used? Name at least one
  concrete alternative and explain the trade-off.
- **Assumptions & Limitations**: What conditions must hold for this equation to be valid?
  When would it produce unreliable results?

## Mathematical Flow
Trace the full mathematical pipeline: how does raw input data flow through the equations
to produce the final results? Which equations are foundational and which are derived?
Draw explicit connections: "Equation 1 produces X, which feeds into Equation 3 as input Y."

## Assessment
- Are the mathematical choices well-justified for this research problem?
- Are there any mathematical gaps, missing justifications, or questionable choices?
- How sensitive are the results to the specific mathematical formulations used?

IMPORTANT: Be thorough — aim for 800-1200 words minimum. Use Markdown with clear headers.
Every claim must be grounded in the actual equations found.
{lang_instruction}

Article Text (for reference):
---
{text}
---

Deep mathematical analysis:"""


# ── Fallback when no equations found ─────────────────────────────────

QUANTITATIVE_PROMPT = """You are a research analyst. The following article does not contain
explicit mathematical equations, but it may contain quantitative reasoning, numerical
results, metrics, comparisons, or data-driven arguments.

Analyze the quantitative aspects of this article:

## Quantitative Framework
What quantitative methods, metrics, or numerical analyses does the paper rely on?
Even without formal equations, how does the paper use numbers to support its claims?

## Key Quantitative Claims
For each major numerical result or comparison:
- What is being measured and how?
- What are the specific numbers reported?
- How do these numbers support the paper's conclusions?
- Are the measurements/comparisons appropriate and sufficient?

## Rigor Assessment
- Are the quantitative claims well-supported by the data presented?
- Are there missing quantitative analyses that would strengthen the paper?
- How confident can we be in the numerical results?
{lang_instruction}

Article Text:
---
{text}
---

Quantitative analysis:"""


class MathAnalysisAgent:
    """Identifies and deeply analyzes mathematical content in research papers."""

    name = "Math Analysis Agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _truncate(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "\n\n[...truncated...]"
        return text

    def _lang(self, language: str) -> str:
        if language == "English":
            return ""
        return f"\n\nIMPORTANT: Write your ENTIRE response in **{language}**.\n"

    def _is_local(self) -> bool:
        return self.llm.__class__.__name__ == "OllamaProvider"

    def _safe_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            return self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature).strip()
        except Exception:
            return ""

    def process(self, text: str, language: str = "English") -> dict:
        """Run the math analysis pipeline.

        Returns:
            {
                "has_math": bool,
                "identified_elements": str,   # raw list of equations found
                "analysis": str,              # deep markdown analysis
            }
        """
        local = self._is_local()
        # Use provider-advertised limits
        max_words = getattr(self.llm, "max_input_words", 5000)
        max_out = getattr(self.llm, "max_output_tokens", 3000)
        max_tokens_id = min(1024, max_out)
        max_tokens_analysis = min(max_out, 2048)

        truncated = self._truncate(text, max_words)
        li = self._lang(language)

        # Step 1: Identify math elements
        identification = self._safe_generate(
            IDENTIFY_PROMPT.format(text=truncated, lang_instruction=li),
            max_tokens=max_tokens_id,
            temperature=0.2,
        )

        if not identification:
            return {
                "has_math": False,
                "identified_elements": "",
                "analysis": "_Math analysis could not be generated — LLM unavailable._",
            }

        # Check if the article has math
        has_math = "NO_MATH_FOUND" not in identification

        if has_math:
            # Step 2: Deep equation analysis
            analysis = self._safe_generate(
                ANALYSIS_PROMPT.format(
                    equations=identification,
                    text=truncated,
                    lang_instruction=li,
                ),
                max_tokens=max_tokens_analysis,
                temperature=0.35,
            )

            # Auto-continuation if too short and budget allows
            continuation_budget = min(max_out, 1024)
            if analysis and len(analysis.split()) < 400 and max_out >= 1024:
                continuation = self._safe_generate(
                    f"Continue your mathematical analysis from where you left off.\n\n"
                    f"You wrote so far:\n---\n{analysis[-1200:]}\n---\n\n"
                    f"Continue covering any remaining equations and the assessment section.{li}",
                    max_tokens=continuation_budget,
                    temperature=0.35,
                )
                if continuation:
                    analysis = analysis + "\n\n" + continuation
        else:
            # Fallback: quantitative analysis for non-math papers
            analysis = self._safe_generate(
                QUANTITATIVE_PROMPT.format(text=truncated, lang_instruction=li),
                max_tokens=max_tokens_analysis,
                temperature=0.35,
            )

        return {
            "has_math": has_math,
            "identified_elements": identification,
            "analysis": analysis or "_Analysis could not be generated._",
        }
