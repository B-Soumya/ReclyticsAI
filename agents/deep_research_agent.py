"""
Deep Research Agent.
Performs in-depth analysis of mathematical equations, statistical methods,
and calculations — explaining WHY they were used and HOW they connect
to the research objectives. Supports multi-language output.
"""

from __future__ import annotations

from agents.llm_provider import LLMProvider


IDENTIFY_PROMPT = """You are a mathematics and research methodology expert.
Carefully read the following research article and check if any mathematical or statistical evidences are present.
If you trace anything then identify EVERY mathematical equation, formula, statistical method, calculation,
metric, algorithm, or quantitative technique used.

For each one found, list it in this exact format:
ITEM: <name or description of the equation/method>
EQUATION: <the equation in plain-text notation, e.g. "L = -sum(y_i * log(p_i))">
LOCATION: <where in the article it appears, e.g. "in the methodology section">

If the article has NO mathematical content, list the key quantitative claims, statistics,
or numerical analyses instead.
{lang_instruction}
Article Text:
---
{text}
---

List all mathematical items:"""


DEEP_ANALYSIS_PROMPT = """You are a senior research scientist with deep expertise in mathematics,
statistics, and the specific domain of the article below.

You have identified the following mathematical elements in this research article:
---
{equations}
---

Now perform a **deep research analysis** covering ALL of the following:

## 1. Mathematical Framework Overview
Provide a comprehensive overview of the mathematical/quantitative framework used in this paper.
What family of methods does it belong to? How do the equations relate to each other?
Draw connections between different mathematical components.

## 2. Equation-by-Equation Deep Dive
For EACH equation, formula, or quantitative method identified:

### [Name of Equation/Method]
- **What it is**: State the equation and define every variable and symbol.
- **Why it was chosen**: Why did the authors use THIS specific formulation over alternatives?
  What are the mathematical properties that make it suitable for this research problem?
  (e.g., convexity, differentiability, interpretability, computational efficiency)
- **How it connects to the research**: Explain exactly how this equation serves the paper's
  research objectives. What role does it play in the overall methodology?
- **Alternatives considered**: What other approaches could have been used? Why might the
  authors have preferred this one?
- **Assumptions & limitations**: What mathematical assumptions does this equation rely on?
  Under what conditions might it break down?

## 3. Mathematical Relationships & Flow
How do the equations feed into each other? Trace the mathematical pipeline from input
data through to final results. Which equations are foundational and which are derived?

## 4. Statistical Rigor Assessment
Assess the statistical rigor of the mathematical framework:
- Are the chosen methods appropriate for the data and research questions?
- Are there potential issues (overfitting, bias, violation of assumptions)?
- How robust are the mathematical conclusions?

## 5. Practical Implications of the Mathematics
What do these mathematical choices mean in practical terms?
How do they affect the reliability, generalizability, and applicability of the results?

IMPORTANT: Write a thorough analysis of at least 1500 words. Be specific, technical,
and insightful. Use Markdown formatting with clear headers and sub-sections.
{lang_instruction}
Article Text (for reference):
---
{text}
---

Deep research analysis:"""


class DeepResearchAgent:
    """Performs deep analysis of mathematical content and quantitative reasoning.
    Supports multi-language output."""

    name = "Deep Research Agent"

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

    def process(self, text: str, language: str = "English") -> dict:
        truncated = self._truncate(text, max_words=5000)
        li = self._lang(language)

        identification = self.llm.generate(
            IDENTIFY_PROMPT.format(text=truncated, lang_instruction=li),
            max_tokens=2048, temperature=0.2,
        )

        analysis = self.llm.generate(
            DEEP_ANALYSIS_PROMPT.format(equations=identification, text=truncated, lang_instruction=li),
            max_tokens=4096, temperature=0.4,
        )

        if len(analysis.split()) < 800:
            continuation = self.llm.generate(
                f"Continue your deep research analysis from where you left off.\n\n"
                f"You wrote so far:\n---\n{analysis[-1000:]}\n---\n\n"
                f"Continue with remaining sections:{li}",
                max_tokens=3000, temperature=0.4,
            )
            analysis = analysis.strip() + "\n\n" + continuation.strip()

        return {
            "identified_elements": identification,
            "deep_analysis": analysis,
        }
