"""
Statistical Analysis Agent.

Identifies every statistical method, test, model, and evidence used in a
research paper, then performs a rigorous assessment explaining:
  - What statistical methods were used and why they are appropriate
  - Whether assumptions (normality, independence, sample size) are met
  - How to interpret the reported p-values, confidence intervals, effect sizes
  - What the statistical evidence actually proves (and what it doesn't)
  - Potential issues: multiple comparisons, overfitting, selection bias, etc.

Supports multi-language output.
"""

from __future__ import annotations

from agents.llm_provider import LLMProvider


# ── Step 1: Identify all statistical elements ────────────────────────

IDENTIFY_STATS_PROMPT = """You are a biostatistician and research methodology expert.

Carefully read the following research article and identify EVERY statistical
element present. This includes:

  - Hypothesis tests (t-test, chi-square, ANOVA, Mann-Whitney, etc.)
  - Regression models (linear, logistic, Cox, mixed-effects, etc.)
  - Machine learning evaluation metrics (accuracy, precision, recall, AUC-ROC, F1, etc.)
  - Confidence intervals and significance levels (p-values, alpha)
  - Effect sizes (Cohen's d, odds ratio, hazard ratio, correlation coefficients)
  - Sample size calculations or power analysis
  - Cross-validation, bootstrapping, or resampling methods
  - Bayesian methods (priors, posteriors, credible intervals)
  - Descriptive statistics (mean, median, SD, IQR, etc.)
  - Data preprocessing (normalization, imputation, outlier handling)
  - Experimental design (randomization, blinding, control groups)

For EACH element found, output in this exact format:

METHOD: <name of the statistical method or test>
DETAILS: <specific parameters, e.g. "two-tailed t-test, alpha=0.05, n=150">
LOCATION: <where it appears in the paper>
RESULT: <the reported result, e.g. "p=0.003, CI=[1.2, 3.4]">
PURPOSE: <what research question this test answers>

If the article contains NO statistical methods, respond with exactly:
NO_STATS_FOUND

Then list any data-driven claims or numerical evidence instead.
{lang_instruction}

Article Text:
---
{text}
---

Identify all statistical elements:"""


# ── Step 2: Deep statistical rigor analysis ──────────────────────────

STATS_ANALYSIS_PROMPT = """You are a senior statistician performing a rigorous peer review.

You identified these statistical elements in the research article:
---
{stats}
---

Now write a **comprehensive statistical analysis** covering ALL of the following:

## Statistical Methods Overview
What is the overall statistical framework? How do the methods relate to the
research design? Is it observational, experimental, quasi-experimental?
What is the primary analysis strategy?

## Method-by-Method Assessment

For EACH statistical method or test identified above:

### [Name of Method/Test]
- **What it does**: Explain in plain language what this test measures or estimates.
- **Why it was chosen**: Is this the right test for the data type (continuous, categorical,
  ordinal) and research question? What assumptions does it require?
- **Assumptions check**: Are the key assumptions likely met?
  - For parametric tests: normality, homoscedasticity, independence
  - For regression: linearity, no multicollinearity, homoscedasticity of residuals
  - For ML metrics: class balance, appropriate threshold selection
  - For Bayesian: prior sensitivity, convergence
- **Result interpretation**: What do the reported numbers (p-value, CI, effect size)
  actually mean in context? Is the effect practically significant, not just statistically?
- **Potential issues**: Multiple comparisons correction? Selection bias? Overfitting?
  Data leakage? Small sample size? Publication bias?

## Statistical Rigor Assessment

### Strengths
- What did the authors do well statistically?
- Are the methods appropriate for the research questions?
- Is the sample size adequate?

### Concerns
- Are there any violated assumptions?
- Were appropriate corrections applied (e.g., Bonferroni, FDR)?
- Is there risk of Type I or Type II error?
- Are effect sizes reported alongside p-values?
- Is there evidence of p-hacking, data dredging, or HARKing?

### Missing Analyses
- What statistical analyses SHOULD have been performed but weren't?
- Would additional sensitivity analyses strengthen the conclusions?
- Are there confounding variables that weren't controlled for?

## Confidence in Results
Rate the overall statistical evidence:
- How reproducible are these findings likely to be?
- What is the practical significance beyond statistical significance?
- What would you recommend in a peer review?

IMPORTANT: Be thorough — aim for 800-1200 words minimum. Use Markdown formatting.
Be specific about which assumptions are met or violated. Cite actual numbers from the paper.
{lang_instruction}

Article Text (for reference):
---
{text}
---

Statistical rigor analysis:"""


# ── Fallback for non-statistical papers ──────────────────────────────

EVIDENCE_PROMPT = """You are a research methodology expert. The following article
does not use formal statistical tests, but it may rely on data, measurements,
comparisons, or evidence-based arguments.

Analyze the evidence quality in this article:

## Evidence Framework
How does the paper support its claims? What types of evidence are presented
(experimental data, case studies, simulations, theoretical proofs, surveys)?

## Evidence Assessment
For each major claim:
- What evidence supports it?
- How strong is that evidence?
- Are there alternative explanations?
- What additional evidence would strengthen the claim?

## Methodological Rigor
- Is the research design appropriate for the questions asked?
- Are there potential biases in data collection or analysis?
- Are the conclusions proportionate to the evidence presented?
- What are the main threats to validity (internal and external)?
{lang_instruction}

Article Text:
---
{text}
---

Evidence analysis:"""


class StatAnalysisAgent:
    """Identifies and rigorously assesses statistical methods in research papers."""

    name = "Statistical Analysis Agent"

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
        """Run the statistical analysis pipeline.

        Returns:
            {
                "has_stats": bool,
                "identified_methods": str,   # raw list of stats methods found
                "analysis": str,             # deep markdown analysis
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

        # Step 1: Identify statistical elements
        identification = self._safe_generate(
            IDENTIFY_STATS_PROMPT.format(text=truncated, lang_instruction=li),
            max_tokens=max_tokens_id,
            temperature=0.2,
        )

        if not identification:
            return {
                "has_stats": False,
                "identified_methods": "",
                "analysis": "_Statistical analysis could not be generated — LLM unavailable._",
            }

        # Check if the article has stats
        has_stats = "NO_STATS_FOUND" not in identification

        if has_stats:
            # Step 2: Deep statistical rigor analysis
            analysis = self._safe_generate(
                STATS_ANALYSIS_PROMPT.format(
                    stats=identification,
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
                    f"Continue your statistical analysis from where you left off.\n\n"
                    f"You wrote so far:\n---\n{analysis[-1200:]}\n---\n\n"
                    f"Continue with the Rigor Assessment and Confidence sections.{li}",
                    max_tokens=continuation_budget,
                    temperature=0.35,
                )
                if continuation:
                    analysis = analysis + "\n\n" + continuation
        else:
            # Fallback: evidence-based analysis for non-statistical papers
            analysis = self._safe_generate(
                EVIDENCE_PROMPT.format(text=truncated, lang_instruction=li),
                max_tokens=max_tokens_analysis,
                temperature=0.35,
            )

        return {
            "has_stats": has_stats,
            "identified_methods": identification,
            "analysis": analysis or "_Analysis could not be generated._",
        }
