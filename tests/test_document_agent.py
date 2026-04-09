"""
Tests for document_agent.py — extract_references, extract_sections, chunk_text.
Covers multiple paper formats: APA, IEEE, numbered, markdown, minimal, and edge cases.
"""

import pytest

from agents.document_agent import chunk_text, extract_sections, extract_references


# ═══════════════════════════════════════════════════════════
# chunk_text
# ═══════════════════════════════════════════════════════════

class TestChunkText:

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        text = "Machine learning is a subset of artificial intelligence. It works well."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert "Machine learning" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        """Text exceeding chunk_size should split into multiple chunks."""
        # ~600 words
        text = ("This is a sentence about deep learning. " * 60).strip()
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_preserves_context(self):
        """Adjacent chunks should share some overlapping words."""
        sentences = [f"Sentence number {i} contains information." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 2
        # The end of chunk 0 and start of chunk 1 should share some content
        last_words_0 = set(chunks[0].split()[-10:])
        first_words_1 = set(chunks[1].split()[:10])
        assert len(last_words_0 & first_words_1) > 0

    def test_empty_text_returns_single_empty_or_no_chunk(self):
        """Empty text should not crash."""
        chunks = chunk_text("", chunk_size=500, overlap=50)
        # Either empty list or single empty-ish chunk
        assert isinstance(chunks, list)


# ═══════════════════════════════════════════════════════════
# extract_sections
# ═══════════════════════════════════════════════════════════

class TestExtractSections:

    def test_numbered_sections(self):
        """Detects numbered headings like '1. Introduction'."""
        text = (
            "1. Introduction\n"
            "This paper presents a novel approach to natural language processing. "
            "We explore transformer architectures and their applications in detail.\n\n"
            "2. Related Work\n"
            "Prior work by Smith et al. (2020) explored similar techniques. "
            "Their approach used convolutional networks for feature extraction.\n\n"
            "3. Methodology\n"
            "We propose a multi-head attention mechanism with improved convergence. "
            "The model is trained on a large-scale dataset of scientific articles."
        )
        sections = extract_sections(text)
        assert len(sections) >= 2

    def test_uppercase_headings(self):
        """Detects ALL-CAPS headings like 'INTRODUCTION'."""
        text = (
            "ABSTRACT\n"
            "This study investigates the effects of temperature on superconductor performance. "
            "We provide comprehensive analysis of multiple material compositions.\n\n"
            "INTRODUCTION\n"
            "Superconductors have been studied extensively since their discovery in 1911. "
            "Recent advances have brought us closer to room-temperature superconductivity.\n\n"
            "METHODS\n"
            "We synthesized samples using chemical vapor deposition techniques. "
            "Each sample was tested under controlled laboratory conditions."
        )
        sections = extract_sections(text)
        assert len(sections) >= 2

    def test_markdown_headings(self):
        """Detects markdown-style headings like '## Introduction'."""
        text = (
            "# Abstract\n"
            "A brief overview of the research conducted in this paper about climate models. "
            "We present findings from a decade-long longitudinal study.\n\n"
            "## Introduction\n"
            "Climate change has been a subject of intense study for the past several decades. "
            "This paper contributes new modeling approaches to the field.\n\n"
            "## Results\n"
            "Our model achieves state-of-the-art accuracy on benchmark datasets. "
            "The improvements are statistically significant across all metrics tested."
        )
        sections = extract_sections(text)
        assert len(sections) >= 2

    def test_keyword_headings(self):
        """Detects common section names like 'Introduction', 'Methodology'."""
        text = (
            "Abstract\n"
            "We present a comprehensive framework for automated code review. "
            "Our system uses large language models to analyze pull requests.\n\n"
            "Introduction\n"
            "Code review is a critical part of the software development lifecycle. "
            "Manual review is time-consuming and prone to inconsistency.\n\n"
            "Conclusion\n"
            "Our framework reduces review time by forty percent on average. "
            "We demonstrate this improvement across multiple open-source projects."
        )
        sections = extract_sections(text)
        assert len(sections) >= 2

    def test_single_block_text_fallback(self):
        """Text with no headings should still return sections (via paragraph or chunk fallback)."""
        text = (
            "This is a long document without any headings. " * 30 + "\n\n" +
            "Another paragraph with different content about machine learning. " * 30
        )
        sections = extract_sections(text)
        assert len(sections) >= 1
        assert all(len(s.strip()) > 0 for s in sections)


# ═══════════════════════════════════════════════════════════
# extract_references
# ═══════════════════════════════════════════════════════════

class TestExtractReferences:

    def test_ieee_numbered_references(self):
        """IEEE style: [1] Author, 'Title,' Journal, ..."""
        text = (
            "Some article body text here discussing various methods.\n\n"
            "References\n"
            "[1] A. Smith, B. Jones, and C. Lee, \"Deep Learning for NLP: A Comprehensive Survey,\" "
            "IEEE Trans. Neural Networks, vol. 30, pp. 1-15, 2021.\n"
            "[2] D. Brown and E. White, \"Transformer Architectures in Computer Vision Applications,\" "
            "Proc. CVPR, pp. 200-210, 2022.\n"
            "[3] F. Garcia, \"Attention Mechanisms for Sequence-to-Sequence Models in Translation,\" "
            "arXiv preprint arXiv:2103.12345, 2021.\n"
        )
        titles = extract_references(text)
        assert len(titles) >= 2
        assert any("Deep Learning" in t for t in titles)

    def test_apa_style_references(self):
        """APA style: Author, A. (2020). Title of paper. Journal."""
        text = (
            "The body of the paper goes here with analysis.\n\n"
            "References\n\n"
            "Smith, J., & Doe, A. (2020). Reinforcement Learning in Robotic Control Systems. "
            "Journal of AI Research, 45, 112-130.\n\n"
            "Brown, K. (2019). A Novel Approach to Semi-Supervised Text Classification. "
            "Proceedings of ACL, 88-99.\n\n"
            "Garcia, M., & Lee, S. (2021). Graph Neural Networks for Molecular Property Prediction. "
            "Nature Machine Intelligence, 3(2), 45-56.\n"
        )
        titles = extract_references(text)
        assert len(titles) >= 2
        assert any("Reinforcement Learning" in t for t in titles)

    def test_numbered_dot_references(self):
        """Numbered with dots: 1. Author, Title, ..."""
        text = (
            "Content of the paper.\n\n"
            "References\n"
            "1. Wang, X. et al. Feature Pyramid Networks for Object Detection. CVPR 2017.\n"
            "2. He, K. et al. Deep Residual Learning for Image Recognition. CVPR 2016.\n"
            "3. Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.\n"
        )
        titles = extract_references(text)
        assert len(titles) >= 1

    def test_no_references_section(self):
        """Paper without a References section returns empty list."""
        text = (
            "This is a short article that does not have any reference section. "
            "It just discusses some general topics about artificial intelligence."
        )
        titles = extract_references(text)
        assert titles == []

    def test_bibliography_heading(self):
        """Detects 'Bibliography' as alternative heading."""
        text = (
            "Article content.\n\n"
            "Bibliography\n"
            "[1] Johnson, R. \"Bayesian Methods for Machine Learning Applications,\" "
            "JMLR, vol. 15, 2019.\n"
            "[2] Williams, T. \"Optimization Techniques for Large-Scale Deep Networks,\" "
            "ICML, 2020.\n"
        )
        titles = extract_references(text)
        assert len(titles) >= 1
        assert any("Bayesian" in t for t in titles)

    def test_references_with_appendix_cutoff(self):
        """References section should stop before Appendix."""
        text = (
            "Article body.\n\n"
            "References\n"
            "[1] Author A, \"Title of First Referenced Paper in the Study,\" Journal, 2020.\n"
            "[2] Author B, \"Title of Second Referenced Paper in the Study,\" Journal, 2021.\n\n"
            "Appendix\n"
            "This appendix contains supplementary material that should not be parsed."
        )
        titles = extract_references(text)
        assert all("supplementary" not in t.lower() for t in titles)

    def test_short_entries_are_filtered(self):
        """Entries shorter than 10 characters should be skipped."""
        text = (
            "Body.\n\n"
            "References\n"
            "[1] A.\n"
            "[2] Smith, J. (2020). Comprehensive Analysis of Federated Learning Approaches. "
            "NeurIPS, 100-115.\n"
        )
        titles = extract_references(text)
        # Only the substantive entry should appear
        assert all(len(t) > 10 for t in titles)
