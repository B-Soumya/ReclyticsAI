"""
Orchestrator.
Coordinates all agents including the new Math and Statistical analysis agents.
"""

from __future__ import annotations

from io import BytesIO

from agents.llm_provider import LLMProvider
from agents.document_agent import DocumentAgent
from agents.summarizer_agent import SummarizerAgent
from agents.segmentation_agent import SegmentationAgent
from agents.recommendation_agent import RecommendationAgent
from agents.chat_agent import ChatAgent
from agents.math_analysis_agent import MathAnalysisAgent
from agents.stat_analysis_agent import StatAnalysisAgent


class Orchestrator:
    """Central coordinator for the multi-agent article analysis pipeline."""

    def __init__(self, llm: LLMProvider, language: str = "English"):
        self.llm = llm
        self.language = language
        self.document_agent = DocumentAgent()
        self.summarizer_agent = SummarizerAgent(llm)
        self.segmentation_agent = SegmentationAgent(llm)
        self.recommendation_agent = RecommendationAgent(llm)
        self.chat_agent = ChatAgent(llm)
        self.chat_agent.language = language
        self.math_agent = MathAnalysisAgent(llm)
        self.stat_agent = StatAnalysisAgent(llm)

        # shared state
        self.doc_data: dict | None = None
        self.summary_data: dict | None = None
        self.segments: list[dict] | None = None
        self.recommendations: dict | None = None
        self.math_analysis: dict | None = None
        self.stat_analysis: dict | None = None

    def parse_document(self, file: BytesIO, filename: str) -> dict:
        self.doc_data = self.document_agent.process(file, filename)
        self.chat_agent.build_index(self.doc_data["chunks"])
        return self.doc_data

    def generate_summary(self) -> dict:
        """Produces a comprehensive overall summary from the full article text."""
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        self.summary_data = self.summarizer_agent.process(
            full_text=self.doc_data["cleaned_text"],
            language=self.language,
        )
        return self.summary_data

    def generate_segments(self) -> list[dict]:
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        self.segments = self.segmentation_agent.process(self.doc_data["raw_text"])
        return self.segments

    def generate_recommendations(self, search_types: list[str] | None = None) -> dict:
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        self.recommendations = self.recommendation_agent.process(
            self.doc_data["cleaned_text"],
            search_types,
            references=self.doc_data.get("references", []),
        )
        return self.recommendations

    def generate_math_analysis(self) -> dict:
        """Analyze mathematical equations and formulas in the paper."""
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        self.math_analysis = self.math_agent.process(
            text=self.doc_data["cleaned_text"],
            language=self.language,
        )
        return self.math_analysis

    def generate_stat_analysis(self) -> dict:
        """Analyze statistical methods and evidence in the paper."""
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        self.stat_analysis = self.stat_agent.process(
            text=self.doc_data["cleaned_text"],
            language=self.language,
        )
        return self.stat_analysis

    def chat(self, question: str) -> str:
        if not self.doc_data:
            raise RuntimeError("Parse a document first.")
        return self.chat_agent.chat(question)
