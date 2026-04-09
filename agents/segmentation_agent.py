"""
Segmentation Agent.
Identifies the key topics / thematic areas covered in an article
using embeddings + clustering + LLM labeling.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from agents.llm_provider import LLMProvider
from agents.document_agent import extract_sections


LABEL_PROMPT = """You are an expert article analyst.
Below are text excerpts that belong to ONE thematic cluster from a larger article.
Provide:
1. A short topic label (3-7 words)
2. A 2-3 sentence description of what this segment covers.

Excerpts:
---
{excerpts}
---

Reply in this exact format:
Topic: <label>
Description: <description>"""


class SegmentationAgent:
    """Clusters document sections into coherent topic segments."""

    name = "Segmentation Agent"

    def __init__(self, llm: LLMProvider, embedding_model=None):
        self.llm = llm
        self._emb_model = embedding_model

    @property
    def embedding_model(self):
        if self._emb_model is None:
            from agents.embeddings import get_embedding_model
            self._emb_model = get_embedding_model()
        return self._emb_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embedding_model.encode(texts, show_progress_bar=False)

    def _optimal_k(self, embeddings: np.ndarray, max_k: int = 8) -> int:
        n = len(embeddings)
        if n <= 3:
            return min(n, 2)
        max_k = min(max_k, n - 1)
        best_k, best_score = 2, -1
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def _label_cluster(self, sections: list[str]) -> dict:
        combined = "\n\n".join(s[:500] for s in sections[:5])
        prompt = LABEL_PROMPT.format(excerpts=combined)
        try:
            raw = self.llm.generate(prompt, max_tokens=256, temperature=0.2)
        except Exception:
            # LLM unavailable — derive a simple label from text
            snippet = sections[0][:200] if sections else ""
            return {"topic": "Topic Segment", "description": snippet}

        topic, description = "General", raw
        for line in raw.split("\n"):
            if line.lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip()
        return {"topic": topic, "description": description}

    def process(self, text: str) -> list[dict]:
        sections = extract_sections(text)
        if len(sections) < 2:
            return [{"topic": "Full Article", "description": "Single-section document.", "sections": sections}]

        embeddings = self._embed(sections)
        k = self._optimal_k(embeddings)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)

        clusters: dict[int, list[str]] = {}
        for idx, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(sections[idx])

        segments = []
        for cluster_id in sorted(clusters):
            info = self._label_cluster(clusters[cluster_id])
            info["sections"] = clusters[cluster_id]
            info["num_sections"] = len(clusters[cluster_id])
            segments.append(info)

        return segments


