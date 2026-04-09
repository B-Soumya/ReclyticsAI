"""
Chat Agent.
RAG-based conversational Q&A over the uploaded article.
Uses FAISS vector store + sentence-transformers for retrieval,
then feeds relevant context to the LLM for answer generation.
Supports multi-language output.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from agents.llm_provider import LLMProvider

CHAT_PROMPT = """You are a helpful research assistant. Answer the user's question
based ONLY on the provided article context. If the context does not contain
enough information to answer, say so honestly.

Article Context:
---
{context}
---

Question: {question}
{lang_instruction}
Answer (be precise and cite relevant parts of the context):"""


class ChatAgent:
    """RAG pipeline: embed chunks -> FAISS index -> retrieve -> generate answer."""

    name = "Chat Agent"

    def __init__(self, llm: LLMProvider, embedding_model=None):
        self.llm = llm
        self._emb_model = embedding_model
        self.index = None
        self.chunks: list[str] = []
        self.language: str = "English"

    @property
    def embedding_model(self):
        if self._emb_model is None:
            from agents.embeddings import get_embedding_model
            self._emb_model = get_embedding_model()
        return self._emb_model

    def build_index(self, chunks: list[str]) -> None:
        import faiss

        self.chunks = chunks
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def _retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if self.index is None or not self.chunks:
            return []
        q_emb = self.embedding_model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def chat(self, question: str, top_k: int = 5) -> str:
        relevant = self._retrieve(question, top_k)
        if not relevant:
            return "I don't have enough context to answer that question."

        lang_instruction = ""
        if self.language != "English":
            lang_instruction = f"\n\nIMPORTANT: Answer in **{self.language}**."

        context = "\n\n---\n\n".join(relevant)
        prompt = CHAT_PROMPT.format(
            context=context, question=question, lang_instruction=lang_instruction,
        )
        return self.llm.generate(prompt, max_tokens=1024, temperature=0.3)
