"""
Recommendation Agent — Semantic Search Pipeline.

Uses multiple ML/NLP models for intelligent article recommendation:
  1. TF-IDF (scikit-learn)           — statistical keyword extraction
  2. Sentence-Transformers           — semantic keyword extraction (KeyBERT-style)
                                       + cosine-similarity reranking of results
  3. NLTK POS Tagger + NP Chunker   — noun-phrase keyphrase extraction
  4. LLM                             — high-level topic / search-query generation
  5. DuckDuckGo                      — free web search backend

All search results are semantically reranked against the source paper so the
most relevant items float to the top, with a relevance score attached.
"""

from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents.llm_provider import LLMProvider


# ── LLM prompt for topic extraction ────────────────────────

TOPIC_PROMPT = """You are a research expert. Analyze the following article and produce:

1. TITLE: The most likely title of this paper (or a descriptive title if unclear)
2. DOMAIN: The broad research domain (e.g. "machine learning", "public health", "materials science")
3. TOPIC: A specific 5-10 word topic description
4. KEYWORDS: 7 highly specific, searchable keywords (comma-separated)
5. SEARCH1 through SEARCH5: Five different search queries designed to find SIMILAR research
   papers on the open internet. Each query should approach the topic from a different angle.
   Make them specific enough to find relevant papers but broad enough to get results.

Return in this EXACT format (one per line):
TITLE: ...
DOMAIN: ...
TOPIC: ...
KEYWORDS: kw1, kw2, kw3, kw4, kw5, kw6, kw7
SEARCH1: ...
SEARCH2: ...
SEARCH3: ...
SEARCH4: ...
SEARCH5: ...

Article Text:
---
{text}
---"""


# ── Shared embedding model (cached across Streamlit reruns) ─
# Imported from agents.embeddings to avoid duplication


def _ensure_nltk_resources():
    """Download NLTK data lazily and quietly."""
    import nltk
    for resource, pkg in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)


# ── Agent ───────────────────────────────────────────────────

class RecommendationAgent:
    """Finds similar research papers, blogs, and videos using a multi-model
    semantic search pipeline.  Supports selective search types."""

    name = "Recommendation Agent"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @property
    def embedding_model(self):
        from agents.embeddings import get_embedding_model
        return get_embedding_model()

    # ────────────────────────────────────────────────────────
    # 1. LLM-based topic extraction (existing, unchanged)
    # ────────────────────────────────────────────────────────

    def _analyze_article(self, text: str) -> dict:
        """Extract title, domain, topic, keywords via LLM.

        Falls back to a purely statistical extraction when the LLM call
        fails (e.g. quota exhausted, network error, bad response).
        """
        result: dict = {"title": "", "domain": "", "topic": "", "keywords": [], "searches": []}

        try:
            truncated = " ".join(text.split()[:3000])
            raw = self.llm.generate(
                TOPIC_PROMPT.format(text=truncated), max_tokens=400, temperature=0.2,
            )
            for line in raw.strip().split("\n"):
                line = line.strip()
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key, val = key.strip().upper(), val.strip()
                if key == "TITLE":
                    result["title"] = val
                elif key == "DOMAIN":
                    result["domain"] = val
                elif key == "TOPIC":
                    result["topic"] = val
                elif key == "KEYWORDS":
                    result["keywords"] = [
                        k.strip().strip("\"'") for k in val.split(",") if k.strip()
                    ][:7]
                elif key.startswith("SEARCH") and val:
                    result["searches"].append(val)
        except Exception:
            # LLM unavailable — fall back to statistical extraction
            result = self._analyze_article_fallback(text)

        return result

    def _analyze_article_fallback(self, text: str) -> dict:
        """Extract title, domain, and keywords without any LLM call.

        Uses the first substantial line as a title guess, and TF-IDF +
        NLTK noun phrases for keywords.
        """
        result: dict = {"title": "", "domain": "", "topic": "", "keywords": [], "searches": []}

        # Guess title from first non-trivial line
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 15 and len(line) < 250:
                result["title"] = line
                break

        # Gather keywords from TF-IDF
        try:
            tfidf_kws = self._extract_tfidf_keywords(text, top_n=10)
        except Exception:
            tfidf_kws = []

        # Gather keywords from NLTK
        try:
            nltk_kws = self._extract_nltk_keyphrases(text, top_n=10)
        except Exception:
            nltk_kws = []

        all_kws = list(dict.fromkeys(tfidf_kws + nltk_kws))[:10]
        result["keywords"] = all_kws

        # Build a rough topic / domain from top keywords
        if all_kws:
            result["topic"] = " ".join(all_kws[:5])
            result["domain"] = all_kws[0]

        # Build search queries from title + keywords
        title = result["title"]
        if title:
            result["searches"].append(f"{title} research paper")
            result["searches"].append(f"{title} arxiv")
        for kw in all_kws[:3]:
            result["searches"].append(f"{kw} research paper")

        return result

    # ────────────────────────────────────────────────────────
    # 2. TF-IDF keyword extraction  (scikit-learn)
    # ────────────────────────────────────────────────────────

    def _extract_tfidf_keywords(self, text: str, top_n: int = 15) -> list[str]:
        """Extract statistically significant uni/bi/trigrams via TF-IDF."""
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 20]
        if len(sentences) < 3:
            sentences = [text[i : i + 500] for i in range(0, min(len(text), 5000), 400)]
        if not sentences:
            return []
        try:
            tfidf = TfidfVectorizer(
                max_features=300,
                stop_words="english",
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.85,
            )
            matrix = tfidf.fit_transform(sentences)
            features = tfidf.get_feature_names_out()
            scores = np.asarray(matrix.sum(axis=0)).flatten()
            ranked = sorted(zip(features, scores), key=lambda x: -x[1])
            return [term for term, _ in ranked[:top_n]]
        except Exception:
            return []

    # ────────────────────────────────────────────────────────
    # 3. NLTK noun-phrase keyphrase extraction
    # ────────────────────────────────────────────────────────

    def _extract_nltk_keyphrases(self, text: str, top_n: int = 12) -> list[str]:
        """Extract multi-word noun phrases using NLTK POS tagging."""
        import nltk

        _ensure_nltk_resources()
        truncated = " ".join(text.split()[:2000])
        try:
            sentences = nltk.sent_tokenize(truncated)
        except Exception:
            return []

        phrases: list[str] = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            # Accumulate runs of adjectives + nouns → noun phrases
            current: list[str] = []
            for word, tag in tagged:
                if tag in ("JJ", "NN", "NNS", "NNP", "NNPS"):
                    current.append(word)
                else:
                    if len(current) >= 2:
                        phrases.append(" ".join(current))
                    current = []
            if len(current) >= 2:
                phrases.append(" ".join(current))

        freq = Counter(p.lower() for p in phrases)
        return [phrase for phrase, _ in freq.most_common(top_n)]

    # ────────────────────────────────────────────────────────
    # 4. Semantic keyword extraction  (KeyBERT-style)
    #    Uses sentence-transformers + MMR diversity selection
    # ────────────────────────────────────────────────────────

    def _extract_semantic_keywords(
        self, text: str, candidates: list[str], top_n: int = 10,
    ) -> list[str]:
        """Pick candidate phrases most semantically similar to the paper,
        with MMR-style diversity so keywords are non-redundant."""
        if not candidates:
            return []

        truncated = " ".join(text.split()[:1000])
        model = self.embedding_model
        paper_emb = model.encode([truncated], show_progress_bar=False)
        cand_embs = model.encode(candidates, show_progress_bar=False)

        # Cosine similarity: paper <-> each candidate
        paper_sims = cosine_similarity(paper_emb, cand_embs)[0]
        # Pairwise similarity among candidates (for diversity)
        cand_sims = cosine_similarity(cand_embs)

        selected_idx: list[int] = []
        remaining = set(range(len(candidates)))
        lam = 0.7  # balance: relevance vs diversity

        for _ in range(min(top_n, len(candidates))):
            if not remaining:
                break
            best_idx, best_score = -1, -float("inf")
            for idx in remaining:
                relevance = paper_sims[idx]
                redundancy = (
                    max(cand_sims[idx][j] for j in selected_idx) if selected_idx else 0.0
                )
                mmr = lam * relevance - (1 - lam) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected_idx.append(best_idx)
            remaining.discard(best_idx)

        return [candidates[i] for i in selected_idx]

    # ────────────────────────────────────────────────────────
    # 5. Merge keywords from all models (semantic dedup)
    # ────────────────────────────────────────────────────────

    def _merge_keywords(
        self,
        llm_kws: list[str],
        tfidf_kws: list[str],
        semantic_kws: list[str],
        nltk_kws: list[str],
    ) -> list[str]:
        """Combine keywords from every extractor, deduplicate semantically."""
        all_kws: list[str] = []
        seen_lower: set[str] = set()

        # Priority: semantic > LLM > NLTK > TF-IDF
        for kw_list in [semantic_kws, llm_kws, nltk_kws, tfidf_kws]:
            for kw in kw_list:
                kw_clean = kw.strip().lower()
                if kw_clean and kw_clean not in seen_lower and len(kw_clean) > 2:
                    seen_lower.add(kw_clean)
                    all_kws.append(kw.strip())

        # Semantic dedup: drop near-duplicate phrases (cosine > 0.85)
        if len(all_kws) > 5:
            embs = self.embedding_model.encode(all_kws, show_progress_bar=False)
            sims = cosine_similarity(embs)
            keep = [True] * len(all_kws)
            for i in range(len(all_kws)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(all_kws)):
                    if keep[j] and sims[i][j] > 0.85:
                        keep[j] = False
            all_kws = [kw for kw, k in zip(all_kws, keep) if k]

        return all_kws[:15]

    # ────────────────────────────────────────────────────────
    # 6. Semantic reranking of search results
    # ────────────────────────────────────────────────────────

    def _semantic_rerank(
        self, paper_text: str, results: list[dict], top_n: int = 15,
    ) -> list[dict]:
        """Rerank search results by cosine similarity to the paper."""
        if not results:
            return []

        truncated = " ".join(paper_text.split()[:1500])
        model = self.embedding_model
        paper_emb = model.encode([truncated], show_progress_bar=False)

        result_texts = [
            f"{r.get('title', '')} {r.get('snippet', '')}".strip() or "unknown"
            for r in results
        ]
        result_embs = model.encode(result_texts, show_progress_bar=False)
        sims = cosine_similarity(paper_emb, result_embs)[0]

        for r, score in zip(results, sims):
            r["relevance_score"] = round(float(score) * 100)  # 0-100 scale

        results.sort(key=lambda x: -x.get("relevance_score", 0))

        # Keep items above a minimum threshold (15 on 0-100 scale)
        filtered = [r for r in results if r.get("relevance_score", 0) >= 15]
        return filtered[:top_n] if filtered else results[:top_n]

    # ────────────────────────────────────────────────────────
    # Web search helpers (DuckDuckGo — free, no API key)
    # ────────────────────────────────────────────────────────

    def _search_web(self, query: str, max_results: int = 5) -> list[dict]:
        from ddgs import DDGS
        try:
            time.sleep(0.2)
            results = []
            for r in DDGS().text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
            return results
        except Exception:
            return []

    def _search_videos(self, query: str, max_results: int = 3) -> list[dict]:
        from ddgs import DDGS
        try:
            time.sleep(0.2)
            results = []
            for r in DDGS().videos(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("content", ""),
                    "snippet": r.get("description", ""),
                    "publisher": r.get("publisher", ""),
                })
            return results
        except Exception:
            return []

    def _deduplicate(self, items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique = []
        for item in items:
            url = item.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)
        return unique

    # ────────────────────────────────────────────────────────
    # Search strategies (enhanced with merged keywords)
    # ────────────────────────────────────────────────────────

    def _find_papers(self, info: dict, references: list[str] | None = None) -> list[dict]:
        papers: list[dict] = []

        if references:
            # ═══════════════════════════════════════════════════════
            # PATH A — References section found: search each cited
            # paper title to find its URL + discover related work
            # ═══════════════════════════════════════════════════════

            # A1 — Search each reference title directly
            for ref_title in references[:10]:
                papers.extend(self._search_web(f"{ref_title} paper", 3))
            papers = self._deduplicate(papers)

            # A2 — Search reference titles on arxiv specifically
            if len(papers) < 5:
                for ref_title in references[:5]:
                    papers.extend(self._search_web(f"{ref_title} arxiv", 3))
                papers = self._deduplicate(papers)

            # A3 — Use the article's own title to find related work
            title = info.get("title", "")
            if title:
                papers.extend(self._search_web(f"{title} similar research", 5))
                papers.extend(self._search_web(f"{title} related papers", 5))
            papers = self._deduplicate(papers)

            # A4 — Fill with keyword searches if still under 5
            if len(papers) < 5:
                for kw in info.get("keywords", [])[:4]:
                    papers.extend(self._search_web(f"{kw} research paper", 5))
                papers = self._deduplicate(papers)

        else:
            # ═══════════════════════════════════════════════════════
            # PATH B — No references section: use article title +
            # keywords for similarity search
            # ═══════════════════════════════════════════════════════

            # B1 — Article title similarity search
            title = info.get("title", "")
            if title:
                papers.extend(self._search_web(f"{title} research paper", 5))
                papers.extend(self._search_web(f"{title} similar papers arxiv", 5))
                papers.extend(self._search_web(f"{title} related work", 5))
            papers = self._deduplicate(papers)
            if len(papers) >= 8:
                return papers

            # B2 — LLM-generated search queries
            for q in info.get("searches", []):
                papers.extend(self._search_web(f"{q} research paper", 5))
            papers = self._deduplicate(papers)
            if len(papers) >= 8:
                return papers

            # B3 — Keyword combos
            keywords = info.get("keywords", [])
            for i in range(0, len(keywords), 2):
                combo = " ".join(keywords[i : i + 2])
                papers.extend(self._search_web(f"{combo} research paper", 5))
            papers = self._deduplicate(papers)
            if len(papers) >= 8:
                return papers

            # B4 — Domain + topic
            domain, topic = info.get("domain", ""), info.get("topic", "")
            if domain and topic:
                papers.extend(self._search_web(f"{domain} {topic} paper PDF", 8))
            papers = self._deduplicate(papers)
            if len(papers) >= 8:
                return papers

            # B5 — Broad domain fallback
            domain = info.get("domain", "")
            if domain:
                papers.extend(self._search_web(f"{domain} latest research 2024 2025", 10))
            papers = self._deduplicate(papers)

        # Filter out Wikipedia pages — those belong in the wiki section
        papers = [p for p in papers if "wikipedia.org" not in p.get("url", "")]
        return papers

    def _find_blogs(self, info: dict) -> list[dict]:
        blogs: list[dict] = []
        for kw in info.get("keywords", [])[:4]:
            blogs.extend(self._search_web(f"{kw} blog article explanation tutorial", 3))
        for kw in info.get("semantic_keywords", [])[:2]:
            blogs.extend(self._search_web(f"{kw} guide blog post", 3))
        blogs = self._deduplicate(blogs)
        # Filter out Wikipedia pages — those belong in the wiki section
        blogs = [b for b in blogs if "wikipedia.org" not in b.get("url", "")]
        return blogs

    def _find_videos(self, info: dict) -> list[dict]:
        videos: list[dict] = []
        topic = info.get("topic", "")
        if topic:
            videos.extend(self._search_videos(f"{topic} explained youtube", 4))
        for kw in info.get("keywords", [])[:2]:
            videos.extend(self._search_videos(f"{kw} tutorial youtube", 3))
        for kw in info.get("semantic_keywords", [])[:1]:
            videos.extend(self._search_videos(f"{kw} lecture youtube", 3))
        return self._deduplicate(videos)

    def _find_podcasts(self, info: dict) -> list[dict]:
        podcasts: list[dict] = []
        topic = info.get("topic", "")
        domain = info.get("domain", "")
        if topic:
            podcasts.extend(
                self._search_web(f"{topic} podcast episode", 5),
            )
            podcasts.extend(
                self._search_web(f"{topic} podcast spotify apple", 4),
            )
        if domain:
            podcasts.extend(
                self._search_web(f"{domain} podcast discussion", 4),
            )
        for kw in info.get("keywords", [])[:2]:
            podcasts.extend(
                self._search_web(f"{kw} podcast", 3),
            )
        return self._deduplicate(podcasts)

    def _find_newsletters(self, info: dict) -> list[dict]:
        """Search for newsletter / Substack / email-digest content related to the paper."""
        newsletters: list[dict] = []
        topic = info.get("topic", "")
        domain = info.get("domain", "")
        if topic:
            newsletters.extend(self._search_web(f"{topic} newsletter substack", 5))
            newsletters.extend(self._search_web(f"{topic} newsletter email digest", 3))
        if domain:
            newsletters.extend(self._search_web(f"{domain} newsletter substack", 4))
        for kw in info.get("keywords", [])[:3]:
            newsletters.extend(self._search_web(f"{kw} newsletter", 3))
        newsletters = self._deduplicate(newsletters)
        # Filter out Wikipedia pages
        newsletters = [n for n in newsletters if "wikipedia.org" not in n.get("url", "")]
        return newsletters

    def _find_wiki(self, info: dict) -> list[dict]:
        """Search for Wikipedia pages related to the paper's topics."""
        wiki: list[dict] = []
        topic = info.get("topic", "")
        domain = info.get("domain", "")
        if topic:
            wiki.extend(self._search_web(f"{topic} site:en.wikipedia.org", 5))
        if domain:
            wiki.extend(self._search_web(f"{domain} site:en.wikipedia.org", 4))
        for kw in info.get("keywords", [])[:4]:
            wiki.extend(self._search_web(f"{kw} site:en.wikipedia.org", 3))
        for kw in info.get("semantic_keywords", [])[:2]:
            wiki.extend(self._search_web(f"{kw} wikipedia", 2))
        # Keep only actual Wikipedia links
        wiki = [w for w in wiki if "wikipedia.org" in w.get("url", "")]
        return self._deduplicate(wiki)

    # ────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────

    def process(
        self,
        text: str,
        search_types: list[str] | None = None,
        references: list[str] | None = None,
    ) -> dict:
        """Run the full multi-model recommendation pipeline.

        search_types — any subset of ['papers', 'blogs', 'videos'].
        references  — paper titles extracted from the References section
                      (empty list or None when the section is absent).
        """
        if search_types is None:
            search_types = ["papers", "blogs", "videos", "podcasts", "newsletters", "wiki"]

        # ── Step 1: LLM topic extraction ──
        info = self._analyze_article(text)

        # ── Step 2: Multi-model keyword extraction (with fallback) ──
        tfidf_kws: list[str] = []
        nltk_kws: list[str] = []
        semantic_kws: list[str] = []

        try:
            tfidf_kws = self._extract_tfidf_keywords(text)
        except Exception:
            pass

        try:
            nltk_kws = self._extract_nltk_keyphrases(text)
        except Exception:
            pass

        try:
            all_candidates = list({
                c.lower().strip(): c
                for c in (tfidf_kws + nltk_kws + info.get("keywords", []))
                if len(c.strip()) > 2
            }.values())
            semantic_kws = self._extract_semantic_keywords(text, all_candidates)
        except Exception:
            pass

        # Merge all keyword sources (LLM keywords always available as fallback)
        try:
            merged_kws = self._merge_keywords(
                info.get("keywords", []), tfidf_kws, semantic_kws, nltk_kws,
            )
        except Exception:
            merged_kws = info.get("keywords", [])

        # Inject merged keywords back into info for search strategies
        info["keywords"] = merged_kws[:10] if merged_kws else info.get("keywords", [])
        info["semantic_keywords"] = semantic_kws
        info["nltk_keyphrases"] = nltk_kws

        # ── Step 3: Search the web ──
        papers = (
            self._find_papers(info, references=references)[:25]
            if "papers" in search_types else []
        )
        blogs = self._find_blogs(info)[:15] if "blogs" in search_types else []
        videos = self._find_videos(info) if "videos" in search_types else []
        podcasts = self._find_podcasts(info) if "podcasts" in search_types else []
        newsletters = self._find_newsletters(info) if "newsletters" in search_types else []
        wiki = self._find_wiki(info) if "wiki" in search_types else []

        # ── Step 4: Semantic reranking (with fallback to unranked) ──
        try:
            if papers:
                papers = self._semantic_rerank(text, papers, top_n=15)
        except Exception:
            papers = papers[:15]

        try:
            if blogs:
                blogs = self._semantic_rerank(text, blogs, top_n=10)
        except Exception:
            blogs = blogs[:10]

        try:
            if videos:
                videos = self._semantic_rerank(text, videos, top_n=8)
        except Exception:
            videos = videos[:8]

        try:
            if podcasts:
                podcasts = self._semantic_rerank(text, podcasts, top_n=8)
        except Exception:
            podcasts = podcasts[:8]

        try:
            if newsletters:
                newsletters = self._semantic_rerank(text, newsletters, top_n=8)
        except Exception:
            newsletters = newsletters[:8]

        try:
            if wiki:
                wiki = self._semantic_rerank(text, wiki, top_n=10)
        except Exception:
            wiki = wiki[:10]

        return {
            "keywords": merged_kws[:10],
            "paper_title": info.get("title", ""),
            "domain": info.get("domain", ""),
            "topic": info.get("topic", ""),
            "research_papers": papers,
            "blogs": blogs,
            "videos": videos,
            "podcasts": podcasts,
            "newsletters": newsletters,
            "wiki": wiki,
        }
