"""
Shared embedding model loader.
Single source of truth — all agents import from here.
"""

from __future__ import annotations

import streamlit as st


@st.cache_resource
def get_embedding_model():
    """Load and cache the sentence-transformer model once across all agents."""
    import logging
    import warnings
    from sentence_transformers import SentenceTransformer

    # Suppress the harmless "embeddings.position_ids UNEXPECTED" warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*position_ids.*")
        logging.getLogger("safetensors").setLevel(logging.ERROR)
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
