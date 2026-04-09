"""Configuration for PaperAI Multi-Agent Framework."""

# --------------- Embedding Model (always local) ---------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------- HuggingFace Inference API (Free Tier) ---------------
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# --------------- Groq (Free Tier) ---------------
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"

# --------------- Google Gemini (Free Tier) ---------------
GEMINI_MODEL = "gemini-2.0-flash"

# --------------- Text Processing ---------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS_FOR_CONTEXT = 5

# --------------- Search / Recommendations ---------------
MAX_SEARCH_RESULTS_PER_QUERY = 5
MAX_KEYWORDS = 5

# --------------- Agent Names ---------------
AGENT_NAMES = {
    "document": "Document Parser Agent",
    "summarizer": "Summarizer Agent",
    "segmentation": "Segmentation Agent",
    "recommendation": "Recommendation Agent",
    "chat": "Chat Agent",
}
