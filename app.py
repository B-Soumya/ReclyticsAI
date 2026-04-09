"""
ReclyticsAI — Multi-Agent Article Analyzer
Run:  streamlit run app.py
"""

import os
from io import BytesIO

from dotenv import load_dotenv
import streamlit as st

# Load .env file (API keys, etc.)
load_dotenv()

# ───────────────────── Page config ──────────────────────
st.set_page_config(
    page_title="ReclyticsAI - AI Research Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────── Supported languages ──────────────────
LANGUAGES = [
    "English",
    "Chinese (中文)",
    "Hindi (हिन्दी)",
    "Spanish (Español)",
    "French (Français)",
    "Arabic (العربية)",
    "Bengali (বাংলা)",
    "Portuguese (Português)",
    "Russian (Русский)",
    "Japanese (日本語)",
]

# ───────────────────── Custom CSS ───────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════
   ReclyticsAI — Dark Glassmorphism Theme
   ══════════════════════════════════════════════════════ */

/* ── Background — deep dark with colored light blobs ─ */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 700px 500px at 15% 20%, rgba(124,106,255,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 50%, rgba(139,92,246,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 600px 400px at 50% 85%, rgba(99,102,241,0.06) 0%, transparent 70%),
        linear-gradient(135deg, #06060f 0%, #0e0b2b 30%, #140f3a 55%, #0a1628 100%);
    color: #f0f0f5;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar — frosted glass panel ───────────────── */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
            rgba(20, 16, 48, 0.85) 0%,
            rgba(16, 14, 42, 0.90) 50%,
            rgba(12, 18, 38, 0.85) 100%) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(124, 106, 255, 0.16);
    box-shadow: 4px 0 40px rgba(0, 0, 0, 0.4),
                1px 0 15px rgba(124, 106, 255, 0.06),
                inset -1px 0 0 rgba(255,255,255,0.03);
}
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    max-height: 100vh !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stCheckbox label span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #d4daf0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(124, 106, 255, 0.08) !important;
    margin: 12px 0 !important;
}

/* ── Sidebar brand ────────────────────────────────── */
.sidebar-brand {
    text-align: center; padding: 24px 12px 12px;
    border-bottom: 1px solid rgba(124, 106, 255, 0.08);
    margin-bottom: 8px;
}
.sidebar-brand .logo {
    width: 50px; height: 50px; margin: 0 auto 10px;
    background: linear-gradient(135deg, #7c6aff, #a855f7, #c084fc);
    border-radius: 15px; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 16px rgba(124, 106, 255, 0.4),
                0 0 40px rgba(124, 106, 255, 0.15),
                0 4px 20px rgba(124, 106, 255, 0.3);
}
.sidebar-brand .logo svg { width: 28px; height: 28px; }
.sidebar-brand .title {
    font-size: 1.5rem; font-weight: 800; letter-spacing: 1.5px;
    background: linear-gradient(135deg, #a5b4fc, #c084fc, #f0abfc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-brand .subtitle {
    font-size: 0.7rem; color: #6a6f95; margin-top: 2px; letter-spacing: 2.5px;
}

.sidebar-section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #7c6aff !important;
    margin: 14px 0 6px; padding-left: 2px;
}

/* ── Tabs — glass bar ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background:
        linear-gradient(135deg,
            rgba(20, 16, 46, 0.65) 0%,
            rgba(14, 16, 38, 0.6) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 106, 255, 0.15);
    border-radius: 16px; padding: 6px;
    box-shadow: 0 0 12px rgba(124, 106, 255, 0.06),
                0 4px 16px rgba(0, 0, 0, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px; padding: 10px 28px;
    font-weight: 600; color: #8a8fb5; font-size: 0.95rem;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #c4b5fd;
    background: rgba(124, 106, 255, 0.06);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: #fff !important; border-radius: 12px;
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.35),
                inset 0 1px 0 rgba(255,255,255,0.1);
}

/* ══════════════════════════════════════════════════════
   Glass Card — Frosted glass containers
   ══════════════════════════════════════════════════════ */

/* Tab panels — main glass card with glow border */
.stTabs [data-baseweb="tab-panel"] {
    background:
        linear-gradient(135deg,
            rgba(22, 18, 50, 0.75) 0%,
            rgba(16, 18, 42, 0.70) 50%,
            rgba(20, 16, 48, 0.75) 100%);
    backdrop-filter: blur(16px) saturate(1.2);
    -webkit-backdrop-filter: blur(16px) saturate(1.2);
    border: 1px solid rgba(124, 106, 255, 0.18);
    border-radius: 20px;
    padding: 32px 28px;
    margin-top: 10px;
    box-shadow: 0 0 15px rgba(124, 106, 255, 0.08),
                0 0 40px rgba(124, 106, 255, 0.04),
                0 8px 40px rgba(0, 0, 0, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
    position: relative;
}
.stTabs [data-baseweb="tab-panel"]::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(124, 106, 255, 0.4) 20%,
        rgba(192, 132, 252, 0.3) 50%,
        rgba(124, 106, 255, 0.4) 80%, transparent 100%);
    border-radius: 20px 20px 0 0;
}

/* Main content area */
[data-testid="stMainBlockContainer"] {
    background: transparent;
}

/* Status / pipeline steps — glass panels with glow */
[data-testid="stStatusWidget"],
div[data-testid="stExpander"],
details[data-testid="stExpander"] {
    background:
        linear-gradient(135deg,
            rgba(18, 16, 44, 0.7) 0%,
            rgba(14, 16, 38, 0.65) 100%) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 106, 255, 0.15) !important;
    border-radius: 14px !important;
    margin-bottom: 8px;
    box-shadow: 0 0 12px rgba(124, 106, 255, 0.06),
                0 4px 20px rgba(0, 0, 0, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
details[data-testid="stExpander"] summary {
    color: #d4daf0 !important;
}

/* Alert boxes — glass */
[data-testid="stAlert"] {
    background:
        linear-gradient(135deg,
            rgba(16, 14, 38, 0.65) 0%,
            rgba(18, 16, 42, 0.60) 100%) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 12px !important;
    color: #d4daf0 !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

/* ══════════════════════════════════════════════════════
   Recommendations — Redesigned UI
   ══════════════════════════════════════════════════════ */

/* ── Rec header — glass banner with paper metadata ─── */
.rec-header {
    background: linear-gradient(135deg, rgba(22, 18, 54, 0.8), rgba(16, 20, 46, 0.7));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(124, 106, 255, 0.18);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.rec-header::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #7c6aff, #a855f7, #c084fc, transparent);
}
.rec-header .rec-title {
    font-size: 1.1rem; font-weight: 700; color: #e8e8f0; margin-bottom: 4px;
}
.rec-header .rec-meta {
    display: flex; gap: 20px; margin-top: 8px; flex-wrap: wrap;
}
.rec-header .rec-meta-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; color: #8a8fb5;
}
.rec-header .rec-meta-item .meta-icon {
    width: 18px; height: 18px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
}

/* ── Stats bar — summary counts ──────────────────── */
.rec-stats {
    display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
}
.rec-stat-card {
    flex: 1; min-width: 120px;
    background: linear-gradient(135deg, rgba(20, 18, 48, 0.7), rgba(16, 18, 42, 0.6));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 106, 255, 0.14);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
    transition: all 0.25s ease;
    box-shadow: 0 0 10px rgba(124, 106, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.rec-stat-card:hover {
    border-color: rgba(124, 106, 255, 0.3);
    box-shadow: 0 0 18px rgba(124, 106, 255, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
.rec-stat-card .stat-num {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, var(--stat-c1, #a5b4fc), var(--stat-c2, #c084fc));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.rec-stat-card.active {
    border-color: rgba(124, 106, 255, 0.45);
    box-shadow: 0 0 24px rgba(124, 106, 255, 0.18),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
    background: linear-gradient(135deg, rgba(124, 106, 255, 0.15), rgba(16, 18, 42, 0.7));
}
.rec-stat-card .stat-label {
    font-size: 0.7rem; font-weight: 600; color: #7a7fa5;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;
}

/* ── Stat card button — minimal style, card is the visual ── */
.rec-stats + div button[kind="secondary"],
.rec-stats + div button[kind="primary"],
div[data-testid="column"] button[key^="rec_tab_"] {
    margin-top: -6px !important;
    padding: 4px 8px !important;
    font-size: 0.65rem !important;
    border-radius: 8px !important;
    opacity: 0.7;
    letter-spacing: 0.5px;
}

/* ── Section header with icon ────────────────────── */
.rec-section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 16px; padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.rec-section-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.rec-section-icon.purple { background: linear-gradient(135deg, #6c5ce7, #a855f7); }
.rec-section-icon.green  { background: linear-gradient(135deg, #059669, #34d399); }
.rec-section-icon.red    { background: linear-gradient(135deg, #dc2626, #f87171); }
.rec-section-icon.orange { background: linear-gradient(135deg, #d97706, #fb923c); }
.rec-section-icon.cyan   { background: linear-gradient(135deg, #0891b2, #22d3ee); }
.rec-section-icon.pink   { background: linear-gradient(135deg, #db2777, #f472b6); }
.rec-section-title {
    font-size: 1.05rem; font-weight: 700; color: #e8e8f0;
}
.rec-section-count {
    font-size: 0.72rem; color: #6a6f95;
    background: rgba(255,255,255,0.04);
    padding: 3px 10px; border-radius: 10px;
    margin-left: auto;
}

/* ── Recommendation cards — enhanced glass ───────── */
.rec-card {
    border-radius: 16px;
    padding: 20px 24px 20px 28px;
    margin-bottom: 12px;
    position: relative;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    display: flex;
    gap: 16px;
}
.rec-card:hover {
    transform: translateY(-3px) scale(1.005);
}
/* Top-edge glow line — always visible */
.rec-card::after {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--card-glow, rgba(124,106,255,0.3)), transparent);
    opacity: 1;
}

/* Left glow border — colored bar with blurred glow behind it */
.rec-card::before {
    content: "";
    position: absolute; top: 8px; bottom: 8px; left: 0; width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--left-glow-solid, #7c6aff);
    box-shadow: 0 0 8px 2px var(--left-glow-blur, rgba(124,106,255,0.5)),
                0 0 20px 4px var(--left-glow-spread, rgba(124,106,255,0.2));
}

/* Card variants — always-on glow borders + left glow */
.rec-card.purple {
    background: linear-gradient(135deg, rgba(108,92,231,0.14), rgba(139,92,246,0.05));
    border: 1px solid rgba(124,106,255,0.22);
    --card-glow: rgba(124,106,255,0.5);
    --left-glow-solid: #7c6aff;
    --left-glow-blur: rgba(124,106,255,0.55);
    --left-glow-spread: rgba(124,106,255,0.2);
    box-shadow: 0 0 14px rgba(124,106,255,0.10),
                0 0 36px rgba(124,106,255,0.04),
                inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.purple:hover {
    border-color: rgba(124,106,255,0.4);
    box-shadow: 0 0 22px rgba(124,106,255,0.18),
                0 0 50px rgba(124,106,255,0.07),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
.rec-card.green {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(52,211,153,0.04));
    border: 1px solid rgba(52,211,153,0.20);
    --card-glow: rgba(52,211,153,0.5);
    --left-glow-solid: #34d399;
    --left-glow-blur: rgba(52,211,153,0.55);
    --left-glow-spread: rgba(52,211,153,0.2);
    box-shadow: 0 0 14px rgba(52,211,153,0.09),
                0 0 36px rgba(52,211,153,0.03),
                inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.green:hover {
    border-color: rgba(52,211,153,0.4);
    box-shadow: 0 0 22px rgba(52,211,153,0.16),
                0 0 50px rgba(52,211,153,0.06),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
.rec-card.red {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(248,113,113,0.04));
    border: 1px solid rgba(248,113,113,0.20);
    --card-glow: rgba(248,113,113,0.5);
    --left-glow-solid: #f87171;
    --left-glow-blur: rgba(248,113,113,0.55);
    --left-glow-spread: rgba(248,113,113,0.2);
    box-shadow: 0 0 14px rgba(248,113,113,0.09),
                0 0 36px rgba(248,113,113,0.03),
                inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.red:hover {
    border-color: rgba(248,113,113,0.4);
    box-shadow: 0 0 22px rgba(248,113,113,0.16),
                0 0 50px rgba(248,113,113,0.06),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
.rec-card.orange {
    background: linear-gradient(135deg, rgba(251,146,60,0.12), rgba(251,191,36,0.04));
    border: 1px solid rgba(251,146,60,0.20);
    --card-glow: rgba(251,146,60,0.5);
    --left-glow-solid: #fb923c;
    --left-glow-blur: rgba(251,146,60,0.55);
    --left-glow-spread: rgba(251,146,60,0.2);
    box-shadow: 0 0 14px rgba(251,146,60,0.09),
                0 0 36px rgba(251,146,60,0.03),
                inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.orange:hover {
    border-color: rgba(251,146,60,0.4);
    box-shadow: 0 0 22px rgba(251,146,60,0.16),
                0 0 50px rgba(251,146,60,0.06),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
.rec-card.cyan {
    background: linear-gradient(135deg, rgba(8,145,178,0.14), rgba(34,211,238,0.05));
    border: 1px solid rgba(34,211,238,0.22);
    --card-glow: rgba(34,211,238,0.5);
    --left-glow-solid: #22d3ee;
    --left-glow-blur: rgba(34,211,238,0.55);
    --left-glow-spread: rgba(34,211,238,0.2);
    box-shadow: 0 0 14px rgba(34,211,238,0.10), 0 0 36px rgba(34,211,238,0.04), inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.cyan:hover {
    border-color: rgba(34,211,238,0.4);
    box-shadow: 0 0 22px rgba(34,211,238,0.22),
                0 0 50px rgba(34,211,238,0.08),
                inset 0 1px 0 rgba(255,255,255,0.07);
}
.rec-card.pink {
    background: linear-gradient(135deg, rgba(219,39,119,0.14), rgba(244,114,182,0.05));
    border: 1px solid rgba(244,114,182,0.22);
    --card-glow: rgba(244,114,182,0.5);
    --left-glow-solid: #f472b6;
    --left-glow-blur: rgba(244,114,182,0.55);
    --left-glow-spread: rgba(244,114,182,0.2);
    box-shadow: 0 0 14px rgba(244,114,182,0.10), 0 0 36px rgba(244,114,182,0.04), inset 0 1px 0 rgba(255,255,255,0.05);
}
.rec-card.pink:hover {
    border-color: rgba(244,114,182,0.4);
    box-shadow: 0 0 22px rgba(244,114,182,0.22),
                0 0 50px rgba(244,114,182,0.08),
                inset 0 1px 0 rgba(255,255,255,0.07);
}

/* Card rank number */
.rec-rank {
    flex-shrink: 0;
    width: 32px; height: 32px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 800; color: #e8e8f0;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 2px;
}

/* Card body */
.rec-card-body { flex: 1; min-width: 0; }
.rec-card-body .card-title {
    font-size: 0.95rem; font-weight: 600; line-height: 1.4;
    margin-bottom: 6px; display: flex; align-items: flex-start; gap: 8px;
    flex-wrap: wrap;
}
.rec-card-body .card-title a {
    text-decoration: none; transition: color 0.2s;
}
.rec-card.purple .card-title a { color: #a5b4fc; }
.rec-card.purple .card-title a:hover { color: #c4b5fd; }
.rec-card.green .card-title a { color: #6ee7b7; }
.rec-card.green .card-title a:hover { color: #a7f3d0; }
.rec-card.red .card-title a { color: #fca5a5; }
.rec-card.red .card-title a:hover { color: #fecaca; }
.rec-card.orange .card-title a { color: #fdba74; }
.rec-card.orange .card-title a:hover { color: #fed7aa; }
.rec-card.cyan .card-title a { color: #67e8f9; }
.rec-card.cyan .card-title a:hover { color: #a5f3fc; }
.rec-card.pink .card-title a { color: #f9a8d4; }
.rec-card.pink .card-title a:hover { color: #fbcfe8; }

.rec-card-body .card-snippet {
    font-size: 0.82rem; color: #9a9ec0; line-height: 1.6;
    margin-bottom: 8px;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.rec-card-body .card-footer {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}
.rec-card-body .card-domain {
    font-size: 0.72rem; color: #6a6f95;
    background: rgba(255,255,255,0.04);
    padding: 3px 10px; border-radius: 8px;
    max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.rec-card-body .card-publisher {
    font-size: 0.72rem; color: #8a8fb5; font-style: italic;
}

/* ── Relevance bar (replaces old badge) ──────────── */
.rel-bar-wrap {
    display: flex; align-items: center; gap: 8px;
    margin-left: auto; flex-shrink: 0;
}
.rel-bar-track {
    width: 60px; height: 5px; border-radius: 4px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}
.rel-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.rel-bar-fill.high   { background: linear-gradient(90deg, #34d399, #6ee7b7); }
.rel-bar-fill.medium { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.rel-bar-fill.low    { background: linear-gradient(90deg, #6b7280, #9ca3af); }
.rel-bar-pct {
    font-size: 0.7rem; font-weight: 700; min-width: 32px; text-align: right;
}
.rel-bar-pct.high   { color: #6ee7b7; }
.rel-bar-pct.medium { color: #fcd34d; }
.rel-bar-pct.low    { color: #9ca3af; }

/* ── Keyword chips — enhanced ────────────────────── */
.kw-wrap {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-bottom: 20px;
}
.kw-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(124, 106, 255, 0.06);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(124, 106, 255, 0.15);
    color: #a5b4fc;
    padding: 6px 16px; border-radius: 22px; font-size: 0.8rem;
    font-weight: 500;
    transition: all 0.25s ease;
    cursor: default;
}
.kw-chip::before {
    content: "#"; font-weight: 700; opacity: 0.5; font-size: 0.75rem;
}
.kw-chip:hover {
    background: rgba(124, 106, 255, 0.14);
    border-color: rgba(124, 106, 255, 0.35);
    box-shadow: 0 2px 16px rgba(124, 106, 255, 0.12);
    transform: translateY(-1px);
}

/* ── Empty state ─────────────────────────────────── */
.rec-empty {
    text-align: center; padding: 48px 24px;
    color: #6a6f95;
}
.rec-empty .empty-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.5; }
.rec-empty p { font-size: 0.9rem; max-width: 400px; margin: 0 auto; }

/* ══════════════════════════════════════════════════════
   Hero / Landing Page
   ══════════════════════════════════════════════════════ */
.hero { text-align: center; padding: 80px 20px 30px; }
.hero h1 {
    font-size: 3.6rem; font-weight: 800; margin-bottom: 14px;
    background: linear-gradient(135deg, #7c6aff 0%, #a855f7 40%, #f0abfc 80%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 8px rgba(124, 106, 255, 0.15));
}
.hero p {
    font-size: 1.12rem; color: #8a8fb5;
    max-width: 640px; margin: 0 auto 40px; line-height: 1.75;
}

/* ── Agent grid — floating glass cards ────────────── */
.agent-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 14px; margin: 28px auto; max-width: 700px;
}
@keyframes float {
    0%   { transform: translateY(0px); }
    50%  { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}
@keyframes glow-pulse {
    0%   { box-shadow: 0 0 12px rgba(124,106,255,0.08), 0 0 30px rgba(124,106,255,0.04), inset 0 1px 0 rgba(255,255,255,0.06); }
    50%  { box-shadow: 0 0 18px rgba(124,106,255,0.16), 0 0 44px rgba(124,106,255,0.07), inset 0 1px 0 rgba(255,255,255,0.08); }
    100% { box-shadow: 0 0 12px rgba(124,106,255,0.08), 0 0 30px rgba(124,106,255,0.04), inset 0 1px 0 rgba(255,255,255,0.06); }
}
.agent-item {
    background:
        linear-gradient(135deg,
            rgba(22, 18, 50, 0.65) 0%,
            rgba(18, 20, 46, 0.55) 50%,
            rgba(22, 18, 50, 0.65) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(124, 106, 255, 0.18);
    border-radius: 14px; padding: 16px 12px; text-align: center;
    animation: float 3.5s ease-in-out infinite, glow-pulse 4s ease-in-out infinite;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 12px rgba(124, 106, 255, 0.08),
                0 0 30px rgba(124, 106, 255, 0.04),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
.agent-item::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.agent-item:nth-child(1) { animation-delay: 0s, 0s; }
.agent-item:nth-child(2) { animation-delay: 0.5s, 0.8s; }
.agent-item:nth-child(3) { animation-delay: 1s, 1.6s; }
.agent-item:nth-child(4) { animation-delay: 1.5s, 0.4s; }
.agent-item:nth-child(5) { animation-delay: 2s, 1.2s; }
.agent-item:nth-child(6) { animation-delay: 2.5s, 2s; }
.agent-item:hover {
    border-color: rgba(124, 106, 255, 0.35);
    box-shadow: 0 8px 32px rgba(124, 106, 255, 0.18);
    background: rgba(124, 106, 255, 0.05);
    transform: translateY(-4px);
}
.agent-icon { font-size: 1.5rem; margin-bottom: 6px; }
.agent-name { font-weight: 700; color: #e8e8f0; margin-bottom: 4px; font-size: 0.82rem; }
.agent-desc { font-size: 0.72rem; color: #7a7fa5; line-height: 1.35; }

/* ── Typography ───────────────────────────────────── */
.stMarkdown h1 { color: #f0f0f5; }
.stMarkdown h2 { color: #e0e2f0; }
.stMarkdown h3 { color: #d0d4e8; }
.stMarkdown p, .stMarkdown li { color: #c8cce0; line-height: 1.7; }
.stMarkdown strong { color: #e8e8f0; }
.stMarkdown code { color: #c4b5fd; }
.stMarkdown a { color: #a5b4fc; }

/* ── Chat messages — glass bubbles with glow ─────── */
.stChatMessage {
    border-radius: 16px;
    background:
        linear-gradient(135deg,
            rgba(20, 16, 46, 0.6) 0%,
            rgba(16, 18, 40, 0.5) 100%) !important;
    border: 1px solid rgba(124, 106, 255, 0.14);
    box-shadow: 0 0 10px rgba(124, 106, 255, 0.06),
                0 4px 16px rgba(0, 0, 0, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* ── Primary button — glass accent ────────────────── */
button[kind="primary"],
[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6c5ce7, #8b5cf6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(108, 92, 231, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.25s ease;
    border-radius: 10px !important;
}
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #7d6ff0, #a06ef8) !important;
    color: #fff !important;
    box-shadow: 0 6px 28px rgba(124, 106, 255, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
    transform: translateY(-1px);
}
button[kind="primary"]:focus,
[data-testid="baseButton-primary"]:focus,
.stButton > button[kind="primary"]:focus {
    box-shadow: 0 0 0 3px rgba(124, 106, 255, 0.3) !important;
}
button[kind="primary"]:active,
[data-testid="baseButton-primary"]:active {
    transform: translateY(0);
}

/* ── Secondary buttons ────────────────────────────── */
.stButton > button:not([kind="primary"]) {
    border-color: rgba(124, 106, 255, 0.2) !important;
    color: #c4b5fd !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 10px !important;
    transition: all 0.25s ease;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(124, 106, 255, 0.45) !important;
    color: #e0d4ff !important;
    background: rgba(124, 106, 255, 0.06) !important;
    box-shadow: 0 4px 16px rgba(124, 106, 255, 0.1);
}

/* ── Toggle / Switch ──────────────────────────────── */
[data-testid="stToggle"] span[data-checked="true"],
[data-testid="stToggle"] [role="switch"][aria-checked="true"],
[data-testid="stToggle"] label > div[data-checked="true"],
div[role="switch"][aria-checked="true"] {
    background-color: #7c6aff !important;
}
[data-testid="stToggle"] span[data-checked="false"],
div[role="switch"][aria-checked="false"] {
    background-color: #2a2a4a !important;
}

/* ── Radio buttons ────────────────────────────────── */
[role="radiogroup"] [data-testid="stMarkdownContainer"] + div,
.stRadio > div[role="radiogroup"] label > div:first-child > div {
    border-color: #4a4a6a !important;
}
[role="radio"][aria-checked="true"] > div:first-child,
.stRadio [aria-checked="true"] > div:first-child > div {
    background-color: #7c6aff !important;
    border-color: #7c6aff !important;
    box-shadow: 0 0 8px rgba(124, 106, 255, 0.3);
}

/* ── Checkboxes ───────────────────────────────────── */
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] svg,
.stCheckbox [aria-checked="true"] svg {
    fill: #7c6aff !important;
    stroke: #7c6aff !important;
}
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"],
.stCheckbox [aria-checked="true"] > div:first-child {
    background-color: #7c6aff !important;
    border-color: #7c6aff !important;
}
[data-testid="stCheckbox"] [role="checkbox"][aria-checked="false"],
.stCheckbox [aria-checked="false"] > div:first-child {
    border-color: #4a4a6a !important;
}

/* ── Selectbox — glass dropdown ───────────────────── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:
        linear-gradient(135deg,
            rgba(18, 16, 42, 0.7) 0%,
            rgba(14, 14, 36, 0.65) 100%) !important;
    border-color: rgba(124, 106, 255, 0.15) !important;
    color: #d4daf0 !important;
    border-radius: 10px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: rgba(124, 106, 255, 0.35) !important;
}

/* ── Text input — glass field ─────────────────────── */
[data-testid="stTextInput"] input {
    background:
        linear-gradient(135deg,
            rgba(18, 16, 42, 0.7) 0%,
            rgba(14, 14, 36, 0.65) 100%) !important;
    border-color: rgba(124, 106, 255, 0.15) !important;
    color: #d4daf0 !important;
    border-radius: 10px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c6aff !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 255, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}

/* ── File uploader — glass dropzone ───────────────── */
[data-testid="stFileUploader"] section {
    background:
        linear-gradient(135deg,
            rgba(18, 16, 42, 0.5) 0%,
            rgba(14, 14, 36, 0.45) 100%) !important;
    border: 1px dashed rgba(124, 106, 255, 0.2) !important;
    border-radius: 14px;
    transition: all 0.25s ease;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(124, 106, 255, 0.4) !important;
    background:
        linear-gradient(135deg,
            rgba(24, 20, 52, 0.55) 0%,
            rgba(18, 18, 42, 0.5) 100%) !important;
}
[data-testid="stFileUploader"] button {
    color: #c4b5fd !important;
    border-color: rgba(124, 106, 255, 0.25) !important;
}

/* ── Chat input — glass bar ───────────────────────── */
[data-testid="stChatInput"] textarea {
    background:
        linear-gradient(135deg,
            rgba(18, 16, 42, 0.6) 0%,
            rgba(14, 14, 36, 0.55) 100%) !important;
    border-color: rgba(124, 106, 255, 0.12) !important;
    color: #f0f0f5 !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7c6aff !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 255, 0.18),
                inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}

/* ── Spinner ──────────────────────────────────────── */
.stSpinner > div > div { border-top-color: #7c6aff !important; }

/* ── Scrollbar — subtle ──────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(124, 106, 255, 0.2);
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(124, 106, 255, 0.35);
}

/* ── Chat – blue & white glow ────────────── */
@keyframes chat-glow {
    0%,100% { box-shadow: 0 0 14px rgba(59,130,246,0.12), 0 0 40px rgba(59,130,246,0.04); }
    50%     { box-shadow: 0 0 22px rgba(59,130,246,0.24), 0 0 54px rgba(59,130,246,0.08); }
}
@keyframes mic-ring {
    0%,100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); }
    50%     { box-shadow: 0 0 0 8px rgba(59,130,246,0); }
}
/* Chat container */
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stChatMessage"]) {
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 16px;
    padding: 14px 8px;
    animation: chat-glow 4s ease-in-out infinite;
    background: linear-gradient(180deg, rgba(59,130,246,0.04) 0%, transparent 100%);
}
/* User messages — blue bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(59,130,246,0.08)) !important;
    border: 1px solid rgba(96,165,250,0.22);
    border-left: 3px solid #3b82f6;
    border-radius: 14px; margin-bottom: 6px;
    transition: all 0.3s;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]):hover {
    border-color: rgba(96,165,250,0.40);
    box-shadow: 0 0 18px rgba(59,130,246,0.12);
}
/* Assistant messages — white/light bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(241,245,249,0.03)) !important;
    border: 1px solid rgba(255,255,255,0.10);
    border-left: 3px solid rgba(255,255,255,0.30);
    border-radius: 14px; margin-bottom: 6px;
    transition: all 0.3s;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]):hover {
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 0 18px rgba(255,255,255,0.06);
}
/* Chat input — blue glow */
[data-testid="stChatInput"] {
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.10), 0 0 40px rgba(59,130,246,0.03);
    transition: all 0.3s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(59,130,246,0.50) !important;
    box-shadow: 0 0 24px rgba(59,130,246,0.25), 0 0 54px rgba(59,130,246,0.08);
}
/* Voice recorder widget — compact blue accent */
[data-testid="stAudioInput"] {
    max-width: 160px;
}
[data-testid="stAudioInput"] > div {
    border: 1px solid rgba(59,130,246,0.30) !important;
    border-radius: 24px !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.10);
    background: rgba(59,130,246,0.06) !important;
    padding: 2px 6px !important;
}
[data-testid="stAudioInput"] button {
    color: #60a5fa !important;
}
/* Voice + chat bottom bar */
.voice-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
}
.voice-bar .voice-label {
    color: rgba(148,163,184,0.7);
    font-size: 0.78rem;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────── Sidebar ──────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="logo">'
        '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="4" y="1" width="16" height="22" rx="2" fill="#fff" opacity="0.15" stroke="#fff" stroke-width="1"/>'
        '<line x1="7" y1="5" x2="17" y2="5" stroke="#fff" stroke-width="0.8" opacity="0.4"/>'
        '<line x1="7" y1="8" x2="14" y2="8" stroke="#fff" stroke-width="0.8" opacity="0.3"/>'
        '<circle cx="12" cy="15" r="4.5" fill="none" stroke="#c084fc" stroke-width="1.2"/>'
        '<circle cx="10.5" cy="14" r="1" fill="#c084fc"/>'
        '<circle cx="13.5" cy="14" r="1" fill="#c084fc"/>'
        '<circle cx="12" cy="16.8" r="0.8" fill="#c084fc"/>'
        '<line x1="10.5" y1="14" x2="13.5" y2="14" stroke="#c084fc" stroke-width="0.6"/>'
        '<line x1="10.5" y1="14" x2="12" y2="16.8" stroke="#c084fc" stroke-width="0.6"/>'
        '<line x1="13.5" y1="14" x2="12" y2="16.8" stroke="#c084fc" stroke-width="0.6"/>'
        '</svg></div>'
        '<div class="title">RECLYTICSAI</div>'
        '<div class="subtitle">AI RESEARCH ANALYZER</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Upload
    st.markdown('<p class="sidebar-section-label">Document</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx", "doc"],
                                     label_visibility="collapsed")

    st.markdown("---")

    # Recommendations
    st.markdown('<p class="sidebar-section-label">Recommendations</p>', unsafe_allow_html=True)
    enable_recs = st.toggle("Enable Recommendations", value=False)
    rec_papers = rec_blogs = rec_videos = rec_podcasts = rec_newsletters = rec_wiki = False
    if enable_recs:
        rec_papers = st.checkbox("Research Papers", value=True)
        rec_blogs = st.checkbox("Blog Articles", value=True)
        rec_videos = st.checkbox("YouTube Videos", value=True)
        rec_podcasts = st.checkbox("Podcasts", value=True)
        rec_newsletters = st.checkbox("Newsletters", value=True)
        rec_wiki = st.checkbox("Wikipedia Pages", value=True)

    st.markdown("---")
    analyze_btn = st.button("🚀  Analyze Article", use_container_width=True,
                            type="primary", disabled=uploaded_file is None)

    st.markdown("---")

    # ── Settings (collapsible) ──
    with st.expander("⚙️  Settings", expanded=False):
        # Language
        st.markdown('<p class="sidebar-section-label">Language</p>', unsafe_allow_html=True)
        language = st.selectbox("Output language", LANGUAGES, index=0, label_visibility="collapsed")

        st.markdown("---")

        # LLM Configuration
        st.markdown('<p class="sidebar-section-label">LLM Provider</p>', unsafe_allow_html=True)
        provider = st.radio(
            "Select LLM Provider (all free)",
            ["Ollama (Local - No Key)", "Groq (Multi-Model)", "Google Gemini", "HuggingFace (Qwen 2.5)"],
            label_visibility="collapsed",
        )
        api_key = ""
        ollama_model = ""
        if "Ollama" in provider:
            provider_name = "ollama"
            # Dynamically fetch installed Ollama models with sizes
            _ollama_models = []
            try:
                import requests as _rq
                _resp = _rq.get("http://localhost:11434/api/tags", timeout=3).json()
                for m in _resp.get("models", []):
                    name = m["name"]
                    size_gb = m.get("size", 0) / 1e9
                    param = m.get("details", {}).get("parameter_size", "")
                    label = f"{name}  ({param}, {size_gb:.1f} GB)"
                    _ollama_models.append((name, label))
            except Exception:
                pass
            if _ollama_models:
                _labels = [label for _, label in _ollama_models]
                _sel = st.selectbox("Ollama Model", _labels, index=0)
                ollama_model = _ollama_models[_labels.index(_sel)][0]
            else:
                st.error("Ollama not running or no models found.")
                st.caption("Install [ollama.com](https://ollama.com) → `ollama pull llama3` → restart app")
        elif "Groq" in provider:
            api_key = st.text_input("Groq API Key", type="password",
                                    value=os.getenv("GROQ_API_KEY", ""),
                                    help="Free at console.groq.com — or set GROQ_API_KEY in .env")
            provider_name = "groq"
            # Store key so voice input can use it
            if api_key:
                st.session_state["_groq_key_voice"] = api_key
        elif "Gemini" in provider:
            api_key = st.text_input("Gemini API Key", type="password",
                                    value=os.getenv("GEMINI_API_KEY", ""),
                                    help="Free at aistudio.google.com — or set GEMINI_API_KEY in .env")
            provider_name = "gemini"
        else:
            api_key = st.text_input("HuggingFace Token", type="password",
                                    value=os.getenv("HF_API_KEY", ""),
                                    help="Free at huggingface.co/settings/tokens — or set HF_API_KEY in .env")
            provider_name = "huggingface"


# ───────────────── Session state ────────────────────────
for key in ("orchestrator", "doc_parsed", "summary_done", "segments_done",
            "recs_done", "chat_messages", "recs_enabled",
            "math_done", "stat_done"):
    if key not in st.session_state:
        st.session_state[key] = None if key != "chat_messages" else []
if "rec_active_tab" not in st.session_state:
    st.session_state.rec_active_tab = "papers"


def get_orchestrator():
    from agents.llm_provider import get_llm_provider
    from agents.orchestrator import Orchestrator
    if provider_name in ("groq", "gemini", "huggingface") and not api_key:
        st.sidebar.error("Please enter your API key.")
        st.stop()
    lang_name = language.split("(")[0].strip()
    model_override = ollama_model if provider_name == "ollama" else ""
    llm = get_llm_provider(provider_name, api_key, model_override)
    return Orchestrator(llm, language=lang_name)


# ───────── Helper: relevance badge HTML ─────────────────
def _rel_bar(score: int) -> str:
    """Generate a relevance bar with animated fill and percentage label."""
    if score >= 55:
        level = "high"
    elif score >= 35:
        level = "medium"
    else:
        level = "low"
    return (
        f'<span class="rel-bar-wrap">'
        f'<span class="rel-bar-track">'
        f'<span class="rel-bar-fill {level}" style="width:{score}%"></span>'
        f'</span>'
        f'<span class="rel-bar-pct {level}">{score}%</span>'
        f'</span>'
    )


def _extract_domain(url: str) -> str:
    """Extract short domain from URL for display."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


# ───────────────── Analysis pipeline ────────────────────
if analyze_btn and uploaded_file is not None:
    orch = get_orchestrator()
    st.session_state.orchestrator = orch
    st.session_state.recs_enabled = enable_recs

    for k in ("doc_parsed", "summary_done", "segments_done", "recs_done",
              "math_done", "stat_done"):
        st.session_state[k] = None
    st.session_state.chat_messages = []

    file_bytes = BytesIO(uploaded_file.read())
    filename = uploaded_file.name

    # Agent 1 — Document Parser
    with st.status("📄 **Document Parser** — extracting text & sections...", expanded=True) as s:
        try:
            doc_data = orch.parse_document(file_bytes, filename)
            st.session_state.doc_parsed = doc_data
            ref_count = doc_data.get("num_references", 0)
            ref_info = f", {ref_count} references" if ref_count else ""
            s.update(
                label=f"📄 Parsed — {doc_data['word_count']:,} words, "
                      f"{doc_data['num_pages']} pages, "
                      f"{doc_data['num_sections']} sections{ref_info}",
                state="complete",
            )
        except ValueError as e:
            s.update(label="📄 Could not read the document", state="error")
            st.error(f"**Document parsing failed:** {e}\n\nPlease upload a valid PDF or Word file.")
            st.stop()
        except Exception as e:
            s.update(label="📄 Parsing failed", state="error")
            st.error(f"**Unexpected error while parsing:** {e}")
            st.stop()

    # Agent 2 — Segmentation
    with st.status("🔖 **Segmentation Agent** — identifying topic structure...", expanded=True) as s:
        try:
            segments = orch.generate_segments()
            st.session_state.segments_done = segments
            s.update(
                label=f"🔖 Segmented — {len(segments)} topic clusters identified",
                state="complete",
            )
        except Exception as e:
            s.update(label="🔖 Segmentation failed (continuing anyway)", state="error")
            st.session_state.segments_done = []

    # Agent 3 — Summarizer
    with st.status("📝 **Summarizer Agent** — generating comprehensive summary...", expanded=True) as s:
        try:
            summary_data = orch.generate_summary()
            st.session_state.summary_done = summary_data
            word_count = len(summary_data.get("summary", "").split())
            s.update(
                label=f"📝 Summary complete — {word_count:,} words",
                state="complete",
            )
        except Exception as e:
            s.update(label="📝 Summary failed", state="error")
            st.error(f"**Summarizer error:** {e}")
            st.session_state.summary_done = {"summary": "_Summary could not be generated._"}

    # Agent 4 — Math Analysis
    with st.status("🧮 **Math Analysis Agent** — analyzing equations & formulas...", expanded=True) as s:
        try:
            math_data = orch.generate_math_analysis()
            st.session_state.math_done = math_data
            if math_data.get("has_math"):
                s.update(
                    label="🧮 Math Analysis complete — equations identified & analyzed",
                    state="complete",
                )
            else:
                s.update(
                    label="🧮 No formal equations found — quantitative analysis generated",
                    state="complete",
                )
        except Exception as e:
            s.update(label="🧮 Math analysis failed (continuing anyway)", state="error")
            st.session_state.math_done = None

    # Agent 5 — Statistical Analysis
    with st.status("📊 **Statistical Analysis Agent** — assessing methods & rigor...", expanded=True) as s:
        try:
            stat_data = orch.generate_stat_analysis()
            st.session_state.stat_done = stat_data
            if stat_data.get("has_stats"):
                s.update(
                    label="📊 Stats Analysis complete — methods identified & assessed",
                    state="complete",
                )
            else:
                s.update(
                    label="📊 No formal stats found — evidence analysis generated",
                    state="complete",
                )
        except Exception as e:
            s.update(label="📊 Statistical analysis failed (continuing anyway)", state="error")
            st.session_state.stat_done = None

    # Agent 6 — Recommendations (optional)
    if enable_recs:
        search_types = []
        if rec_papers:
            search_types.append("papers")
        if rec_blogs:
            search_types.append("blogs")
        if rec_videos:
            search_types.append("videos")
        if rec_podcasts:
            search_types.append("podcasts")
        if rec_newsletters:
            search_types.append("newsletters")
        if rec_wiki:
            search_types.append("wiki")

        with st.status("🔍 **Recommendation Agent** — searching for similar content...", expanded=True) as s:
            try:
                recs = orch.generate_recommendations(search_types)
                st.session_state.recs_done = recs
                counts = []
                for key, label in [("research_papers", "papers"), ("blogs", "blogs"),
                                   ("videos", "videos"), ("podcasts", "podcasts"),
                                   ("newsletters", "newsletters"), ("wiki", "wiki pages")]:
                    n = len(recs.get(key, []))
                    if n:
                        counts.append(f"{n} {label}")
                count_str = ", ".join(counts) if counts else "0 results"
                s.update(
                    label=f"🔍 Found {count_str}",
                    state="complete",
                )
            except Exception as e:
                s.update(label="🔍 Recommendation search failed", state="error")
                st.session_state.recs_done = {}


# ───────────────── Hero / Landing Page ──────────────────
if st.session_state.doc_parsed is None:
    st.markdown("""
    <div class="hero">
        <h1>ReclyticsAI</h1>
        <p>Upload a research paper or article and let our AI agents analyze it —
        producing structured summaries, topic maps, related content recommendations,
        and an interactive Q&amp;A chatbot.</p>
    </div>
    <div class="agent-grid">
        <div class="agent-item">
            <div class="agent-icon">📄</div>
            <div class="agent-name">Document Parser</div>
            <div class="agent-desc">Extracts text, sections &amp; references</div>
        </div>
        <div class="agent-item">
            <div class="agent-icon">🔖</div>
            <div class="agent-name">Segmentation</div>
            <div class="agent-desc">Clusters topics via embeddings</div>
        </div>
        <div class="agent-item">
            <div class="agent-icon">📝</div>
            <div class="agent-name">Summarizer</div>
            <div class="agent-desc">Comprehensive multi-section summary</div>
        </div>
        <div class="agent-item">
            <div class="agent-icon">🧮</div>
            <div class="agent-name">Math &amp; Stats Analyzer</div>
            <div class="agent-desc">Equations, formulas &amp; statistical rigor</div>
        </div>
        <div class="agent-item">
            <div class="agent-icon">🔍</div>
            <div class="agent-name">Recommender</div>
            <div class="agent-desc">Finds similar papers &amp; content</div>
        </div>
        <div class="agent-item">
            <div class="agent-icon">💬</div>
            <div class="agent-name">Chat Agent</div>
            <div class="agent-desc">RAG-powered Q&amp;A over article</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ───────────────── Results Tabs ─────────────────────────
tab_labels = ["📝 Summary"]
if st.session_state.math_done or st.session_state.stat_done:
    tab_labels.append("🧮 Math & Stats")
if st.session_state.recs_enabled and st.session_state.recs_done:
    tab_labels.append("🔍 Recommendations")
tab_labels.append("💬 Chat")

tabs = st.tabs(tab_labels)
tab_idx = 0

# ── Tab: Summary ──────────────────────────────────────
with tabs[tab_idx]:
    tab_idx += 1

    # Show topic segments
    segments = st.session_state.segments_done
    if segments:
        st.markdown("### Topic Structure")
        cols = st.columns(min(len(segments), 3))
        for i, seg in enumerate(segments):
            with cols[i % len(cols)]:
                st.markdown(f"**{seg.get('topic', 'Topic')}**")
                st.caption(seg.get("description", ""))
                st.caption(f"{seg.get('num_sections', 0)} sections")
        st.markdown("---")

    # Show summary
    summary_data = st.session_state.summary_done
    if summary_data:
        st.markdown(summary_data.get("summary", "_No summary available._"))

# ── Tab: Math & Stats (combined) ─────────────────────
if st.session_state.math_done or st.session_state.stat_done:
    with tabs[tab_idx]:
        tab_idx += 1

        # ── Math section ──
        if st.session_state.math_done:
            math_data = st.session_state.math_done
            if math_data.get("has_math"):
                st.markdown("### 🧮 Equations & Formulas Identified")
                with st.expander("View identified mathematical elements", expanded=False):
                    st.markdown(math_data.get("identified_elements", ""))
                st.markdown("---")
                st.markdown("### Deep Mathematical Analysis")
                st.markdown(math_data.get("analysis", "_No analysis available._"))
            else:
                st.info("No formal mathematical equations were found in this paper. "
                        "A quantitative reasoning analysis has been generated instead.")
                st.markdown("---")
                st.markdown("### Quantitative Analysis")
                st.markdown(math_data.get("analysis", "_No analysis available._"))

        # Divider between math and stats
        if st.session_state.math_done and st.session_state.stat_done:
            st.markdown("")
            st.divider()
            st.markdown("")

        # ── Stats section ──
        if st.session_state.stat_done:
            stat_data = st.session_state.stat_done
            if stat_data.get("has_stats"):
                st.markdown("### 📊 Statistical Methods Identified")
                with st.expander("View identified statistical methods", expanded=False):
                    st.markdown(stat_data.get("identified_methods", ""))
                st.markdown("---")
                st.markdown("### Statistical Rigor Assessment")
                st.markdown(stat_data.get("analysis", "_No analysis available._"))
            else:
                st.info("No formal statistical tests were found in this paper. "
                        "An evidence quality analysis has been generated instead.")
                st.markdown("---")
                st.markdown("### Evidence Analysis")
                st.markdown(stat_data.get("analysis", "_No analysis available._"))

# ── Tab: Recommendations ──────────────────────────────
if st.session_state.recs_enabled and st.session_state.recs_done: # type: ignore
    with tabs[tab_idx]:
        tab_idx += 1
        recs = st.session_state.recs_done

        # ── Rec header banner with paper metadata ──
        paper_title = recs.get("paper_title", "")
        domain_name = recs.get("domain", "")
        topic = recs.get("topic", "")
        meta_items = ""
        if domain_name:
            meta_items += (
                '<span class="rec-meta-item">'
                '<span class="meta-icon" style="background:rgba(124,106,255,0.15);color:#a5b4fc;">&#9670;</span>'
                f'{domain_name}</span>'
            )
        if topic:
            meta_items += (
                '<span class="rec-meta-item">'
                '<span class="meta-icon" style="background:rgba(52,211,153,0.15);color:#6ee7b7;">&#9733;</span>'
                f'{topic}</span>'
            )
        header_title = paper_title if paper_title else "Related Content"
        st.markdown(
            f'<div class="rec-header">'
            f'<div class="rec-title">{header_title}</div>'
            f'<div class="rec-meta">{meta_items}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Keyword chips ──
        keywords = recs.get("keywords", [])
        if keywords:
            kw_html = " ".join(f'<span class="kw-chip">{kw}</span>' for kw in keywords)
            st.markdown(f'<div class="kw-wrap">{kw_html}</div>', unsafe_allow_html=True)

        # ── Data ──
        papers = recs.get("research_papers", [])
        blogs = recs.get("blogs", [])
        videos = recs.get("videos", [])
        podcasts = recs.get("podcasts", [])
        newsletters = recs.get("newsletters", [])
        wiki = recs.get("wiki", [])

        # ── Clickable stat cards (filter tabs) ──
        card_defs = [
            ("papers",   len(papers),   "Papers",   "#a5b4fc", "#c084fc"),
            ("blogs",    len(blogs),    "Blogs",    "#6ee7b7", "#34d399"),
            ("videos",   len(videos),   "Videos",   "#fca5a5", "#f87171"),
            ("podcasts", len(podcasts), "Podcasts", "#fdba74", "#fb923c"),
            ("newsletters", len(newsletters), "Newsletters", "#f9a8d4", "#f472b6"),
            ("wiki",     len(wiki),     "Wiki",     "#67e8f9", "#22d3ee"),
        ]
        stat_cols = st.columns(len(card_defs))
        for col, (key, count, label, c1, c2) in zip(stat_cols, card_defs):
            is_active = st.session_state.rec_active_tab == key
            active_cls = " active" if is_active else ""
            with col:
                st.markdown(
                    f'<div class="rec-stat-card{active_cls}" style="--stat-c1:{c1};--stat-c2:{c2};cursor:pointer;">'
                    f'<div class="stat-num">{count}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    label, key=f"rec_tab_{key}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.rec_active_tab = key
                    st.rerun()

        # ── Section rendering helper ──
        def _render_section(items, emoji, title, color_class, show_publisher=False):
            if not items:
                st.markdown(
                    '<div class="rec-empty">'
                    f'<div class="rec-empty-icon">{emoji}</div>'
                    f'<div class="rec-empty-title">No {title.lower()} found</div>'
                    '<div class="rec-empty-desc">Try enabling this search type in the sidebar.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                return
            st.markdown(
                f'<div class="rec-section-header">'
                f'<div class="rec-section-icon {color_class}">{emoji}</div>'
                f'<div class="rec-section-title">{title}</div>'
                f'<div class="rec-section-count">{len(items)} found</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for idx, item in enumerate(items, 1):
                score = item.get("relevance_score", 0)
                bar = _rel_bar(score)
                t = item.get("title", "Untitled")
                u = item.get("url", "#")
                snip = item.get("snippet", "")
                dom = _extract_domain(u)
                publisher = item.get("publisher", "")

                footer_parts = ""
                if dom:
                    footer_parts += f'<span class="card-domain">{dom}</span>'
                if show_publisher and publisher:
                    footer_parts += f'<span class="card-publisher">{publisher}</span>'

                st.markdown(
                    f'<div class="rec-card {color_class}">'
                    f'<div class="rec-rank">{idx}</div>'
                    f'<div class="rec-card-body">'
                    f'<div class="card-title">'
                    f'<a href="{u}" target="_blank">{t}</a>'
                    f'{bar}'
                    f'</div>'
                    f'<div class="card-snippet">{snip}</div>'
                    f'<div class="card-footer">{footer_parts}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Render only the active section ──
        active = st.session_state.rec_active_tab
        if active == "papers":
            _render_section(papers, "&#128196;", "Research Papers", "purple")
        elif active == "blogs":
            _render_section(blogs, "&#128240;", "Blog Articles", "green")
        elif active == "videos":
            _render_section(videos, "&#127916;", "Videos", "red", show_publisher=True)
        elif active == "podcasts":
            _render_section(podcasts, "&#127908;", "Podcasts", "orange", show_publisher=True)
        elif active == "newsletters":
            _render_section(newsletters, "&#128232;", "Newsletters", "pink")
        elif active == "wiki":
            _render_section(wiki, "&#127760;", "Wikipedia Pages", "cyan")

        if not any([papers, blogs, videos, podcasts, newsletters, wiki]):
            st.markdown(
                '<div class="rec-empty">'
                '<div class="rec-empty-icon">&#128269;</div>'
                '<div class="rec-empty-title">No recommendations found</div>'
                '<div class="rec-empty-desc">Try enabling more search types in the sidebar.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ── Tab: Chat ─────────────────────────────────────────
with tabs[tab_idx]:
    orch = st.session_state.orchestrator
    if orch is None:
        st.warning("Please upload a document first.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # ── Process pending voice question (set on previous rerun) ──
        _pending_voice = st.session_state.pop("_voice_pending_q", None)
        if _pending_voice:
            st.session_state.chat_history.append(
                {"role": "user", "content": _pending_voice}
            )
            with st.spinner("Thinking..."):
                try:
                    _ans = orch.chat(_pending_voice)
                except Exception as e:
                    _ans = f"Sorry, I encountered an error: {e}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": _ans}
            )

        # ── Chat history ──
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Bottom bar: Voice mic + keyboard input ──
        _mic_col, _spacer = st.columns([1, 5])
        with _mic_col:
            audio_bytes = st.audio_input(
                "🎤", label_visibility="collapsed", key="chat_voice_mic"
            )

        # Voice transcription → save to session & rerun
        if audio_bytes is not None:
            # Only transcribe if this is new audio (not a stale widget)
            _audio_id = hash(audio_bytes.getvalue()[:256])
            if _audio_id != st.session_state.get("_last_audio_id"):
                st.session_state["_last_audio_id"] = _audio_id
                groq_key = (
                    os.getenv("GROQ_API_KEY", "")
                    or st.session_state.get("_groq_key_voice", "")
                )
                if not groq_key:
                    st.info(
                        "Set GROQ_API_KEY in .env or select Groq in the sidebar to use voice."
                    )
                else:
                    try:
                        from groq import Groq as _GroqClient

                        _gclient = _GroqClient(api_key=groq_key)
                        _af = BytesIO(audio_bytes.getvalue())
                        _af.name = "voice.wav"
                        _transcription = _gclient.audio.transcriptions.create(
                            model="whisper-large-v3",
                            file=_af,
                            language="en",
                        )
                        _vtext = _transcription.text.strip()
                        if _vtext:
                            st.session_state["_voice_pending_q"] = _vtext
                            st.rerun()
                        else:
                            st.warning("Could not detect speech. Try again.")
                    except Exception as e:
                        st.error(f"Voice error: {e}")

        # ── Keyboard chat input (pinned to bottom by Streamlit) ──
        user_q = st.chat_input("Ask a question about the paper...")
        if user_q:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_q}
            )
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = orch.chat(user_q)
                    except Exception as e:
                        answer = f"Sorry, I encountered an error: {e}"
                st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
