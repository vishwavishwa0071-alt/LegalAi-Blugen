import streamlit as st
import os
import re
import json
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# ─────────────────────────────────────────────
#  MUST be the very first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Legal AI Expert",
    page_icon="⚖️ ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pathlib import Path

# ─────────────────────────────────────────────
#  Environment & Paths
# ─────────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

# All paths are relative to this file — works both locally and on Streamlit Cloud
# (PDFs must be committed to the repo under a pdf/ folder next to app.py)
LOCAL_DIR = Path(__file__).parent.absolute()

UNIFIED_INDEX_DIR = os.path.join(LOCAL_DIR, "unified_vector_store")
PDF_DIR = os.path.join(LOCAL_DIR, "pdf")
HISTORY_FILE = os.path.join(LOCAL_DIR, "chat_history.json")

# FAISS L2 distance threshold – lower = stricter.
# Normalised Gemini embeddings range [0, 2]; good legal hits are typically < 0.9.
RELEVANCE_SCORE_THRESHOLD = 0.9

# Max sources shown per answer (quality > quantity)
MAX_SOURCES = 4

# ─────────────────────────────────────────────
#  CSS – Light Legal Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ── Root Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f4f6fb !important;
    font-family: 'Inter', sans-serif;
    color: #162044;
}
[data-testid="stHeader"] { background: rgba(244,246,251,0.92) !important; backdrop-filter: blur(10px); border-bottom: 1px solid #d0daf0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #eef1f8 !important;
    border-right: 1px solid #d0daf0 !important;
}
[data-testid="stSidebarContent"] { background: transparent !important; }

/* ── Fix Bottom Bar ── */
[data-testid="stBottom"] {
    background: #f4f6fb !important;
    border-top: 1px solid #d0daf0;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e8ecf5; }
::-webkit-scrollbar-thumb { background: #a0b4d8; border-radius: 4px; }

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0a3e 50%, #0d3347 100%);
    border-bottom: 1px solid rgba(250,200,80,0.25);
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(13,27,62,0.18);
    position: sticky;
    top: 0;
    z-index: 100;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f5c842, #e8a020, #f5c842);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.header-badge {
    background: linear-gradient(135deg, #f5c842, #d4900a);
    color: #0d1b3e;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.header-sub {
    color: #c8d4e8;
    font-size: 0.82rem;
    margin-top: 0.15rem;
}

/* ── Chat Container ── */
.chat-container {
    background: #ffffff;
    border: 1px solid #d0daf0;
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(13,27,62,0.06);
}
.chat-scroll-area {
    height: 65vh !important;
    overflow-y: auto !important;
    padding-right: 10px;
}

/* ── Message Bubbles ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0px); }
}

.msg-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 2rem;
    width: 100%;
    animation: fadeSlideIn 0.4s ease forwards;
}
.msg-row.user { 
    flex-direction: row-reverse; 
}

.avatar {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.avatar.ai {
    background: linear-gradient(135deg, #1a3a6e, #0e5a8a);
    border: 2px solid rgba(200,137,10,0.5);
    margin-right: 0.8rem;
}
.avatar.user {
    background: linear-gradient(135deg, #3d1c6e, #6b21a8);
    border: 2px solid rgba(168,85,247,0.5);
    margin-left: 0.8rem;
}

.bubble {
    max-width: 85%;
    padding: 1rem 1.35rem;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.95rem;
    box-shadow: 0 3px 14px rgba(13,27,62,0.10);
    flex-grow: 0;
}
.bubble.ai {
    background: #ffffff;
    border: 1px solid #d0daf0;
    color: #162044;
    border-top-left-radius: 4px;
}
.bubble.user {
    background: linear-gradient(135deg, #3b0e72, #5b21b6);
    border: 1px solid rgba(167,84,207,0.4);
    color: #f0e4ff;
    border-top-right-radius: 4px;
}

/* ── Inline Sources Panel (per message) ── */
.inline-sources-panel {
    background: #ffffff;
    border: 1px solid #d0daf0;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    margin: 0;
    animation: fadeSlideIn 0.4s ease forwards;
    height: 100%;
    box-shadow: 0 2px 10px rgba(13,27,62,0.07);
}
.inline-sources-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #e0e8f4;
}
.inline-sources-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
/* Horizontal category label */
.cat-col-label {
    font-size: 0.67rem;
    color: #c8890a;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.055em;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.cat-col-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 10px;
    background: #c8890a;
    border-radius: 2px;
    flex-shrink: 0;
}
/* Active source chip */
.src-active-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: #fdf3dc;
    border: 1px solid rgba(200,137,10,0.5);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.76rem;
    color: #c8890a;
    margin-bottom: 4px;
    line-height: 1.3;
}
/* Category divider label */
.cat-divider {
    font-size: 0.67rem;
    font-weight: 700;
    color: #c8890a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0.55rem 0 0.3rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid rgba(200,137,10,0.2);
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

/* ── Welcome Screen ── */
.welcome-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    animation: fadeSlideIn 0.6s ease forwards;
}
.welcome-seal {
    font-size: 5rem;
    filter: drop-shadow(0 0 20px rgba(200,137,10,0.3));
    animation: pulse-glow 2.5s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(200,137,10,0.25)); }
    50% { filter: drop-shadow(0 0 28px rgba(200,137,10,0.55)); }
}
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #162044;
    margin-top: 1rem;
    text-align: center;
}
.welcome-sub {
    color: #4a6080;
    text-align: center;
    max-width: 520px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Source Pills ── */
.source-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #eef1f8;
    border: 1px solid #c0cfe8;
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    color: #3a5080;
    cursor: pointer;
    transition: all 0.2s;
}
.source-pill:hover {
    border-color: #c8890a;
    color: #c8890a;
    background: #fdf3dc;
}

/* ── PDF Preview Panel (right column) ── */
.pdf-panel {
    background: #ffffff;
    border: 1px solid #d0daf0;
    border-radius: 16px;
    padding: 1.2rem;
    height: 100%;
    animation: fadeSlideIn 0.35s ease forwards;
    box-shadow: 0 2px 12px rgba(13,27,62,0.07);
}
.pdf-panel-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #e0e8f4;
}
.pdf-panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #162044;
}
.pdf-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.8rem;
}
.pdf-meta-chip {
    background: #eef1f8;
    border: 1px solid #c0cfe8;
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #4a6080;
}
.pdf-meta-chip span { color: #1a3a6e; font-weight: 500; }
.pdf-path-box {
    background: #f4f6fb;
    border: 1px solid #c0cfe8;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.7rem;
    color: #6080a0;
    font-family: 'Courier New', monospace;
    word-break: break-all;
    margin-bottom: 1rem;
}
.pdf-path-box span { color: #c8890a; }
.highlight-text-box {
    background: #fdf3dc;
    border-left: 3px solid #c8890a;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.8rem;
    font-size: 0.78rem;
    color: #2a3a5c;
    line-height: 1.6;
    margin-bottom: 1rem;
    font-style: italic;
}

/* ── Input Area Reset ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid #c0cfe8 !important;
    background: #ffffff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a7e, #0e5a8a) !important;
    border: 1px solid rgba(56,100,180,0.4) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    border-color: #c8890a !important;
    color: #fdf3dc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(13,27,62,0.18), 0 0 8px rgba(200,137,10,0.15) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}
/* Clear button — distinct red-tinted variant */
[data-testid="stBaseButton-secondary"].clear-btn {
    background: rgba(220,38,38,0.06) !important;
    border: 1px solid rgba(220,38,38,0.2) !important;
    color: #7a1a1a !important;
}
[data-testid="stBaseButton-secondary"].clear-btn:hover {
    background: rgba(220,38,38,0.12) !important;
    border-color: rgba(220,38,38,0.45) !important;
    color: #b91c1c !important;
}

/* ── Spinners / loading ── */
@keyframes thinking-dots {
    0%, 80%, 100% { opacity: 0; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
.thinking-row { display: flex; align-items: center; gap: 8px; padding: 0.8rem 1rem; }
.dot {
    width: 8px; height: 8px;
    background: #6080b0;
    border-radius: 50%;
    animation: thinking-dots 1.4s infinite ease-in-out both;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Dividers ── */
hr { border-color: #d0daf0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d0daf0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #4a6080 !important;
    font-size: 0.8rem !important;
}

/* ── Chat History Sidebar ── */
.hist-item {
    background: #ffffff;
    border: 1px solid #d0daf0;
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.45rem;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(13,27,62,0.05);
}
.hist-item:hover {
    border-color: #c8890a;
    box-shadow: 0 2px 8px rgba(200,137,10,0.12);
}
.hist-date {
    font-size: 0.68rem;
    color: #7090b0;
    margin-bottom: 0.2rem;
}
.hist-preview {
    font-size: 0.78rem;
    color: #2a3a5c;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Category stat pills ── */
.stat-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
}
.stat-pill {
    background: #eef1f8;
    border: 1px solid #c0cfe8;
    color: #4a6080;
    font-size: 0.72rem;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
}
.stat-pill b { color: #162044; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
def load_chat_history() -> list:
    """Load all past sessions from the history file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat_session(messages: list):
    """Append the current conversation to the history file (max 20 sessions)."""
    if not messages:
        return
    first_user = next((m for m in messages if m["role"] == "user"), None)
    if not first_user:
        return
    preview_text = first_user["content"]
    preview = (preview_text[:55] + "…") if len(preview_text) > 55 else preview_text
    session = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().strftime("%b %d, %Y  %I:%M %p"),
        "preview": preview,
        # Store only role + content; sources are heavyweight and not needed for history
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    history = load_chat_history()
    history.insert(0, session)
    history = history[:20]  # keep last 20 sessions
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # silent fail (e.g. read-only filesystem on cloud)


def init_state():
    defaults = {
        "messages": [],          # [{role, content, sources}]
        "preview_source": None,  # Source dict currently shown in right-panel PDF viewer
        "preview_msg_idx": None, # Which message's source is being previewed
        "is_thinking": False,
        "chat_history": None,    # cached history list (loaded once per session)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Load history once per session
    if st.session_state.chat_history is None:
        st.session_state.chat_history = load_chat_history()

init_state()


# ─────────────────────────────────────────────
#  Cached Resource Loaders
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Load the unified FAISS vector store once and cache it."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )
    if not os.path.exists(UNIFIED_INDEX_DIR):
        st.error(f"Unified vector store not found at: {UNIFIED_INDEX_DIR}\nRun `ingest_unified.py` first.")
        st.stop()
    return FAISS.load_local(UNIFIED_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=API_KEY)


# ─────────────────────────────────────────────
#  PDF Loader — local path first, GCS fallback
#  Cached so each PDF is read from disk / network only once per session.
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=30)
def load_pdf_bytes(pdf_path: str) -> bytes | None:
    """Read PDF from the repo's pdf/ folder and cache the bytes for the session."""
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            return f.read()
    return None


# ─────────────────────────────────────────────
#  Content relevance filter
# ─────────────────────────────────────────────
def is_index_or_toc_content(content: str) -> bool:
    """
    Return True if a chunk looks like a Table of Contents, Arrangement of Sections,
    or other navigational boilerplate that adds no answer value.
    """
    c = content.lower().strip()

    # Explicit TOC / index phrases
    toc_phrases = [
        "arrangement of sections",
        "table of contents",
        "index of sections",
        "list of sections",
        "list of forms",
        "contents",
    ]
    for phrase in toc_phrases:
        if phrase in c:
            # Allow only if content also contains substantive legal language
            legal_signals = ["shall", "court", "plaintiff", "defendant",
                             "decree", "suit", "order", "section", "rule"]
            if not any(sig in c for sig in legal_signals):
                return True

    # Too short to be useful (header/footer fragment)
    if len(content.strip()) < 80:
        return True

    # TOC pattern: many short numbered lines like "5. Jurisdiction ... 12"
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) >= 8:
        toc_like = sum(1 for l in lines
                       if len(l) < 80 and re.match(r"^\d+\.?\s", l))
        if toc_like / len(lines) > 0.55:
            return True

    return False


# ─────────────────────────────────────────────
#  Page verification — confirm text is on page
# ─────────────────────────────────────────────
def verify_page_relevance(raw_bytes: bytes, claimed_page: int, content: str,
                          radius: int = 2) -> int | None:
    """
    Confirm that the chunk's text physically exists on (or near) the claimed PDF page.

    Strategy:
      1. Extract 3 candidate phrases from different positions in the chunk and
         search for each with fitz (exact text layer match).
      2. If no exact hit, fall back to a word-overlap ratio: if ≥ 40% of the
         chunk's significant words appear in the page's text, accept that page.
         (Handles OCR'd PDFs where exact spacing differs.)
      3. Search the claimed page first, then ±radius pages.

    Returns the verified page number, or None if the text cannot be located.
    """
    if not raw_bytes:
        return None
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        total = len(doc)
        clean = re.sub(r'\s+', ' ', content).strip()
        words = clean.split()
        n = len(words)

        # Build candidate search phrases (skip leading section-header words)
        phrases = []
        if n >= 8:
            mid = n // 3
            phrases.append(" ".join(words[mid: mid + 7]))       # ~30% in
        if n >= 6:
            phrases.append(" ".join(words[2:8]))                # near start
        if n >= 6:
            phrases.append(" ".join(words[max(0, n-8): n-1]))   # near end
        phrases = list(dict.fromkeys(p for p in phrases if p))  # deduplicate

        # Word-overlap fallback: significant words (len > 4) in chunk
        sig_words = [w.lower() for w in words if len(w) > 4]

        # Search order: claimed page, then alternating ±1, ±2 …
        order = [claimed_page]
        for d in range(1, radius + 1):
            if claimed_page - d >= 0:     order.append(claimed_page - d)
            if claimed_page + d < total:  order.append(claimed_page + d)

        for p in order:
            page = doc[p]

            # Stage 1 — exact phrase match
            for phrase in phrases:
                if page.search_for(phrase):
                    doc.close()
                    return p

            # Stage 2 — word-overlap ratio (≥ 40% of significant words present)
            if sig_words:
                page_text = page.get_text().lower()
                overlap = sum(1 for w in sig_words if w in page_text)
                if overlap / len(sig_words) >= 0.40:
                    doc.close()
                    return p

        doc.close()
        return None   # text not found near the claimed page → skip this source
    except Exception:
        return None


# ─────────────────────────────────────────────
#  PDF: Extract single page + highlight → bytes
# ─────────────────────────────────────────────
def get_highlighted_pdf_bytes(raw_pdf_bytes: bytes, page_num: int, search_text: str) -> bytes | None:
    """
    Extract the target page and highlight relevant text from the retrieved context.
    Uses a multi-tier matching strategy to ensure the actual passage is found
    while avoiding 'dirty' highlights on common words/headers.
    """
    try:
        if not raw_pdf_bytes:
            return None

        src = fitz.open(stream=raw_pdf_bytes, filetype="pdf")
        if page_num >= len(src):
            page_num = len(src) - 1

        out = fitz.open()
        out.insert_pdf(src, from_page=page_num, to_page=page_num)
        src.close()
        page = out[0]

        # ── Normalize text ──
        clean_text = re.sub(r'\s+', ' ', search_text).strip()
        
        # ── Filter out very generic header/footer markers if they appear in chunk ──
        # (e.g. page numbers, common document titles that cause 'irrelevant' highlights)
        noise_patterns = [
            r'THE CODE OF CIVIL PROCEDURE', r'ARRANGEMENT OF SECTIONS',
            r'TAMIL NADU GOVERNMENT GAZETTE', r'PUBLISHED BY AUTHORITY'
        ]
        text_to_process = clean_text
        for pat in noise_patterns:
            text_to_process = re.sub(pat, '', text_to_process, flags=re.IGNORECASE)

        # ── Split into significant sentences (min 20 chars) ──
        raw_sentences = re.split(r'(?<=[.;!?])\s+|\n+', text_to_process)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= 20]

        if not sentences:
            sentences = [clean_text[:200]]

        def _highlight_phrase(text: str, is_fallback=False) -> int:
            if len(text.strip()) < 12 and not is_fallback:
                return 0
            hits = page.search_for(text, quads=True)
            count = 0
            for quad in hits:
                # Add highlighting
                ann = page.add_highlight_annot(quad)
                ann.set_colors(stroke=[1.0, 0.82, 0.0]) # Gold
                ann.set_opacity(0.55 if not is_fallback else 0.3)
                ann.update()
                count += 1
            return count

        total_hits = 0
        for sentence in sentences:
            words = sentence.split()
            if not words: continue
            
            # Tier 1: Try the full sentence (the best & cleanest match)
            if _highlight_phrase(sentence):
                total_hits += 1
                continue
                
            # Tier 2: Try longest contiguous sub-phrases (avoiding small fragments)
            # Try 75% of the sentence, then 50%
            matched_tier2 = False
            for ratio in [0.75, 0.5]:
                length = int(len(words) * ratio)
                if length < 6: continue
                
                phrase = " ".join(words[:length])
                if _highlight_phrase(phrase):
                    total_hits += 1
                    matched_tier2 = True
                    break
            
            if matched_tier2: continue

            # Tier 3: Sliding window (only as last resort, min 6 words)
            if len(words) >= 7:
                window_size = 6
                for start in range(0, len(words) - window_size, 4):
                    phrase = " ".join(words[start:start + window_size])
                    if _highlight_phrase(phrase):
                        total_hits += 1
                        break # Stop after first window match to avoid 'over-highlighting'

        # Tier 4: Global fallback only if NOTHING was found
        if total_hits == 0 and len(clean_text) > 30:
            # Try a safe-length unique-ish chunk from the middle
            middle = len(clean_text) // 2
            fallback_chunk = clean_text[middle:middle+50]
            _highlight_phrase(fallback_chunk, is_fallback=True)

        # Render the highlighted page to a PNG image (2× zoom for sharpness)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        out.close()
        return img_bytes
    except Exception as e:
        st.warning(f"Highlighting notice: {e}")
        return None



# ─────────────────────────────────────────────
#  RAG Query Function
# ─────────────────────────────────────────────
LEGAL_PROMPT = """\
You are a Senior Civil Procedural Law Expert AI. You are provided with a specific question and a set of retrieved context chunks from legal documents.

Your task is to answer the user's question comprehensively using *only* the provided context.

**Guidelines for your response:**

1.  **Citation is Mandatory:** You must cite the specific legal source for every claim you make. Use the format: "Section X," "Order Y, Rule Z," or "Appendix [Letter], Form [Number]."
2.  **Structure:**
    *   **Direct Answer:** Start with a clear, direct response to the user's question.
    *   **Procedural Details:** Explain the steps, requirements, or conditions found in the text.
    *   **Exceptions/Provisos:** Explicitly mention any "Provided that" clauses or exceptions found in the context.
3.  **Sections vs. Orders:** Distinguish between substantive law (Sections) and procedural rules (Orders/Rules) if the context contains both.
4.  **State Amendments:** If the retrieved context contains "State Amendments" (e.g., for Tamil Nadu, Maharashtra, etc.), explicitly state that these apply only to those specific regions.
5.  **Definitions:** If the user asks for a definition, use exact definitions provided in the context if available.
6.  **Tone:** Maintain a professional, objective, and legal tone. Do not offer personal legal advice or opinions.
7.  **Missing Information:** If the provided context does not contain the answer, state: "The provided context does not contain sufficient information to answer this specific question." Do not hallucinate.
8. **briefness:** Keep the answer concise and to the point. Do not provide unnecessary information.
**Context:**
{context}

**User Question:**
{query}

**Answer:**
"""

RELEVANCE_JUDGE_PROMPT = """\
You are a strict relevance judge for a legal document retrieval system.

Given a user's legal question and a numbered list of retrieved document chunks, \
return ONLY the indices of chunks that are **directly relevant** to answering the question.

A chunk is relevant if it contains substantive information that directly answers, \
defines, or provides procedural detail for the question. \
Reject chunks that are only tangentially related, repeat generic boilerplate, \
or discuss a different legal topic entirely.

User Question: {query}

Retrieved Chunks:
{chunks_list}

Respond with ONLY a JSON array of the relevant 0-based indices. Examples:
  [0, 2]      ← chunks 0 and 2 are relevant
  [1, 2, 3]   ← chunks 1, 2 and 3 are relevant
  []           ← none are relevant

JSON array:"""


def run_rag_query(query: str):
    """
    Run the unified RAG pipeline and return (answer_text, sources_list).

    Two LLM calls run in parallel:
      • Answer LLM — generates the legal answer from all verified candidates.
      • Judge LLM  — selects which candidates are relevant enough to show as
                     source preview buttons.
    """
    vs = load_vector_store()
    llm = load_llm()

    docs_and_scores = vs.similarity_search_with_score(query, k=12)

    # ── Phase 1: build verified candidate list (gates 1-4) ───────────────
    candidates = []
    seen_pages = set()

    for doc, score in docs_and_scores:
        # Gate 1 – score
        if float(score) > RELEVANCE_SCORE_THRESHOLD:
            continue
        # Gate 2 – content quality
        if is_index_or_toc_content(doc.page_content):
            continue

        source_file = doc.metadata.get("source_file", "Unknown")
        page_num    = int(doc.metadata.get("page", 0))
        pdf_path    = os.path.join(PDF_DIR, source_file)

        # Gate 3 – page verification (load_pdf_bytes is cached)
        raw_bytes     = load_pdf_bytes(pdf_path)
        verified_page = verify_page_relevance(raw_bytes, page_num, doc.page_content)
        if verified_page is None:
            continue
        page_num = verified_page

        # Gate 4 – deduplicate
        key = (source_file, page_num)
        if key in seen_pages:
            continue
        seen_pages.add(key)

        candidates.append({
            "category":     doc.metadata.get("category", "Unknown"),
            "sub_category": doc.metadata.get("sub_category", "Unknown"),
            "source_file":  source_file,
            "pdf_path":     pdf_path,
            "page":         page_num,
            "content":      doc.page_content,
            "score":        float(score),
        })

        if len(candidates) >= MAX_SOURCES:
            break

    # ── Phase 2: two parallel LLM calls ──────────────────────────────────
    context_chunks = "".join(
        f"--- Source: {c['source_file']} (Category: {c['category']} | "
        f"Sub-category: {c['sub_category']}) ---\n{c['content']}\n\n"
        for c in candidates
    )

    def call_answer_llm():
        return llm.invoke(
            LEGAL_PROMPT.format(context=context_chunks, query=query)
        ).content

    def call_judge_llm():
        if not candidates:
            return []
        chunks_list = "\n\n".join(
            f"[{i}] {c['source_file']}  p.{c['page'] + 1}:\n"
            f"{c['content'][:300]}{'...' if len(c['content']) > 300 else ''}"
            for i, c in enumerate(candidates)
        )
        raw = llm.invoke(
            RELEVANCE_JUDGE_PROMPT.format(query=query, chunks_list=chunks_list)
        ).content.strip()

        try:
            match = re.search(r'\[[\d,\s]*\]', raw)
            if match:
                indices = json.loads(match.group())
                return [i for i in indices
                        if isinstance(i, int) and 0 <= i < len(candidates)]
        except Exception:
            pass
        return list(range(len(candidates)))   # fallback: show all

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_answer = executor.submit(call_answer_llm)
        future_judge  = executor.submit(call_judge_llm)
        answer           = future_answer.result()
        relevant_indices = future_judge.result()

    sources = [candidates[i] for i in relevant_indices]
    return answer, sources


# ─────────────────────────────────────────────
#  UI Helpers
# ─────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="header-banner">
        <div style="font-size:2.2rem;filter:drop-shadow(0 0 10px rgba(245,200,66,0.5))">⚖️</div>
        <div>
            <div class="header-title">Legal AI Expert</div>
            <div class="header-sub">Tamil Nadu Laws &amp; Code of Civil Procedure — Powered by Gemini AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div class="welcome-wrapper">
        <div class="welcome-seal">⚖️</div>
        <div class="welcome-title">Ask Any Legal Question</div>
        <div class="welcome-sub">
            I have deep knowledge of Tamil Nadu State Laws, Code of Civil Procedure 1908, 
            and related legislation — all backed by verified source documents.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str):
    """Render a single chat message bubble."""
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user">👤</div>
            <div class="bubble user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        formatted = content
        formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted)
        formatted = re.sub(r'^\s*[\*\-]\s+', r'• ', formatted, flags=re.MULTILINE)
        formatted = formatted.replace("\n", "<br>")

        st.markdown(f"""
        <div class="msg-row">
            <div class="avatar ai">⚖️</div>
            <div class="bubble ai">{formatted}</div>
        </div>
        """, unsafe_allow_html=True)


def render_inline_sources(sources: list, msg_idx: int):
    """
    Render sources grouped by category directly beside the AI response.
    Lives in its own column (parallel to the response bubble), so a clean
    vertical list per category is the right UX - no nested column tricks needed.
    """
    if not sources:
        return

    categories: dict[str, list] = {}
    for src in sources:
        categories.setdefault(src["category"], []).append(src)

    num_cats = len(categories)
    num_srcs  = len(sources)

    # Panel header
    st.markdown(f"""
    <div class="inline-sources-panel">
        <div class="inline-sources-header">
            <span style="font-size:1rem">&#128218;</span>
            <div class="inline-sources-title">
                {num_srcs} source{"s" if num_srcs != 1 else ""}
                &nbsp;&middot;&nbsp;
                {num_cats} categor{"ies" if num_cats != 1 else "y"}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # One section per category - vertical list of buttons
    for cat_name, cat_sources in categories.items():
        short_cat = cat_name if len(cat_name) <= 30 else cat_name[:27] + "..."
        st.markdown(
            f'<div class="cat-divider">&#128194; {short_cat}</div>',
            unsafe_allow_html=True,
        )
        for si, src in enumerate(cat_sources):
            page_disp = src["page"] + 1
            filename  = src["source_file"]
            short_fn  = filename if len(filename) <= 22 else filename[:19] + "..."
            btn_key   = f"isrc_{msg_idx}_{cat_name[:6]}_{si}"

            is_active = (
                st.session_state.preview_source is not None
                and st.session_state.preview_source.get("source_file") == src["source_file"]
                and st.session_state.preview_source.get("page")        == src["page"]
                and st.session_state.preview_msg_idx == msg_idx
            )

            if is_active:
                st.markdown(
                    f'<div class="src-active-chip">'
                    f'<span>&#9989;</span>'
                    f'<span>{short_fn}&nbsp;<b style="color:#f5c842">p.{page_disp}</b></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"\U0001f4c4  {short_fn}  -  p.{page_disp}",
                    key=btn_key,
                    use_container_width=True,
                ):
                    st.session_state.preview_source = src
                    st.session_state.preview_msg_idx = msg_idx
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def render_pdf_panel(src: dict):
    """Render the right-side full PDF preview panel."""
    pdf_path = src["pdf_path"]
    page_num = src["page"]
    content = src["content"]
    source_file = src["source_file"]
    category = src["category"]
    sub_cat = src["sub_category"]

    st.markdown("""
    <div class="pdf-panel">
        <div class="pdf-panel-header">
            <span style="font-size:1.3rem">&#128196;</span>
            <div>
                <div class="pdf-panel-title">PDF Source Preview</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pdf-meta-row">
        <div class="pdf-meta-chip">&#128194; Category: <span>{category}</span></div>
        <div class="pdf-meta-chip">&#128209; Sub-cat: <span>{sub_cat}</span></div>
        <div class="pdf-meta-chip"> File: <span>{source_file}</span></div>
        <div class="pdf-meta-chip">&#128214; Page: <span>{page_num + 1}</span></div>
    </div>
    """, unsafe_allow_html=True)

    snippet = content[:400].strip().replace("\n", " ")
    st.markdown(f"""
    <div class="highlight-text-box">
        <b style="color:#f5c842;font-style:normal">&#128269; Highlighted Paragraph:</b><br><br>
        {snippet}{"..." if len(content) > 400 else ""}
    </div>
    """, unsafe_allow_html=True)

    raw_bytes = load_pdf_bytes(pdf_path)
    if raw_bytes:
        with st.spinner("Loading PDF..."):
            pdf_bytes = get_highlighted_pdf_bytes(raw_bytes, page_num, content)

        if pdf_bytes:
            st.markdown(
                f"**Page {page_num + 1} of** `{source_file}` — "
                "matching text highlighted in **gold** ✦"
            )
            st.image(pdf_bytes, use_container_width=True)
        else:
            st.warning("Could not prepare the PDF preview.")
    else:
        st.error(f"PDF not found: `{source_file}`\n\nEnsure it is committed to the `pdf/` folder in the repo.")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("✕ Close Preview", key="close_preview"):
        st.session_state.preview_source = None
        st.session_state.preview_msg_idx = None
        st.rerun()


def render_thinking_indicator():
    st.markdown("""
    <div class="msg-row">
      <div class="avatar ai">⚖️</div>
      <div class="bubble ai" style="padding:0.6rem 1.2rem">
        <div class="thinking-row">
          <span style="color:#4a6080;font-size:0.82rem">Analysing legal documents</span>
          <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem; margin-bottom:1rem;">⚖️</div>
        <div style="font-family:'Playfair Display', serif; font-size:1.2rem; color:#f5c842;">Legal AI Control</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", key="clear_btn_sidebar", use_container_width=True):
        save_chat_session(st.session_state.messages)  # persist before wiping
        st.session_state.messages = []
        st.session_state.preview_source = None
        st.session_state.preview_msg_idx = None
        st.session_state.chat_history = load_chat_history()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Click any document source next to an AI response to view the verified PDF excerpt.")

    # ── Chat History ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.85rem;font-weight:600;color:#8099c5;text-transform:uppercase;"
        "letter-spacing:0.07em;margin-bottom:0.6rem'>📜 Chat History</div>",
        unsafe_allow_html=True,
    )

    history_list = st.session_state.chat_history or []
    if not history_list:
        st.caption("No saved conversations yet.")
    else:
        for session in history_list[:15]:
            sid = session["id"]
            # Render a styled card; clicking the button loads the session
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-date">🕐 {session["timestamp"]}</div>'
                f'<div class="hist-preview">{session["preview"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Load", key=f"hist_load_{sid}", use_container_width=True):
                save_chat_session(st.session_state.messages)
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"], "sources": None}
                    for m in session["messages"]
                ]
                st.session_state.preview_source = None
                st.session_state.preview_msg_idx = None
                st.session_state.chat_history = load_chat_history()
                st.rerun()

# ─────────────────────────────────────────────
#  Main Chat Layout (Full Width)
# ─────────────────────────────────────────────
render_header()

if not st.session_state.messages:
    render_welcome()
else:
    # Use a container for the whole chat to maintain structure
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                render_message(role="user", content=msg["content"])
            else:
                sources = msg.get("sources") or []
                # Split AI response and Sources/Preview into two columns *within* the message row
                if sources:
                    ai_cols = st.columns([1, 1], gap="large") # 50/50 split for clear context
                    with ai_cols[0]:
                        render_message(role="assistant", content=msg["content"])
                    with ai_cols[1]:
                        # Always show the source buttons
                        render_inline_sources(sources, msg_idx=idx)
                        
                        # IF this specific message's PDF preview is active, render it right here
                        if st.session_state.preview_msg_idx == idx and st.session_state.preview_source:
                            render_pdf_panel(st.session_state.preview_source)
                else:
                    render_message(role="assistant", content=msg["content"])

    # Thinking bubble (while processing)
    if st.session_state.is_thinking:
        render_thinking_indicator()

st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True) # Spacer for chat input

# ── Voice Input — injected directly into the page DOM, floated next to send button ──
st.markdown("""
<style>
#legalai-mic-btn {
    position: fixed;
    bottom: 13px;
    right: 60px;
    z-index: 9999;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1.5px solid #c0cfe8;
    background: #ffffff;
    cursor: pointer;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all .2s;
    box-shadow: 0 2px 8px rgba(13,27,62,.12);
    padding: 0;
    line-height: 1;
}
#legalai-mic-btn:hover  { border-color: #1a3a7e; transform: scale(1.08); }
#legalai-mic-btn:active { transform: scale(0.96); }
#legalai-mic-btn.listening {
    border-color: #dc2626;
    background: #fff5f5;
    animation: legalai-pulse 1s infinite;
}
@keyframes legalai-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,.35); }
    50%      { box-shadow: 0 0 0 7px rgba(220,38,38,0); }
}
</style>

<button id="legalai-mic-btn" title="Voice input (Chrome / Edge)">🎤</button>

<script>
(function () {
    const btn = document.getElementById('legalai-mic-btn');
    let recog = null;

    btn.addEventListener('click', () => {
        if (recog) { recog.stop(); return; }

        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { alert('Voice input requires Chrome or Edge.'); return; }

        recog = new SR();
        recog.lang = 'en-IN';
        recog.continuous = false;
        recog.interimResults = false;

        recog.onstart = () => {
            btn.classList.add('listening');
            btn.textContent = '⏹';
        };

        recog.onresult = (e) => {
            const text = e.results[0][0].transcript;
            // Fill Streamlit's chat textarea and submit using React synthetic event trick
            const ta = document.querySelector('[data-testid="stChatInputTextArea"]');
            if (!ta) return;
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeSetter.call(ta, text);
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            // Give React a tick to register the value, then click Send
            setTimeout(() => {
                const sendBtn = document.querySelector('[data-testid="stChatInputSubmitButton"]');
                if (sendBtn) sendBtn.click();
            }, 80);
        };

        recog.onerror = () => {
            btn.classList.remove('listening');
            btn.textContent = '🎤';
            recog = null;
        };
        recog.onend = () => {
            btn.classList.remove('listening');
            btn.textContent = '🎤';
            recog = null;
        };

        recog.start();
    });
})();
</script>
""", unsafe_allow_html=True)

# ── Run the actual RAG engine ─────────────────────
if st.session_state.is_thinking:
    last_user_msg = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    if last_user_msg:
        with st.spinner(" "): # Spinner handled by render_thinking_indicator
            try:
                answer, sources = run_rag_query(last_user_msg["content"])
            except Exception as e:
                answer  = f"⚠️ An error occurred: {str(e)}"
                sources = []
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })
    st.session_state.is_thinking = False
    st.rerun()

# ── Global Chat Input ────────────────────────────────────────────────────────
user_query = st.chat_input("Ask a legal question...")
if user_query:
    st.session_state.messages.append({
        "role":    "user",
        "content": user_query.strip(),
        "sources": None,
    })
    st.session_state.preview_source = None
    st.session_state.preview_msg_idx = None
    st.session_state.is_thinking = True
    st.rerun()
