import streamlit as st
import os
import re
import base64
import fitz  # PyMuPDF
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

# ─────────────────────────────────────────────
#  Environment & Paths
# ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_DIR = r"c:\Users\kousi\Downloads\RAG"
UNIFIED_INDEX_DIR = os.path.join(BASE_DIR, "unified_vector_store")
PDF_DIR = os.path.join(BASE_DIR, "pdf")

# ─────────────────────────────────────────────
#  CSS – Dark Legal Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ── Root Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080c18 !important;
    font-family: 'Inter', sans-serif;
    color: #d4e4ff;
}
[data-testid="stHeader"] { background: rgba(8, 12, 24, 0.9) !important; backdrop-filter: blur(10px); }

/* ── Fix White Bottom Bar ── */
[data-testid="stBottom"] {
    background: #080c18 !important;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2d4a7a; border-radius: 4px; }

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0a3e 50%, #0d3347 100%);
    border-bottom: 1px solid rgba(250,200,80,0.25);
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.6);
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
    color: #8099c5;
    font-size: 0.82rem;
    margin-top: 0.15rem;
}

/* ── Chat Container ── */
.chat-container {
    background: rgba(10, 18, 40, 0.4);
    border: 1px solid rgba(56, 100, 180, 0.2);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
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
    border: 2px solid rgba(245,200,66,0.4);
    margin-right: 0.8rem;
}
.avatar.user {
    background: linear-gradient(135deg, #3d1c6e, #6b21a8);
    border: 2px solid rgba(168,85,247,0.4);
    margin-left: 0.8rem;
}

.bubble {
    max-width: 85%;
    padding: 1rem 1.35rem;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.95rem;
    box-shadow: 0 6px 25px rgba(0,0,0,0.45);
    flex-grow: 0;
}
.bubble.ai {
    background: linear-gradient(145deg, rgba(16,30,60,0.9), rgba(10,22,48,0.95));
    border: 1px solid rgba(56,100,180,0.35);
    color: #d4e4ff;
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
    background: linear-gradient(145deg, rgba(8,14,32,0.97), rgba(5,10,24,0.99));
    border: 1px solid rgba(56,100,180,0.3);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    margin: 0;
    animation: fadeSlideIn 0.4s ease forwards;
    height: 100%;
}
.inline-sources-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(56,100,180,0.2);
}
.inline-sources-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8099c5;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
/* Horizontal category label */
.cat-col-label {
    font-size: 0.67rem;
    color: #f5c842;
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
    background: #f5c842;
    border-radius: 2px;
    flex-shrink: 0;
}
/* Active source chip */
.src-active-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(245,200,66,0.12);
    border: 1px solid rgba(245,200,66,0.5);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.76rem;
    color: #f5c842;
    margin-bottom: 4px;
    line-height: 1.3;
}
/* Category divider label */
.cat-divider {
    font-size: 0.67rem;
    font-weight: 700;
    color: #f5c842;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0.55rem 0 0.3rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid rgba(245,200,66,0.2);
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
    filter: drop-shadow(0 0 20px rgba(245,200,66,0.4));
    animation: pulse-glow 2.5s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 12px rgba(245,200,66,0.3)); }
    50% { filter: drop-shadow(0 0 30px rgba(245,200,66,0.7)); }
}
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5c842;
    margin-top: 1rem;
    text-align: center;
}
.welcome-sub {
    color: #8099c5;
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
    background: rgba(20,40,80,0.8);
    border: 1px solid rgba(56,100,180,0.4);
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    color: #a0c0ff;
    cursor: pointer;
    transition: all 0.2s;
}
.source-pill:hover {
    border-color: rgba(245,200,66,0.6);
    color: #f5c842;
    background: rgba(30,55,100,0.9);
}

/* ── PDF Preview Panel (right column) ── */
.pdf-panel {
    background: linear-gradient(185deg, rgba(10,18,40,0.98), rgba(6,14,32,0.99));
    border: 1px solid rgba(56,100,180,0.35);
    border-radius: 16px;
    padding: 1.2rem;
    height: 100%;
    animation: fadeSlideIn 0.35s ease forwards;
}
.pdf-panel-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(56,100,180,0.25);
}
.pdf-panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #d4e4ff;
}
.pdf-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.8rem;
}
.pdf-meta-chip {
    background: rgba(20,40,80,0.7);
    border: 1px solid rgba(56,100,180,0.3);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #8099c5;
}
.pdf-meta-chip span { color: #a8c4f0; font-weight: 500; }
.pdf-path-box {
    background: rgba(8,14,32,0.8);
    border: 1px solid rgba(30,60,120,0.5);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.7rem;
    color: #5a78b0;
    font-family: 'Courier New', monospace;
    word-break: break-all;
    margin-bottom: 1rem;
}
.pdf-path-box span { color: #f5c842; }
.highlight-text-box {
    background: rgba(245,200,66,0.06);
    border-left: 3px solid #f5c842;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.8rem;
    font-size: 0.78rem;
    color: #c8d8f0;
    line-height: 1.6;
    margin-bottom: 1rem;
    font-style: italic;
}

/* ── Input Area Reset ── */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a7e, #0e5a8a) !important;
    border: 1px solid rgba(56,100,180,0.5) !important;
    color: #d4e4ff !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    border-color: rgba(245,200,66,0.8) !important;
    color: #f5c842 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5), 0 0 10px rgba(245,200,66,0.2) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}
/* Specifically for the Clear button to make it look distinct but premium */
[data-testid="stBaseButton-secondary"].clear-btn {
    background: rgba(40, 20, 20, 0.4) !important;
    border: 1px solid rgba(255, 100, 100, 0.2) !important;
}
[data-testid="stBaseButton-secondary"].clear-btn:hover {
    background: rgba(60, 20, 20, 0.6) !important;
    border-color: rgba(255, 100, 100, 0.5) !important;
    color: #ff8080 !important;
}

/* ── Spinners / loading ── */
@keyframes thinking-dots {
    0%, 80%, 100% { opacity: 0; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
.thinking-row { display: flex; align-items: center; gap: 8px; padding: 0.8rem 1rem; }
.dot {
    width: 8px; height: 8px;
    background: #5a8fc5;
    border-radius: 50%;
    animation: thinking-dots 1.4s infinite ease-in-out both;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Dividers ── */
hr { border-color: rgba(56,100,180,0.2) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(10,18,40,0.6) !important;
    border: 1px solid rgba(56,100,180,0.25) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #8099c5 !important;
    font-size: 0.8rem !important;
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
    background: rgba(15,30,70,0.8);
    border: 1px solid rgba(56,100,180,0.3);
    color: #8099c5;
    font-size: 0.72rem;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
}
.stat-pill b { color: #c8d8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # [{role, content, sources}]
        "preview_source": None,  # Source dict currently shown in right-panel PDF viewer
        "preview_msg_idx": None, # Which message's source is being previewed
        "is_thinking": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
#  Cached Resource Loaders
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Load the unified FAISS vector store once and cache it."""
    if not API_KEY:
        st.error("GOOGLE_API_KEY not found in .env")
        st.stop()
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
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)


# ─────────────────────────────────────────────
#  PDF: Extract single page + highlight → bytes
# ─────────────────────────────────────────────
def get_highlighted_pdf_bytes(pdf_path: str, page_num: int, search_text: str) -> bytes | None:
    """
    Extract the target page and highlight relevant text from the retrieved context.
    Uses a multi-tier matching strategy to ensure the actual passage is found
    while avoiding 'dirty' highlights on common words/headers.
    """
    try:
        if not os.path.exists(pdf_path):
            return None
            
        src = fitz.open(pdf_path)
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

        pdf_bytes = out.tobytes(garbage=3, deflate=True)
        out.close()
        return pdf_bytes
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


def run_rag_query(query: str):
    """Run the unified RAG pipeline and return (answer_text, sources_list)."""
    vs = load_vector_store()
    llm = load_llm()

    docs_and_scores = vs.similarity_search_with_score(query, k=8)

    context_chunks = ""
    sources = []
    seen = set()

    for doc, score in docs_and_scores:
        cat = doc.metadata.get("category", "Unknown")
        sub_cat = doc.metadata.get("sub_category", "Unknown")
        source_file = doc.metadata.get("source_file", "Unknown")
        page_num = int(doc.metadata.get("page", 0))
        pdf_path = os.path.join(PDF_DIR, source_file)

        context_chunks += (
            f"--- Source: {source_file} (Category: {cat} | Sub-category: {sub_cat}) ---\n"
            f"{doc.page_content}\n\n"
        )

        key = f"{source_file}:{page_num}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "category": cat,
                "sub_category": sub_cat,
                "source_file": source_file,
                "pdf_path": pdf_path,
                "page": page_num,
                "content": doc.page_content,
                "score": float(score),
            })

    response = llm.invoke(LEGAL_PROMPT.format(context=context_chunks, query=query))
    return response.content, sources


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

    if os.path.exists(pdf_path):
        with st.spinner("Loading PDF..."):
            pdf_bytes = get_highlighted_pdf_bytes(pdf_path, page_num, content)

        if pdf_bytes:
            st.markdown(
                f"**Page {page_num + 1} of** `{source_file}` — "
                "matching text highlighted in **gold** ✦"
            )
            b64 = base64.b64encode(pdf_bytes).decode()
            pdf_url = f"data:application/pdf;base64,{b64}#view=FitH&pagemode=none&navpanes=0&toolbar=0"
            st.markdown(
                f"""
                <div style="height:700px; width:100%;">
                    <iframe
                        src="{pdf_url}"
                        width="100%"
                        height="700"
                        style="border:none; border-radius:14px;
                               box-shadow:0 12px 40px rgba(0,0,0,0.7);
                               background: #0d1117;"
                        title="PDF preview — {source_file}"
                    ></iframe>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Could not prepare the PDF preview.")
    else:
        st.error(f"PDF file not found:\n`{pdf_path}`")

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
          <span style="color:#5a8fc5;font-size:0.82rem">Analysing legal documents</span>
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
        st.session_state.messages = []
        st.session_state.preview_source = None
        st.session_state.preview_msg_idx = None
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Click any document source next to an AI response to view the verified PDF excerpt.")

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
